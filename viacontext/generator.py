import numpy as np
import evaluate
import torch
from torch import nn
from typing import Dict, Any
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import DatasetDict, Dataset


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

class CustomTrainerWithDiscriminator(Seq2SeqTrainer):
    def __init__(self, discriminator, tokenizer, model, args, train_dataset=None, eval_dataset=None, data_collator=None, compute_metrics=None):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, compute_metrics=compute_metrics)
        self.discriminator = discriminator
        self.tokenizer = tokenizer


    def compute_metrics(self, inputs, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        discriminator_predictions = self.discriminator.predict(decoded_labels, decoded_preds)
        labels = torch.tensor(discriminator_predictions, dtype=torch.long, device=self.model.device)

        outputs = self.model(**inputs)
        logits = outputs.logits

        # Compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return loss



## Assuming the dataset is a json in the format [{lang:" ", target:" "}, {lang:" ", target:" "},...]
## Also assuming that the user_model is a vaild model for text generation
class Generator():
    def __init__(self, user_model=None, _tokenizer=None, dataset=None, discriminator=None, lang='lang', target='target', output_dir="discriminator", learning_rate=2e-5,
                 per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=2, weight_decay=0.01,
                 evaluation_strategy="epoch", save_strategy="epoch", split=0.3,):

        # main stuff
        if user_model is None:
            user_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.user_model = user_model

        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self._tokenizer = _tokenizer

        self.dataset = dataset
        self.discriminator = discriminator

        self.data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)


        # Args blablabla
        self.lang = lang
        self.target = target
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.split = split


    ## Function for preprocessing
    def _model_inputs(self):

        source_lang = self.lang 
        target_lang = self.target 

        def preprocess_function(examples):
          inputs = [example[source_lang] for example in examples['translation']]
          targets = [example[target_lang] for example in examples['translation']]
          model_inputs = self._tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
          return model_inputs

        len_of_dataset = len(self.dataset)
        ratio = int(len_of_dataset * self.split)
        temp_train = self.dataset[0:ratio]
        temp_eval = self.dataset[ratio:len_of_dataset]

        train_id = [str(index) for index, value in enumerate(temp_train)]
        eval_id = [str(index) for index, value in enumerate(temp_eval)]

        train_dataset = {
                          'id':train_id,
                          'translation':temp_train
                      }
        eval_dataset = {
                  'id':eval_id,
                  'translation':temp_eval
              }

        train_dataset = DatasetDict({"train": Dataset.from_dict(train_dataset)})
        eval_dataset = DatasetDict({"eval": Dataset.from_dict(eval_dataset)})

        token_train = train_dataset.map(preprocess_function, batched=True)
        token_eval = eval_dataset.map(preprocess_function, batched=True)

        return token_train, token_eval


    # Training
    def train(self):

      def compute_metrics(eval_preds):
        metric = evaluate.load("sacrebleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self._tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self._tokenizer.pad_token_id)
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self._tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
        
      # Prepare the training arguments
      training_args = Seq2SeqTrainingArguments(
      output_dir = self.output_dir,
      learning_rate = self.learning_rate,
      per_device_train_batch_size = self.per_device_train_batch_size,
      per_device_eval_batch_size = self.per_device_eval_batch_size,
      num_train_epochs = self.num_train_epochs,
      weight_decay = self.weight_decay,
      evaluation_strategy = self.evaluation_strategy,
      save_strategy = self.save_strategy,
      load_best_model_at_end = True,
      )

      token_train, token_eval = self._model_inputs()

      # Create a Trainer and train the model
      # trainer = CustomTrainerWithDiscriminator(
      #     discriminator=self.discriminator,
      #     tokenizer=self._tokenizer,
      #     model=self.user_model,
      #     args=training_args,
      #     train_dataset=token_train['train'],
      #     eval_dataset=token_eval['eval'],
      #     data_collator=self.data_collator,
      #  )

      trainer = Seq2SeqTrainer(
          model=self.user_model,
          args=training_args,
          train_dataset=token_train['train'],
          eval_dataset=token_eval['eval'],
          tokenizer=self._tokenizer,
          data_collator=self.data_collator,
          compute_metrics=compute_metrics,
      )

      trainer.train()

    ## Prediction
    def predict(self, text):
        inputs = self._tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.user_model.generate(**inputs, max_length=50, num_return_sequences=1)
            generated_text = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
