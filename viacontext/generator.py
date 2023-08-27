import numpy as np
import evaluate
import torch
from torch import nn
from typing import Dict, Any
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
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
        metric = evaluate.load("sacrebleu")
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

        # Compute BLEU score
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred !=  self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return loss, result



## Assuming the dataset is a json in the format [{lang:" ", target:" "}, {lang:" ", target:" "},...] in a DataDict
## Also assuming that the user_model is a vaild model for text generation
class Generator():
    def __init__(self, user_model=None, _tokenizer=None, dataset=None, discriminator=None, lang='lang', target='target', output_dir="generator", learning_rate=2e-5,
                 per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=2, weight_decay=0.01,
                 evaluation_strategy="epoch", save_strategy="epoch", split=0.3,):

        # main stuff
        if user_model is None:
            user_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        self.user_model = user_model

        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        self._tokenizer = _tokenizer

        self.dataset = dataset
        self.discriminator = discriminator

        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self._tokenizer, model=self.user_model)


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
          model_inputs = self._tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding=True, return_tensors="pt")
          return model_inputs

        token_dataset = self.dataset.map(preprocess_function, batched=True)

        token_train = token_dataset['train']
        token_eval = token_dataset['test']

        return token_train, token_eval


    # Training
    def train(self):

      training_args = Seq2SeqTrainingArguments(
          output_dir = self.output_dir,
          evaluation_strategy = self.evaluation_strategy,
          save_strategy = self.save_strategy,
          learning_rate = self.learning_rate,
          per_device_train_batch_size= self.per_device_train_batch_size,
          per_device_eval_batch_size= self.per_device_eval_batch_size,
          weight_decay = self.weight_decay,
          save_total_limit = 3,
          num_train_epochs = self.num_train_epochs,
          predict_with_generate=True,
          load_best_model_at_end = True,
      )

      token_train, token_eval = self._model_inputs()

      # Create a Trainer and train the model
      trainer = CustomTrainerWithDiscriminator(
          discriminator=self.discriminator,
          tokenizer=self._tokenizer,
          model=self.user_model,
          args=training_args,
          train_dataset=token_train['train'],
          eval_dataset=token_eval['eval'],
          data_collator=self.data_collator,
       )

      trainer.train()

    ## Prediction
    def predict(self, text):
        inputs = self._tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.user_model.device) for k, v in inputs.items()}  # Move inputs to the same device as the model


        generated_ids = self.user_model.generate(**inputs, max_length=50, num_return_sequences=1)
        generated_text = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
