import numpy as np
import evaluate
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,TrainingArguments, Trainer

## Assuming the dataset is a json in the format [{first_lang:" ", second_lang:" ", context:" "}, {first_lang:" ", second_lang:" ", context:" "},...] in a DataDict
## Also assuming that the user_model is a vaild model for classification
class Discriminator():
    def __init__(self, user_model: str, dataset=None, output_dir="discriminator", first_lang='first_lang', second_lang='second_lang', target='target', learning_rate=2e-5,
                 per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=2, weight_decay=0.01,
                 evaluation_strategy="epoch", save_strategy="epoch",):

        # main stuff
        self.dataset = dataset
        self._tokenizer = AutoTokenizer.from_pretrained(user_model, truncation=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(user_model, num_labels=2)
        self.data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

        # Args blablabla
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.first_lang = first_lang
        self.second_lang = second_lang
        self.target = target



    ## Encode the dataset let's goooo
    def _model_inputs(self):

        lang1 = self.first_lang
        lang2 = self.second_lang
        target = self.target

        def preprocess_function(examples):
          inputs = [example[lang1] + ' ' + example[lang2] for example in examples['translation']]
          labels = [int(example[target]) for example in examples['translation']]

          model_inputs = self._tokenizer(inputs, padding="max_length", truncation=True)
          model_inputs['labels'] = labels
          model_inputs['text'] = inputs
          return model_inputs

        token_dataset = self.dataset.map(preprocess_function, batched=True)

        token_train = token_dataset['train']
        token_eval = token_dataset['test']

        return token_train, token_eval

    #Training
    def train(self):

      accuracy = evaluate.load("accuracy")
      def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


      # Prepare the training arguments
      training_args = TrainingArguments(
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

      # Create eval and train datasets from the encoded data with labels
      token_train, token_eval = self._model_inputs()

      # Create a Trainer and train the model
      trainer = Trainer(
          model=self.model,
          args=training_args,
          train_dataset=token_train,
          eval_dataset=token_eval,
          data_collator=self.data_collator,
          compute_metrics=compute_metrics,
      )

      trainer.train()

    ## Prediction
    def predict(self, text1, text2, trained_model_dir=None):
        if trained_model_dir == None:
          trained_model_dir = self.output_dir
        # Initialize the tokenizer and model
        tokenizer = self._tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(trained_model_dir)

        # Tokenize the input text
        inputs = tokenizer(text1, text2, padding="max_length", truncation=True, return_tensors="pt")

        # Ensure the model is in evaluation mode
        model.eval()

        # Perform the prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

        return predictions

