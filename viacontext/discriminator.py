import numpy as np
import evaluate
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,TrainingArguments, Trainer

## Assuming the dataset is a json in the format [{first_lang:" ", second_lang:" ", context:int 1 or 0}, {first_lang:" ", second_lang:" ", context:int 1 or 0},...]
## Also assuming that the user_model is a vaild model for classification
class Discriminator():
    def __init__(self, user_model: str, dataset: Dict[str, Any], output_dir="discriminator", learning_rate=2e-5,
                 per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=2, weight_decay=0.01,
                 evaluation_strategy="epoch", save_strategy="epoch", split=0.3,):
      
        # main stuff
        self.dataset = dataset
        self._tokenizer = AutoTokenizer.from_pretrained(user_model, truncation=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(user_model, num_labels=2)
        self.data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        self.encoded_data_with_labels = self._encode_data_with_labels()

        # Args blablabla
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.split = split


    ## Encode the dataset let's goooo
    def _encode_data_with_labels(self):
        encoded_data_with_labels = []
        for item in self.dataset:
            first_lang = list(item.values())[0]
            second_lang = list(item.values())[1]
            context = list(item.values())[2]

            encoded = self._tokenizer(
                first_lang,
                second_lang,
                padding="max_length",
                truncation=True
            )

            data_dict = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': context
            }
            encoded_data_with_labels.append(data_dict)

        return encoded_data_with_labels

    ## Training 
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
      len_of_encoded = len(self.encoded_data_with_labels)
      ratio = int(len_of_encoded * self.split)
      train_dataset = self.encoded_data_with_labels[0:ratio]
      eval_dataset = self.encoded_data_with_labels[ratio:len_of_encoded]

      # Create a Trainer and train the model
      trainer = Trainer(
          model=self.model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          data_collator=self.data_collator,
          compute_metrics=compute_metrics,
      )

      trainer.train()
    
    ## Prediction, retuns a tensor
    def predict(self, text1, text2):
        inputs = self._tokenizer(text1, text2, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
        return predictions
   
    

