import numpy as np
import evaluate
import torch
from torch import nn
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,TrainingArguments, Trainer


class CustomTrainerWithDiscriminator(Trainer):
    def __init__(self, discriminator, model, args, train_dataset=None, eval_dataset=None, data_collator=None, compute_metrics=None):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, compute_metrics=compute_metrics)
        self.discriminator = discriminator

    def compute_loss(self, model, inputs, return_outputs=False):
        # Use the discriminator to get predictions
        discriminator_predictions = self.discriminator.predict(inputs)  # Assuming the Discriminator has a predict method

        # Convert 0s and 1s to class indices (0 for real, 1 for fake)
        labels = torch.tensor(discriminator_predictions, dtype=torch.long, device=model.device)

        # Forward pass through the model
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

## Assuming the dataset is a json in the format [{lang:" ", target:" "}, {lang:" ", target:" "},...]
## Also assuming that the user_model is a vaild model for text generation
class Generator():
    def __init__(self, user_model: str, dataset: Dict[str, Any], discriminator, output_dir="discriminator", learning_rate=2e-5,
                 per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=2, weight_decay=0.01,
                 evaluation_strategy="epoch", save_strategy="epoch", split=0.3,):
      
        # Main stuff
        self.discriminator = discriminator
        self.dataset = dataset
        self._tokenizer = AutoTokenizer.from_pretrained(user_model, truncation=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(user_model, num_labels=1)
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
            lang = list(item.values())[0]
            target = list(item.values())[1]

            encoded = self._tokenizer(
                lang,
                padding="max_length",
                truncation=True
            )

            data_dict = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': target
            }
            encoded_data_with_labels.append(data_dict)

        return encoded_data_with_labels

    ## Training
    def train(self):
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

      ## Create eval and train datasets from the encoded data with labels
      len_of_encoded = len(self.encoded_data_with_labels)
      ratio = int(len_of_encoded * self.split)
      train_dataset = self.encoded_data_with_labels[0:ratio]
      eval_dataset = self.encoded_data_with_labels[ratio:len_of_encoded]

      ## Create a Trainer and train the model
      trainer = CustomTrainerWithDiscriminator(
          discriminator=self.discriminator,
          model=self.model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          data_collator=self.data_collator,
      )

      trainer.train()
    
    ## Prediction
    def predict(self, text):
        inputs = self._tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=50, num_return_sequences=1)
            generated_text = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text