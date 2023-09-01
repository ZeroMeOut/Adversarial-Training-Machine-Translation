# Adversarial-Training-Machine-Translation
The aim of this project is to enhance the accuracy of translation models by considering contextual correctness. Many translation models often struggle with producing translations that are contextually accurate. For instance, Google Translate may translate "I speak Chinese" to "我说中文," which is incorrect.

My solution involves training a discriminator to determine the correctness of the context between two languages. Each data sample consists of pairs like {"en": "I speak Chinese", "zh": "我说中文", "context": 0}, where the 'context' flag indicates whether the translation context is incorrect (0) or correct (1).

To achieve this, I employ an adversarial approach. The generator produces translations, and the discriminator evaluates their contextual correctness. A cross-entropy loss is computed based on the discriminator's predictions to train the generator.

My project aims to improve the quality of machine translations by ensuring that they not only capture the literal translation but also consider the broader context, ultimately leading to more accurate and contextually meaningful translations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, you'll need to install the required Python libraries and dependencies. You can use the following command to install them:

```bash
pip install transformers==4.28.0 datasets evaluate torch sentencepiece tokenizers sacrebleu
```

## Usage
Use  ```generate_discriminate.ipynb``` as reference. Please note that format of the dataset should be in a DatasetDict format, with train and test Dataset.from_dict respectively.

## Training the Discriminator
To train the discriminator, you can use the following code:
```python
user_model_name = "bert-base-uncased"
discriminator = Discriminator(user_model_name, context_dataset, first_lang='en', second_lang='zh', target='context')
discriminator.train()
```
The first_lang, second_lang and target should be changed according to the context dataset. The models accepted can be seen [here](https://huggingface.co/docs/transformers/v4.31.0/en/tasks/sequence_classification#text-classification).

## Training the Generator
To train the generator, you can use the following code:
```python
from transformers import BartForConditionalGeneration, BartTokenizer

discriminator_dir = '/content/discriminator/checkpoint-2000'
user_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0)
_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

generator = Generator(dataset=opus_smol_dataset, user_model=user_model, _tokenizer=_tokenizer, discriminator=discriminator, discriminator_dir=discriminator_dir, lang='en', target='zh', num_train_epochs=2,  output_dir="generator")
```
The lang and target should be changed according to the lang dataset, and discriminator_dir to the appropriate dir. The models accepted can be seen [here](https://huggingface.co/docs/transformers/tasks/translation)

## Making Predictions
To make predictions using the trained generator, you can use the following code:
```python
generator.predict("I just got in my head a little bit.", "/content/generator/checkpoint-1000")
```
## Contributing
If you would like to contribute to this project, I welcome contributions from the community. Here's how you can get involved:

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Create a new branch for your contributions.
4. Make changes, improvements, or bug fixes.
5. Test your changes thoroughly.
6. Commit your changes and push them to your fork on GitHub.
7. Submit a pull request to the main repository.
   
Please ensure that your code adheres to any coding standards and guidelines set by the project.





