# Adversarial-Training-Machine-Translation
The aim of this project is to enhance the accuracy of translation models by considering contextual correctness. Many translation models often struggle with producing translations that are contextually accurate. For instance, Google Translate may translate "I speak Chinese" to "我说中文," which is incorrect.

My solution involves training a discriminator to determine the correctness of the context between two languages. Each data sample consists of pairs like {"en": "I speak Chinese", "zh": "我说中文", "context": 0}, where the 'context' flag indicates whether the translation context is incorrect (0) or correct (1).

To achieve this, I employ an adversarial approach. The generator produces translations, and the discriminator evaluates their contextual correctness. A cross-entropy loss is computed based on the discriminator's predictions to train the generator.

Our project aims to improve the quality of machine translations by ensuring that they not only capture the literal translation but also consider the broader context, ultimately leading to more accurate and contextually meaningful translations.

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


