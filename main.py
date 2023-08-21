import json
from viacontext.discriminator import Discriminator
from viacontext.generator import Generator

## Create two dummy datasets

dataset_for_discriminator = [
    {
        "first_lang": "Hello",
        "second_lang": "Bonjour",
        "context": 1
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
        "context": 0
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
        "context": 0
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
        "context": 0
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
        "context": 0
    }
]

dataset_for_generator = [
    {
        "first_lang": "Hello",
        "second_lang": "Bonjour",
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
    },
    {
        "first_lang": "Hi",
        "second_lang": "Salut",
    }
]