from datasets import load_dataset
import json


dataset = load_dataset("wmt18", name="zh-en") 

# with open('dataset.json', 'w') as f:
#     json.dump(dataset['train']['translation'][0:500000], f)

# with open('dataset2.json', 'w') as f:
#     json.dump(dataset['train']['translation'][500001:1000000], f)