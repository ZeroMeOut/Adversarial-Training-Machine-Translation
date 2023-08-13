import json

with open('dataset\dataset-context.json') as file:
     data = json.load(file)

with open('dataset\dataset-context-10k.json', 'w') as f:
    json.dump(data[0:10000], f)
    