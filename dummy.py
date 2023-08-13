import json

with open('dataset\dataset-wrong-context-10k.json') as file:
     data = json.load(file)

print(data[9998])
    