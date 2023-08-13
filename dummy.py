import json

with open('dataset\dataset-context-10k.json') as file:
     data1 = json.load(file)

with open('dataset\dataset-wrong-context-10k.json') as file:
     data2 = json.load(file)

def merge_json_files(file_paths):
    merged_contents = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file_in:
            merged_contents.extend(json.load(file_in))

    with open('dataset\dataset-context-final.json', 'w', encoding='utf-8') as file_out:
        json.dump(merged_contents, file_out)

paths = [
    'dataset\dataset-context-10k.json', # I made these in the the wmt8 dataset
    'dataset\dataset-wrong-context-10k.json'
]

merge_json_files(paths)