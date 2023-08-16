import json
from time import sleep
from googletrans import Translator

# with open('dataset\dataset.json') as file:
#     data = json.load(file)


# for index, values in enumerate(data):
#     data[index]['context'] = '1'

# with open('dataset-context.json', 'w') as f:
#     json.dump(data, f)

# with open('dataset\dataset2.json') as file:
#      data = json.load(file)

# translator = Translator()

# target = 99 + 100
# i = 0
# j = 99
# while j != target:
#     for index, values in enumerate(data[i:j]):
#         translation = translator.translate(data[index]['en'], dest='zh-cn')
#         data[index]['zh'] = translation.text
#         data[index]['context'] = '0'
#     i = j + 1
#     j = j + 100
#     sleep(60)

# It would be best to use another method to find the wrong context but eh, it works

