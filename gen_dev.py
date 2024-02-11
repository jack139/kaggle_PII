import os
import json
from tqdm import tqdm
from copy import deepcopy


train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'

test_doc = []

dev_data = []

data = json.load(open(test_file))
for l in tqdm(data):
    #print(f"---> {l['document']}")
    test_doc.append(l['document'])


data = json.load(open(train_file))
for l in tqdm(data):
    #print(f"---> {l['document']}")
    if l['document'] in test_doc:
    	dev_data.append(l)

json.dump(
    dev_data,
    open("data/gen_dev.json", 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)

