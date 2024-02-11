import os
import random
import json
from tqdm import tqdm
from copy import deepcopy


train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'

random.seed(816)

train_doc = []
test_doc = []
dev_data = []

data = json.load(open(test_file))
for l in tqdm(data):
    #print(f"---> {l['document']}")
    test_doc.append(l['document'])


data = json.load(open(train_file))

random.shuffle(data)

# dev数据集
split_n = int(len(data) * 0.1)

for n, l in tqdm(enumerate(data)):
    #print(f"---> {l['document']}")
    if (l['document'] in test_doc) or (n<split_n):
    	dev_data.append(l)

json.dump(
    dev_data,
    open("data/gen_dev.json", 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)

print("test document: ", test_doc)
print("dev_n= ", len(dev_data))
