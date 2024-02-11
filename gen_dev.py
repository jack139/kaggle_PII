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

for x in dev_data:
	x['submission'] = []
	for n, i in enumerate(x['labels']):
		if i!='O':
			x['submission'].append((n, i))

	if x['document'] in test_doc:
		print('\n'.join([f"{x['document']}, {i[0]}, {i[1]}" for i in x['submission']]))

json.dump(
    dev_data,
    open("data/gen_dev.json", 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)

print("test document: ", test_doc)
print("dev_n= ", len(dev_data))


'''
7, 9, B-NAME_STUDENT
7, 10, I-NAME_STUDENT
7, 482, B-NAME_STUDENT
7, 483, I-NAME_STUDENT
7, 741, B-NAME_STUDENT
7, 742, I-NAME_STUDENT

10, 0, B-NAME_STUDENT
10, 1, I-NAME_STUDENT
10, 464, B-NAME_STUDENT
10, 465, I-NAME_STUDENT

16, 4, B-NAME_STUDENT
16, 5, I-NAME_STUDENT

20, 5, B-NAME_STUDENT
20, 6, I-NAME_STUDENT

56, 12, B-NAME_STUDENT
56, 13, I-NAME_STUDENT

86, 6, B-NAME_STUDENT
86, 7, I-NAME_STUDENT

93, 0, B-NAME_STUDENT
93, 1, I-NAME_STUDENT

104, 8, B-NAME_STUDENT
104, 9, I-NAME_STUDENT

112, 5, B-NAME_STUDENT
112, 6, I-NAME_STUDENT

123, 32, B-NAME_STUDENT
123, 33, I-NAME_STUDENT
'''