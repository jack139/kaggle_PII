
import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras

from bert4keras.tokenizers import Tokenizer

import random
import json
import pandas as pd
from tqdm import tqdm
from copy import deepcopy


# 来源: https://huggingface.co/datasets/ai4privacy/pii-masking-65k/blob/main/english_balanced_10k.jsonl

file_10k = 'data/external/english_balanced_10k.jsonl'


dict_path = '../nlp_model/bert_wwm_uncased_L-24_H-1024_A-16/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

'''
  {
    "text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。",
    "entities": [
      {
        "start_idx": 3,
        "end_idx": 9,
        "type": "bod",
        "entity": "SARST细胞"
      },
      {
        "start_idx": 19,
        "end_idx": 24,
        "type": "dis",
        "entity": "成人SARS"
      }
    ]
  },

10k_label:

['EMAIL', 'SEX', 'ACCOUNTNUMBER', 'BIC', 'DATE', 'MIDDLENAME', 'IPV4', 'TIME', 'USERNAME', 
'CURRENCYCODE', 'STREETADDRESS', 'JOBTITLE', 'SECONDARYADDRESS', 'STATE', 'PHONE_NUMBER', 
'JOBAREA', 'CURRENCY', 'LASTNAME', 'IPV6', 'VEHICLEVIN', 'CREDITCARDNUMBER', 'PHONEIMEI', 
'URL', 'USERAGENT', 'IBAN', 'MAC', 'FULLNAME', 'PREFIX', 'CREDITCARDISSUER', 'ACCOUNTNAME', 
'MASKEDNUMBER', 'CREDITCARDCVV', 'VEHICLEVRM', 'LITECOINADDRESS', 'SSN', 'STREET', 'AMOUNT', 
'CURRENCYSYMBOL', 'JOBDESCRIPTOR', 'ZIPCODE', 'PIN', 'JOBTYPE', 'IP', 'BITCOINADDRESS', 'COUNTY', 
'COMPANY_NAME', 'CITY', 'CURRENCYNAME', 'FIRSTNAME', 'PASSWORD', 'ETHEREUMADDRESS', 
'BUILDINGNUMBER', 'SUFFIX']


labels 映射：

'EMAIL' --> EMAIL
'ID_NUM' --> PIN, ACCOUNTNUMBER, CREDITCARDNUMBER, MASKEDNUMBER
'NAME_STUDENT' --> LASTNAME, MIDDLENAME, FIRSTNAME, FULLNAME
'PHONE_NUM' --> PHONE_NUMBER
'STREET_ADDRESS' --> SECONDARYADDRESS, STREET, BUILDINGNUMBER, CITY, STATE, ZIPCODE, COUNTY, STREETADDRESS
'URL_PERSONAL' --> URL
'USERNAME' --> USERNAME

'''

labels_to = {
    'EMAIL' : ['EMAIL'],
    'ID_NUM' : ['PIN', 'ACCOUNTNUMBER', 'CREDITCARDNUMBER', 'MASKEDNUMBER'],
    'NAME_STUDENT' : ['LASTNAME', 'MIDDLENAME', 'FIRSTNAME', 'FULLNAME'],
    'PHONE_NUM' : ['PHONE_NUMBER'],
    'STREET_ADDRESS' : ['SECONDARYADDRESS', 'STREET', 'BUILDINGNUMBER', 'CITY', 'STATE', 'ZIPCODE', 'COUNTY', 'STREETADDRESS'],
    'URL_PERSONAL' : ['URL'],
    'USERNAME' : ['USERNAME'],
}

to_labels = {}

for x in labels_to.keys():
    for y in labels_to[x]:
        to_labels[y] = x


def assemble(infile, outfile_path, max_len=500, include_blank=False):
    total = 0
    D = []
    L = set()

    with open(infile) as f:
        for l in tqdm(f):
            l = json.loads(l)

            #l['unmasked_text'] = l['Filled Template'].replace('₨', '$').replace('´', "'").replace('﷼', '$').replace('…', '.')

            total += 1

            if len(l['token_entity_labels'])>max_len:
                continue # 放弃大于 max_len 的

            tokens = tokenizer.tokenize(l['unmasked_text'])
            assert tokens[1:-1]==l['tokenised_unmasked_text'], f"tokens are not EQUAL! \n{l['unmasked_text']}\n{tokens[1:-1]}\n{l['tokenised_unmasked_text']}"

            entities = []

            mapping = tokenizer.rematch(l['unmasked_text'], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}


            for start_idx, end_idx, label in zip(start_mapping, end_mapping, l['token_entity_labels']):

                if label=="O":
                    continue

                _label = label.split('-')[1]
                if _label not in to_labels.keys():
                    continue

                # 转换 label
                _label = to_labels[_label]
                L.add(_label)

                entities.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "type": _label,
                    "entity": l['unmasked_text'][start_idx:end_idx+1],
                })

            if include_blank or len(entities)>0:
                D.append({
                    'text' : l['unmasked_text'],
                    'entities' : entities,
                })

            #break # for test

    # 处理 entities
    for d in D:
        d['entities'] = sorted(d['entities'], key=lambda x: x['start_idx'], reverse=False)
        new_e = []
        last_label = None
        last_end = -100
        last_entity = None
        for e in d['entities']:
            if e['type'] != last_label:
                if last_entity is not None:
                    new_e.append(last_entity)
                last_entity = deepcopy(e)
                last_label = e['type']
                last_end = e['end_idx']
            else:
                if last_end == e['start_idx'] - 1:
                    last_entity['end_idx'] = last_end = e['end_idx']
                    last_entity['entity'] += e['entity']
                elif last_end == e['start_idx'] - 2:
                    last_entity['end_idx'] = last_end = e['end_idx']
                    last_entity['entity'] += (' ' + e['entity'])
                elif last_label=='STREET_ADDRESS' and d['text'][last_end+1]==',' and last_end == e['start_idx'] - 3:
                    last_entity['end_idx'] = last_end = e['end_idx']
                    last_entity['entity'] += (', ' + e['entity'])
                else:
                    new_e.append(last_entity)
                    last_entity = deepcopy(e)
                    last_end = e['end_idx']

        new_e.append(last_entity)
        d['entities'] = []
        for x in new_e:
            if x['type']=='STREET_ADDRESS' and x['entity'].isdigit():
                continue
            elif x['type']=='STREET_ADDRESS' and len(x['entity'].split())<4:
                continue
            else:
                d['entities'].append(x)

    D = [d for d in D if len(d['entities'])>0]

    print(list(L))

    print(f"total= {total}\t D= {len(D)}")

    json.dump(
        D,
        open(os.path.join(outfile_path, "dataset_10k.json"), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

if __name__ == '__main__':
    assemble(file_10k, 'data', max_len=250)
