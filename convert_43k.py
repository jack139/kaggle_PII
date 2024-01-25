import os
import random
import json
from tqdm import tqdm
from copy import deepcopy

file_43k = 'data/external/english_pii_43k.jsonl'


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

43k_label:

['IBAN', 'BITCOINADDRESS', 'COMPANYNAME', 'COUNTY', 'EYECOLOR', 'IPV4', 'PIN', 'DATE', 'USERNAME', 
 'CURRENCY', 'CREDITCARDCVV', 'ETHEREUMADDRESS', 'BIC', 'VEHICLEVIN', 'PHONENUMBER', 'GENDER', 
 'LASTNAME', 'USERAGENT', 'IPV6', 'LITECOINADDRESS', 'MASKEDNUMBER', 'NEARBYGPSCOORDINATE', 
 'AMOUNT', 'SEX', 'BUILDINGNUMBER', 'PHONEIMEI', 'CREDITCARDNUMBER', 'SSN', 'FIRSTNAME', 
 'ZIPCODE', 'EMAIL', 'AGE', 'VEHICLEVRM', 'STATE', 'CURRENCYSYMBOL', 'PASSWORD', 'TIME', 
 'CURRENCYCODE', 'JOBTYPE', 'HEIGHT', 'CURRENCYNAME', 'SECONDARYADDRESS', 'ORDINALDIRECTION', 
 'JOBAREA', 'ACCOUNTNUMBER', 'MIDDLENAME', 'ACCOUNTNAME', 'JOBTITLE', 'IP', 'DOB', 'STREET', 
 'CREDITCARDISSUER', 'MAC', 'PREFIX', 'URL', 'CITY']

labels 映射：

'EMAIL' --> EMAIL
'ID_NUM' --> PIN, ACCOUNTNUMBER, SSN
'NAME_STUDENT' --> LASTNAME, MIDDLENAME, FIRSTNAME
'PHONE_NUM' --> PHONENUMBER
'STREET_ADDRESS' --> SECONDARYADDRESS, STREET, BUILDINGNUMBER, CITY, STATE, ZIPCODE
'URL_PERSONAL' --> URL
'USERNAME' --> USERNAME

'''

labels_to = {
    'EMAIL' : ['EMAIL'],
    'ID_NUM' : ['PIN', 'ACCOUNTNUMBER', 'SSN'],
    'NAME_STUDENT' : ['LASTNAME', 'MIDDLENAME', 'FIRSTNAME'],
    'PHONE_NUM' : ['PHONENUMBER'],
    'STREET_ADDRESS' : ['SECONDARYADDRESS', 'STREET', 'BUILDINGNUMBER', 'CITY', 'STATE', 'ZIPCODE'],
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

    #data = json.load(open(infile))
    with open(infile) as f:
        for l in tqdm(f):
            l = json.loads(l)

            #print(f"---> {l['document']}")

            total += 1

            if len(l['tokenised_text'])>max_len:
                continue # 放弃大于 max_len 的

            entities = []

            for span_label in json.loads(l['span_labels'].replace("'",'"')):
                start_idx, end_idx, label = span_label

                if label=="O":
                    continue

                _label = label.split('_')[0]
                if _label not in to_labels.keys():
                    continue

                # 转换 label
                _label = to_labels[_label]
                L.add(_label)

                entities.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx - 1,
                    "type": _label,
                    "entity": l['unmasked_text'][start_idx:end_idx],
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
                if last_end == e['start_idx'] - 2:
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
            else:
                d['entities'].append(x)

    D = [d for d in D if len(d['entities'])>0]

    print(list(L))

    print(f"total= {total}\t D= {len(D)}")

    json.dump(
        D,
        open(os.path.join(outfile_path, "dataset_43k.json"), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

if __name__ == '__main__':
    assemble(file_43k, 'data')
