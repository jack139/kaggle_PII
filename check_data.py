'''
import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras
from bert4keras.tokenizers import Tokenizer

dict_path = '../nlp_model/bert_wwm_uncased_L-24_H-1024_A-16/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
'''

import json
import pandas as pd
from tqdm import tqdm

train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'
file_43k = 'data/external/english_pii_43k.jsonl'
file_43k_csv = 'data/external/PII43k.csv'
file_10k = 'data/external/english_balanced_10k.jsonl'



def check_data(filename):
    n1 = n2 = 0
    doc = {}

    data = json.load(open(filename))
    for l in data:
        if len(l['entities'])>0:
            if l['entities'][0]['start_idx']==0:
                n1 += 1
                print(1, l.get('document'), l['text'][:30])
                if l.get('document'):
                    if l.get('document') in doc.keys():
                        doc[l.get('document')] += 1
                    else:
                        doc[l.get('document')] = 1
            if l['entities'][-1]['end_idx']==len(l['text'])-1:
                n2 += 1
                print(2, l.get('document'), l['text'][:30])
                if l.get('document'):
                    if l.get('document') in doc.keys():
                        doc[l.get('document')] += 1
                    else:
                        doc[l.get('document')] = 1

            for x in l['entities']:
                if "\n\n" in l['text'][x['start_idx']:x['end_idx']+1]:
                    print(3, l['text'][:30])

    print(f"n1={n1} n2={n2} ")
    for x in doc.keys():
        if doc[x]>1:
            print(x, doc[x])
    

def load_data(filename, text_name='full_text'):
    """加载数据
    单条格式：(文本, 标签id)
    """
    max_len = 0    
    max_cnt = 0
    total = 0

    labels = set()

    #data = json.load(open(filename))

    with open(filename) as f:
        for l in tqdm(f):
            l = json.loads(l)

            total += 1

            if len(l[text_name])>250:
                print(len(l[text_name]), l[text_name][:10])
                max_cnt += 1
            if len(l[text_name])>max_len:
                max_len = len(l[text_name])

            '''
            tokens = tokenizer.tokenize(l[text_name])
            if len(tokens)>500:
                print(len(tokens), l[text_name][:10])
                max_cnt += 1
            if len(tokens)>max_len:
                max_len = len(tokens)
            '''

            for x in l['token_entity_labels']:
                if x=='O':
                    continue
                labels.add(x.split('-')[1])


    return max_len, max_cnt, total, list(labels)


def assemble(filename, max_len=500):
    total = text_break = tmp_break = 0

    data = json.load(open(filename))
    for l in data:
        print(f"---> {l['document']}")

        total += 1

        text = ''
        tmp_text = ''
        n = n_text = n_tmp = 0
        while n<len(l['tokens']):
            token = l['tokens'][n]
            if l['trailing_whitespace'][n]:
                token += ' ' 

            if n_tmp + 1 > max_len:
                text += tmp_text
                tmp_text = ''
                n_text += n_tmp
                n_tmp = 0

                tmp_break += 1

                #print(text)
                #assert False, f"tmp_text is too long: {len(text)}, {len(tmp_text)}, {len(token)}"
                

            if n_text + n_tmp + 1 > max_len:
                assert len(text)>0, f"too long: {n_text}, {n_tmp}"
                print(text)
                print('-'*20)
                text = ''
                n_text = 0

                text_break += 1

            tmp_text += token
            n_tmp += 1

            if token=='\n\n':
                text += tmp_text
                tmp_text = ''
                n_text += n_tmp
                n_tmp = 0

            n += 1

        if n_text + n_tmp > 0:
            text += tmp_text
            n_text += n_tmp
            print(text)
            print('-'*20)

            text_break += 1            

        break # for test


    print(f"total= {total}\ttext_break= {text_break}\ttmp_break= {tmp_break}")


def load_data_csv(filename, text_name='Tokenised Filled Template'):
    """加载数据
    单条格式：(文本, 标签id)
    """
    max_len = 0    
    max_cnt = 0
    total = 0

    labels = set()

    pd_data = pd.read_csv(filename)

    for index, l in tqdm(pd_data.iterrows()):

        total += 1

        #print(l[text_name])
        tokenized = eval(l[text_name])
        if len(tokenized)>250:
            print(len(tokenized), tokenized[:10])
            max_cnt += 1
        if len(tokenized)>max_len:
            max_len = len(tokenized)

        tokens = eval(l['Tokens'])
        for x in tokens:
            if x=='O':
                continue
            labels.add(x.split('-')[1])

    return max_len, max_cnt, total, list(labels)


if __name__ == '__main__':

    #assemble(train_file)
    #assemble(test_file)

    #load_data(file_43k, "tokenised_text")
    #load_data(file_43k, "unmasked_text")

    #check_data('data/dataset_43k.json')
    check_data('data/train.json')
    #check_data('data/dev.json')
    #check_data('data/train_43k.json')

    #print(load_data_csv(file_43k_csv))

    #print(load_data(file_10k, "tokenised_unmasked_text"))