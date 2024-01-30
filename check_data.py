'''
import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras
from bert4keras.tokenizers import Tokenizer

dict_path = '../nlp_model/bert_wwm_uncased_L-24_H-1024_A-16/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
'''

import json

train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'
file_43k = 'data/external/english_pii_43k.jsonl'




def load_data(filename, text_name='full_text'):
    """加载数据
    单条格式：(文本, 标签id)
    """
    max_len = 0    
    max_cnt = 0
    total = 0

    #data = json.load(open(filename))

    with open(filename) as f:
        for l in f:
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

    return max_len, max_cnt, total


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


if __name__ == '__main__':

    #assemble(train_file)
    #assemble(test_file)

    load_data(file_43k, "tokenised_text")
    #load_data(file_43k, "unmasked_text")

