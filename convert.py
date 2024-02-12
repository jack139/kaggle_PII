import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras

from bert4keras.tokenizers import Tokenizer

import random
import json
from tqdm import tqdm
from copy import deepcopy


train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'
dev_file = 'data/gen_dev.json'

train_43k = 'data/dataset_43k.json'
train_43k_csv = 'data/dataset_43k_csv.json'
train_10k = 'data/dataset_10k.json'


random.seed(1608)

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

'''


def __convert(indata, include_blank=False):

    text = ''
    entities = []
    all_idx = 0
    start_idx = 0
    etype = ''

    text = indata['sentence']

    for n, label in enumerate(indata['BIO_label']):
        if label[0]=='O':
            if etype!='':
                entities.append({
                    "start_idx": len(''.join(text[:start_idx])),
                    "end_idx": len((''.join(text[:all_idx])).rstrip()) - 1, # rstrio() 去掉右侧的空格
                    "type": etype,
                    "entity": (''.join(text[start_idx:all_idx])).rstrip(),
                })
            start_idx = 0
            etype = ''
        elif label[0]=='B':
            if etype!='':
                entities.append({
                    "start_idx": len(''.join(text[:start_idx])),
                    "end_idx": len((''.join(text[:all_idx])).rstrip()) - 1,
                    "type": etype,
                    "entity": (''.join(text[start_idx:all_idx])).rstrip(),
                })                
            start_idx = all_idx
            etype = label.split('-')[1]
        elif label[0]=='I':
            pass
        else:
            print('unknown label: ', label)

        all_idx += 1


    # 一行text结束
    if etype!='':
        entities.append({
            "start_idx": len(''.join(text[:start_idx])),
            "end_idx": len((''.join(text[:all_idx])).rstrip()) - 1,
            "type": etype,
            "entity": (''.join(text[start_idx:all_idx])).rstrip(),
        })

    # 加入数据集
    if include_blank or len(entities)>0:
        # 检查 token 长度
        tokens = tokenizer.tokenize(''.join(text))
        if len(tokens)>512: 
            print(f"wrong length of tokens, {len(tokens)}: {''.join(text)[:50]}")

        return {
            'text' : ''.join(text),
            'entities' : entities,
        }
    else:
        return None




def assemble(infile, outfile_path, max_len, is_train, include_blank, split_ratio, extra_data):

    os.makedirs(outfile_path, exist_ok=True)

    total = text_break = tmp_break = 0
    D_train, D_dev, data_doc = [], [], []

    data = json.load(open(infile))
    for l in tqdm(data):
        #print(f"---> {l['document']}")
        data_doc.append(l['document'])

    # 拆分数据集
    split_n = int(len(data_doc) * split_ratio)
    random.shuffle(data_doc)

    # 转换数据
    for l in tqdm(data):
        #print(f"---> {l['document']}")

        total += 1

        text = []
        tmp_text = []
        n = n_text = n_tmp = 0
        while n<len(l['tokens']):
            token = l['tokens'][n].replace('ﬄ', 'ffl').replace('ﬃ', 'ffi').replace('ﬂ', 'fl').replace('ﬁ', 'fi').replace('ﬀ', 'ff')
            #    .replace('…', '.').replace('´', "'").replace('²', '2')\
            #    .replace('΅', "'").replace('¨', "'").replace(';', ';').replace('．', '.')\
            #    .replace('³', '3').replace('‑', '-').replace('¹', '1').replace('½', '1/2')\
            #    .replace('¾', '3/4').replace('¼', '1/4').replace('\xad', '-')\
            #    .replace('™', 'TM').replace('№', 'No')

            if l['trailing_whitespace'][n]:
                token += ' ' 

            token_len = len(tokenizer.tokenize(token))

            if n_tmp + token_len > max_len:
                text += deepcopy(tmp_text)
                tmp_text = []
                n_text += n_tmp
                n_tmp = 0

                #tmp_break += 1
                is_tmp_break = True

                #print(text)
                #assert False, f"tmp_text is too long: {len(text)}, {len(tmp_text)}, {len(token)}"
            else:
                is_tmp_break = True

            if n_text + n_tmp + token_len > max_len:
                assert n_text>0, f"too long: {n_text}, {n_tmp}, {max_len}"
                #print(text)
                #print('-'*20)
                dd = __convert({
                        'sentence'  : [x[0] for x in text],
                        'BIO_label' : [x[1] for x in text],
                    }, include_blank=include_blank)
                if dd:
                    dd['document'] = l['document']
                    #if not is_train:
                    dd['tokens'] = [x[0] for x in text]

                    if dd['document'] in data_doc[:split_n]:
                        D_train.append(dd)
                    else:
                        D_dev.append(dd)

                    if is_tmp_break: # 只记录有label的tmp_break
                        tmp_break += 1
                        if len(dd['entities'])>0 and dd['entities'][-1]['end_idx']==len(dd['text'])-1:
                            print('is_tmp_break in the tail: ', dd['text'][:50])

                text = []
                n_text = 0

                text_break += 1

            if is_train:
                tmp_text += [(token, l['labels'][n])] # token, label
            else:
                tmp_text += [(token, 'O')] # token, blank-label
            #n_tmp += 1
            n_tmp += token_len

            if token=='\n\n':
                text += deepcopy(tmp_text)
                tmp_text = []
                n_text += n_tmp
                n_tmp = 0

            n += 1

        if n_text + n_tmp > 0:
            text += deepcopy(tmp_text)
            n_text += n_tmp
            #print(text)
            #print('-'*20)
            dd = __convert({
                    'sentence'  : [x[0] for x in text],
                    'BIO_label' : [x[1] for x in text],
                }, include_blank=include_blank)
            if dd:
                dd['document'] = l['document']
                #if not is_train:
                dd['tokens'] = [x[0] for x in text]

                if dd['document'] in data_doc[:split_n]:
                    D_train.append(dd)
                else:
                    D_dev.append(dd)

                if is_tmp_break: # 只记录有label的tmp_break
                    tmp_break += 1
                    if len(dd['entities'])>0 and dd['entities'][-1]['end_idx']==len(dd['text'])-1:
                        print(dd['text'][:30])

            text_break += 1

        #break # for test

    blank = 0
    for x in D_train+D_dev:
        if len(x['entities'])==0:
            blank += 1

    print(f"total= {total}\ttext_break= {text_break}\ttmp_break= {tmp_break}\tblank= {blank}")

    if is_train:
        # 增加外部数据
        if extra_data:
            data_43k = json.load(open(train_43k))
            data_43k_csv = json.load(open(train_43k_csv))
            data_10k = json.load(open(train_10k))        

            data_more = data_43k + data_43k_csv + data_10k
            data_more += D_train
            
            random.shuffle(data_more)

            json.dump(
                data_more,
                open(os.path.join(outfile_path, "train_more.json"), 'w', encoding='utf-8'),
                indent=4,
                ensure_ascii=False
            )

            print(f"train_more set: {len(data_more)}")

        random.shuffle(D_train)
        random.shuffle(D_dev)

        json.dump(
            D_train,
            open(os.path.join(outfile_path, "train.json"), 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

        json.dump(
            D_dev,
            open(os.path.join(outfile_path, "dev.json"), 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

        print(f"train set: {len(D_train)}\tdev set: {len(D_dev)}")

    else:
        json.dump(
            D_train + D_dev,
            open(os.path.join(outfile_path, "test.json"), 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

        print(f"test set: {len(D_train)+len(D_dev)}")

if __name__ == '__main__':
    # 随机拆分 train 和 dev
    assemble(train_file, 'data/wo_blank', max_len=500, include_blank=False, is_train=True, split_ratio=0.8, extra_data=True)
    assemble(train_file, 'data/w_blank', max_len=500, include_blank=True, is_train=True, split_ratio=0.9, extra_data=False)
    # 生成测试数据 test
    assemble(test_file, 'data', max_len=500, include_blank=True, is_train=False, split_ratio=1, extra_data=False)



'''
test 先验结果：

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