import os
import random
import json
from tqdm import tqdm
from copy import deepcopy

train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'

split_ratio = 0.9

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


def __convert(indata, include_blank=True):

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
                    "end_idx": len(''.join(text[:all_idx])) - 1,
                    "type": etype,
                    "entity": ''.join(text[start_idx:all_idx]),
                })
            start_idx = 0
            etype = ''
        elif label[0]=='B':
            if etype!='':
                entities.append({
                    "start_idx": len(''.join(text[:start_idx])),
                    "end_idx": len(''.join(text[:all_idx])) - 1,
                    "type": etype,
                    "entity": ''.join(text[start_idx:all_idx]),
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
            "end_idx": len(''.join(text[:all_idx])) - 1,
            "type": etype,
            "entity": ''.join(text[start_idx:all_idx]),
        })

    # 加入数据集
    if include_blank or len(entities)>0:
        return {
            'text' : ''.join(text),
            'entities' : entities,
        }
    else:
        return None




def assemble(infile, outfile_path, max_len=500, is_train=True):
    total = text_break = tmp_break = 0
    D = []

    data = json.load(open(infile))
    for l in tqdm(data):
        #print(f"---> {l['document']}")

        total += 1

        text = []
        tmp_text = []
        n = n_text = n_tmp = 0
        while n<len(l['tokens']):
            token = l['tokens'][n]
            if l['trailing_whitespace'][n]:
                token += ' ' 

            if n_tmp + 1 > max_len:
                text += deepcopy(tmp_text)
                tmp_text = []
                n_text += n_tmp
                n_tmp = 0

                tmp_break += 1

                #print(text)
                #assert False, f"tmp_text is too long: {len(text)}, {len(tmp_text)}, {len(token)}"

            if n_text + n_tmp + 1 > max_len:
                assert n_text>0, f"too long: {n_text}, {n_tmp}, {max_len}"
                #print(text)
                #print('-'*20)
                dd = __convert({
                        'sentence'  : [x[0] for x in text],
                        'BIO_label' : [x[1] for x in text],
                    })
                if dd:
                    dd['document'] = l['document']
                    dd['tokens'] = [x[0] for x in text]
                    D.append(dd)
                text = []
                n_text = 0

                text_break += 1

            if is_train:
                tmp_text += [(token, l['labels'][n])] # token, label
            else:
                tmp_text += [(token, 'O')] # token, blank-label
            n_tmp += 1

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
                })
            if dd:
                dd['document'] = l['document']
                dd['tokens'] = [x[0] for x in text]
                D.append(dd)

            text_break += 1            

        #break # for test

    blank = 0
    for x in D:
        if len(x['entities']):
            blank += 1

    print(f"total= {total}\ttext_break= {text_break}\ttmp_break= {tmp_break}\tblank= {blank}")

    if is_train:
        random.shuffle(D)

        # 拆分数据集
        split_n = int(len(D) * split_ratio)

        json.dump(
            D[:split_n],
            open(os.path.join(outfile_path, "train.json"), 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )


        json.dump(
            D[split_n:],
            open(os.path.join(outfile_path, "dev.json"), 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

        print(f"train set: {split_n}\tdev set: {len(D)-split_n}")

    else:
        json.dump(
            D,
            open(os.path.join(outfile_path, "test.json"), 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

        print(f"test set: {len(D)}")

if __name__ == '__main__':
    #assemble(train_file, 'data')
    assemble(test_file, 'data', is_train=False)
