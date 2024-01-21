import os
import json
from copy import deepcopy

train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'


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

def convert(infile, outfile, include_blank=True):

    print(f"{infile} --> {outfile}")

    D = []
    text = ''
    entities = []
    all_idx = 0
    start_idx = 0
    etype = ''

    data = json.load(open(infile))

    for k in data.keys():
        for diag in data[k]['dialogue']:
            text = diag['sentence']

            for label in diag['BIO_label'].split():
                if label[0]=='O':
                    if etype!='':
                        entities.append({
                            "start_idx": start_idx,
                            "end_idx": all_idx - 1,
                            "type": etype,
                            "entity": text[start_idx:all_idx],
                        })
                    start_idx = 0
                    etype = ''
                elif label[0]=='B':
                    if etype!='':
                        entities.append({
                            "start_idx": start_idx,
                            "end_idx": all_idx - 1,
                            "type": etype,
                            "entity": text[start_idx:all_idx],
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
                    "start_idx": start_idx,
                    "end_idx": all_idx - 1,
                    "type": etype,
                    "entity": text[start_idx:all_idx],
                })

            # 加入数据集
            if include_blank or len(entities)>0:
                D.append({
                    'text' : text,
                    'entities' : entities,
                })

            text = ''
            entities = []
            all_idx = 0
            start_idx = 0
            etype = ''

    json.dump(
        D,
        open(os.path.join(outfile), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print(len(D))


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
                    "end_idx": len(''.join(text[:all_idx])),
                    "type": etype,
                    "entity": ''.join(text[start_idx:all_idx]),
                })
            start_idx = 0
            etype = ''
        elif label[0]=='B':
            if etype!='':
                entities.append({
                    "start_idx": len(''.join(text[:start_idx])),
                    "end_idx": len(''.join(text[:all_idx])),
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
            "end_idx": len(''.join(text[:all_idx])),
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




def assemble(infile, outfile, max_len=500):
    total = text_break = tmp_break = 0
    D = []

    data = json.load(open(infile))
    for l in data:
        print(f"---> {l['document']}")

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
                assert len(text)>0, f"too long: {n_text}, {n_tmp}"
                #print(text)
                #print('-'*20)
                dd = __convert({
                        'sentence'  : [x[0] for x in text],
                        'BIO_label' : [x[1] for x in text],
                    })
                if dd:
                    D.append(dd)
                text = []
                n_text = 0

                text_break += 1

            tmp_text += [(token, l['labels'][n])] # token, label
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
                D.append(dd)

            text_break += 1            

        break # for test

    json.dump(
        D,
        open(os.path.join(outfile), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print(f"total= {total}\ttext_break= {text_break}\ttmp_break= {tmp_break}")


if __name__ == '__main__':
    #convert('../dataset/IMCS-IR/new_split/data/IMCS-V2_train.json', './data/train.json', True)
    #convert('../dataset/IMCS-IR/new_split/data/IMCS-V2_dev.json', './data/dev.json')
    #convert('../dataset/3.0/IMCS-V2/IMCS-V2_train.json', './data/train.json', True)
    #convert('../dataset/3.0/IMCS-V2/IMCS-V2_dev.json', './data/dev.json')
    assemble(train_file, 'data/train.json')
