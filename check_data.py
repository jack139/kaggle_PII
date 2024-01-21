import json

train_file = 'data/pii-detection-removal-from-educational-data/train.json'
test_file = 'data/pii-detection-removal-from-educational-data/test.json'

def load_data(filename, text_name='full_text'):
    """加载数据
    单条格式：(文本, 标签id)
    """
    max_len = 0    
    max_cnt = 0

    data = json.load(open(filename))

    for l in data:
        if len(l[text_name])>512:
            print(len(l[text_name]), l[text_name][:10])
            max_cnt += 1
        if len(l[text_name])>max_len:
            max_len = len(l[text_name])
    return max_len, max_cnt, len(data)


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
    #print('train:', load_data(train_file))
    #print('test:', load_data(test_file))

    assemble(train_file)
    #assemble(test_file)
