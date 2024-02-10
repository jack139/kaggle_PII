#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别

import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras

import json
import numpy as np
import math
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
#from bert4keras.layers import GlobalPointer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from tqdm import tqdm

keras.utils.set_random_seed(816)

maxlen = 512
batch_size = 16 # 16 for base / 4 for large
#maxlen = 256
#batch_size = 32 # 32 for base / 8 for large 
epochs = 30
learning_rate = 2e-5 #* (0.8 ** 10)
categories = set()

# bert配置

config_path = '../nlp_model/bert_uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../nlp_model/bert_uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../nlp_model/bert_uncased_L-12_H-768_A-12/vocab.txt'
'''
config_path = '../nlp_model/bert_wwm_uncased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '../nlp_model/bert_wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '../nlp_model/bert_wwm_uncased_L-24_H-1024_A-16/vocab.txt'
'''

def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    for d in json.load(open(filename)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
            categories.add(label)
    return D


# 标注数据
#train_data = load_data('data/train.json')
train_data = load_data('data/train_more.json')
valid_data = load_data('data/dev.json')
categories = list(sorted(categories))

print("labels: ", categories)
# labels:  ['EMAIL', 'ID_NUM', 'NAME_STUDENT', 'PHONE_NUM', 'STREET_ADDRESS', 'URL_PERSONAL', 'USERNAME']


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred, epsilon=1e-10):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / (K.sum(y_true + y_pred) + epsilon)


model = build_transformer_model(config_path, checkpoint_path)
output = GlobalPointer(len(categories), 64)(model.output)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=learning_rate)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=optimizer, #Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        entities = sorted(entities, key=lambda x: x[0], reverse=False) # sort by start_idx
        return entities


NER = NamedEntityRecognizer()

name_dataset = json.load(open("data/names.json"))


def post_filter(label, text):
    """结果进行后处理
    """
    if label=='NAME_STUDENT':
        if text.lower() in name_dataset:
            return True
    elif label=='URL_PERSONAL':
        if text.endswith("/"):
            return True
        elif text.endswith(".org"):
            return True
        elif text.endswith(".net"):
            return True
        elif text.endswith(".biz"):
            return True
        elif text.endswith(".com"):
            return True
    elif label=='STREET_ADDRESS':
        if len(text.split())<4:
            return True
    return False


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = NER.recognize(d[0])
        R = [ e for e in R if not post_filter(e[2], d[0][e[0]:e[1]+1]) ]
        R = set(R)
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(f'ckpt/pii_gp_best_b{batch_size}_l{maxlen}_e{epoch:02d}_f1_{f1:.5f}.h5')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


###### generate result & evaluating

def predict_to_file(in_file, out_file):
    """预测到文件
    """
    data = json.load(open(in_file))

    document = ""
    last_pos = 0
    last_type = None
    D = []

    for d in tqdm(data, ncols=100):

        if document != d['document']:
            document = d['document']
            last_pos = 0
            last_type = None

        # 初始化 BIO 标记
        label = ['O']*len(d['tokens'])
        this_last_type = None

        # 识别
        entities = NER.recognize(d['text'])
        for e in entities:
            filt_this = post_filter(e[2], d['text'][e[0]:e[1]+1])
            d['entities'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2],
                'entity': d['text'][e[0]:e[1]+1],
                'filt_this': filt_this
            })

            if not filt_this:
                # 生成 BIO标记, 依据 原始 tokens
                pos = last = 0
                for n, x in enumerate(d['tokens']):
                    if pos >= e[0] and pos <= e[1]+1:
                        if last==0:
                            if pos==0 and e[2]==last_type:
                                label[n] = 'I-'+e[2] # 第一个与上一条最后一个type一样，type继续
                            elif n>0 and (label[n-1]=='B-'+e[2] or label[n-1]=='I-'+e[2]):
                                label[n] = 'I-'+e[2]
                            else:
                                label[n] = 'B-'+e[2]
                            last += 1
                        else:
                            label[n] = 'I-'+e[2]
                        D.append((document, last_pos+n, label[n]))

                        if n==len(d['tokens'])-1:
                            this_last_type = e[2] # 记录末尾的 type
                    else:
                        last = 0

                    pos += len(x)

                    if pos > e[1]:
                        break

        d['labels'] = label

        last_pos += len(d['tokens'])
        last_type = this_last_type

    # 保存json格式
    json.dump(
        data,
        open("data/output.json", 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    with open(out_file, "w") as f:
        f.write("row_id,document,token,label\n")
        for n, x in enumerate(D):
            f.write(f"{n},{x[0]},{x[1]},{x[2]}\n")


def evl_to_file(in_file, out_file):
    """评估到文件
    """
    data = json.load(open(in_file))

    for d in tqdm(data, ncols=100):
        d['entities_2'] = []
        d['tokens'] = []

        # 识别
        entities = NER.recognize(d['text'])
        for e in entities:
            filt_this = post_filter(e[2], d['text'][e[0]:e[1]+1])
            d['entities_2'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2],
                'entity': d['text'][e[0]:e[1]+1],
                'filt_this': filt_this
            })

    # 保存json格式
    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


def lr_step_decay(epoch):
    """ lr 衰减函数
    """
    drop = 0.8
    epochs_drop = 1.0
    lrate = learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


if __name__ == '__main__':

    evaluator = Evaluator()
    lrate = LearningRateScheduler(lr_step_decay)

    train_generator = data_generator(train_data, batch_size)

    #model.load_weights('ckpt/pii_gp_best_b4_l512_e11_f1_0.93980.h5')

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator, lrate]
    )

else:
    model.load_weights('ckpt/pii_gp_best_b4_l512_e11_f1_0.93980.h5')
    predict_to_file('data/test2.json', 'data/submission.csv')

    #evl_to_file('data/test/diff_output2.json', 'data/output2.json')
    #evl_to_file('data/dev.json', 'data/output2.json')
