#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别

import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras

import json
import numpy as np
import unicodedata
from bert4keras.snippets import is_py2
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
#from bert4keras.layers import GlobalPointer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer#, Tokenizer
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm

maxlen = 512
epochs = 30
batch_size = 4 # 16 for base, 4 for xxlarge
learning_rate = 2e-5
categories = set()

# bert配置
'''
config_path = '../nlp_model/albert_base_v2/albert_config.json'
checkpoint_path = '../nlp_model/albert_base_v2/model.ckpt-best'
#dict_path = '../nlp_model/albert_base_v2/30k-clean.vocab'
spm_path = '../nlp_model/albert_base_v2/30k-clean.model'
'''
config_path = '../nlp_model/albert_xxlarge_v2/albert_config.json'
checkpoint_path = '../nlp_model/albert_xxlarge_v2/model.ckpt-best'
spm_path = '../nlp_model/albert_xxlarge_v2/30k-clean.model'


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
train_data = load_data('data/train_43k.json')
valid_data = load_data('data/dev.json')
categories = list(sorted(categories))

print("labels: ", categories)
# labels:  ['EMAIL', 'ID_NUM', 'NAME_STUDENT', 'PHONE_NUM', 'STREET_ADDRESS', 'URL_PERSONAL', 'USERNAME']


# 自定义Tokenizer, 实现 rematch
class MySpTokenizer(SpTokenizer):
    def __init__(self, sp_model_path, **kwargs):
        super(MySpTokenizer, self).__init__(sp_model_path, **kwargs)

        self._stranges = [
            ('ﬄ', 'ffl'),
            ('ﬃ', 'ffi'),
            ('ﬂ', 'fl'),
            ('ﬁ', 'fi'),
            ('ﬀ', 'ff'),
            ('™', 'TM'),
            ('№', 'No'),
        ]

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是"▁"开头，则自动去掉"▁"）
        """
        if token[:1] == '▁':
            return token[1:]
        elif token[:1] == '\xad':
            return token[1:]
        else:
            return token

    @staticmethod
    def _is_special2(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if is_py2:
            text = unicode(text)

        #if self._do_lower_case:
        #    text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            #if self._do_lower_case:
            #    ch = lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for n, token in enumerate(tokens):
            if self._is_special2(token):
                token_mapping.append([])
            else:
                token = self.stem(token)

                search_token = token

                span = len(token)

                # 处理 奇怪的符号
                for strange in self._stranges:
                    if strange[0] in (text[offset:].lstrip())[:span]:
                        if strange[1] in token:
                            search_token = token.replace(strange[1], strange[0])
                        elif token[:-len(strange[1])+1]+strange[0] in (text[offset:].lstrip())[:span]:
                            search_token = token[:-len(strange[1])+1]
                        elif strange[0]+token[1:] == text[offset:offset+len(token)]:
                            search_token = strange[0] + token[1:]
                        elif token == strange[1][-len(strange[1])+1] + (text[offset:].lstrip())[len(strange[1])-1:len(token)]:
                            search_token = strange[0] + token[len(strange[1])-1:]
                        break

                if search_token in text[offset:]:
                    start = text[offset:].index(search_token) + offset

                    end = start + len(search_token)
                    token_mapping.append(char_mapping[start:end])
                    offset = end

                else:
                    print(f"2 ---> {offset}")
                    print(f"3 ---> [{token}]")
                    print(f"4 ---> [{search_token}]")
                    print(f"5 ---> [{text[offset:offset+20]}]")

                    token_mapping.append([])

        return token_mapping

# 建立分词器
tokenizer = MySpTokenizer(spm_path)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            try:
                mapping = tokenizer.rematch(d[0], tokens)
            except Exception as e:
                print(f"-----> [{d[0]}]")
                raise e
            
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


model = build_transformer_model(config_path, checkpoint_path, model='albert')
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
            if l>len(categories):
                print("categories out:", len(categories), l)
            else:
                entities.append(
                    (mapping[start][0], mapping[end][-1], categories[l])
                )
        return entities


NER = NamedEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
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
            model.save_weights('ckpt/pii_albert_gp_best_f1_%.5f.h5'%f1)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    """
    data = json.load(open(in_file))

    document = ""
    last_pos = 0
    D = []

    for d in tqdm(data, ncols=100):

        if document != d['document']:
            document = d['document']
            last_pos = 0

        # 初始化 BIO 标记
        label = ['O']*len(d['tokens'])

        # 识别
        entities = NER.recognize(d['text'])
        for e in entities:
            d['entities'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2]
            })

            # 生成 BIO标记, 依据 原始 tokens
            pos = last = 0
            for n, x in enumerate(d['tokens']):
                if pos >= e[0] and pos <= e[1]+1:
                    if last==0:
                        label[n] = 'B-'+e[2]
                        last += 1
                    else:
                        label[n] = 'I-'+e[2]
                    D.append((document, last_pos+n, label[n]))
                else:
                    last = 0

                pos += len(x)

                if pos > e[1]:
                    break

        d['labels'] = label

        last_pos += len(d['tokens'])

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
    """预测到文件
    """
    data = json.load(open(in_file))

    for d in tqdm(data, ncols=100):
        d['entities_2'] = []

        # 识别
        entities = NER.recognize(d['text'])
        for e in entities:
            d['entities_2'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2],
                'entity': d['text'][e[0]:e[1]+1]
            })

    # 保存json格式
    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    #model.load_weights('ckpt/pii_albert_gp_best_f1_0.82844.h5')

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('ckpt/pii_gp_best_f1_0.92476.h5')
    #predict_to_file('data/test.json', 'data/submission.csv')
    evl_to_file('data/dev.json', 'data/output.json')
