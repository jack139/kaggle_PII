
import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras

import unicodedata
from bert4keras.snippets import is_py2
from bert4keras.tokenizers import SpTokenizer#, Tokenizer


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

        print(f"1 ---> {text} {len(text)}")

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

                '''
                # 处理 '…'
                if 'fi' in token and 'ﬁ' in text[offset:offset+20]:
                    search_token = token.replace('fi', 'ﬁ')
                elif 'fl' in token and 'ﬂ' in text[offset:offset+20]:
                    search_token = token.replace('fl', 'ﬂ')
                elif 'ff' in token and 'ﬀ' in text[offset:offset+20]:
                    search_token = token.replace('ff', 'ﬀ')
                else:
                    search_token = token
                '''

                print(f"2 ---> {offset}")
                print(f"3 ---> [{token}]")
                print(f"4 ---> [{search_token}]")
                print(f"5 ---> [{text[offset:offset+20]}]")

                start = text[offset:].index(search_token) + offset

                end = start + len(search_token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


    def token_to_text(self, tokens):
        return self.sp_model.decode_pieces(tokens)


spm_path = '../nlp_model/albert_base_v2/30k-clean.model'

tokenizer = MySpTokenizer(spm_path)



text = """
We talked after the brieﬁng and we agreed that we could`t

"""



tokens = tokenizer.tokenize(text, maxlen=512)


print(text)
print(tokens)

mapping = tokenizer.rematch(text, tokens)
print(mapping)

#print(tokenizer.token_to_text(tokens))
