
import os
os.environ["TF_KERAS"] = "1" # use tf 2.7 keras

import unicodedata
from bert4keras.snippets import is_py2
from bert4keras.tokenizers import SpTokenizer#, Tokenizer


class MySpTokenizer(SpTokenizer):
    def __init__(self, sp_model_path, **kwargs):
        super(MySpTokenizer, self).__init__(sp_model_path, **kwargs)

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

                skip_offset = False

                # 处理 '…'
                if n+2<len(tokens) and ''.join(tokens[n:n+3])=='...' and '…' in text[offset:offset+3]:
                    search_token = '…'
                    skip_offset = True
                elif n+1<len(tokens) and ''.join(tokens[n-1:n+2])=='...' and '…' in text[offset:offset+3]:
                    search_token = '…'
                    skip_offset = True
                elif n<len(tokens) and ''.join(tokens[n-2:n+1])=='...' and '…' in text[offset:offset+2]:
                    search_token = '…'
                elif n+1<len(tokens) and ''.join(tokens[n:n+2])=='Rs' and '₨' in text[offset:offset+2]:
                    search_token = '₨'
                    skip_offset = True
                elif n<len(tokens) and ''.join(tokens[n-1:n+1])=='Rs' and '₨' in text[offset:offset+2]:
                    search_token = '₨'
                elif '́' in token:
                    search_token = token.replace('́', '´')
                elif 'ریال' in token:
                    search_token = token.replace('ریال', '﷼')
                elif 'fi' in token and 'ﬁ' in text[offset:offset+20]:
                    search_token = token.replace('fi', 'ﬁ')
                elif 'fl' in token and 'ﬂ' in text[offset:offset+20]:
                    search_token = token.replace('fl', 'ﬂ')
                elif 'ff' in token and 'ﬀ' in text[offset:offset+20]:
                    search_token = token.replace('ff', 'ﬀ')
                else:
                    search_token = token

                print(f"2 ---> {offset}")
                print(f"3 ---> [{token}]")
                print(f"4 ---> [{search_token}]")

                start = text[offset:].index(search_token) + offset

                if not skip_offset:
                    end = start + len(search_token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

spm_path = '../nlp_model/albert_base_v2/30k-clean.model'

tokenizer = MySpTokenizer(spm_path)

#text = "A suspicious login was detected on user 50855527 which used IP address e55a:0e26:37ec:ebaa:e4b4:aa03:a9e8:de11. The tool 'Education Measure Lite' was in use at 3:30 PM. Please confirm."

#text = "A… …suspicious … \n\n login was detected\n\non …user 50855527 which\n\n"

#text = "Obesity-induced ailment has been linked to altered hormonal patterns. To further explore the issue, make a deposit of ₨... to 34320673 named Personal Loan Account."

#text = "Alessandro Giorgio\n\nDesign\t\r  Thinking\t\r  Innovation\t\r  -­‐\t\r  Mind\t\r  Mapping\n\n"

#text = """In 2011 a colleague and I embarked in a difﬁcult task, building the most advanced"""
text = "DESIGN THINKING ASSINGMENT\n\nDARDEN BUSINESS SCHOOL – COURSERA\n\nOFF-GRID REFRIGERATION CHALLENGE IN ABUJA-NIGERIA\n\nHabibu George\n\nAbuja, October 30, 2016\n\nCONTENTS\n\n1. CHALLENGE; Market penetration  ……………………………………………  3\n\n2. TOOL SELECTED; Learning Launches …………………………………..........  3\n\n3. APPLICATION and INSIGHTS ……………………………………………………..  3\n\n4. APPROACH  ………………………………………………………………..….. ..  4\n\nCHALLENGE; Market penetration\n\nPower is a significant challenge in Nigeria. In about two decades this nation has expended  over twenty billion U.S. Dollars yet our total power output is about Five thousand  megawatts. This is a country with a population of about a hundred and seventy million  people occupying about a million square kilometres of land. Many people need power to  run their fridges, freezers, air conditioning etc but this lack of it created a challenge and an  opportunity. So we set up a factory to produce blocks of ice that we could provide the  people in our state to use in meeting some of their refrigeration needs. Because of the  burning nature of the need and huge population we believed this would be a no brainer  and would hit the market in a huge wave. This did not happen. As a matter of fact, we  struggled to make sales of 20 blocks a day from a 1000 block a day factory. This was not  sustainable and we had to take significant action and fast if were not going to shelve the  business.    This way we would get hard facts on why the market was reacting with such poor reception  to a product that was designed (as we thought) to meet an ardent refrigeration need.\n\nTOOL SELECTED; Learning Launches\n\n"

tokens = tokenizer.tokenize(text, maxlen=512)


print(text)
print(tokens)

mapping = tokenizer.rematch(text, tokens)
print(mapping)
