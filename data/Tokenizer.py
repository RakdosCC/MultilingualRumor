import re
import unicodedata
import sys
import pynlpir
import nltk

class tokenizer():
    def __init__(self):
        pynlpir.open()

    def tokenize(self, text, cn=True):
        text = text.lower()
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'urltag', text)
        text = re.sub('(\@[^\s]+)', 'usertag', text)

        if cn:
            try:
                tokens = pynlpir.segment(text, pos_tagging=True)
                format = ' '.join([_[0] for _ in tokens])
                return format
            except:
                try:
                    return ' '.join(nltk.word_tokenize(text))
                except:
                    print text
                    return text
        else:
            try:
                return ' '.join(nltk.word_tokenize(text))
            except:
                return text
