"""
Dataset loading
"""
import random
import pynlpir
import re
import json
from Tokenizer import tokenizer


def preprocess():
    blog_en, blog_cn = [], []
    t = tokenizer()
    with open('UM_Corpus/Bi-Microblog.txt', 'rU') as f:
        lines = f.readlines()
        for i in xrange(0, len(lines), 2):
            en = t.tokenize(lines[i].strip())
            cn = t.tokenize(lines[i + 1].strip())
            if en is None or cn is None:
                continue
            blog_en.append(en)
            blog_cn.append(cn)

    news_en, news_cn = [], []
    with open('UM_Corpus/Bi-News.txt', 'rU') as f:
        lines = f.readlines()
        for i in xrange(0, len(lines), 2):
            en = t.tokenize(lines[i].strip())
            cn = t.tokenize(lines[i + 1].strip())
            if en is None or cn is None:
                continue
            news_en.append(en)
            news_cn.append(cn)

    with open('UM_Corpus/blog_en', 'w') as f1, open('UM_Corpus/blog_cn', 'w') as f2, open(
            'UM_Corpus/news_en', 'w') as f3, open('UM_Corpus/news_cn', 'w') as f4:
        json.dump(blog_en, f1)
        json.dump(blog_cn, f2)
        json.dump(news_en, f3)
        json.dump(news_cn, f4)


def load_dataset(name='UM_Corpus', load_test=False):
    """
    Load en and cn sentences
    """
    random.seed(777)
    with open('UM_Corpus/blog_en') as f1, open('UM_Corpus/blog_cn') as f2, open(
            'UM_Corpus/news_en') as f3, open('UM_Corpus/news_cn') as f4:
        blog_en = json.load(f1)
        blog_cn = json.load(f2)
        news_en = json.load(f3)
        news_cn = json.load(f4)

    # shuffle UM_Corpus
    blog = zip(blog_en, blog_cn)
    random.shuffle(blog)
    blog_en = [_[0] for _ in blog]
    blog_cn = [_[1] for _ in blog]
    news = zip(news_en, news_cn)
    random.shuffle(news)
    news_en = [_[0] for _ in news]
    news_cn = [_[1] for _ in news]

    if load_test:
        test_en = blog_en[:500] + news_en[:500]
        test_cn = blog_cn[:500] + news_cn[:500]
        return (test_en, test_cn)

    else:
        train_en = blog_en[1000:5000] + news_en[1000:]
        train_cn = blog_cn[1000:5000] + news_cn[1000:]

        dev_en = blog_en[500:1000] + news_en[500:1000]
        dev_cn = blog_cn[500:1000] + news_cn[500:1000]

        return (train_en, train_cn), (dev_en, dev_cn)




if __name__ == '__main__':
    preprocess()
