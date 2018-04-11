import json
from newspaper import Article
import nltk
import re

def tokenize(text):
    text = text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'urltag', text)
    text = re.sub('(\@[^\s]+)', 'usertag', text)

    try:
        return ' '.join(nltk.word_tokenize(text))
    except:
        return text

with open('google_results.txt') as f:
    google = json.load(f)

for id in google:
    for s in google[id]:
        try:
            article = Article(s['url'])
            article.download()
            article.parse()
            s['body']=tokenize(article.text[:300])
            print (s['body'])
        except:
            s['body']="NaN"

with open('google_results.txt','w') as f:
    json.dump(google,f)
