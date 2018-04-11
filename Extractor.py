import sent2vec
import json
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from textblob import TextBlob
import os
from google.cloud import translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "MV.json"

class intermid:
    def __init__(self):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model('wiki_bigrams.bin')

    def load(self, data):
        try:
            with open('mids/intermid.txt', 'r') as f:
                mid = json.load(f)
                return mid
        except:
            mid = self.process(data)
            return mid

    def embed(self, text):
        emb = self.model.embed_sentence(text)
        return emb

    def process(self,  data):
        title_data = []
        tweet_data = []
        for elem in data:
            relation = []
            for s in elem['google']:
                tweet, title = self.translate(elem['content'], s['title'])
                id=elem['tweet_id']
                # generate csv for FNC
                title_data.append([title,id ])
                if [id, tweet] in tweet_data:
                    continue
                else:
                    tweet_data.append([id, tweet])

                #compute distance
                dist = cosine(self.embed(tweet), self.embed(title))
                relation.append((dist, s['trust']))

            elem['relation']=relation
            # if elem['label'] == 'fake':
            #     mid.append((relation, 1))
            # else:
            #     mid.append((relation, 0))

        #save distance
        with open('mids/intermid.txt', 'w') as f:
            json.dump(data, f)

        #output csv
        title_data = np.array(title_data)
        title_data = pd.DataFrame(title_data, columns=['Headline', 'Body ID'])
        title_data.to_csv('test_stances_unlabeled.csv', index=False, )

        tweet_data = np.array(tweet_data)
        tweet_data = pd.DataFrame(tweet_data, columns=['Body ID', 'articleBody'])
        tweet_data.to_csv('FNC/test_bodies.csv', index=False)

        return mid

    def translate(self, text1, text2):
        l1 = TextBlob(text1).detect_language()
        l2 = TextBlob(text2).detect_language()

        t = translator()

        if l1 == 'en' and l2 == 'en':
            return text1, text2
        elif l1 == 'en' and l2 != 'en':
            text1 = t.translate(text1, l2)
            text1 = t.translate(text1, 'en')
            text2 = t.translate(text2, 'en')
            return text1, text2
        elif l1 != 'en' and l2 == 'en':
            text2 = t.translate(text2, l2)
            text2 = t.translate(text2, 'en')
            text1 = t.translate(text1, 'en')
            return text1, text2
        else:
            text1 = t.translate(text1, 'en')
            text2 = t.translate(text2, 'en')
            return text1, text2

class translator():
    def __init__(self):
        self.client = translate.Client()

    def translate(self, text, target):
        translation = self.client.translate(text, target)
        print(u'Text: {}'.format(text))
        print(u'Translation: {}'.format(translation['translatedText']))
        return translation['translatedText']

