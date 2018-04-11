import torch
import numpy as np
from torch.autograd import Variable
import cPickle as pkl
import math
import json
from Tokenizer import tokenizer
from tools import encode_sentences_np, encode_sentences
from model import LIUMCVC_Encoder


class embedder():
    def __init__(self, run):
        data = 'UM_Corpus'
        saveto = 'save/%s' % (data)
        with open('%s_params_%s.pkl' % (saveto, run), 'rb') as f:
            model_options = pkl.load(f)
        with open('%s.dictionary_%s.pkl' % (saveto, run), 'rb') as f:
            worddict = pkl.load(f)
        model = LIUMCVC_Encoder(model_options)
        model.load_state_dict(torch.load('%s_model_%s.pkl' % (saveto, run)))
        model = model.cuda()

        best_model = {}
        best_model['options'] = model_options
        best_model['en_cn_model'] = model
        best_model['worddict'] = worddict
        self.model = best_model

        self.baidu_path = 'baidu_results.txt'
        self.google_path = 'google_results.txt'
        self.twitter_path = 'dataset.txt'

    def embed(self, text1, text2):
        p = tokenizer()
        text1 = [p.tokenize(t, cn=False) for t in text1]
        text2 = [p.tokenize(t, cn=False) for t in text2]
        feats1, feats2 = encode_sentences(self.model, (text1, text2), test=True)
        return feats1, feats2

    def embed_google(self):
        with open(self.google_path) as f:
            google = json.load(f)
        texts = []
        for id in google:
            for s in google[id]:
                texts.append(s['title'])

        feats, _ = encode_sentences_np(self.model, (texts, texts), test=True)

        # compute dists and assign
        i = 0
        for id in google:
            for s in google[id]:
                s['embed'] = feats[i].tolist()
                i += 1

        # embed body
        # bodies = []
        # for id in google:
        #     for s in google[id]:
        #         bodies.append(s['body'])
        # print bodies
        # bfeats, _ = encode_sentences_np(self.model, (bodies, bodies), test=True)
        # i = 0
        # for id in google:
        #     for s in google[id]:
        #         s['bembed'] = bfeats[i].tolist()
        #         i += 1

        with open('google_embed.txt', 'w') as f:
            json.dump(google, f)

    def embed_baidu(self):
        with open(self.baidu_path) as f:
            baidu = json.load(f)
        texts = []
        for id in baidu:
            for s in baidu[id]:
                texts.append(s['title'])

        feats, _ = encode_sentences_np(self.model, (texts, texts), test=True)

        # compute dists and assign
        i = 0
        for id in baidu:
            for s in baidu[id]:
                s['embed'] = feats[i].tolist()
                i += 1

        with open('baidu_embed.txt', 'w') as f:
            json.dump(baidu, f)

    def embed_twitter(self):
        with open(self.twitter_path) as f:
            twitter = json.load(f)

        texts = []
        for tweet in twitter:
            texts.append(tweet['content'])

        feats, _ = encode_sentences_np(self.model, (texts, texts), test=True)

        # compute feats and assign
        i = 0
        for tweet in twitter:
            tweet['embed'] = feats[i].tolist()
            i += 1

        with open('twitter_embed.txt', 'w') as f:
            json.dump(twitter, f)
