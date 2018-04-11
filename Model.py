import re
import numpy as np
from tools import imputer, standardize, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import cPickle
from proposed_models import NN

crossplatform = ['dist mean', 'dist var', 'agree mean', 'agree var', 'disagree mean',
                 'disagree var', 'discuss mean', 'discuss var', 'unrelated mean', 'unrelated var']


class Supervise():
    def __init__(self):
        # load baseline features
        with open('resources/add_dict.txt', 'rb') as f:
            add_dict = cPickle.load(f)
        self.base = add_dict

    def verfiy(self, train, test):
        self.base(test)
        X, Y, x, y = self.prepare(train, test)
        self.classify(X, Y, x, y)

    def prepare(self, train, test, feat='1'):
        if feat == '1':
            X = [self.compute(_) for _ in train]
            Y = [_['label'] for _ in train]

            x = [self.compute(_) for _ in test]
            y = [_['label'] for _ in test]

        elif feat == '2':
            X = np.array([self.compute(elem) + self.base[elem['tweet_id']] for elem in train], dtype=float)
            Y = [_['label'] for _ in train]
            # replace all with 0
            for i in xrange(X[0, :].size):
                if np.all(np.isnan(X[:, i])):
                    X[:, i].fill(0)

            x = np.array([self.compute(elem) + self.base[elem['tweet_id']] for elem in test], dtype=float)
            y = [_['label'] for _ in test]
            # replace all with 0
            for i in xrange(x[0, :].size):
                if np.all(np.isnan(x[:, i])):
                    x[:, i].fill(0)

        # prepare
        X = standardize(imputer(X))
        Y = np.array(Y)
        x = standardize(imputer(x))
        y = np.array(y)

        return X, Y, x, y

    def compute(self, elem):
        if not elem['dist']:
            dist = [0, 0]
        # dist feats
        else:
            mean = np.average(np.array(elem['dist']))
            variance = np.var(np.array(elem['dist']))
            dist = [mean, variance]
        # agree feats
        if not elem['agree']:
            agree = [0 for _ in xrange(8)]
        else:
            means = np.average(elem['agree'], axis=0).tolist()
            vars = np.var(elem['agree'], axis=0).tolist()
            agree = [means[0], vars[0], means[1], vars[1], means[2], vars[2], means[3], vars[3]]

        return dist + agree


class MMMLRV():
    def __init__(self, miss, threshold):
        self.neu = 10
        self.miss = miss
        self.threshold = threshold

    def verify(self, elem):
        score = self.compute(elem['relation'])
        self.neu = elem['trust']
        if elem['trust'] == 'NaN':
            self.neu = self.miss
        if score > self.threshold:
            return 0
        else:
            return 1

    def compute(self, relation):
        pos = []
        neg = []
        for pair in relation[:20]:
            dist = pair[0]
            if pair[1] == 'NaN':
                trust = self.miss
            else:
                trust = float(pair[1])
            if trust > self.neu:
                pos.append((dist, trust - self.neu))
            else:
                neg.append((dist, self.neu - trust))
        score = .0
        if pos:
            w = .0
            s_pos = .0
            for obj in pos:
                w += obj[1]
                s_pos += (2 - obj[0]) * obj[1]
            if w == 0:
                s_pos = 0
            else:
                s_pos = s_pos / w
            score += s_pos

        if neg:
            w = .0
            s_neg = .0
            for obj in neg:
                w += obj[1]
                s_neg += (2 - obj[0]) * obj[1]
            if w == 0:
                s_neg = 0
            else:
                s_neg = s_neg / w
            score -= s_neg

        return score


class Clustering:
    def __init__(self, miss=0, top=5):
        self.miss = miss
        self.top = 5

    def build_vec(self, mid):
        for elem in mid:
            if len(elem[0]) < self.top:
                continue


class RE:
    def __init__(self):
        self.pattern = r'fake|liar|cheat|wrong'

    def verify(self, google):
        for s in google:
            # for p in self.pattern:
            if re.match(self.pattern, s['title']):
                # print(s['title'])
                return 1
        return 0
