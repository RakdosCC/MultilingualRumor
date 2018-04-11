from data.Manager import manager
from Model import Supervise
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
import numpy as np
from tools import imputer, standardize,metrics
from proposed_models import NN


import json

events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback',
          'columbianChemicals', 'passport', 'elephant',
          'underwater', 'livr', 'pigFish', 'eclipse', 'samurai',
          'nepal', 'garissa', 'syrianboy', 'varoufakis']


def main():
    with open('data/processed_twitter.txt', 'r') as f:
        twitter = json.load(f)

    with open('data/processed_baidu.txt', 'r') as f:
        baidu = json.load(f)

    verify_Baidu_event_via_Twitter(baidu, twitter)

    verify_Baidu_event_via_Random(baidu)



def verify_Baidu_event_via_Random(baidu):
    print 'Randonly guessing!'
    np.random.seed(70)
    scores=[]
    for event in events:
        print event
        test = []
        for elem in baidu:
            if elem['event'] == event:
                test.append(elem)
        y=[]
        y_pred=[]
        if not test:
            print "None"
            continue
        for elem in test:
            y.append(elem['label'])
            y_pred.append(np.random.randint(2))
        F1, p, recall=metrics(y,y_pred)
        print 'F1: ', F1

        scores.append([F1, p, recall])
    print 'Average F1', np.average(scores[:][0])



def verify_Baidu_event_via_Twitter(baidu, twitter):
    print 'verify_Baidu_event_via_Twitter'
    scores = []
    s = Supervise()
    train = twitter
    for event in events:
        print event
        test = []
        for elem in baidu:
            if elem['event'] == event:
                test.append(elem)
        if not test:
            print 'None'
            continue
        X, Y, x, y = s.prepare(train, test)
        F1, p, recall = NN(X, Y, x, y)
        print 'F1: ', F1

        scores.append([F1, p, recall])
    print 'Average F1', np.average(scores[:][0])


def PCC(dataset):
    crossplatform = ['dist mean', 'dist var', 'agree mean', 'agree var', 'disagree mean',
                     'disagree var', 'discuss mean', 'discuss var', 'unrelated mean', 'unrelated var']
    s = Supervise()
    fs, ls = [], []
    for elem in dataset:
        feats = s.compute(elem)
        fs.append(feats)
        ls.append(elem['label'])

    ms = np.array(fs, dtype=float)
    # replace all with 0
    for i in xrange(ms[0, :].size):
        if np.all(np.isnan(ms[:, i])):
            ms[:, i].fill(0)
    ms = imputer(ms)
    ms = standardize(ms)
    ls = np.array(ls)
    # print('mean: ', pearsonr(ms, ls))
    # print('var: ', pearsonr(vs, ls))
    order = []
    for i in xrange(10):
        order.append((crossplatform[i], pearsonr(ms[:, i], ls)[0]))
    order = sorted(order, key=lambda x: abs(x[1]), reverse=True)
    print 'Features with the highest PCC:'
    print order


if __name__ == '__main__':
    main()
