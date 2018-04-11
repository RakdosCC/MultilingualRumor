from data.Manager import manager
from Model import MMMLRV, MnV
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
import numpy as np
from sklearn import preprocessing
import json

def main(data='twitter'):
    if data == 'twitter':
        with open('data/processed_twitter.txt', 'r') as f:
            dataset = json.load(f)

    elif data == 'baidu':
        with open('data/processed_baidu.txt', 'r') as f:
            dataset = json.load(f)

    else:
        return

    verify_by_event(dataset)
    PCC(dataset)


def imputer(data):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    data = imp.transform(data)
    return data


def PCC(dataset):
    mnv = MnV(trust=False, base=False)
    ms, vs, ls = [], [], []
    for elem in dataset:
        # if elem['lng']!='en':
        #     continue
        mean, variance = mnv.compute(elem)
        ms.append(mean)
        vs.append(variance)
        ls.append(elem['label'])

    ms = np.array(ms).reshape(-1, 1)
    ms = imputer(ms)
    vs = np.array(vs).reshape(-1, 1)
    vs = imputer(vs)
    ls = np.array(ls).reshape(-1, 1)
    print('mean: ', pearsonr(ms, ls))
    print('var: ', pearsonr(vs, ls))


def verify_by_event(dataset):
    events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
              'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']
    scores = []
    for event in events:
        print event
        data = []
        for elem in dataset:
            if elem['event'] == event:
                # if elem['lng'] != 'en':
                #     continue
                data.append(elem)
        if not data:
            continue
        model = MMMLRV(miss=0, threshold=1.2)

        F1 = evaluate(data, model)
        base(data)


def evaluate(data, model):
    y = []
    y_pred = []
    for elem in data:
        y_pred.append(model.verify(elem))
        if len(elem) == 0:
            print(elem['image_id'])
            y_pred.append(1)
        y.append(elem['label'])

    p, recall, F1 = metrics(y, y_pred)
    print(confusion_matrix(y, y_pred))
    print('p: ', p)
    print('recall: ', recall)
    print('F1: ', F1)

    return p, recall, F1


def base(data):
    y = []
    y_pred = []
    for elem in data:
        y_pred.append(1)
        y.append(elem['label'])
    p, recall, F1 = metrics(y, y_pred)

    print('base: ', F1)


def metrics(y, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y[i] == 0 and y_pred[i] == 0:
            pass
        else:
            print('error!')
    if (tp + fp) == 0:
        p = 1
    else:
        p = float(tp) / (tp + fp)
    if (tp + fn) == 0:
        recall = 1
    else:
        recall = float(tp) / (tp + fn)
    F1 = 2 * float(tp) / (2 * tp + fp + fn)
    return p, recall, F1


if __name__ == '__main__':
    main(data='twitter')
    #main(data='baidu')
