import json
import cPickle
import numpy as np
from proposed_models import RF, NN, standardize, imputer, readf
from sklearn.model_selection import KFold


# divide data into test and train set
def devide(dataset, event):
    test = []
    train = []
    for elem in dataset:
        if elem['event'] == event:
            test.append(elem)
        else:
            train.append(elem)
    return train, test


def task_divide(dataset):
    test = []
    train = []
    for elem in dataset:
        if elem['event'] in ['eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']:
            test.append(elem)
        else:
            train.append(elem)
    return train, test


def get_input(train, test, feat):
    def feats_prep(dataset):
        data = []
        label = []
        # the index of features importance is based on the input here
        for elem in dataset:
            if feat == '1':
                data.append(elem['dist'] + elem['agree'])
            elif feat == '2':
                data.append(elem['base+add'])
            elif feat == '3':
                data.append(elem['dist'] + elem['agree'] + elem['base+add'])
            else:
                print 'input error!'
                return
            label.append(readf(elem['label']))

        label = np.array(label, dtype=float)
        data = np.array(data, dtype=float)
        # replace all with 1 if there is no valid value in a featuresour
        for i in xrange(data[0, :].size):
            if np.all(np.isnan(data[:, i])):
                data[:, i].fill(1)
        data = imputer(data)
        data = standardize(data)
        return data, label

    X, Y = feats_prep(train)
    x, y = feats_prep(test)
    return X, Y, x, y


def model_selection(elembase, events, clf, feat):
    print 'model selection...'
    best = 0
    if clf == 'RF':
        fractions = [0.2, 0.25, 0.3, 0.35]
        for frac in fractions:
            scores = []
            for event in events:
                train, test = devide(elembase, event)
                X, Y, x, y = get_input(train, test, feat)
                precision, F = RF(X, Y, x, y, frac)
                scores.append(F)
            score = np.average(scores)
            if score > best:
                best = score
                p = [frac]
        print 'the best frac: ' + str(p[0])
        return p

    elif clf == 'NN':
        dropouts = [0.85, 0.9]
        for d in dropouts:
            scores = []
            for event in events:
                train, test = devide(elembase, event)
                X, Y, x, y = get_input(train, test, feat)
                F = NN(X, Y, x, y, d)
                scores.append(F)
            score = np.average(scores)
            if score > best:
                best = score
                p = [d]
        print 'the best dropout: ' + str(p[0])
        return p

    else:
        print 'input error!'
        return


def run_task(elembase, clf, feat):
    print 'start VMU 2015 task!'
    train, test = task_divide(elembase)
    X, Y, x, y = get_input(train, test, feat)
    print 'K-Fold cross-validation...'
    if clf == 'RF':
        p = None
        best = 0
        fractions = [0.2, 0.25, 0.3, 0.35]
        kf = KFold(n_splits=5, shuffle=True, random_state=777)
        for frac in fractions:
            scores = []
            for train_index, test_index in kf.split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                precision, F = RF(X_train, Y_train, X_test, Y_test, frac)
                scores.append(F)
            score = np.average(scores)
            if score > best:
                best = score
                p = frac
        print 'the best frac: ' + str(p)

        precision, F = RF(X, Y, x, y, p, weight=True, feat=feat)
        print 'F1: ' + str(F)
        print 'precision ' + str(precision)
        print ''

    elif clf == 'NN':
        p = None
        best = 0
        dropouts = [0.85, 0.9]
        kf = KFold(n_splits=5, shuffle=True, random_state=777)
        for d in dropouts:
            scores = []
            for train_index, test_index in kf.split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                precision, F = NN(X_train, Y_train, X_test, Y_test, d)
                scores.append(F)
            score = np.average(scores)
            if score > best:
                best = score
                p = d
        print 'the best dropout: ' + str(p)

        precision, F = NN(X, Y, x, y, p, weight=True, feat=feat)
        print 'F1: ' + str(F)
        print 'precision ' + str(precision)
        print ''

    else:
        print 'input error!'
        return


def get_scores(elembase, clf, feat):
    print 'start leave-one-event-out cross-validation!'

    # VMU 2015
    events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
              'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']

    scores = []
    if clf == 'RF':
        p = model_selection(elembase, events, clf, feat)

    elif clf == 'NN':
        p = model_selection(elembase, events, clf, feat)

    else:
        print 'input error!'
        return

    print 'F1 scores for each event:'
    for i in xrange(len(events)):
        train, test = devide(elembase, events[i])
        X, Y, x, y = get_input(train, test, feat)
        if clf == 'RF':
            precision, F = RF(X, Y, x, y, p[0])
        elif clf == 'NN':
            precision, F = NN(X, Y, x, y, p[0])
        else:
            print 'input error!'
            return
        print events[i] + ' : ' + str(F)
        scores.append(F)

    mean = np.average(scores)

    print clf + ' average F1 score for all events:' + str(mean)
    print ''


def run(clf='', task=None, feat='1'):
    f = open('resources/dataset.txt', 'rb')
    elembase = json.load(f)
    f.close()

    f = open('resources/dist_feats.txt', 'rb')
    dist_feats = cPickle.load(f)
    f.close()

    f = open('resources/agree_feats.txt', 'rb')
    agree_feats = cPickle.load(f)
    f.close()

    # load baseline features
    f = open('resources/add_dict.txt', 'rb')
    add_dict = cPickle.load(f)
    f.close()

    for elem in elembase:
        id = elem['tweet_id']
        elem['dist'] = dist_feats[id]
        elem['agree'] = agree_feats[id]
        elem['base+add'] = add_dict[id]
    if task:
        run_task(elembase, clf, feat)
        return

    get_scores(elembase, clf, feat)


if __name__ == '__main__':
    # task with RF
    run(clf='RF', task=True)

    # event with RF
    run(clf='RF')

    # task with NN
    run(clf='NN', task=True)

    # event with NN
    run(clf='NN')
