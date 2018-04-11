import json
import cPickle
import numpy as np
from proposed_models import RF, NN, MCG
from sklearn.model_selection import KFold
from Model import Supervise
from sklearn.metrics import f1_score, make_scorer
from sklearn import svm
from sklearn.model_selection import GridSearchCV

with open('data/processed_twitter.txt', 'r') as f:
    twitter = json.load(f)
with open('data/processed_baidu.txt', 'r') as f:
    baidu = json.load(f)


# divide data into test and train set
def devide(dataset, event, cross=False):
    test = []
    train = []
    if cross:
        train = baidu
        for elem in dataset:
            if elem['event'] == event:
                test.append(elem)
    else:
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


def model_selection(elembase, events, clf, feat):
    print 'model selection...'
    supervise = Supervise()
    best = 0
    if clf == 'RF':
        fractions = [0.2, 0.25, 0.3, 0.35]
        for frac in fractions:
            scores = []
            for event in events:
                train, test = devide(elembase, event)
                X, Y, x, y = supervise.prepare(train, test, feat)
                F, precision, recall = RF(X, Y, x, y, frac)
                scores.append(F)
            score = np.average(scores)
            if score > best:
                best = score
                p = [frac]
        print 'the best frac: ' + str(p[0])
        return p

    elif clf == 'NN':
        dropouts = [0.4, 0.5, 0.6]
        for d in dropouts:
            scores = []
            for event in events:
                train, test = devide(elembase, event)
                X, Y, x, y = supervise.prepare(train, test, feat)
                F = NN(X, Y, x, y, dropout=d)
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
    X, Y, x, y = Supervise().prepare(train, test, feat)
    # print 'K-Fold cross-validation...'
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
                F, precision, recall = RF(X_train, Y_train, X_test, Y_test, frac)
                scores.append(F)
            score = np.average(scores)
            if score > best:
                best = score
                p = frac
        print 'the best frac: ' + str(p)

        F, precision, recall = RF(X, Y, x, y, p, weight=True, feat=feat)
        print 'F1: ' + str(F)
        print 'precision ' + str(precision)
        print ''

    elif clf == 'NN':
        F, precision, recall = NN(X, Y, x, y, epochs=30)
        print 'F1', F


    elif clf == 'MCG':

        F, precision, recall = MCG(train, test, 0.3, 0.4)

        print 'best F1', F


    elif clf == 'SVM':
        parameters = {'C': [0.5, 1, 1.5], 'gamma': [0.1, 0.01]}
        svc = svm.SVC(random_state=70)
        gs = GridSearchCV(svc, parameters, verbose=1, n_jobs=-1, cv=5, scoring=make_scorer(f1_score))
        gs.fit(X, Y)
        print gs.best_params_
        CLF = gs.best_estimator_

        CLF.fit(X, Y)
        y_pred = CLF.predict(x)
        F = f1_score(y, y_pred)

        print 'F1: ', F
        print ''


def get_scores(elembase, clf, feat):
    print 'start leave-one-event-out cross-validation!'

    # VMU 2015
    events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
              'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']

    scores = []
    if clf == 'RF':
        p = model_selection(elembase, events, clf, feat)

    elif clf == 'NN':
        # p = model_selection(elembase, events, clf, feat)
        pass
    else:
        print 'input error!'
        return

    print 'F1 scores for each event:'
    supervise = Supervise()
    for i in xrange(len(events)):
        train, test = devide(elembase, events[i])
        X, Y, x, y = supervise.prepare(train, test, feat)
        if clf == 'RF':
            F, precision, recall = RF(X, Y, x, y, p[0])
        elif clf == 'NN':
            F, precision, recall = NN(X, Y, x, y)
        else:
            print 'input error!'
            return
        print events[i] + ' : ' + str(F)
        scores.append(F)

    mean = np.average(scores)

    print clf + ' average F1 score for all events:', mean
    print ''


def run(clf='', task=None, feat='1'):
    if task:
        run_task(twitter, clf, feat)
        return

    get_scores(twitter, clf, feat)


if __name__ == '__main__':
    # task with NN
    run(clf='NN', feat='1', task=True)

    # event with NN
    run(clf='NN')
