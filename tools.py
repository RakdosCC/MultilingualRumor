from sklearn import preprocessing
from sklearn.preprocessing import normalize

def imputer(data):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(data)
    data = imp.transform(data)
    return data

# scale to (0, 1)
def standardize(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    data = scaler.transform(data)
    return data


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
    if (tp + fp + fn)==0:
        F1=1
    else:
        F1 = 2 * float(tp) / (2 * tp + fp + fn)
    return F1, p, recall
