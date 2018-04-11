import numpy as np
import json
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn import preprocessing
from data.Manager import manager


def Clustering():
    with open('data/twitter_embed.txt') as f:
        twitter = json.load(f)

    data = []
    for elem in twitter:
        if elem['event'] == 'sandy':
            if elem['lng']!='en':
                continue
            data.append(elem)

    vecs = []
    for elem in data:
        vecs.append(elem['embed'])

    vecs = np.array(vecs)
    vecs = preprocessing.normalize(vecs)
    cluster = SpectralClustering(n_clusters=10, eigen_solver=None,
                                 random_state=None, n_init=10, gamma=1.0,
                                 affinity='rbf', n_neighbors=10, eigen_tol=0.0,
                                 assign_labels='kmeans', degree=3, coef0=1,
                                 kernel_params=None, n_jobs=-1)
    cluster_dict=defaultdict(list)
    preds = cluster.fit_predict(vecs).tolist()
    for elem, pred in zip(data, preds):
        cluster_dict[pred].append(elem)

    for pred in cluster_dict:
        if cluster_dict[pred]:
            fake, real =0 ,0
            for elem in cluster_dict[pred]:
                if elem['label']=='fake':
                    fake+=1
                else:
                    real+=1
            print 'cluster ', pred
            print 'fake ',fake
            print 'real', real

if __name__ == '__main__':
    Clustering()