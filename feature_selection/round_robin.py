import sys, os
import numpy as np
from joblib import Parallel, delayed
from feature_selection.tsr_function import *
import time
from os.path import join
import pickle

class FeatureSelectorFromRank:
    def __init__(self, k, features_rank):
        self._k=k
        self._features_rank=features_rank

    def fit(self, X, y=None):
        _, self.nF = X.shape
        if len(self._features_rank) != self.nF: raise ValueError("Error: features rank incomplete")
        self._k_best_feats = self._features_rank[:self._k]
        self._k_best_feats.sort()

    def transform(self, X):
        _,nF=X.shape
        if nF != self.nF: raise ValueError("Error: feature domain is not compatible with the fit domain")
        if not hasattr(self, '_k_best_feats'): raise NameError('Transform method called before fit.')
        return X[:, self._k_best_feats]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class RoundRobin:
    def __init__(self, k, score_func=information_gain, n_jobs=-1):
        self._score_func = score_func
        self._k = k
        self.n_jobs=n_jobs

    def fit(self, X, y):
        nF = X.shape[1]
        nC = y.shape[1]
        self.supervised_4cell_matrix = get_supervised_matrix(X, y, n_jobs=self.n_jobs)
        tsr_matrix = get_tsr_matrix(self.supervised_4cell_matrix, self._score_func)

        #enhance the tsr_matrix with the feature index
        tsr_matrix = [[(tsr_matrix[c,f], f) for f in range(nF)] for c in range(nC)]
        for c in range(nC):
            tsr_matrix[c].sort(key=lambda x: x[0])

        sel_feats = set()
        self._features_rank = []
        round = 0
        #while len(sel_feats) < self._k:
        while len(self._features_rank) < nF:
            feature_index = tsr_matrix[round].pop()[1]
            if feature_index not in sel_feats:
                sel_feats.add(feature_index)
                self._features_rank.append(feature_index)
            round = (round+1) % nC

        self.fs_rank = FeatureSelectorFromRank(k=self._k, features_rank=self._features_rank)
        self.fs_rank.fit(X,y)
        self._k_best_feats = self.fs_rank._k_best_feats
        self.supervised_4cell_matrix = self.supervised_4cell_matrix[:, self._k_best_feats]

    def transform(self, X):
        if not hasattr(self, 'fs_rank'): raise NameError('Transform method called before fit.')
        return self.fs_rank.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)



def fit_round_robin(X, Y, k, score_func=information_gain, features_rank_pickle_path=None):
    if k==None: return
    nD,nF = X.shape
    if isinstance(k, float):
        if k <=0.0 or k>1.0: raise ValueError("Feature selection ratio should be contained in (0,1]")
        k = int(k * nF)
    if k >= nF:
        print("Warning: number of features to select is greater than the actual number of features (ommitted).")
        return
    print('Feature selection: round robin, %s, select %d/%d features.' % (score_func.__name__, k, nF))

    #check if the ranking of features for this setting has already been calculated
    if features_rank_pickle_path and os.path.exists(features_rank_pickle_path):
        features_rank = pickle.load(open(features_rank_pickle_path, 'rb'))
        fs = FeatureSelectorFromRank(k=k, features_rank=features_rank)
    else:
        fs = RoundRobin(score_func=score_func, k=k)

    fs.fit(X, Y)

    if features_rank_pickle_path and not os.path.exists(features_rank_pickle_path):
        print("Pickling ranked features for faster subsequent runs in %s" % features_rank_pickle_path)
        pickle.dump(fs._features_rank, open(features_rank_pickle_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    return fs