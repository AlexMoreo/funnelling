import numpy as np
import dill
import sklearn
from sklearn.svm import SVC, LinearSVC
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


class CLESA(object):

    supported_similarity = ['dot', 'cosine']

    def __init__(self, similarity='dot', centered=False, post_norm=False):
        if similarity not in self.supported_similarity:
            raise ValueError("Similarity method %s is not supported" % similarity)
        self.similarity = similarity
        self.centered = centered
        self.post_norm=post_norm

    # lW is a dictionary of (language, doc-by-term matrix)
    # the matrix is already processed (i.e., weighted, reduced, etc.)
    def fit(self, lW):
        self.l = lW.keys()
        self.lW = lW
        self.dimensionality() #checks consistency in the vector space across languages

    # lX is a dictionary of (language, doc-by-term matrix) that
    # is to be transformed into the CL-ESA space. Returns a matrix in this space
    # containing all documents from all languages, and the label-matrix stacked consistently.
    def transform(self, lX, lY):
        if not hasattr(self, "lW"):
            raise ValueError("Error, transform method called before fit")

        for lang in lX.keys():
            if lang not in self.l:
                raise ValueError("Language %s not in scope" % lang)

        _clesaX = []
        _clesaY = []
        #computes the CL-ESA representation
        for lang in lX.keys():
            X = lX[lang]
            W = self.lW[lang]
            _X = self.ESA(X,W)
            _clesaX.append(_X)
            _clesaY.append(lY[lang])

        return np.vstack(_clesaX), np.vstack(_clesaY)

    def fit_transform(self, lW, lX, lY):
        self.fit(lW)
        return self.transform(lX, lY)

    def ESA(self,X,W):
        if X.shape[1] != W.shape[1]:
            raise ValueError("The feature spaces for X=%s and W=%s do not agree" % (str(X.shape),str(W.shape)))

        if self.similarity in ['dot', 'cosine']:
            if self.similarity == 'cosine':
                X = sklearn.preprocessing.normalize(X, norm='l2', axis=1, copy=True)
                W = sklearn.preprocessing.normalize(W, norm='l2', axis=1, copy=True)

            XW = (X * W.transpose()).toarray()
            if self.centered:
                pX = np.sum(X, axis=1)/X.shape[1]
                pW = np.sum(W, axis=1)/W.shape[1]
                pXpW= np.sqrt(pX*pW.transpose())
                out = XW - pXpW
            else:
                out = XW

            if self.post_norm:
                return sklearn.preprocessing.normalize(out, norm='l2', axis=1, copy=True)
            else:
                return out

    def dimensionality(self):
        if hasattr(self, "dimensions"):
            return self.dimensions

        for lang in self.l:
            if not hasattr(self, "dimensions"):
                self.dimensions = self.lW[lang].shape[0]
            elif self.dimensions != self.lW[lang].shape[0]:
                raise ValueError("The dimensionality of the W matrix is inconsistent across languages")

    def learner(self, **kwargs):
        if not hasattr(self, "svm"):
            self.svm = LinearSVC(**kwargs)
        return self.svm


class CLESA_PPindex(CLESA):

    def __init__(self, num_permutations, similarity='dot'):
        super(CLESA_PPindex, self).__init__(similarity)
        self.num_permutations = num_permutations

    def transform(self, lX, lY):
        lX_,lY_ = super(CLESA_PPindex, self).transform(lX, lY)
        return np.argsort(-lX_, axis=1)[:,:self.num_permutations], lY_

    def learner(self, **kwargs):
        if not hasattr(self, "svm"):
            distance_kernel = lambda X, W: cdist(X, W)
            kendalltau_kernel = lambda X, W: cdist(X, W, lambda u, v: stats.kendalltau(u, v)[0])
            spearmanr_kernel  = lambda X, W: cdist(X, W, lambda u, v: stats.spearmanr(u, v)[0])
            self.svm = SVC(kernel=kendalltau_kernel, **kwargs)
        return self.svm
