import numpy as np
import dill
import sklearn
from sklearn.svm import SVC, LinearSVC
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky

class ESA(object):

    supported_similarity = ['dot', 'cosine']

    def __init__(self, similarity='dot', centered=False, post_norm=False, distributional_aware=False):
        if similarity not in self.supported_similarity:
            raise ValueError("Similarity method %s is not supported" % similarity)
        self.similarity = similarity
        self.centered = centered
        self.post_norm = post_norm
        self.distributional_aware = distributional_aware # this is a prototype!

    # W is a doc-by-term matrix of wikipedia documents
    # the matrix is already processed (i.e., weighted, reduced, etc.)
    def fit(self, W):
        self.W = W
        # if self.distributional_aware:
        #     try:
        #         Rpn = W.dot(W.transpose())
        #         #R = np.absolute(Rpn.toarray())
        #         R=Rpn.toarray() # <-- prueba
        #         self.L = cholesky(R)
        #     except np.linalg.linalg.LinAlgError:
        #         print('error: matrix is not semidefinite positive!!!')
        #         self.L = None
        #         #raise
        return self

    # X is a doc-by-term matrix that is to be transformed into the ESA space.
    def transform(self, X, Y=None):
        if not hasattr(self, "W"):
            raise ValueError("Error, transform method called before fit")

        W = self.W
        if X.shape[1] != W.shape[1]:
            raise ValueError("The feature spaces for X=%s and W=%s do not agree" % (str(X.shape), str(W.shape)))

        if self.similarity in ['dot', 'cosine']:
            if self.similarity == 'cosine':
                X = sklearn.preprocessing.normalize(X, norm='l2', axis=1, copy=True)
                W = sklearn.preprocessing.normalize(W, norm='l2', axis=1, copy=True)

            esa = (X.dot(W.transpose())).toarray()
            # if self.distributional_aware:
            #     if self.L is not None:
            #         esa = esa.dot(self.L)
            #     else:
            #         raise ValueError('The L matrix could not be found. [Error]')
            if self.centered:
                pX = (X > 0).sum(1) / float(X.shape[1])
                pW = (W > 0).sum(1) / float(W.shape[1])
                pXpW = np.sqrt(pX.dot(pW.transpose()))
                esa = esa - pXpW

            if self.post_norm:
                esa = sklearn.preprocessing.normalize(esa, norm='l2', axis=1, copy=True)

            return esa

    def fit_transform(self, W, X, Y=None):
        self.fit(W)
        return self.transform(X, Y)

    def dimensionality(self):
        return self.W.shape[0]

    def learner(self, **kwargs):
        if self.distributional_aware: # and self.L is None:
            R = np.absolute(self.W.dot(self.W.transpose()).toarray())
            self.svm = SVC(kernel=lambda X, Y: (X.dot(R)).dot(Y.T), **kwargs)
        else:
            #self.svm = LinearSVC(**kwargs)
            self.svm = SVC(kernel=lambda X, Y: X.dot(Y.T), **kwargs)
        return self.svm

class ESA_PPindex(ESA):

    def __init__(self, num_permutations, similarity='dot', centered=False, post_norm=False):
        super(ESA_PPindex, self).__init__(similarity, centered, post_norm)
        self.num_permutations = num_permutations

    def transform(self, X, Y=None):
        X_ = super(ESA_PPindex, self).transform(X, Y)
        return np.argsort(-X_, axis=1)[:,:self.num_permutations]

    def learner(self, **kwargs):
        if not hasattr(self, "svm"):
            # distance_kernel = lambda X, W: cdist(X, W)
            kendalltau_kernel = lambda X, W: cdist(X, W, lambda u, v: stats.kendalltau(u, v)[0])
            # spearmanr_kernel  = lambda X, W: cdist(X, W, lambda u, v: stats.spearmanr(u, v)[0])
            self.svm = SVC(kernel=kendalltau_kernel, **kwargs)
        return self.svm

class CLESA(ESA):

    def __init__(self, similarity='dot', centered=False, post_norm=False, distributional_aware=False):
        super(CLESA, self).__init__(similarity, centered, post_norm, distributional_aware)

    # lW is a dictionary of (language, doc-by-term matrix)
    # the matrix is already processed (i.e., weighted, reduced, etc.)
    def fit(self, lW):
        self.cl = {}
        for lang in lW.keys():
            l_esa = ESA(self.similarity, self.centered, self.post_norm, distributional_aware=self.distributional_aware)
            self.cl[lang] = l_esa.fit(lW[lang])

        self.dimensionality() #checks consistency in the vector space across languages

    # lX is a dictionary of (language, doc-by-term matrix) that
    # is to be transformed into the CL-ESA space. Returns a matrix in this space
    # containing all documents from all languages, and the label-matrix stacked consistently.
    def transform(self, lX, lY):
        if not hasattr(self, "cl"):
            raise ValueError("Error, transform method called before fit")

        langs = lX.keys()
        for lang in langs:
            if lang not in self.languages():
                raise ValueError("Language %s not in scope" % lang)

        return np.vstack([self.cl[lang].transform(lX[lang]) for lang in langs]), np.vstack([lY[lang] for lang in langs])

    def fit_transform(self, lW, lX, lY):
        self.fit(lW)
        return self.transform(lX, lY)

    def languages(self):
        return self.cl.keys()

    def dimensionality(self):
        if not hasattr(self, "dimensions"):
            for lang in self.languages():
                if not hasattr(self, "dimensions"):
                    self.dimensions = self.cl[lang].dimensionality()
                elif self.dimensions != self.cl[lang].dimensionality():
                    raise ValueError("The dimensionality of the W matrix is inconsistent across languages")

        return self.dimensions


class CLESA_PPindex(CLESA):

    def __init__(self, num_permutations, similarity='dot', centered=False, post_norm=False):
        super(CLESA_PPindex, self).__init__(similarity, centered, post_norm)
        self.num_permutations = num_permutations

    def transform(self, lX, lY):
        lX_,lY_ = super(CLESA_PPindex, self).transform(lX, lY)
        return np.argsort(-lX_, axis=1)[:,:self.num_permutations], lY_

    def learner(self, **kwargs):
        if not hasattr(self, "svm"):
            # distance_kernel = lambda X, W: cdist(X, W)
            kendalltau_kernel = lambda X, W: cdist(X, W, lambda u, v: stats.kendalltau(u, v)[0])
            # spearmanr_kernel  = lambda X, W: cdist(X, W, lambda u, v: stats.spearmanr(u, v)[0])
            self.svm = SVC(kernel=kendalltau_kernel, **kwargs)
        return self.svm
