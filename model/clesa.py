import numpy as np
import sklearn

class CLESA(object):

    supported_similarity = ['dot', 'cosine']

    def __init__(self, similarity='dot'):
        if similarity not in self.supported_similarity:
            raise ValueError("Similarity method %s is not supported" % similarity)
        self.similarity = similarity

    # lW is a dictionary of (language, doc-by-term matrix)
    # the matrix is already processed (i.e., weighted, reduced, etc.)
    def fit(self, lW):
        self.l = lW.keys()
        self.lW = lW
        self.dimensionality() #checks consistency in the vector space across languages

    # lX is a dictionary of (language, doc-by-term matrix) that
    # is to be transformed into the CL-ESA space. Returns a single matrix in this space
    # containing all documents from all languages
    def transform(self, lX):
        if not hasattr(self, "lW"):
            raise ValueError("Error, transform method called before fit")

        langs = lX.keys()
        for lang in langs:
            if lang not in self.l:
                raise ValueError("Language %s not in scope" % lang)

        _clesaX = []
        #computes the CL-ESA representation
        for lang in langs:
            X = lX[lang]
            W = self.lW[lang]
            _X = self.ESA(X,W)
            _clesaX.append(_X)

        return np.vstack(_clesaX)

    def fit_transform(self, lW, lX):
        self.fit(lW)
        return self.transform(lX)

    def ESA(self,X,W):
        if X.shape[1] != W.shape[1]:
            raise ValueError("The feature space of X and W does not agree")

        if self.similarity in ['dot', 'cosine']:
            if self.similarity == 'cosine':
                X = sklearn.preprocessing.normalize(X, norm='l2', axis=1, copy=True)
                W = sklearn.preprocessing.normalize(W, norm='l2', axis=1, copy=True)
            return (X * W.transpose()).toarray()

    def dimensionality(self):
        if hasattr(self, "dimensions"):
            return self.dimensions

        for lang in self.l:
            if not hasattr(self, "dimensions"):
                self.dimensions = self.lW[lang].shape[0]
            elif self.dimensions != self.lW[lang].shape[0]:
                raise ValueError("The dimensionality of the W matrix is inconsistent across languages")





