import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

class DCI:

    valid_dcf = ['cosine']

    # pivots is a dictionary {domain:[feat-index]}
    def __init__(self, pivots, dcf):
        self.domains = pivots.keys()
        for d in self.domains:
            if not hasattr(self, "dimensions"):
                self.dimensions = len(pivots[d])
            else:
                if self.dimensions != len(pivots[d]):
                    raise ValueError("Dimensions across domains do not agree")
        if dcf not in self.valid_dcf:
            raise ValueError("Distributional Correspondence Function should be in [%s]" % ' '.join(self.valid_dcf))
        self.dcf = getattr(DCI, dcf)

    # dU is a dictionary of {domain:dsm}, where dsm (distributional semantic model) is, e.g., a document-by-term csr_matrix
    def fit(self, dU):
        if not dU.keys().issubset(self.domains):
            raise ValueError("Domains in dU are not aligned with the initialization")
        self.dFP = {}
        for d in dU.keys:
            U = dU[d]
            d_pivots = self.pivots[d]
            self.dFP[d] = cdist(U.transpose(), U[:,d_pivots].transpose(), self.dcf)

    # dX is a dictionary of {domain:dsm}, where dsm (distributional semantic model) is, e.g., a document-by-term csr_matrix
    def transform(self, dX):
        if not hasattr(self, "dFP"):
            raise ValueError("Transform method called before fit")
        if not dX.keys().issubset(self.domains):
            raise ValueError("Domains in dX are not aligned with the initialization")

        _dX = {}
        for d in dX.keys:
            X = dX[d]
            dFP = self.dFP[d]
            _X = np.zeros(X.shape[0], self.dimensions)
            for doc,feat in X.nonzero():
                weight = X[doc,feat]
                _X[doc] += (weight*dFP[feat])
            _X = normalize(_X, norm='l2', axis=1)
            _dX[d] = _X

        return _dX

    def fit_transform(self, dX, dU):
        self.fit(dU)
        return self.transform(dX)

    def _prevalence(self, v):
        if isinstance(v, csr_matrix):
            print "prev---"
            return v.nnz /len(v)
        else:
            print "unexpected"

    def cosine(self, u, v):
        pu = self._prevalence(u)
        pv = self._prevalence(v)
        return cosine(u,v) - np.sqrt(pu*pv)

