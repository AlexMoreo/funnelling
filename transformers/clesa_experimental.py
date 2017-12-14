from model.clesa import ESA, CLESA
from sklearn.svm import SVC
from scipy.spatial.distance import cdist

class ESA_PPindex(ESA): # experimental

    def __init__(self, num_permutations, similarity='dot', centered=False, post=False):
        super(ESA_PPindex, self).__init__(similarity, centered, post)
        self.num_permutations = num_permutations

    def transform(self, X, Y=None):
        X_ = super(ESA_PPindex, self).transform(X, Y)
        return np.argsort(-X_, axis=1)[:,:self.num_permutations]

    def learner(self, **kwargs):
        if not hasattr(self, "svm"):
            kendalltau_kernel = lambda X, W: cdist(X, W, lambda u, v: stats.kendalltau(u, v)[0])
            # spearmanr_kernel  = lambda X, W: cdist(X, W, lambda u, v: stats.spearmanr(u, v)[0])
            self.svm = SVC(kernel=kendalltau_kernel, **kwargs)
        return self.svm


class CLESA_PPindex(CLESA):

    def __init__(self, num_permutations, similarity='dot', centered=False, post=False):
        super(CLESA_PPindex, self).__init__(similarity, centered, post)
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
