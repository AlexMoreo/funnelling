from clesa import CLESA
import numpy as np

class CLESA_PPindex(CLESA):

    def __init__(self, num_permutations, similarity='dot'):
        super(CLESA_PPindex, self).__init__(similarity)
        self.num_permutations = num_permutations

    def transform(self, lX):
        lX_ = super(CLESA_PPindex, self).transform(lX)
        return np.argsort(-lX_, axis=1)[:,:self.num_permutations]



