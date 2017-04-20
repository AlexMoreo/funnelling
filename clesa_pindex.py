from clesa import CLESA
import numpy as np

class CLESA_Pindex(CLESA):

    def __init__(self, num_permutations, similarity='dot'):
        super(CLESA_Pindex, self).__init__(similarity)
        self.num_permutations = num_permutations

    def transform(self, lX):
        lX_ = super(CLESA_Pindex, self).transform(lX)
        return np.argsort(-lX_, axis=1)[:,:self.num_permutations]



