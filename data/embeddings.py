import os
import pickle
import numpy as np

class WordEmbeddings:

    def __init__(self, basedir, lang):
        self.lang = lang
        filename = 'wiki.multi.{}.vec'.format(lang)
        we_path = os.path.join(basedir, filename)

        if not os.path.exists(we_path + '.pkl'):
            lines = open(we_path).readlines()
            nwords, dims = [int(x) for x in lines[0].split()]
            we = np.zeros((nwords, dims), dtype=float)
            worddim = {}
            for i, line in enumerate(lines[1:]):
                if (i + 1) % 100 == 0:
                    print('\r{}/{}'.format(i + 1, len(lines)), end='')
                word, *vals = line.split()
                if len(vals) == dims:
                    worddim[word] = i
                    we[i, :] = np.array(vals).astype(float)

            print('saving...')
            pickle.dump((worddim, we), open(we_path + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            print('loading pkl in {}'.format(we_path + '.pkl'))
            (worddim, we) = pickle.load(open(we_path + '.pkl', 'rb'))

        self.worddim = worddim
        self.we = we
        self.dimword = {v:k for k,v in self.worddim.items()}

    def __getitem__(self, key):
        return self.we[self.worddim[key]]

    def dim(self):
        return self.we.shape[1]

    def __contains__(self, key):
        return key in self.worddim

    def most_similar(self, word_vect, k):
        if word_vect.ndim == 1:
            word_vect = word_vect.reshape(1,-1)
        assert word_vect.shape[1] == self.dim(), 'inconsistent dimensions'

        sim = np.dot(word_vect,self.we.T)
        order = np.argsort(-1*sim, axis=1)[:,:k]

        similar_words = [[self.dimword[order[vi,ki]] for ki in range(k)] for vi in range(word_vect.shape[0])]
        sim_scores = sim[:,order]
        return similar_words, sim_scores

    def get_vectors(self, wordlist):
        indexes = np.array([self.worddim[w] for w in wordlist])
        return self.we[indexes]
