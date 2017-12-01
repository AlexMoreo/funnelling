from data.languages import NLTK_LANGMAP
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

class NLTKLemmaTokenizer(object):

    def __init__(self, lang, verbose=False):
        if lang not in NLTK_LANGMAP:
            raise ValueError('Language %s is not supported in NLTK' % lang)
        self.verbose=verbose
        self.called = 0
        self.wnl = SnowballStemmer(NLTK_LANGMAP[lang])
        self.cache = {}

    def __call__(self, doc):
        self.called += 1
        if self.verbose and self.called%10==0:
            print("\r\t\t[documents processed %d]" % (self.called), end="")
        tokens = word_tokenize(doc)
        stems = []
        for t in tokens:
            if t not in self.cache:
                self.cache[t] = self.wnl.stem(t)
            stems.append(self.cache[t])
        return stems