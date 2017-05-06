from __future__ import print_function
def warn(*args, **kwargs):
   pass
import warnings
warnings.warn = warn
import os
from os.path import join
from data.reader.jrcacquis_reader import fetch_jrcacquis, inspect_eurovoc
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from time import time as T
from data.reader.wikipedia_tools import fetch_wikipedia_multilingual
from model.clesa import CLESA
from data.languages import *
from sklearn.preprocessing import MultiLabelBinarizer

"""
Required nltk.download() packages:
    - punkt: for word_tokenize
    - snoball_data:
    - stopwords
"""

#Languages which can be stemmed by means of NLTK
#danish dutch english finnish french german hungarian italian norwegian porter portuguese romanian russian spanish swedish

JRC_ACQUIS = "JRCAcquis"
DATASETS_SUPPORTED = [JRC_ACQUIS]

class CLESA_Data:
    def __init__(self, tr_years=None, te_years=None, langmap='Unknown', notes=""):
        self.lXtr={}
        self.lYtr={}
        self.lXte={}
        self.lYte={}
        self.lW={}
        self.tr_years=tr_years
        self.te_years=te_years
        self.langmap=langmap
        self.notes=notes
        self.labels_processed = False

    def add(self,lang,Xtr,Ytr,Xte,Yte,W):
        self.lXtr[lang] = Xtr
        self.lYtr[lang] = Ytr
        self.lXte[lang] = Xte
        self.lYte[lang] = Yte
        self.lW[lang] = W

    def process_labels(self):
        if self.labels_processed: return
        langs = self.lYtr.keys()
        def _expandlabels(y,langs):
            doclabels = []
            for lang in langs: doclabels.extend(y[lang])
            return doclabels
        tr_doclabels = _expandlabels(self.lYtr, langs)
        te_doclabels = _expandlabels(self.lYte, langs)
        mlb = MultiLabelBinarizer()
        Ytr = mlb.fit_transform(tr_doclabels)
        Yte = mlb.transform(te_doclabels)

        def _reallocatelabels(y,langs,lX,lY):
            offset = 0
            for lang in langs:
                l_num_docs = lX[lang].shape[0]
                lY[lang] = y[offset:offset + l_num_docs]
                offset += l_num_docs

        _reallocatelabels(Ytr, langs, self.lXtr, self.lYtr)
        _reallocatelabels(Yte, langs, self.lXte, self.lYte)

        self.labels_processed = True


def show_classification_scheme(Y):
    class_count = {}
    for y in Y:
        for c in y:
            if c not in class_count:
                class_count[c] = 0
            class_count[c] += 1
    print("\nnum classes %d" % len(class_count))
    class_count = class_count.items()
    class_count.sort(key=lambda x: x[1], reverse=True)
    print(class_count)

def as_lang_dict(doc_list, langs):
    return {lang:[(d.text,d.categories) for d in doc_list if d.lang == lang] for lang in langs}

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
        #return [self.wnl.stem(t) for t in word_tokenize(doc)]

def year_list_as_str(years):
    y_str = list(years)
    y_str.sort()
    return '_'.join([str(y) for y in y_str])

def clesa_data_generator(dataset, langs, tr_years, te_years, jrcacquis_datapath, wikipedia_datapath, cat_threshold=50, pickle_name="", langmap="unknown"):

    if dataset not in DATASETS_SUPPORTED:
        raise ValueError("Dataset %s is not supportd" % dataset)

    if os.path.exists(pickle_name):
        print("unpickling %s" % pickle_name)
        return pickle.load(open(pickle_name, 'rb'))

    clesa_data = CLESA_Data(tr_years=year_list_as_str(tr_years), te_years=year_list_as_str(te_years), langmap=langmap, notes="from "+dataset)

    wiki_docs, n_pivots = fetch_wikipedia_multilingual(wikipedia_datapath, langs, min_words=100)
    if n_pivots == 0:
        raise ValueError("Wikipedia documents were not loaded correctly.")

    print("Fetching "+dataset+" data...")
    if dataset == JRC_ACQUIS:
        cat_list = inspect_eurovoc(jrcacquis_datapath, pickle_name="broadest_concepts.pickle")
        tr_request, final_cats = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=tr_years, cat_filter=cat_list, cat_threshold=cat_threshold)
        te_request, _ = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=te_years, cat_filter=final_cats)

        print("Training request length: %d" % len(tr_request))
        print("Test request length: %d" % len(te_request))
        print("Effective categories: %d" % len(final_cats))

        tr_request = as_lang_dict(tr_request, langs)
        te_request = as_lang_dict(te_request, langs)

    for lang in langs:
        print("\nprocessing %d training, %d test, and %d wikipedia documents for language <%s>" % (len(tr_request[lang]), len(te_request[lang]), n_pivots, lang))

        tr_data, tr_labels = zip(*tr_request[lang])
        te_data, te_labels = zip(*te_request[lang])

        tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                tokenizer=NLTKLemmaTokenizer(lang, verbose=True), stop_words=stopwords.words(NLTK_LANGMAP[lang]))
        XWtr= tfidf.fit_transform(list(tr_data)+wiki_docs[lang])
        Xte = tfidf.transform(te_data)
        Xtr = XWtr[:len(tr_data)]
        W   = XWtr[len(tr_data):]

        clesa_data.add(lang=lang,Xtr=Xtr,Ytr=tr_labels,Xte=Xte,Yte=te_labels,W=W)

    print("\nProcessing labels...")
    clesa_data.process_labels()

    print("Pickling CLESA data object in %s" % pickle_name)
    pickle.dump(clesa_data, open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    print("Done!")
    return clesa_data

"""
Read 65454 documents in total
filtered 37267 documents out without categories in the filter list
filtered 1374 documents out without categories in the filter list
Read 77144 documents in total
filtered 43338 documents out without categories in the filter list
filtered 0 documents out without categories in the filter list
training request length: 26813
test request length: 33806
Effective categories: 146
"""