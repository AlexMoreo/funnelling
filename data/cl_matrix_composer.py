from __future__ import print_function
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
from model.clesa_ppindex import CLESA_PPindex

"""
Required nltk.download() packages:
    - punkt: for word_tokenize
    - snoball_data:
    - stopwords
"""

#Languages which can be stemmed by means of NLTK
#danish dutch english finnish french german hungarian italian norwegian porter portuguese romanian russian spanish swedish
LANGS_WITH_NLTK_STEMMING = ['da', 'nl', 'en', 'fi', 'fr', 'de', 'hu', 'it', 'pt', 'ro', 'es', 'sv']

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

    def add(self,lang,Xtr,Ytr,Xte,Yte,W):
        self.lXtr[lang] = Xtr
        self.lYtr[lang] = Ytr
        self.lXte[lang] = Xte
        self.lYte[lang] = Yte
        self.lW[lang] = W


def show_classification_scheme(Y):
    class_count = {}
    for y in Y:
        for c in y:
            if c not in class_count:
                class_count[c] = 0
            class_count[c] += 1
    print("\nnum classes %d" % len(class_count))
    print(class_count)

def as_lang_dict(doc_list, langs):
    return {lang:[(d.text,d.categories) for d in doc_list if d.lang == lang] for lang in langs}

nltk_langmap = {'da': 'danish', 'nl': 'dutch', 'en': 'english', 'fi': 'finnish', 'fr': 'french', 'de': 'german',
                    'hu': 'hungarian', 'it': 'italian', 'pt': 'portuguese', 'ro': 'romanian', 'es': 'spanish',
                    'sv': 'swedish'}

class NLTKLemmaTokenizer(object):

    def __init__(self, lang, verbose=False):
        if lang not in nltk_langmap:
            raise ValueError('Language %s is not supported in NLTK' % lang)
        self.verbose=verbose
        self.called = 0
        self.wnl = SnowballStemmer(nltk_langmap[lang])
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

if __name__ == "__main__":
    dataset = "JRCAcquis"
    langmap = "DEBUG"
    lang_set = {'NLTK':LANGS_WITH_NLTK_STEMMING, 'DEBUG':['en', 'es', 'it']}
    langs = lang_set[langmap]
    tr_years = [2005]
    te_years = [2006]
    cat_threshold=50

    jrcacquis_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"
    wikipedia_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia/multilingual_docs"

    tr_years_str = '_'.join([str(y) for y in tr_years])
    te_years_str = '_'.join([str(y) for y in te_years])

    pickle_name = join(jrcacquis_datapath, 'preprocessed_' + langmap
                               + '_tr_' + tr_years_str
                               + '_te_' + te_years_str
                               + '_broadests.pickle')

    if os.path.exists(pickle_name):
        print("unpickling %s" % pickle_name)
        clesa_data = pickle.load(open(pickle_name, 'rb'))
    else:
        clesa_data = CLESA_Data(tr_years=tr_years_str, te_years=te_years_str, langmap=langmap, notes="from "+dataset)

        wiki_docs, n_pivots = fetch_wikipedia_multilingual(wikipedia_datapath, langs, min_words=100)
        if n_pivots == 0:
            raise ValueError("Wikipedia documents were not loaded correctly.")

        print("Fetching JRC-Acquis V.3 data...")
        cat_list = inspect_eurovoc(jrcacquis_datapath, pickle_name="broadest_concepts.pickle")
        tr_request, final_cats = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=tr_years, cat_filter=cat_list, cat_threshold=cat_threshold)
        te_request, _ = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=te_years, cat_filter=final_cats)

        print("Training request length: %d" % len(tr_request))
        print("Test request length: %d" % len(te_request))
        print("Effective categories: %d" % len(final_cats))

        tr_request = as_lang_dict(tr_request, langs)
        te_request = as_lang_dict(te_request, langs)

        for lang in langs:
            print("\nprocessing %d training, %d test, and %d wikipedia documentsfor language <%s>" % (len(tr_request[lang]), len(te_request[lang]), n_pivots, lang))

            tr_data, tr_labels = zip(*tr_request[lang])
            te_data, te_labels = zip(*te_request[lang])

            tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                    tokenizer=NLTKLemmaTokenizer(lang, verbose=True), stop_words=stopwords.words(nltk_langmap[lang]))
            Xtr = tfidf.fit_transform(tr_data)
            Xte = tfidf.transform(te_data)
            W   = tfidf.transform(wiki_docs[lang])

            show_classification_scheme(tr_labels)
            show_classification_scheme(te_labels)

            clesa_data.add(lang=lang,Xtr=Xtr,Ytr=tr_labels,Xte=Xte,Yte=te_labels,W=W)

        print("Pickling CLESA data object in %s" % pickle_name)
        pickle.dump(clesa_data, open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    print("Done!")

    #clesa = CLESA()
    clesa = CLESA_PPindex(101)
    print("fitting...")
    clesa.fit(clesa_data.lW)
    print("transform training...")
    Xtr_ = clesa.transform(clesa_data.lXtr)
    print("transform test...")
    Xte_ = clesa.transform(clesa_data.lXte)

    print(Xtr_.shape)
    print(Xte_.shape)





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