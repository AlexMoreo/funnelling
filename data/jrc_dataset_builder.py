from os.path import join, exists
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from data.reader.jrcacquis_reader import *
from data.languages import lang_set, NLTK_LANGMAP
from data.reader.wikipedia_tools import fetch_wikipedia_multilingual, random_wiki_sample
from data.text_preprocessor import NLTKLemmaTokenizer
import pickle
from learner.learners import *
from random import shuffle

JRC_DATAPATH = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"
WIKI_DATAPATH = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia/multilingual_docs_JRC_NLTK"


class MultilingualDataset:

    def __init__(self):
        self.multiling_dataset = {}

    def add(self, lang, Xtr, Ytr, Xte, Yte, tr_ids=None, te_ids=None):
        self.multiling_dataset[lang] = ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids))

    def save(self, file):
        pickle.dump(self.multiling_dataset, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)
        return self

    def __getitem__(self, item):
        return self.multiling_dataset[item]

    @classmethod
    def load(cls, file):
        data = MultilingualDataset()
        data.multiling_dataset = pickle.load(open(file, 'rb'))
        return data

    def lXtr(self):
        return {lang:Xtr for (lang, ((Xtr,_,_),_)) in self.multiling_dataset.items()}

    def lXte(self):
        return {lang:Xte for (lang, (_,(Xte,_,_))) in self.multiling_dataset.items()}

    def lYtr(self):
        return {lang:Ytr for (lang, ((_,Ytr,_),_)) in self.multiling_dataset.items()}

    def lYte(self):
        return {lang:Yte for (lang, (_,(_,Yte,_))) in self.multiling_dataset.items()}

    def langs(self):
        langs =  list(self.multiling_dataset.keys())
        langs.sort()
        return langs

    def show_dimensions(self):
        for (lang, ((Xtr, Ytr, IDtr), (Xte, Yte, IDte))) in self.multiling_dataset.items():
            print("Lang {}, Xtr={}, ytr={}, Xte={}, yte={}".format(lang, Xtr.shape, Ytr.shape, Xte.shape, Yte.shape))


def _group_by_lang(doc_list, langs):
    return {lang:[(d.text, d.categories, d.parallel_id+'__'+d.id) for d in doc_list if d.lang == lang] for lang in langs}

def _preprocess(documents, lang):
    tokens = NLTKLemmaTokenizer(lang, verbose=True)
    sw = stopwords.words(NLTK_LANGMAP[lang])
    return [' '.join([w for w in tokens(doc) if w not in sw]) for doc in documents]


# creates a dictionary of {<lang>:((trX,trY),(teX,teY)) where trX,teX are csr_matrices with independent language feature spaces
def prepare_dataset_independent_matrices(langs, training_docs, test_docs, label_names, wiki_docs=[]):

    training_docs = _group_by_lang(training_docs, langs)
    test_docs = _group_by_lang(test_docs, langs)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    lW = {}

    multiling_dataset = MultilingualDataset()
    for lang in langs:
        print("\nprocessing %d training, %d test, %d wiki for language <%s>" %
              (len(training_docs[lang]), len(test_docs[lang]), len(wiki_docs[lang]) if wiki_docs else 0, lang))

        tr_data, tr_labels, IDtr = zip(*training_docs[lang])
        te_data, te_labels, IDte = zip(*test_docs[lang])

        tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                tokenizer=NLTKLemmaTokenizer(lang, verbose=True),
                                stop_words=stopwords.words(NLTK_LANGMAP[lang]))

        Xtr = tfidf.fit_transform(tr_data)
        Xte = tfidf.transform(te_data)
        if wiki_docs:
            lW[lang] = tfidf.transform(wiki_docs[lang])
        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)

        multiling_dataset.add(lang, Xtr, Ytr, Xte, Yte, IDtr, IDte)

    multiling_dataset.show_dimensions()

    if wiki_docs:
        return multiling_dataset, lW
    else:
        return multiling_dataset

# creates a dictionary of {<lang>:((trX,trY),(teX,teY)) where trX,teX are csr_matrices sharing a single yuxtaposed feature space
def prepare_dataset_juxtaposed_matrices(langs, training_docs, test_docs, label_names):

    training_docs = _group_by_lang(training_docs, langs)
    test_docs = _group_by_lang(test_docs, langs)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    multiling_dataset = MultilingualDataset()
    tr_data_stack = []
    for lang in langs:
        print("\nprocessing %d training and %d test for language <%s>" % (len(training_docs[lang]), len(test_docs[lang]), lang))
        tr_data, tr_labels, tr_ID = zip(*training_docs[lang])
        te_data, te_labels, te_ID = zip(*test_docs[lang])
        tr_data = _preprocess(tr_data, lang)
        te_data = _preprocess(te_data, lang)
        tr_data_stack.extend(tr_data)
        multiling_dataset.add(lang, tr_data, tr_labels, te_data, te_labels, tr_ID, te_ID)

    tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True)
    tfidf.fit(tr_data_stack)

    for lang in langs:
        print("\nweighting documents for language <%s>" % (lang))
        (tr_data, tr_labels, tr_ID), (te_data, te_labels, te_ID) = multiling_dataset[lang]
        Xtr = tfidf.transform(tr_data)
        Xte = tfidf.transform(te_data)
        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)
        multiling_dataset.add(lang,Xtr,Ytr,Xte,Yte,tr_ID,te_ID)

    multiling_dataset.show_dimensions()
    return multiling_dataset


def filter_by_lang(doclist, lang):
    return [d for d in doclist if d.lang == lang]

#generates the "feature-independent" and the "yuxtaposed" datasets
def prepare_datasets(langs, train_years, test_years, cat_policy, most_common_cat=-1, max_wiki=5000):

    config_name = 'jrc_nltk_' + __years_to_str(train_years) + 'vs' + __years_to_str(test_years) + \
                  '_' + cat_policy + ('_top' + str(most_common_cat) if most_common_cat!=-1 else '')+ '_noparallel_processed'
    indep_path = join(JRC_DATAPATH, config_name + '.pickle')
    upper_path = join(JRC_DATAPATH, config_name + '_upper.pickle')
    yuxta_path = join(JRC_DATAPATH, config_name + '_yuxtaposed.pickle')
    wiki_path  = join(JRC_DATAPATH, config_name + '.wiki.pickle')
    if exists(indep_path) and exists(upper_path) and exists(yuxta_path):
        print(config_name + " already calculated. Skipping.")
        return

    cat_list = inspect_eurovoc(JRC_DATAPATH, select=cat_policy)
    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=JRC_DATAPATH, years=train_years, cat_filter=cat_list, cat_threshold=1, parallel=None, most_frequent=most_common_cat)
    test_docs, _ = fetch_jrcacquis(langs=langs, data_path=JRC_DATAPATH, years=test_years, cat_filter=label_names, parallel='force')

    print('Generating feature-independent dataset...')
    training_docs_no_parallel = random_sampling_avoiding_parallel(training_docs)
    if not exists(indep_path):
        wiki_docs = fetch_wikipedia_multilingual(WIKI_DATAPATH, langs, min_words=50, deletions=False)
        wiki_docs = random_wiki_sample(wiki_docs, max_wiki)

        lang_data, wiki_docs = prepare_dataset_independent_matrices(langs, training_docs_no_parallel, test_docs, label_names, wiki_docs)
        lang_data.save(indep_path)
        pickle.dump(wiki_docs, open(wiki_path, 'wb'), pickle.HIGHEST_PROTOCOL)

    print('Generating upper-bound (English-only) dataset...')
    if not exists(upper_path):
        training_docs_eng_only = filter_by_lang(training_docs, 'en')
        test_docs_eng_only = filter_by_lang(test_docs, 'en')
        prepare_dataset_independent_matrices(['en'], training_docs_eng_only, test_docs_eng_only, label_names).save(upper_path)

    print('Generating yuxtaposed dataset...')
    if not exists(yuxta_path):
        prepare_dataset_juxtaposed_matrices(langs, training_docs_no_parallel, test_docs, label_names).save(yuxta_path)

#-----------------------------------------------------------------------------------------------------------------------
def __years_to_str(years):
    if isinstance(years, list):
        if len(years) > 1:
            return str(years[0])+'-'+str(years[-1])
        return str(years[0])
    return str(years)

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    langs = lang_set['JRC_NLTK']
    prepare_datasets(langs, train_years=list(range(1986, 2006)), test_years=[2006], cat_policy='all', most_common_cat=300)
    prepare_datasets(langs, train_years=list(range(1986, 2006)), test_years=[2006], cat_policy='broadest')
    prepare_datasets(langs, train_years=list(range(1986, 2006)), test_years=[2006], cat_policy='all')
