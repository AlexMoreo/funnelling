from os.path import join, exists
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from data.reader.jrcacquis_reader import *
from data.languages import lang_set, NLTK_LANGMAP
from data.text_preprocessor import NLTKLemmaTokenizer
import pickle
from learner.learners import *
from random import shuffle

jrcacquis_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"

def _group_by_lang(doc_list, langs):
    return {lang:[(d.text, d.categories) for d in doc_list if d.lang == lang] for lang in langs}

def _preprocess(documents, lang):
    tokens = NLTKLemmaTokenizer(lang, verbose=True)
    sw = stopwords.words(NLTK_LANGMAP[lang])
    return [' '.join([w for w in tokens(doc) if w not in sw]) for doc in documents]

class MultilingualDataset:

    def __init__(self):
        self.multiling_dataset = {}

    def add(self, lang, Xtr,Ytr,Xte,Yte):
        self.multiling_dataset[lang] = ((Xtr, Ytr), (Xte, Yte))

    def save(self, file):
        pickle.dump(self.multiling_dataset, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item):
        return self.multiling_dataset[item]

    @classmethod
    def load(cls, file):
        data = MultilingualDataset()
        data.multiling_dataset = pickle.load(open(file, 'rb'))
        return data

    def lXtr(self):
        return {lang:Xtr for (lang, ((Xtr,_),(_,_))) in self.multiling_dataset.items()}

    def lXte(self):
        return {lang:Xte for (lang, ((_,_),(Xte,_))) in self.multiling_dataset.items()}

    def lYtr(self):
        return {lang:Ytr for (lang, ((_,Ytr),(_,_))) in self.multiling_dataset.items()}

    def lYte(self):
        return {lang:Yte for (lang, ((_,_),(_,Yte))) in self.multiling_dataset.items()}

    def langs(self):
        langs =  list(self.multiling_dataset.keys())
        langs.sort()
        return langs

    def show_dimensions(self):
        for (lang, ((Xtr, Ytr), (Xte, Yte))) in self.multiling_dataset.items():
            print("Lang {}, Xtr={}, ytr={}, Xte={}, yte={}".format(lang, Xtr.shape, Ytr.shape, Xte.shape, Yte.shape))


# creates a dictionary of {<lang>:((trX,trY),(teX,teY)) where trX,teX are csr_matrices with independent language feature spaces
def prepare_dataset_independent_matrices(langs, train_years, test_years, jrcacquis_datapath, cat_selection_policy,
                                         tr_parallel='avoid', te_parallel='force'):

    cat_list = inspect_eurovoc(jrcacquis_datapath, select=cat_selection_policy)
    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=train_years,
                                                 cat_filter=cat_list, cat_threshold=1, parallel=tr_parallel)
    test_docs, _ = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=test_years, cat_filter=label_names,
                                   parallel=te_parallel)

    training_docs = _group_by_lang(training_docs, langs)
    test_docs = _group_by_lang(test_docs, langs)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    multiling_dataset = MultilingualDataset()
    for lang in langs:
        print("\nprocessing %d training and %d test for language <%s>" % (len(training_docs[lang]), len(test_docs[lang]), lang))

        tr_data, tr_labels = zip(*training_docs[lang])
        te_data, te_labels = zip(*test_docs[lang])

        tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                tokenizer=NLTKLemmaTokenizer(lang, verbose=True),
                                stop_words=stopwords.words(NLTK_LANGMAP[lang]))

        Xtr = tfidf.fit_transform(tr_data)
        Xte = tfidf.transform(te_data)
        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)

        multiling_dataset.add(lang,Xtr,Ytr,Xte,Yte)

    multiling_dataset.show_dimensions()
    return multiling_dataset

# creates a dictionary of {<lang>:((trX,trY),(teX,teY)) where trX,teX are csr_matrices sharing a single yuxtaposed feature space
def prepare_dataset_juxtaposed_matrices(langs, train_years, test_years, jrcacquis_datapath, cat_selection_policy,
                                        tr_parallel='avoid', te_parallel='force'):

    cat_list = inspect_eurovoc(jrcacquis_datapath, select=cat_selection_policy)
    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=train_years,
                                                 cat_filter=cat_list, cat_threshold=1, parallel=tr_parallel)
    test_docs, _ = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=test_years, cat_filter=label_names,
                                   parallel=te_parallel)

    training_docs = _group_by_lang(training_docs, langs)
    test_docs = _group_by_lang(test_docs, langs)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    multiling_dataset = MultilingualDataset()
    tr_data_stack = []
    for lang in langs:
        print("\nprocessing %d training and %d test for language <%s>" % (len(training_docs[lang]), len(test_docs[lang]), lang))
        tr_data, tr_labels = zip(*training_docs[lang])
        te_data, te_labels = zip(*test_docs[lang])
        tr_data = _preprocess(tr_data, lang)
        te_data = _preprocess(te_data, lang)
        tr_data_stack.extend(tr_data)
        multiling_dataset.add(lang, tr_data, tr_labels, te_data, te_labels)

    tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True)
    tfidf.fit(tr_data_stack)

    for lang in langs:
        print("\nweighting documents for language <%s>" % (lang))
        (tr_data, tr_labels), (te_data, te_labels) = multiling_dataset[lang]
        Xtr = tfidf.transform(tr_data)
        Xte = tfidf.transform(te_data)
        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)
        multiling_dataset.add(lang,Xtr,Ytr,Xte,Yte)

    multiling_dataset.show_dimensions()
    return multiling_dataset

#generates the "feature-independent" and the "yuxtaposed" datasets
def prepare_both_datasets(save_path, langs, train_years, test_years, jrcacquis_datapath, cat_selection_policy):
    print('Generating feature-independent dataset...')
    if not exists(save_path):
        prepare_dataset_independent_matrices(langs, train_years, test_years, jrcacquis_datapath,
                                             cat_selection_policy).save(save_path)

    print('Generating yuxtaposed dataset...')
    save_path = save_path.replace('_processed.pickle','_yuxtaposed_processed.pickle')
    if not exists(save_path):
        prepare_dataset_juxtaposed_matrices(langs, train_years, test_years, jrcacquis_datapath,
                                            cat_selection_policy).save(save_path)

#-----------------------------------------------------------------------------------------------------------------------
def __years_to_str(years):
    if isinstance(years, list):
        if len(years) > 1:
            return str(years[0])+'-'+str(years[-1])
        return str(years[0])
    return str(years)

def __prepare_jrc_datasplit(train_years, test_years, cat_selection_policy):
    langset = lang_set['JRC_NLTK']
    config_name = 'jrc_nltk_'+__years_to_str(train_years)+'vs'+\
                  __years_to_str()+'_'+cat_selection_policy+'_noparallel_processed.pickle'

    jrc_processed_path = join(jrcacquis_datapath, config_name)
    prepare_both_datasets(jrc_processed_path, langset, train_years, test_years, jrcacquis_datapath, cat_selection_policy)

#-----------------------------------------------------------------------------------------------------------------------
def __prepare_jrc_noparallel_big(cat_selection_policy):
    train_years = list(range(1996,2006)) #10 years before 2006
    test_years = [2006]
    __prepare_jrc_datasplit(train_years, test_years, cat_selection_policy)

# -----------------------------------------------------------------------------------------------------------------------
def __prepare_jrc_noparallel_medium(cat_selection_policy):
    train_years = [2003, 2004, 2005]
    test_years = [2006]
    __prepare_jrc_datasplit(train_years, test_years, cat_selection_policy)

#-----------------------------------------------------------------------------------------------------------------------
def __prepare_jrc_noparallel_small(cat_selection_policy):
    train_years = [2005]
    test_years = [2006]
    __prepare_jrc_datasplit(train_years, test_years, cat_selection_policy)


if __name__=='__main__':
    for policy in ['broadest', 'all']:
        __prepare_jrc_noparallel_small(policy)
        __prepare_jrc_noparallel_medium(policy)
        __prepare_jrc_noparallel_big(policy)
