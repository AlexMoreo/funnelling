from os.path import join, exists
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from data.reader.jrcacquis_reader import *
from data.languages import lang_set, NLTK_LANGMAP, RCV2_LANGS_WITH_NLTK_STEMMING
from data.reader.rcv_reader import fetch_RCV1, fetch_RCV2, fetch_topic_hierarchy
from data.reader.reuters21578_reader import fetch_reuters21579
from data.reader.wikipedia_tools import fetch_wikipedia_multilingual, random_wiki_sample
from data.text_preprocessor import NLTKLemmaTokenizer, preprocess_documents
import pickle
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
import itertools

class MultilingualDataset:

    def __init__(self):
        self.dataset_name = ""
        self.multiling_dataset = {}

    def add(self, lang, Xtr, Ytr, Xte, Yte, tr_ids=None, te_ids=None):
        self.multiling_dataset[lang] = ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids))

    def save(self, file):
        self.sort_indexes()
        pickle.dump(self, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)
        return self

    def __getitem__(self, item):
        if item in self.langs():
            return self.multiling_dataset[item]
        return None

    @classmethod
    def load(cls, file):
        data = pickle.load(open(file, 'rb'))
        data.sort_indexes()
        return data

    @classmethod
    def load_ids(cls, file):
        data = pickle.load(open(file, 'rb'))
        tr_ids = {lang:tr_ids for (lang,((_,_,tr_ids), (_,_,_))) in data.multiling_dataset.items()}
        te_ids = {lang: te_ids for (lang, ((_, _, _), (_, _, te_ids))) in data.multiling_dataset.items()}
        return tr_ids, te_ids


    def sort_indexes(self):
        for (lang, ((Xtr,_,_),(Xte,_,_))) in self.multiling_dataset.items():
            if issparse(Xtr): Xtr.sort_indices()
            if issparse(Xte): Xte.sort_indices()

    def set_view(self, categories=None, languages=None):
        if categories is not None:
            if isinstance(categories, int):
                categories = np.array([categories])
            elif isinstance(categories, list):
                categories = np.array(categories)
            self.categories_view = categories
        if languages is not None:
            self.languages_view = languages


    def lXtr(self):
        return {lang:Xtr for (lang, ((Xtr,_,_),_)) in self.multiling_dataset.items() if lang in self.langs()}

    def lXte(self):
        return {lang:Xte for (lang, (_,(Xte,_,_))) in self.multiling_dataset.items() if lang in self.langs()}

    def lYtr(self):
        return {lang:self.cat_view(Ytr) for (lang, ((_,Ytr,_),_)) in self.multiling_dataset.items() if lang in self.langs()}

    def lYte(self):
        return {lang:self.cat_view(Yte) for (lang, (_,(_,Yte,_))) in self.multiling_dataset.items() if lang in self.langs()}

    def cat_view(self, Y):
        if hasattr(self, 'categories_view'):
            return Y[:,self.categories_view]
        else:
            return Y

    def langs(self):
        if hasattr(self, 'languages_view'):
            langs = self.languages_view
        else:
            langs =  list(self.multiling_dataset.keys())
        langs.sort()
        return langs

    def num_categories(self):
        return self.lYtr()[self.langs()[0]].shape[1]

    def show_dimensions(self):
        for (lang, ((Xtr, Ytr, IDtr), (Xte, Yte, IDte))) in self.multiling_dataset.items():
            if lang not in self.langs(): continue
            if hasattr(Xtr, 'shape') and hasattr(Xte, 'shape'):
                print("Lang {}, Xtr={}, ytr={}, Xte={}, yte={}".format(lang, Xtr.shape, self.cat_view(Ytr).shape, Xte.shape, self.cat_view(Yte).shape))

    def show_category_prevalences(self):
        #pass
        nC = self.num_categories()
        accum_tr = np.zeros(nC, dtype=np.int)
        accum_te = np.zeros(nC, dtype=np.int)
        in_langs = np.zeros(nC, dtype=np.int) #count languages with at least one positive example (per category)
        for (lang, ((Xtr, Ytr, IDtr), (Xte, Yte, IDte))) in self.multiling_dataset.items():
            if lang not in self.langs(): continue
            prev_train = np.sum(self.cat_view(Ytr), axis=0)
            prev_test = np.sum(self.cat_view(Yte), axis=0)
            accum_tr += prev_train
            accum_te += prev_test
            in_langs += (prev_train>0)*1
            print(lang+'-train', prev_train)
            print(lang+'-test', prev_test)
        print('all-train', accum_tr)
        print('all-test', accum_te)

        return accum_tr, accum_te, in_langs


    def set_labels(self, labels):
        self.labels = labels

def apply_single_label_fragment_selection(train_doclist, test_doclist):
    train_doclist = single_label_fragment(train_doclist)
    test_doclist = single_label_fragment(test_doclist)
    active_labels = get_active_labels(train_doclist)
    filter_by_categories(test_doclist, active_labels)
    test_doclist = [d for d in test_doclist if d.categories]
    return train_doclist, test_doclist, active_labels

def single_label_fragment(doclist):
    single = [d for d in doclist if len(d.categories)==1]
    final_categories = set([d.categories[0] for d in single if d.categories])
    print('{} single-label documents ({} categories) from the original {} documents'.format(len(single), len(final_categories), len(doclist)))
    return single

def get_active_labels(doclist):
    cat_list = set()
    for d in doclist:
        cat_list.update(d.categories)
    return list(cat_list)

def filter_by_categories(doclist, keep_categories):
    catset = frozenset(keep_categories)
    for d in doclist:
        d.categories = list(set(d.categories).intersection(catset))


# creates a MultilingualDataset where matrices lie in language-specific feature spaces
def build_independent_matrices(dataset_name, langs, training_docs, test_docs, label_names, wiki_docs=[], preprocess=True, singlelabel=False):
    """
    :param langs: list of languages (str)
    :param training_docs: map {lang:doc-list} where each doc is a tuple (text, categories, id)
    :param test_docs: map {lang:doc-list} where each doc is a tuple (text, categories, id)
    :param label_names: list of names (str)
    :param wiki_docs: doc-list
    :return:
    """

    # if not singlelabel:
    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    lW = {}

    multiling_dataset = MultilingualDataset()
    multiling_dataset.dataset_name=dataset_name
    multiling_dataset.set_labels(mlb.classes_)
    for lang in langs:
        print("\nprocessing %d training, %d test, %d wiki for language <%s>" %
              (len(training_docs[lang]), len(test_docs[lang]), len(wiki_docs[lang]) if wiki_docs else 0, lang))

        tr_data, tr_labels, IDtr = zip(*training_docs[lang])
        te_data, te_labels, IDte = zip(*test_docs[lang])

        if preprocess:
            tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                    tokenizer=NLTKLemmaTokenizer(lang, verbose=True),
                                    stop_words=stopwords.words(NLTK_LANGMAP[lang]))
        else:
            tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True)

        Xtr = tfidf.fit_transform(tr_data)
        Xte = tfidf.transform(te_data)
        if wiki_docs:
            lW[lang] = tfidf.transform(wiki_docs[lang])

        # if singlelabel:
        #     Ytr = np.array(tr_labels).flatten()
        #     Yte = np.array(te_labels).flatten()
        #     assert len(Ytr) == Xtr.shape[0] and len(Yte) == Xte.shape[0], 'wrong size encountered in single-label fragment'
        # else:
        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)

        multiling_dataset.add(lang, Xtr, Ytr, Xte, Yte, IDtr, IDte)

    multiling_dataset.show_dimensions()
    multiling_dataset.show_category_prevalences()

    if wiki_docs:
        return multiling_dataset, lW
    else:
        return multiling_dataset

# creates a MultilingualDataset where matrices shares a single yuxtaposed feature space
def build_juxtaposed_matrices(dataset_name, langs, training_docs, test_docs, label_names, preprocess=True, singlelabel=False):

    multiling_dataset = MultilingualDataset()
    multiling_dataset.dataset_name = dataset_name

    # if not singlelabel:
    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    multiling_dataset.set_labels(mlb.classes_)

    tr_data_stack = []
    for lang in langs:
        print("\nprocessing %d training and %d test for language <%s>" % (len(training_docs[lang]), len(test_docs[lang]), lang))
        tr_data, tr_labels, tr_ID = zip(*training_docs[lang])
        te_data, te_labels, te_ID = zip(*test_docs[lang])
        if preprocess:
            tr_data = preprocess_documents(tr_data, lang)
            te_data = preprocess_documents(te_data, lang)
        tr_data_stack.extend(tr_data)
        multiling_dataset.add(lang, tr_data, tr_labels, te_data, te_labels, tr_ID, te_ID)

    tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True)
    tfidf.fit(tr_data_stack)

    for lang in langs:
        print("\nweighting documents for language <%s>" % (lang))
        (tr_data, tr_labels, tr_ID), (te_data, te_labels, te_ID) = multiling_dataset[lang]
        Xtr = tfidf.transform(tr_data)
        Xte = tfidf.transform(te_data)
        # if singlelabel:
        #     Ytr = np.array(tr_labels).flatten()
        #     Yte = np.array(te_labels).flatten()
        #     assert len(Ytr) == Xtr.shape[0] and len(Yte) == Xte.shape[0], 'wrong size encountered in single-label fragment'
        # else:home
        Ytr = mlb.transform(tr_labels)
        Yte = mlb.transform(te_labels)
        multiling_dataset.add(lang,Xtr,Ytr,Xte,Yte,tr_ID,te_ID)

    multiling_dataset.show_dimensions()
    return multiling_dataset

def __years_to_str(years):
    if isinstance(years, list):
        if len(years) > 1:
            return str(years[0])+'-'+str(years[-1])
        return str(years[0])
    return str(years)

"""
This method has been added a posteriori, to create document embeddings using the polylingual embeddings of the recent
article 'Word Translation without Parallel Data'; basically, it takes oone of the splits and retrieves the RCV documents
from the doc ids and then pickles an object (tr_docs, te_docs, label_names) in the outpath
"""
def retrieve_rcv_documents_from_dataset(datasetpath, rcv1_data_home, rcv2_data_home, outpath):

    tr_ids, te_ids = MultilingualDataset.load_ids(datasetpath)
    assert tr_ids.keys() == te_ids.keys(), 'inconsistent keys tr vs te'
    langs = list(tr_ids.keys())

    print('fetching the datasets')
    rcv1_documents, labels_rcv1 = fetch_RCV1(rcv1_data_home, split='train')
    rcv2_documents, labels_rcv2 = fetch_RCV2(rcv2_data_home, [l for l in langs if l != 'en'])

    filter_by_categories(rcv1_documents, labels_rcv2)
    filter_by_categories(rcv2_documents, labels_rcv1)

    label_names = get_active_labels(rcv1_documents + rcv2_documents)
    print('Active labels in RCV1/2 {}'.format(len(label_names)))

    print('rcv1: {} train, {} test, {} categories'.format(len(rcv1_documents), 0, len(label_names)))
    print('rcv2: {} documents'.format(len(rcv2_documents)), Counter([doc.lang for doc in rcv2_documents]))

    all_docs = rcv1_documents + rcv2_documents
    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    dataset = MultilingualDataset()
    for lang in langs:
        analyzer = CountVectorizer(strip_accents='unicode', min_df=3,
                                   stop_words=stopwords.words(NLTK_LANGMAP[lang])).build_analyzer()

        Xtr,Ytr,IDtr = zip(*[(d.text,d.categories,d.id) for d in all_docs if d.lang == lang and d.id in tr_ids[lang]])
        Xte,Yte,IDte = zip(*[(d.text,d.categories,d.id) for d in all_docs if d.lang == lang and d.id in te_ids[lang]])
        Xtr = [' '.join(analyzer(d)) for d in Xtr]
        Xte = [' '.join(analyzer(d)) for d in Xte]
        Ytr = mlb.transform(Ytr)
        Yte = mlb.transform(Yte)
        dataset.add(lang, Xtr, Ytr, Xte, Yte, IDtr, IDte)

    dataset.save(outpath)

"""
Same thing but for jrc
"""
def retrieve_jrc_documents_from_dataset(datasetpath, jrc_data_home, train_years, test_years, cat_policy, most_common_cat, outpath):

    tr_ids, te_ids = MultilingualDataset.load_ids(datasetpath)
    assert tr_ids.keys() == te_ids.keys(), 'inconsistent keys tr vs te'
    langs = list(tr_ids.keys())

    print('fetching the datasets')

    cat_list = inspect_eurovoc(jrc_data_home, select=cat_policy)
    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=train_years,
                                                 cat_filter=cat_list, cat_threshold=1, parallel=None,
                                                 most_frequent=most_common_cat)
    test_docs, _ = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=test_years, cat_filter=label_names,
                                   parallel='force')


    def filter_by_id(doclist, ids):
        ids_set = frozenset(itertools.chain.from_iterable(ids.values()))
        return [x for x in doclist if (x.parallel_id+'__'+x.id) in ids_set]

    training_docs = filter_by_id(training_docs, tr_ids)
    test_docs = filter_by_id(test_docs, te_ids)

    print('jrc: {} train, {} test, {} categories'.format(len(training_docs), len(test_docs), len(label_names)))

    mlb = MultiLabelBinarizer()
    mlb.fit([label_names])

    dataset = MultilingualDataset()
    for lang in langs:
        analyzer = CountVectorizer(strip_accents='unicode', min_df=3,
                                   stop_words=stopwords.words(NLTK_LANGMAP[lang])).build_analyzer()

        Xtr,Ytr,IDtr = zip(*[(d.text,d.categories,d.parallel_id+'__'+d.id) for d in training_docs if d.lang == lang])
        Xte,Yte,IDte = zip(*[(d.text,d.categories,d.parallel_id+'__'+d.id) for d in test_docs if d.lang == lang])
        Xtr = [' '.join(analyzer(d)) for d in Xtr]
        Xte = [' '.join(analyzer(d)) for d in Xte]
        Ytr = mlb.transform(Ytr)
        Yte = mlb.transform(Yte)
        dataset.add(lang, Xtr, Ytr, Xte, Yte, IDtr, IDte)

    dataset.save(outpath)


#generates the "feature-independent" and the "yuxtaposed" datasets
def prepare_jrc_datasets(jrc_data_home, wiki_data_home, langs, train_years, test_years, cat_policy, most_common_cat=-1, max_wiki=5000, single_fragment=False, run=0):
    name = 'JRCacquis'+('-singlelabel' if single_fragment else '')

    config_name = 'jrc_nltk_' + __years_to_str(train_years) + 'vs' + __years_to_str(test_years) + \
                  '_' + cat_policy + ('_top' + str(most_common_cat) if most_common_cat!=-1 else '')+ '_noparallel_processed' +\
                  ('_singlefragment' if single_fragment else '')
    run = '_run'+str(run)
    indep_path = join(jrc_data_home, config_name + run + '.pickle')
    upper_path = join(jrc_data_home, config_name + run + '_upper.pickle')
    yuxta_path = join(jrc_data_home, config_name + run + '_yuxtaposed.pickle')
    wiki_path  = join(jrc_data_home, config_name + run + '.wiki.pickle')
    wiki_docs_path = join(jrc_data_home, config_name + '.wiki.raw.pickle')

    cat_list = inspect_eurovoc(jrc_data_home, select=cat_policy)
    training_docs, label_names = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=train_years, cat_filter=cat_list, cat_threshold=1, parallel=None, most_frequent=most_common_cat)
    test_docs, _ = fetch_jrcacquis(langs=langs, data_path=jrc_data_home, years=test_years, cat_filter=label_names, parallel='force')

    if single_fragment:
        training_docs, test_docs, label_names = apply_single_label_fragment_selection(training_docs, test_docs)

    print('Generating feature-independent dataset...')
    training_docs_no_parallel = random_sampling_avoiding_parallel(training_docs)

    def _group_by_lang(doc_list, langs):
        return {lang: [(d.text, d.categories, d.parallel_id + '__' + d.id) for d in doc_list if d.lang == lang]
                for lang in langs}

    training_docs = _group_by_lang(training_docs, langs)
    training_docs_no_parallel = _group_by_lang(training_docs_no_parallel, langs)
    test_docs = _group_by_lang(test_docs, langs)
    if True or not exists(indep_path):
        #wiki_docs = []
        #wiki_docs_path = wiki_path.replace('.pickle', '.raw.pickle')
        if not exists(wiki_docs_path):
            wiki_docs = fetch_wikipedia_multilingual(wiki_data_home, langs, min_words=50, deletions=False)
            wiki_docs = random_wiki_sample(wiki_docs, max_wiki)
            pickle.dump(wiki_docs, open(wiki_docs_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            wiki_docs = pickle.load(open(wiki_docs_path, 'rb'))
        wiki_docs = random_wiki_sample(wiki_docs, max_wiki)

        if wiki_docs:
            lang_data, wiki_docs = build_independent_matrices(name, langs, training_docs_no_parallel, test_docs, label_names, wiki_docs, singlelabel=single_fragment)
            pickle.dump(wiki_docs, open(wiki_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            lang_data = build_independent_matrices(name, langs, training_docs_no_parallel, test_docs, label_names, singlelabel=single_fragment)

        lang_data.save(indep_path)


    print('Generating upper-bound (English-only) dataset...')
    if True or not exists(upper_path):
        training_docs_eng_only = {'en':training_docs['en']}
        test_docs_eng_only = {'en':test_docs['en']}
        build_independent_matrices(name, ['en'], training_docs_eng_only, test_docs_eng_only, label_names, singlelabel=single_fragment).save(upper_path)

    print('Generating yuxtaposed dataset...')
    if True or not exists(yuxta_path):
        build_juxtaposed_matrices(name, langs, training_docs_no_parallel, test_docs, label_names, singlelabel=single_fragment).save(yuxta_path)


# generates the "feature-independent" and the "yuxtaposed" datasets
def prepare_rcv_datasets(outpath, rcv1_data_home, rcv2_data_home, wiki_data_home, langs,
                         train_for_lang=1000, test_for_lang=1000, max_wiki=5000, preprocess=True,
                         single_fragment_for=None, run=0):

    assert 'en' in langs, 'English is not in requested languages, but is needed for some datasets'
    assert len(langs)>1, 'the multilingual dataset cannot be built with only one dataset'
    assert not preprocess or set(langs).issubset(set(RCV2_LANGS_WITH_NLTK_STEMMING+['en'])), \
        "languages not in RCV1-v2/RCV2 scope or not in valid for NLTK's processing"

    name = 'RCV1/2'+('-singlelabel' if single_fragment_for else '')
    config_name = 'rcv1-2_nltk_trByLang'+str(train_for_lang)+'_teByLang'+str(test_for_lang)+\
                  ('_processed' if preprocess else '_raw')+('singlefragment' if single_fragment_for else '')
    run = '_run' + str(run)
    indep_path = join(outpath, config_name + run + '.pickle')
    upper_path = join(outpath, config_name + run +'_upper.pickle')
    yuxta_path = join(outpath, config_name + run +'_yuxtaposed.pickle')
    wiki_path = join(outpath, config_name + run + '.wiki.pickle')
    wiki_docs_path = join(outpath, config_name + '.wiki.raw.pickle')

    print('fetching the datasets')
    rcv1_documents, labels_rcv1 = fetch_RCV1(rcv1_data_home, split='train')
    rcv2_documents, labels_rcv2 = fetch_RCV2(rcv2_data_home, [l for l in langs if l!='en'])
    filter_by_categories(rcv1_documents, labels_rcv2)
    filter_by_categories(rcv2_documents, labels_rcv1)

    if single_fragment_for:
        filter_by_categories(rcv1_documents, single_fragment_for)
        filter_by_categories(rcv2_documents, single_fragment_for)
        rcv1_documents = single_label_fragment(rcv1_documents)
        rcv2_documents = single_label_fragment(rcv2_documents)

    label_names = get_active_labels(rcv1_documents+rcv2_documents)
    print('Active labels in RCV1/2 {}'.format(len(label_names)))

    print('rcv1: {} train, {} test, {} categories'.format(len(rcv1_documents), 0, len(label_names)))
    print('rcv2: {} documents'.format(len(rcv2_documents)), Counter([doc.lang for doc in rcv2_documents]))

    lang_docs = {lang: [d for d in rcv1_documents + rcv2_documents if d.lang == lang] for lang in langs}

    #for the upper bound there are no parallel versions, so for the English case, we take as many documents as there
    # would be in the multilingual case -- then we will extract from them only train_for_lang for the other cases
    print('Generating upper-bound (English-only) dataset...')
    train, test = train_test_split(lang_docs['en'], train_size=train_for_lang*len(langs), test_size=test_for_lang, shuffle=True)
    train_lang_doc_map = {'en':[(d.text, d.categories, d.id) for d in train]}
    test_lang_doc_map = {'en':[(d.text, d.categories, d.id) for d in test]}
    build_independent_matrices(name, ['en'], train_lang_doc_map, test_lang_doc_map, label_names, singlelabel=single_fragment_for).save(upper_path)

    train_lang_doc_map['en'] = train_lang_doc_map['en'][:train_for_lang]
    for lang in langs:
        if lang=='en': continue # already split
        test_take = min(test_for_lang, len(lang_docs[lang])-train_for_lang)
        train, test = train_test_split(lang_docs[lang], train_size=train_for_lang, test_size=test_take, shuffle=True)
        train_lang_doc_map[lang] = [(d.text, d.categories, d.id) for d in train]
        test_lang_doc_map[lang]  = [(d.text, d.categories, d.id) for d in test]

    # for lang in train_lang_doc_map:
    #     catcount = Counter()
    #     for _,cats,_ in train_lang_doc_map[lang]:
    #         catcount.update(cats)
    #     print(lang, len(catcount))
    #     for cat,count in catcount.most_common():
    #         print(cat+'\t'+str(count))

    print('Generating feature-independent dataset...')
    #wiki_docs_path = wiki_path.replace('.pickle','.raw.pickle')
    #wiki_docs=[]
    #wiki_docs_matrix=None
    if not exists(wiki_docs_path):
        wiki_docs = fetch_wikipedia_multilingual(wiki_data_home, langs, min_words=50, deletions=False)
        wiki_docs = random_wiki_sample(wiki_docs, max_wiki)
        pickle.dump(wiki_docs, open(wiki_docs_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        wiki_docs = pickle.load(open(wiki_docs_path, 'rb'))
    wiki_docs = random_wiki_sample(wiki_docs, max_wiki)

    #wiki_docs = random_wiki_sample(wiki_docs, 50)
    if wiki_docs:
        lang_data, wiki_docs_matrix = build_independent_matrices(name, langs, train_lang_doc_map, test_lang_doc_map, label_names, wiki_docs, preprocess, singlelabel=single_fragment_for)
    else:
        lang_data = build_independent_matrices(name, langs, train_lang_doc_map, test_lang_doc_map, label_names, wiki_docs, preprocess, singlelabel=single_fragment_for)

    lang_data.save(indep_path)
    if wiki_docs:
        pickle.dump(wiki_docs_matrix, open(wiki_path, 'wb'), pickle.HIGHEST_PROTOCOL)

    print('Generating yuxtaposed dataset...')
    build_juxtaposed_matrices(name, langs, train_lang_doc_map, test_lang_doc_map, label_names, preprocess, singlelabel=single_fragment_for).save(yuxta_path)

def prepare_reuters21578(data_path=None, preprocess=True):

    if not data_path:
        data_path = get_data_home()

    reuters_train = fetch_reuters21579(subset='train')
    reuters_test = fetch_reuters21579(subset='test')

    mlb = MultiLabelBinarizer()
    mlb.fit(reuters_train.target)

    multiling_dataset = MultilingualDataset()

    nDtr = len(reuters_train.data)
    nDte = len(reuters_test.data)
    print("\nprocessing %d training and %d test for language en" % (nDtr, nDte))
    if preprocess:
        tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                    tokenizer=NLTKLemmaTokenizer('en', verbose=True),
                                    stop_words=stopwords.words(NLTK_LANGMAP['en']))
    else:
        tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True)

    Xtr = tfidf.fit_transform(reuters_train.data)
    Xte = tfidf.transform(reuters_test.data)
    Ytr = mlb.transform(reuters_train.target)
    Yte = mlb.transform(reuters_test.target)
    Xtr.sort_indices()
    Xte.sort_indices()

    multiling_dataset.add('en', Xtr, Ytr, Xte, Yte, tr_ids=list(range(nDtr)), te_ids=list(range(nDtr, nDtr + nDte)))

    multiling_dataset.show_dimensions()

    outpath = join(data_path, 'reuters21578-multilingformat'+('-processed' if preprocess else '-raw')+'.pickle')
    print('saving dataset in ' + outpath)
    multiling_dataset.save(outpath)

    return multiling_dataset

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    JRC_DATAPATH = "/media/moreo/1TB Volume/Datasets/JRC_Acquis_v3"
    WIKI_DATAPATH = "/media/moreo/1TB Volume/Datasets/Wikipedia/multilingual_docs_JRC_NLTK"
    RCV1_PATH = '/media/moreo/1TB Volume/Datasets/RCV1-v2/unprocessed_corpus'
    RCV2_PATH = '/media/moreo/1TB Volume/Datasets/RCV2'
    langs = lang_set['JRC_NLTK']
    # rcv1_leave_topics = fetch_topic_hierarchy("/media/moreo/1TB Volume/Datasets/RCV1-v2/rcv1.topics.hier.orig",
    #                                           topics='leaves')

    # prepare_rcv_datasets(RCV2_PATH, RCV1_PATH, RCV2_PATH, WIKI_DATAPATH, RCV2_LANGS_WITH_NLTK_STEMMING + ['en'],
    #                      train_for_lang=1000,
    #                      test_for_lang=1000,
    #                      max_wiki=0, run='X')
    # prepare_jrc_datasets(JRC_DATAPATH, WIKI_DATAPATH, langs, train_years=list(range(1958, 2006)), test_years=[2006],
    #                      cat_policy='all', most_common_cat=300, max_wiki=0, run='X')
    #
    # sys.exit()

    for run in range(0,10):
        # print('Building JRC-Acquis datasets...')
        # prepare_jrc_datasets(JRC_DATAPATH, WIKI_DATAPATH, langs, train_years=list(range(1958, 2006)), test_years=[2006],
        #                      cat_policy='leaves', most_common_cat=300, single_fragment=True, run=run)
        # prepare_jrc_datasets(JRC_DATAPATH, WIKI_DATAPATH, langs, train_years=list(range(1958, 2006)), test_years=[2006], cat_policy='all', most_common_cat=300, run=run)
        # prepare_jrc_datasets(JRC_DATAPATH, WIKI_DATAPATH, langs, train_years=list(range(1986, 2006)), test_years=[2006], cat_policy='broadest')
        # prepare_jrc_datasets(JRC_DATAPATH, WIKI_DATAPATH, langs, train_years=list(range(1986, 2006)), test_years=[2006], cat_policy='all')

        print('Building RCV1-v2/2 datasets run',run)
        # prepare_rcv_datasets(RCV2_PATH, RCV1_PATH, RCV2_PATH, WIKI_DATAPATH, RCV2_LANGS_WITH_NLTK_STEMMING + ['en'],
        #                      train_for_lang=1000,
        #                      test_for_lang=1000,
        #                      max_wiki=5000,
        #                      single_fragment_for=rcv1_leave_topics, run=run)
        # prepare_rcv_datasets(RCV2_PATH, RCV1_PATH, RCV2_PATH, WIKI_DATAPATH, RCV2_LANGS_WITH_NLTK_STEMMING + ['en'],
        #                      train_for_lang=1000,
        #                      test_for_lang=1000,
        #                      max_wiki=5000, run=run)
        datasetpath = join(RCV2_PATH,'rcv1-2_nltk_trByLang1000_teByLang1000_processed_run{}.pickle'.format(run))
        outpath = join(RCV2_PATH, 'rcv1-2_doclist_trByLang1000_teByLang1000_processed_run{}.pickle'.format(run))
        #retrieve_rcv_documents_from_dataset(datasetpath, RCV1_PATH, RCV2_PATH, outpath)

        datasetpath = join(JRC_DATAPATH, 'jrc_nltk_1958-2005vs2006_all_top300_noparallel_processed_run{}.pickle'.format(run))
        outpath = join(JRC_DATAPATH, 'jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run{}.pickle'.format(run))
        retrieve_jrc_documents_from_dataset(datasetpath, JRC_DATAPATH, train_years=list(range(1958, 2006)), test_years=[2006], cat_policy='all', most_common_cat=300, outpath=outpath)


    #print('Building Reuters-21578 dataset...')
    #prepare_reuters21578(preprocess=False)
    #prepare_reuters21578(preprocess=True)

