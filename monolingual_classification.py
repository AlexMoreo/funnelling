from data.clesa_data_generator import *
from data.languages import *
from model.clesa import ESA, ESA_PPindex
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from util.metrics import *
import sys
import time
import numpy as np
from sklearn.svm import LinearSVC
from feature_selection.round_robin import RoundRobin
import random as rn
from sklearn.svm import SVC, LinearSVC

def fit_model_hyperparameters(Xtr, ytr, parameters, model):
    single_class = ytr.shape[1] == 1
    if not single_class:
        parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
        model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters, scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)

    if single_class: ytr = np.squeeze(ytr)
    return model_tunning.fit(Xtr, ytr)

def wiki_rand_doc_selection(n, wiki_docs, seed=None):
    if seed: rn.seed(seed)
    rn.shuffle(wiki_docs)
    return wiki_docs[:n]

def get_data_labels(doc_list):
    return zip(*[(d.text, d.categories) for d in doc_list])

if __name__ == "__main__":

    dataset = "JRCAcquis"
    tr_years = [2005]
    te_years = [2006]
    cat_threshold=30
    jrcacquis_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"
    wikipedia_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia/multilingual_docs_FIVE_LANGS"
    lang = 'en'

    pickle_name = join(jrcacquis_datapath, 'preprocessed_monolingual_' + lang
                       + '_tr_' + year_list_as_str(tr_years) + '_te_' + year_list_as_str(te_years)
                       + '_broadcats.pickle')

    if os.path.exists(pickle_name):
        print("Loading pickled object %s" % pickle_name)
        (W, Xtr, Ytr, Xte, Yte) = pickle.load(open(pickle_name, 'rb'))
    else:
        print("Fetching JRC-Acquis data and Wikipedia documents")
        cat_list = inspect_eurovoc(jrcacquis_datapath, pickle_name="broadest_concepts.pickle")
        tr_request, final_cats = fetch_jrcacquis(langs=[lang], data_path=jrcacquis_datapath, years=tr_years,
                                                 cat_filter=cat_list, cat_threshold=cat_threshold)
        te_request, _ = fetch_jrcacquis(langs=[lang], data_path=jrcacquis_datapath, years=te_years, cat_filter=final_cats)
        wiki_docs = fetch_wikipedia_multilingual(wikipedia_datapath, [lang], min_words=100)
        wiki_docs = wiki_rand_doc_selection(n=10000, wiki_docs=wiki_docs[lang], seed=123456789)

        tr_data, tr_labels = get_data_labels(tr_request)
        te_data, te_labels = get_data_labels(te_request)

        # representation for language lang
        print("\nTransforming tfidf %d docs" % (len(tr_data) + len(te_data) + len(wiki_docs)))
        tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True, tokenizer=NLTKLemmaTokenizer(lang, verbose=True), stop_words=stopwords.words(NLTK_LANGMAP[lang]))
        Xtr = tfidf.fit_transform(tr_data)
        Xte = tfidf.transform(te_data)
        W = tfidf.transform(wiki_docs)

        print("Processing labels...")
        mlb = MultiLabelBinarizer()
        Ytr = mlb.fit_transform(tr_labels)
        Yte = mlb.transform(te_labels)

        print("Feature selection of top 10000 features (Round Robin with Information Gain)")
        rrobin = RoundRobin(k=10000)
        Xtr = rrobin.fit_transform(Xtr, Ytr)
        Xte = rrobin.transform(Xte)
        W   = rrobin.transform(W)

        print("Pickling problem setting for future experiments in %s" % pickle_name)
        pickle.dump((W,Xtr,Ytr,Xte,Yte), open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Running Monolingual ESA transformation for lang %s" % lang)

    for use_distributional_heuristic in [True]:
        esa = ESA(similarity='dot', centered=False, post_norm=False, distributional_aware=use_distributional_heuristic)
        #esa = ESA_PPindex(num_permutations=5, similarity='cosine', centered=False, post_norm=False)
        Xtr_ = esa.fit_transform(W, Xtr)
        Xte_ = esa.transform(Xte)

        # M = (W*W.transpose()).toarray()
        # M = (Xtr*Xtr.transpose()).toarray()
        # WWT = lambda X, V: np.dot(np.dot(X, M), V.T)
        # model = OneVsRestClassifier(SVC(kernel=WWT), n_jobs=1)
        print("Learning ESA")
        for c in [10e3, 10e2, 10e1, 10e0, 10e-1]:
            model = OneVsRestClassifier(esa.learner(C=c), n_jobs=-1)
            t_ini = time.time()
            model.fit(Xtr_, Ytr)
            yte_ = model.predict(Xte_)
            macro_f1 = macroF1(Yte, yte_)
            micro_f1 = microF1(Yte, yte_)
            print("Test scores: %.3f macro-f1, %.3f micro-f1 [took %d s][C=%f][distributional=%s]" % (macro_f1, micro_f1, time.time()-t_ini, c, 'active' if use_distributional_heuristic else 'off'))

    # ESA English: Test scores: 0.572 macro-f1, 0.707 micro-f1 [wikidocs = 10000 random, fs = 10000, centered = False, metric=cosine, postnorm=False]
    # ESA English: Test scores: 0.570 macro-f1, 0.706 micro-f1 [wikidocs = 10000 random, fs = 10000, centered = True, metric=cosine, postnorm=False]
    # ESA English: Test scores: 0.261 macro-f1, 0.542 micro-f1 [took 41 s] [wikidocs = 10000 random, fs = 10000, centered = True, metric=cosine, postnorm=True]
    # ESA English: Test scores: 0.558 macro-f1, 0.681 micro-f1 [took 615 s] [wikidocs = 10000 random, fs = 10000, centered = False, metric=cosine, postnorm=False, kernel = WWT]
    # ESA English: Test scores: 0.576 macro-f1, 0.715 micro-f1 [took 1 s] [BOW]
    # ESA English: Test scores: 0.000 macro-f1, 0.000 micro-f1 [took 31328 s] [with pp-index (only 5 permutations) and kendall-kernel; about 9 hours]

    # ESA English: Test scores: 0.567 macro-f1, 0.704 micro-f1 [took 174 s] with linearsvc, vanilla
    # Esa English: Test scores: 0.554 macro-f1, 0.680 micro-f1 [took 497 s] with svc dist-aware kernel
    # Esa English: Test scores: 0.495 macro-f1, 0.674 micro-f1 [took 68 s]  with svc linear (not-aware) kernel (is running...)
    # No funciona:
"""
ESA with dist-aware kernel
Test scores: 0.503 macro-f1, 0.612 micro-f1 [took 502 s][C=10000.000000][distributional=active]
Test scores: 0.507 macro-f1, 0.617 micro-f1 [took 504 s][C=1000.000000][distributional=active]
Test scores: 0.526 macro-f1, 0.645 micro-f1 [took 505 s][C=100.000000][distributional=active]
Test scores: 0.534 macro-f1, 0.655 micro-f1 [took 540 s][C=10.000000][distributional=active]
Test scores: 0.554 macro-f1, 0.680 micro-f1 [took 498 s][C=1.000000][distributional=active]
ESA without kernel
Test scores: 0.588 macro-f1, 0.690 micro-f1 [took 64 s][C=10000.000000][distributional=off]
Test scores: 0.590 macro-f1, 0.693 micro-f1 [took 64 s][C=1000.000000][distributional=off]
Test scores: 0.610 macro-f1, 0.716 micro-f1 [took 64 s][C=100.000000][distributional=off]
Test scores: 0.616 macro-f1, 0.721 micro-f1 [took 64 s][C=10.000000][distributional=off]
Test scores: 0.495 macro-f1, 0.674 micro-f1 [took 64 s][C=1.000000][distributional=off]
Running with diagonal (funciona igual que sin kernel, pero mas lento, no se porque
"""