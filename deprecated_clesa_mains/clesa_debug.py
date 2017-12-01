from data._clesa_data_generator import *
from data.languages import *
from model.clesa import CLESA, CLESA_PPindex
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from util.metrics import *
import numpy as np
import sklearn
import sys
from sklearn.svm import LinearSVC
from feature_selection.round_robin import fit_round_robin

#reduces the problem to the n_cats most populated categories for language lang
def clesa_data_category_reduction(to_select, lY, lang, n_cats):
    lang_prev = [(cat, sum(lY[lang][:,cat])) for cat in range(n_cats)]
    lang_prev.sort(key=lambda x:x[1], reverse=True)
    most_populated = [cat for cat,prev in lang_prev[:to_select]]

    for lang in lY.keys():
        lY[lang] = lY[lang][:,most_populated]

def wiki_doc_selection_noise(n_wiki_docs, lP):
    Xm = np.array([lP[lang] for lang in lP.keys()])
    print(Xm.shape)
    variance = np.var(Xm, axis=0)
    var_mean = np.mean(variance, axis=0)
    return np.argsort(var_mean)[:n_wiki_docs]

def wiki_rand_doc_selection(n_wiki_docs, lW, lwiki, seed=None):
    if seed: np.random.seed(seed)
    wiki_sel = np.random.permutation(lW.values()[0].shape[0])[:n_wiki_docs]
    for l in lW.keys():
        lW[l] = lW[l][wiki_sel]
        lwiki[l] = [lwiki[l][i] for i in wiki_sel]

def fit_model_hyperparameters(Xtr, ytr, parameters, model):
    single_class = ytr.shape[1] == 1
    if not single_class:
        parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
        model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters, scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)

    if single_class: ytr = np.squeeze(ytr)
    return model_tunning.fit(Xtr, ytr)

#lP is a map of language-ESAprojection
def checkerror(lP):
    print("error checking...")
    error = 0
    for i in range(len(langs)):
        for j in range(i+1, len(langs)):
            error += np.mean(lP[langs[i]] - lP[langs[j]])**2
    print("Error %.10f" % error)


if __name__ == "__main__":

    dataset = "JRCAcquis"
    langmap = "ES_IT"
    tr_years = [2005]
    cat_threshold=30
    jrcacquis_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"
    wikipedia_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia/multilingual_docs_FIVE_LANGS"
    force_parallel = True

    langs = ['es', 'it', 'en']

    picklename = "debug.en_it_es.parallel.pickle"
    if os.path.exists(picklename):
        matrices = pickle.load(open(picklename, 'rb'))
    else:
        cat_list = inspect_eurovoc(jrcacquis_datapath, pickle_name="broadest_concepts.pickle")
        request, final_cats = fetch_jrcacquis(langs=langs, data_path=jrcacquis_datapath, years=tr_years,
                                              cat_filter=cat_list, cat_threshold=cat_threshold,
                                              parallel=force_parallel)
        wiki_docs = fetch_wikipedia_multilingual(wikipedia_datapath, langs, min_words=500, deletions=False,
                                                 pickle_name=join(wikipedia_datapath, "wiki_enites_langs.pickle"))
        request = as_lang_dict(request, langs, force_parallel)
        matrices={}
        for lang in langs:
            data, labels = zip(*request[lang])
            tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=False, tokenizer=NLTKLemmaTokenizer(lang, verbose=True), stop_words=stopwords.words(NLTK_LANGMAP[lang]))
            #tfidf = TfidfVectorizer(min_df=3, sublinear_tf=False)

            #representation for language lang
            print("Transforming tfidf %d docs" % (len(data) + len(wiki_docs[lang])))
            XW = tfidf.fit_transform(list(data)+wiki_docs[lang])
            X = XW[:len(data)]
            W = XW[len(data):]

            #X = tfidf.fit_transform(data)
            #W = tfidf.transform(wiki_docs[lang])
            matrices[lang]={'X':X,'W':W,'data':data,'labels':labels,'wiki':wiki_docs[lang]}

        print("\nProcessing labels...")
        mlb = MultiLabelBinarizer()
        mlb.fit(matrices[langs[0]]['labels'])

        for lang in langs:
            Y = mlb.transform(matrices[lang]['labels'])
            matrices[lang]['Y'] = Y

        pickle.dump(matrices, open(picklename, 'wb'), pickle.HIGHEST_PROTOCOL)

    lW = {lang: matrices[lang]['W'] for lang in langs}
    lX = {lang: matrices[lang]['X'] for lang in langs}
    lY = {lang: matrices[lang]['Y'] for lang in langs}
    lwiki = {lang: matrices[lang]['wiki'] for lang in langs}

    wiki_rand_doc_selection(10000, lW, lwiki, seed=123456789)

    do_feature_selection = True
    if do_feature_selection:
        for lang in langs:
            rrobin = fit_round_robin(lX[lang], lY[lang], k=1000,
                                     features_rank_pickle_path="feature_selection_"+lang+".pickle")
            lX[lang] = sklearn.preprocessing.normalize(rrobin.transform(lX[lang]), norm='l2', axis=1, copy=True)
            lW[lang] = sklearn.preprocessing.normalize(rrobin.transform(lW[lang]), norm='l2', axis=1, copy=True)

    ltr1 = 'en'
    ltr2 = 'es'
    ltr3 = 'it'

    # without distributional-aware experiment -------------------------------------------------------------------------
    # clesa = CLESA(similarity='dot', centered=False, post_norm=False)
    # print("\tprojecting clesa...")
    # clesa.fit(lW)
    # lP = {lang: clesa.transform(lX={lang:lX[lang]}, lY={lang:lY[lang]})[0] for lang in langs}
    #
    # print("CLESA")
    #
    # Xtr, Ytr, Wtr = np.vstack((lP[ltr1], lP[ltr2])), np.vstack((lY[ltr1], lY[ltr2])), np.vstack((lW[ltr1], lW[ltr2]))
    # Xte, Yte, Wte = lP[ltr3], lY[ltr3], lW[ltr3]
    #
    # print("Feature Selection" if do_feature_selection else "Without Feature Selection")
    # for c in [1e3, 1e2, 1e1, 1]:
    #     model = OneVsRestClassifier(sklearn.svm.LinearSVC(C=c), n_jobs=-1)
    #     model.fit(Xtr, Ytr)
    #     yte_ = model.predict(Xte)
    #     macro_f1 = macroF1(Yte, yte_)
    #     micro_f1 = microF1(Yte, yte_)
    #     print("[C=%f] Test scores: %.3f macro-f1, %.3f micro-f1" % (c, macro_f1, micro_f1))

    # with distributional-aware experiment -------------------------------------------------------------------------
    clesa = CLESA(similarity='dot', centered=False, post=False, distributional_aware=True)
    print("\tprojecting clesa...")
    clesa.fit(lW)
    lP = {lang: clesa.transform(lX={lang: lX[lang]}, lY={lang: lY[lang]})[0] for lang in langs}

    print("CLESA")

    Xtr, Ytr, Wtr = np.vstack((lP[ltr1], lP[ltr2])), np.vstack((lY[ltr1], lY[ltr2])), np.vstack(
        (lW[ltr1], lW[ltr2]))
    Xte, Yte, Wte = lP[ltr3], lY[ltr3], lW[ltr3]

    print("Feature Selection" if do_feature_selection else "Without Feature Selection")
    for c in [1e3, 1e2, 1e1, 1]:
        #model = OneVsRestClassifier(sklearn.svm.LinearSVC(C=c), n_jobs=-1)
        gram_tr = Xtr.dot(Xtr.transpose())
        model = OneVsRestClassifier(sklearn.svm.SVC(kernel='precomputed', C=c), n_jobs=-1)
        model.fit(gram_tr, Ytr)
        gram_te = Xte.dot(Xtr.transpose())
        yte_ = model.predict(Xte)
        macro_f1 = macroF1(Yte, yte_)
        micro_f1 = microF1(Yte, yte_)
        print("[C=%f] Test scores: %.3f macro-f1, %.3f micro-f1" % (c, macro_f1, micro_f1))

        """
with feature selection k=1000, train EN-ES, test IT (with CLESA(similarity='cosine', centered=True, post_norm=True))
[C=1000.000000] Test scores: 0.347 macro-f1, 0.294 micro-f1
[C=100.000000] Test scores: 0.399 macro-f1, 0.382 micro-f1
[C=10.000000] Test scores: 0.364 macro-f1, 0.479 micro-f1
[C=1.000000] Test scores: 0.146 macro-f1, 0.323 micro-f1

however it seems to be wrong to perform feature selection; the reason is that most of the wikipedia terms will be discarded
(as they will probably be low-correlated to, say, laws in JRC-Acquis), and therefore the comparison (cosine) between
jrc-documents and wiki-documents is not taking into account the fact that most words have been removed.
E.g.,
wiki1 = w0, w1, w2, w3; wiki2 = w0, w4; jrc-acquis=w0, w5

feature selection--> wiki1 = w0; wiki2 = w0; jrc-acquis=w0, w5 --> wiki1 and wiki2 have been modified in a non consistent manner,
such that <jar-acquis,wiki1> = <jar-acquis,wiki2>

Pues no, estos son los resultados sin feature selection, igual setting el resto:
[C=1000.000000] Test scores: 0.257 macro-f1, 0.170 micro-f1
[C=100.000000] Test scores: 0.203 macro-f1, 0.265 micro-f1
[C=10.000000] Test scores: 0.036 macro-f1, 0.108 micro-f1
[C=1.000000] Test scores: 0.008 macro-f1, 0.032 micro-f1
        """
