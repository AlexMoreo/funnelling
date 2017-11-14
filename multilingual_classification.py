from data.clesa_data_generator import *
from data.languages import *
from model.clesa import CLESA, CLESA_PPindex
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from util.metrics import *
import sys
import time
import numpy as np
from sklearn.svm import LinearSVC
from feature_selection.round_robin import fit_round_robin

def fit_model_hyperparameters(Xtr, ytr, parameters, model):
    single_class = ytr.shape[1] == 1
    if not single_class:
        parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
        model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters, scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)

    if single_class: ytr = np.squeeze(ytr)
    return model_tunning.fit(Xtr, ytr)

def wiki_rand_doc_selection(n_wiki_docs, clesa_data, seed=None):
    if seed: np.random.seed(seed)
    wiki_sel = np.random.permutation(clesa_data.wiki_shape()[0])[:n_wiki_docs]
    for l in clesa_data.langs():
        clesa_data.lW[l] = clesa_data.lW[l][wiki_sel]

def wiki_doc_selection_h1(n_wiki_docs, clesa_data):
    sel = []
    nW,nF = clesa_data.wiki_shape()
    for i in range(nW):
        min_w = nF
        max_w = 0
        for l in clesa_data.langs():
            W = clesa_data.lW[l]
            num_words = W.getnnz(axis=1)[i]#sum([1 for t in range(nF) if W[i,t] > 0])
            min_w = min(min_w, num_words)
            max_w = max(max_w, num_words)
        cross_length = min_w*1.0 / max_w
        sel.append((i,cross_length))
    sel.sort(key=lambda x:x[1],reverse=True)
    sel = [doc_pos for doc_pos,cross_length in sel[:n_wiki_docs]]

    for l in clesa_data.langs():
        clesa_data.lW[l] = clesa_data.lW[l][sel]

def wiki_doc_selection_noise(n_wiki_docs, clesa_data, Xtr, n_langs):
    if Xtr.shape[0] % n_langs != 0:
        raise ValueError("Block size is not consistent. Documents might not be parallel")

    block = Xtr.shape[0] / n_langs
    Xl = np.array([Xtr[i*block:(i+1)*block] for i in range(n_langs)])
    print(Xl.shape)
    variance = np.var(Xl, axis=0)
    var_mean = np.mean(variance, axis=0)
    sel = np.argsort(var_mean)[:n_wiki_docs]
    for l in clesa_data.langs():
        clesa_data.lW[l] = clesa_data.lW[l][sel]


#reduces the problem to the n_cats most populated categories for language lang
def clesa_data_category_reduction(n_cats, clesa_data, lang):
    lang_prev = [(cat, clesa_data.cat_prevalence(cat, lang)) for cat in range(clesa_data.num_categories())]
    lang_prev.sort(key=lambda x:x[1], reverse=True)
    most_populated = [cat for cat,prev in lang_prev[:n_cats]]

    for lang in clesa_data.lYtr.keys():
        clesa_data.lYtr[lang] = clesa_data.lYtr[lang][:,most_populated]
    for lang in clesa_data.lYte.keys():
        clesa_data.lYte[lang] = clesa_data.lYte[lang][:,most_populated]


if __name__ == "__main__":

    dataset = "JRCAcquis"
    langmap = "ES_IT"
    tr_years = [2005]
    te_years = [2006]
    cat_threshold=30
    jrcacquis_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"
    wikipedia_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia/multilingual_docs_FIVE_LANGS"
    force_parallel = True

    langs = lang_set[langmap]

    pickle_name = join(jrcacquis_datapath, 'preprocessed_' + langmap
                       + '_tr_' + year_list_as_str(tr_years) + '_te_' + year_list_as_str(te_years)
                       + ('_parallel' if force_parallel else '') + '_broadcats.pickle')

    clesa_data = clesa_data_generator(dataset, langs, tr_years, te_years,
                                      jrcacquis_datapath, wikipedia_datapath,
                                      cat_threshold=cat_threshold, pickle_name=pickle_name, langmap=langmap, force_parallel=True)

    # tr_langs = ['en']
    # te_langs = ['en']
    # print("Problem setting: training", tr_langs, "test", te_langs)
    # clesa_data.problem_setting(train_langs=tr_langs, test_langs=te_langs)

    clesa_data_category_reduction(10, clesa_data, 'es')

    for lang in clesa_data.langs():
        print("Language %s" % lang)
        print([(cat,clesa_data.cat_prevalence(cat, lang)) for cat in range(clesa_data.num_categories())])


    mode = "CL-ESA"


    if mode == "BOW":
        for lang in clesa_data.langs():
            print("Running BoW baseline on language %s" %lang)
            Xtr_, Ytr_ = clesa_data.lang_train(lang)
            Xte_, Yte_ = clesa_data.lang_test(lang)
            print("\ttrain %s:%s" % (str(Xtr_.shape), str(Ytr_.shape)))
            print("\ttest  %s:%s" % (str(Xte_.shape), str(Yte_.shape)))
            rrobin = fit_round_robin(Xtr_, Ytr_, k=10000,
                  features_rank_pickle_path=join(jrcacquis_datapath,dataset+"_"+lang+"X"+str(Xtr_.shape)+"_Y"+str(Ytr_.shape)+".pickle"))
            Xtr_ = rrobin.transform(Xtr_)
            Xte_ = rrobin.transform(Xte_)
            print("\ttrain %s:%s" % (str(Xtr_.shape), str(Ytr_.shape)))
            print("\ttest  %s:%s" % (str(Xte_.shape), str(Yte_.shape)))
            model = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
            model.fit(Xtr_, Ytr_)
            yte_ = model.predict(Xte_)
            macro_f1 = macroF1(Yte_, yte_)
            micro_f1 = microF1(Yte_, yte_)
            print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))

            """
            Running BoW baseline on language fr: train (2415, 10000):(2415, 10) Test scores: 0.565 macro-f1, 0.766 micro-f1
            Running BoW baseline on language en: train (2405, 10000):(2405, 10) Test scores: 0.564 macro-f1, 0.765 micro-f1
            Running BoW baseline on language de: train (2406, 10000):(2406, 10)	Test scores: 0.555 macro-f1, 0.759 micro-f1
            Running BoW baseline on language it: train (2416, 10000):(2416, 10) Test scores: 0.551 macro-f1, 0.761 micro-f1
            Running BoW baseline on language es: train (2415, 10000):(2415, 10) Test scores: 0.565 macro-f1, 0.766 micro-f1
            """

    elif mode == "ESA":
        lang = 'es'
        print("Running Monolingual ESA transformation for lang %s" % lang)
        wiki_rand_doc_selection(n_wiki_docs=10000, clesa_data=clesa_data, seed=123456789)
        clesa_data.problem_setting(train_langs=[lang], test_langs=[lang])


        #clesa = CLESA_PPindex(5)
        print("\ttrain %s:%s" % (str(clesa_data.lXtr[lang].shape), str(clesa_data.lYtr[lang].shape)))
        print("\ttest  %s:%s" % (str(clesa_data.lXte[lang].shape), str(clesa_data.lYte[lang].shape)))
        print("\twiki  %s" % (str(clesa_data.lW[lang].shape)))
        rrobin = fit_round_robin(clesa_data.lXtr[lang], clesa_data.lYtr[lang], k=10000,
                                 features_rank_pickle_path=join(jrcacquis_datapath, dataset + "_" + lang + "X" + str(
                                     clesa_data.lXtr[lang].shape) + "_Y" + str(clesa_data.lYtr[lang].shape) + ".pickle"))
        clesa_data.lXtr[lang] = rrobin.transform(clesa_data.lXtr[lang])
        clesa_data.lXte[lang] = rrobin.transform(clesa_data.lXte[lang])
        clesa_data.lW[lang] = rrobin.transform(clesa_data.lW[lang])
        print("\ttrain %s:%s" % (str(clesa_data.lXtr[lang].shape), str(clesa_data.lYtr[lang].shape)))
        print("\ttest  %s:%s" % (str(clesa_data.lXte[lang].shape), str(clesa_data.lYte[lang].shape)))
        print("\twiki  %s" % (str(clesa_data.lW[lang].shape)))

        clesa = CLESA()
        clesa.fit(clesa_data.lW)
        Xtr_, Ytr_ = clesa.transform(clesa_data.lXtr, clesa_data.lYtr)
        Xte_, Yte_ = clesa.transform(clesa_data.lXte, clesa_data.lYte) # the lYte is not altered, just stacked in the same order
        model = OneVsRestClassifier(clesa.learner(), n_jobs=-1)
        print("fit")
        model.fit(Xtr_, Ytr_)
        print("predict")
        yte_ = model.predict(Xte_)
        print("eval")
        macro_f1 = macroF1(Yte_, yte_)
        micro_f1 = microF1(Yte_, yte_)
        print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))

        # ESA English: Test scores: 0.372 macro-f1, 0.685 micro-f1 [wikidocs = 1000 random]
        # ESA English: Test scores: 0.584 macro-f1, 0.766 micro-f1 [wikidocs = 10000 random]
        # ESA Spanish: Test scores: 0.569 macro-f1, 0.762 micro-f1 [wikidocs = 10000 random]

    elif mode == "CL-ESA":
        tr_lang = ['es', 'es']
        te_lang = 'es'
        n_wiki_docs = 10000
        print("Running CL-ESA transformation from source lang %s to target lang %s" % (tr_lang, te_lang))
        #wiki_doc_selection_h1(n_wiki_docs=n_wiki_docs, clesa_data=clesa_data)
        #wiki_rand_doc_selection(n_wiki_docs=n_wiki_docs, clesa_data=clesa_data, seed=123456789)
        #clesa_data.problem_setting(train_langs=tr_lang, test_langs=[te_lang])


        print("\ttrain %s:%s" % (str([clesa_data.lXtr[tr_l].shape for tr_l in tr_lang]), str([clesa_data.lYtr[tr_l].shape for tr_l in tr_lang])))
        print("\ttest  %s:%s" % (str(clesa_data.lXte[te_lang].shape), str(clesa_data.lYte[te_lang].shape)))
        print("\twiki (S/T) %s:%s" % (str([clesa_data.lW[tr_l].shape for tr_l in tr_lang]), str(clesa_data.lW[te_lang].shape)))
        # for tr_l in tr_lang:
        #     rrobin = fit_round_robin(clesa_data.lXtr[tr_l], clesa_data.lYtr[tr_l], k=10000,
        #                              features_rank_pickle_path=join(jrcacquis_datapath, dataset + "_" + tr_l + "X" + str(
        #                                  clesa_data.lXtr[tr_l].shape) + "_Y" + str(
        #                                  clesa_data.lYtr[tr_l].shape) + ".pickle"))
        #     clesa_data.lXtr[tr_l] = rrobin.transform(clesa_data.lXtr[tr_l])
        #     clesa_data.lW[tr_l] = rrobin.transform(clesa_data.lW[tr_l])
        #     if tr_l in [te_lang]:
        #         clesa_data.lXte[tr_l] = rrobin.transform(clesa_data.lXte[tr_l])
        # print("\ttrain %s:%s" % (str([clesa_data.lXtr[tr_l].shape for tr_l in tr_lang]), str([clesa_data.lYtr[tr_l].shape for tr_l in tr_lang])))
        # print("\ttest  %s:%s" % (str(clesa_data.lXte[te_lang].shape), str(clesa_data.lYte[te_lang].shape)))
        # print("\twiki (S/T) %s:%s" % (str([clesa_data.lW[tr_l].shape for tr_l in tr_lang]), str(clesa_data.lW[te_lang].shape)))

        clesa = CLESA(similarity='cosine', centered=False, post_norm=False)
        #clesa = CLESA_PPindex(2)
        print("clesa projection")
        clesa.fit(clesa_data.lW)

        #0.665 macro-f1, 0.797 micro-f1 with it in training
        #with cleaning 0.348 macro-f1, 0.485 micro-f1 without it in training!



        # clesa_data.lXte['it'] = clesa_data.lXtr['it']
        # clesa_data.lYte['it'] = clesa_data.lYtr['it']
        # del clesa_data.lXtr['it']
        # del clesa_data.lYtr['it']
        # del clesa_data.lXte['es']
        # del clesa_data.lYte['es']

        Xtr_, Ytr_ = clesa.transform(clesa_data.lXtr, clesa_data.lYtr)

        print("inspecting noise and selecting wikipedia documents")
        wiki_doc_selection_noise(10000, clesa_data, Xtr_, len(tr_lang))

        print("deleting the italian documents from training")
        del clesa_data.lXtr['it']
        del clesa_data.lYtr['it']
        tr_lang.remove('it')
        del clesa_data.lXte['es']
        del clesa_data.lYte['es']


        print("reprojecting")
        clesa.fit(clesa_data.lW)
        print("\ttrain %s:%s" % (
        str([clesa_data.lXtr[tr_l].shape for tr_l in tr_lang]), str([clesa_data.lYtr[tr_l].shape for tr_l in tr_lang])))
        print("\ttest  %s:%s" % (str(clesa_data.lXte[te_lang].shape), str(clesa_data.lYte[te_lang].shape)))
        print("\twiki (S/T) %s:%s" % (
        str([clesa_data.lW[tr_l].shape for tr_l in tr_lang]), str(clesa_data.lW[te_lang].shape)))
        Xtr_, Ytr_ = clesa.transform(clesa_data.lXtr, clesa_data.lYtr)
        Xte_, Yte_ = clesa.transform(clesa_data.lXte, clesa_data.lYte)  # the lYte is not altered, just stacked in the same order




        model = OneVsRestClassifier(clesa.learner(), n_jobs=-1)
        print("fit")
        model.fit(Xtr_, Ytr_)
        print("predict")
        yte_ = model.predict(Xte_)
        print("eval")
        macro_f1 = macroF1(Yte_, yte_)
        micro_f1 = microF1(Yte_, yte_)
        print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))

        # ESA English: Test scores: 0.372 macro-f1, 0.685 micro-f1 [wikidocs = 1000 random]
        # ESA English: Test scores: 0.584 macro-f1, 0.766 micro-f1 [wikidocs = 10000 random]
        # ESA Spanish: Test scores: 0.348 macro-f1, 0.672 micro-f1 [wikidocs = 10000 random; dot-not-centered]
        # ESA Spanish: Test scores: 0.348 macro-f1, 0.671 micro-f1 [wikidocs = 10000 random; dot-not-centered


# IDEA: Elegir los documentos de la wikipedia que en la representacion CLESA (con ellos mismos como documentos pivots)...
        #no es tan facil!



        #parameters = {'C': [1e4, 1e3, 1e2, 1e1, 1, 1e-1], 'loss': ['hinge', 'squared_hinge'], 'dual': [True, False]}

    # print("Tunning hyperparameters...")
    # tunned_model = fit_model_hyperparameters(Xtr_, Ytr_, parameters, model)
    # tunning_time = time.time() - init_time
    # print("\t%s: best parameters %s, best score %.3f, took %.3f seconds" %
    #       (type(model).__name__, tunned_model.best_params_, tunned_model.best_score_, tunning_time))
    #
    # Xte_, Yte_ = clesa.transform(clesa_data.lXte, clesa_data.lYte) #the lYte is not altered, just stacked in the same order
    # print(Xte_.shape)
    # print(Yte_.shape)
    #
    # yte_ = tunned_model.predict(Xte_)
    #
    # macro_f1 = macroF1(Yte_, yte_)
    # micro_f1 = microF1(Yte_, yte_)
    # print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))


    # CLESA: Test scores: 0.284 macro-f1, 0.555 micro-f1
    # Mono-Bow: 0.242 macro-f1, 0.526 micro-f1
    # Mono-ESA:
