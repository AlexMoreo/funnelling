from data.clesa_data_generator import *
from data.languages import *
from model.clesa import CLESA, CLESA_PPindex
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from util.metrics import *
import sys
import time

def fit_model_hyperparameters(Xtr, ytr, parameters, model):
    single_class = ytr.shape[1] == 1
    if not single_class:
        parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
        model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters, scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)

    if single_class: ytr = np.squeeze(ytr)
    return model_tunning.fit(Xtr, ytr)

if __name__ == "__main__":

    dataset = "JRCAcquis"
    langmap = "DEBUG"
    tr_years = [2005]
    te_years = [2006]
    cat_threshold=50
    jrcacquis_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"
    wikipedia_datapath = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia/multilingual_docs"

    langs = lang_set[langmap]

    pickle_name = join(jrcacquis_datapath, 'preprocessed_' + langmap
                               + '_tr_' + year_list_as_str(tr_years)
                               + '_te_' + year_list_as_str(te_years)
                               + '_broadcats.pickle')

    clesa_data = clesa_data_generator(dataset, langs, tr_years, te_years,
                                      jrcacquis_datapath, wikipedia_datapath,
                                      cat_threshold=50, pickle_name=pickle_name, langmap=langmap)

    #clesa = CLESA()
    clesa = CLESA_PPindex(5)
    print("Running CL-ESA transformation")
    clesa.fit(clesa_data.lW)
    Xtr_,Ytr_ = clesa.transform(clesa_data.lXtr, clesa_data.lYtr)
    Xte_, Yte_ = clesa.transform(clesa_data.lXte,
                                 clesa_data.lYte)  # the lYte is not altered, just stacked in the same order

    Xtr_ = Xtr_[:500,:]
    Xte_ = Xte_[:500, :]
    Ytr_ = Ytr_[:500,:5]
    Yte_ = Yte_[:500,:5]
    print(Xtr_.shape)
    print(Ytr_.shape)
    model = clesa.learner()

    init_time = time.time()

    print("fit")
    model = OneVsRestClassifier(model, n_jobs=-1)
    model.fit(Xtr_, Ytr_)
    print("predict")
    yte_ = model.predict(Xte_)
    print("eval")
    macro_f1 = macroF1(Yte_, yte_)
    micro_f1 = microF1(Yte_, yte_)
    print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))

    sys.exit()

    #parameters = {'C': [1e1, 1], 'loss': ['squared_hinge'], 'dual': [True]}
    parameters = {'C': [1e1, 1]}
    #parameters = {'C': [1e4, 1e3, 1e2, 1e1, 1, 1e-1], 'loss': ['hinge', 'squared_hinge'], 'dual': [True, False]}

    print("Tunning hyperparameters...")
    tunned_model = fit_model_hyperparameters(Xtr_, Ytr_, parameters, model)
    tunning_time = time.time() - init_time
    print("\t%s: best parameters %s, best score %.3f, took %.3f seconds" %
          (type(model).__name__, tunned_model.best_params_, tunned_model.best_score_, tunning_time))

    Xte_, Yte_ = clesa.transform(clesa_data.lXte, clesa_data.lYte) #the lYte is not altered, just stacked in the same order
    print(Xte_.shape)
    print(Yte_.shape)

    yte_ = tunned_model.predict(Xte_)

    macro_f1 = macroF1(Yte_, yte_)
    micro_f1 = microF1(Yte_, yte_)
    print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))

    # CLESA: Test scores: 0.284 macro-f1, 0.555 micro-f1
