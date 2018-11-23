import util.disable_sklearn_warnings
import os,sys
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.metrics import *


output = './results_ABLATION_posterior_probabilities_jrc.csv'

for run in [4,9,8,3]:

    #dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processed_run'+str(run)+'.pickle'
    dataset = '/media/moreo/1TB Volume/Datasets/JRC_Acquis_v3/jrc_nltk_1958-2005vs2006_all_top300_noparallel_processed_run'+str(run)+'.pickle'
    data = MultilingualDataset.load(dataset)
    data.show_dimensions()

    print('Init')

    all_langs = list(data.langs())
    for ablated_language in list(data.langs()):
        print('run {}, ablated_lang {}:'.format(run, ablated_language))

        data = MultilingualDataset.load(dataset)
        Xab, Yab = data.lXtr()[ablated_language], data.lYtr()[ablated_language]
        Xabte, Yabte = data.lXte()[ablated_language], data.lYte()[ablated_language]

        rest_langs = list(all_langs)
        rest_langs.remove(ablated_language)
        print(rest_langs, ablated_language)
        data.set_view(languages=rest_langs)
        data.show_dimensions()

        classifier = FunnellingPolylingualClassifier(first_tier_learner=SVC(kernel='linear', probability=True),
                                                     meta_learner=SVC(kernel='rbf', probability=False),
                                                     first_tier_parameters=None, meta_parameters=[{'kernel': ['rbf'], 'C': [1e4, 1e3, 1e2, 1e1, 1]}],
                                                     n_jobs=-1)

        print('Training Fun(TAT) with all but the ablated language')
        classifier.fit(data.lXtr(), data.lYtr())

        print('Training SVM with the ablated language')
        svm = OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=-1)
        svm.fit(Xab, Yab)

        print('getting posterioir probabilities of training examples in the ablation language')
        Z = svm.predict_proba(Xab)
        Zte = svm.predict_proba(Xabte)

        print('classifying them through the Fun(TAT) meta-classifiver')
        Yab_ = classifier.model.predict(Z)
        Yabte_ = classifier.model.predict(Zte)


        print('evaluation')
        MacroF1 = macroF1(Yab, Yab_)
        MicroF1 = microF1(Yab, Yab_)
        MacroK = macroK(Yab, Yab_)
        MicroK = microK(Yab, Yab_)
        MSE = np.mean((Yab - Yab_) ** 2.)
        MAE = np.mean((np.abs(Yab - Yab_)))

        print('Test:',macroF1(Yabte, Yabte_))
        print('Test:',microF1(Yabte, Yabte_))



        with open(output, 'a') as results:
            #out_line = ('%s\t%d\t%s\t%.3f\t%.3f\t%.3f\t%.3f' % (data.dataset_name, run, language, smooth_MF1, smooth_mF1, smooth_MK, smooth_mK))
            out_line = ('%s\t%d\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.6f\t%.6f\n' % (data.dataset_name, run, ablated_language, MacroF1, MicroF1, MacroK, MicroK, MSE, MAE))
            print('Data\trun\tLang\tMSE\tMAE')
            print(out_line+'\n')
            results.write(out_line)



