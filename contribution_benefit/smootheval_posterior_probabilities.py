import util.disable_sklearn_warnings
import os,sys
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.metrics import *


output = './results_MSE_MAE_posterior_probabilities_jrc.csv'

for run in range(10):

    #dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processed_run'+str(run)+'.pickle'
    dataset = '/media/moreo/1TB Volume/Datasets/JRC_Acquis_v3/jrc_nltk_1958-2005vs2006_all_top300_noparallel_processed_run'+str(run)+'.pickle'
    data = MultilingualDataset.load(dataset)

    for language in list(data.langs()):
        print('run {}, lang {}:'.format(run, language))

        data.set_view(languages=[language])
        #data.show_dimensions()

        svm = OneVsRestClassifier(SVC(kernel='linear', probability=True, cache_size=1000), n_jobs=-1)

        print('training')
        Xtr, ytr = data.lXtr()[language], data.lYtr()[language]
        svm.fit(Xtr, ytr)

        print('getting posterioir probabilities')
        Xte,yte = data.lXte()[language], data.lYte()[language]
        Z = svm.predict_proba(Xte)


        print('evaluation')
        # smooth_MF1 = smoothmacroF1(yte, Z)
        # smooth_mF1 = smoothmicroF1(yte, Z)
        # smooth_MK = smoothmacroK(yte, Z)
        # smooth_mK = smoothmicroK(yte, Z)
        MSE = np.mean((Z - yte) ** 2)
        MAE = np.mean((np.abs(Z - yte)))

        with open(output, 'a') as results:
            #out_line = ('%s\t%d\t%s\t%.3f\t%.3f\t%.3f\t%.3f' % (data.dataset_name, run, language, smooth_MF1, smooth_mF1, smooth_MK, smooth_mK))
            out_line = ('%s\t%d\t%s\t%.3f\t%.3f\n' % (data.dataset_name, run, language, MSE, MAE))
            print('Data\trun\tLang\tMSE\tMAE')
            print(out_line+'\n')
            results.write(out_line)



