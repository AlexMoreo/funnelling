import util.disable_sklearn_warnings
from sklearn.svm import SVC
import os,sys
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.results import PolylingualClassificationResults
from util.file import exists
import pickle
from sklearn.utils import shuffle

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")
parser.add_option("-m", "--mode", dest="mode",
                  help="Model to apply", type=str, default=None)
parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results.csv')
parser.add_option("-n", "--note", dest="note",
                  help="A description note to be added to the result file", type=str,  default='')
parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimices hyperparameters", default=False)
parser.add_option("-f", "--force", dest="force", action='store_true',
                  help="Run even if the result was already computed", default=False)
parser.add_option("-j", "--n_jobs", dest="n_jobs",type=int,
                  help="Number of parallel jobs (default is -1, all)", default=-1)
parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=1)


# in this experiment we vary, for each (target) language, the size of its training documents (the rest of training
# languages are left untouched

def get_learner(calibrate=False, kernel='linear'):
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=op.set_c, random_state=1)

def get_params(dense=False):
    if not op.optimc:
        return None
    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    kernel = 'rbf' if dense else 'linear'
    return [{'kernel': [kernel], 'C': c_range}]

if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'

    results = PolylingualClassificationResults(op.output)

    dataset_file = os.path.basename(op.dataset)
    result_id = dataset_file+'_svm'+('_optimC' if op.optimc else ('_setc'+str(op.set_c) if op.set_c!=1. else ''))

    if not op.force and results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

    data = MultilingualDataset.load(op.dataset)
    langs = data.langs()
    for lang in langs:
        if data is None:
            data = MultilingualDataset.load(op.dataset)

        ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids)) = data.multiling_dataset[lang]
        Xtr, Ytr = shuffle(Xtr, Ytr, random_state=0)
        Xtr.sort_indices()
        data.multiling_dataset[lang] = ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids))
        nD = Xtr.shape[0]

        for tr_proportion in np.arange(0., 1.01, 0.1)[::-1]:
            if tr_proportion < 1:
                ndocs = int(nD*tr_proportion)
                Xtr = Xtr[:ndocs]
                Ytr = Ytr[:ndocs]
                data.multiling_dataset[lang] = ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids))

            print('Language {}, reduction {}, shape {}:'.format(lang, tr_proportion, Xtr.shape))
            data.show_dimensions()

            if op.mode == 'fun-tat':
                print('Learning Class-Embedding Poly-lingual Classifier')
                classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=True),
                                                         meta_learner=get_learner(calibrate=False),
                                                         first_tier_parameters=None,
                                                         meta_parameters=get_params(dense=True),
                                                         n_jobs=op.n_jobs)
            elif op.mode == 'naive':
                print('Learning Naive Poly-lingual Classifier')
                classifier = NaivePolylingualClassifier(base_learner=get_learner(),
                                                        parameters=get_params(),
                                                        n_jobs=op.n_jobs)
                print('deleting all other languages in naive...')
                for l in langs:
                    if l != lang and l in data.multiling_dataset:
                        del data.multiling_dataset[l]
            else:
                raise ValueError('unknown mode')


            classifier.fit(data.lXtr(), data.lYtr())
            l_eval = evaluate_method(classifier, {lang:data.lXte()[lang]}, {lang:data.lYte()[lang]})

            #for lang in data.langs():
            macrof1, microf1, macrok, microk = l_eval[lang]
            print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
            notes=op.note + ('C='+str(op.set_c) if op.set_c!=1 else '') + str(classifier.best_params() if op.optimc else '')
            results.add_row(result_id, op.mode, 'svm', op.optimc, data.dataset_name, -1, tr_proportion, classifier.time, lang, macrof1, microf1, macrok, microk, notes=notes)

        data = None