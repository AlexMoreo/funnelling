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

valid_models = ['fun-kfcv', 'fun-tat', 'naive', 'lri', 'clesa', 'kcca', 'dci', 'ple', 'upper', 'fun-mono']

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset", help="Path to the multilingual dataset processed and stored in .pickle format")
parser.add_option("-m", "--mode", dest="mode", help="Model code of the polylingual classifier, valid ones include {}".format(valid_models), type=str, default=None)
parser.add_option("-o", "--output", dest="output", help="Result file", type=str,  default='./results.csv')
parser.add_option("-n", "--note", dest="note", help="A description note to be added to the result file", type=str,  default='')
parser.add_option("-c", "--optimc", dest="optimc", action='store_true', help="Optimice hyperparameters", default=False)
parser.add_option("-b", "--binary", dest="binary",type=int, help="Run experiments on a single category specified with this parameter", default=-1)
parser.add_option("-L", "--lang_ablation", dest="lang_ablation",type=str, help="Removes the language from the training", default=None)
parser.add_option("-f", "--force", dest="force", action='store_true', help="Run even if the result was already computed", default=False)
parser.add_option("-j", "--n_jobs", dest="n_jobs",type=int, help="Number of parallel jobs (default is -1, all)", default=-1)
parser.add_option("-s", "--set_c", dest="set_c",type=float, help="Set the C parameter", default=1)
parser.add_option("-r", "--kccareg", dest="kccareg",type=float, help="Set the regularization parameter for KCCA", default=1)
parser.add_option("-w", "--we-path", dest="we_path", help="Path to the polylingual word embeddings (required only if --mode polyembeddings)")
parser.add_option("-W", "--wiki", dest="wiki", help="Path to Wikipedia raw documents", type=str, default=None)
parser.add_option("--calmode", dest="calmode", help="Calibration mode for the base classifiers (only for class-based models). Valid ones are"
                       "'cal' (default, calibrates the base classifiers and use predict_proba to project), "
                       "'nocal' (does not calibrate, use the decision_function to project)"
                       "'sigmoid' (does not calibrate, use the sigmoid of the decision function to project)",
                  default='cal')


def get_learner(calibrate=False, kernel='linear'):
    return SVC(kernel=kernel, probability=calibrate, cache_size=1000, C=op.set_c, random_state=1)

def get_params(dense=False):
    if not op.optimc:
        return None
    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    kernel = 'rbf' if dense else 'linear'
    return [{'kernel': [kernel], 'C': c_range}]


#-------------------------------------------
# MAIN
#-------------------------------------------
if __name__=='__main__':

    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'
    assert op.mode in valid_models, 'Unknown mode, valid ones are {}'.format(valid_models)

    # if the results file exists, then load it, if not, creates an empty one
    results = PolylingualClassificationResults(op.output)

    dataset_file = os.path.basename(op.dataset)
    result_id = dataset_file+'_'+op.mode+'_svm'+('_optimC' if op.optimc else ('_setc'+str(op.set_c) if op.set_c!=1. else ''))+\
                ('_bin'+str(op.binary) if op.binary != -1 else '')+\
                ('_langablation_'+str(op.lang_ablation) if op.lang_ablation else '')

    # skip the experiment if already calculated (unless force==True)
    if not op.force and results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

    # load the multilingual dataset
    data = MultilingualDataset.load(op.dataset)
    if op.binary != -1:
        assert op.binary < data.num_categories(), 'category not in scope'
        data.set_view(categories=np.array([op.binary]))
    if op.lang_ablation:
        assert op.lang_ablation in data.langs(), 'language for ablation test not in scope'
        languages = list(data.langs())
        languages.remove(op.lang_ablation)
        data.set_view(languages=languages)
    data.show_dimensions()

    lXtr, lytr = data.training()
    lXte, lyte = data.test()

    calibrate = (op.calmode == 'cal')

    # instantiate the learner requested in op.mode
    if op.mode in ['fun-kfcv','fun-tat']:

        if op.mode == 'fun-kfcv':
            print('Learning Fun(KFCV) Poly-lingual Classifier')
            folds = 10
        else:
            print('Learning Fun(TAT) Poly-lingual Classifier')
            folds = 1

        classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=calibrate),
                                                     meta_learner=get_learner(calibrate=False),
                                                     first_tier_parameters=None,
                                                     meta_parameters=get_params(dense=True),
                                                     folded_projections=folds,
                                                     calmode=op.calmode,
                                                     n_jobs=op.n_jobs)
    elif op.mode == 'naive':
        print('Learning Naive Poly-lingual Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(),
                                                parameters=get_params(),
                                                n_jobs=op.n_jobs)
    elif op.mode == 'lri':
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(base_learner=get_learner(),
                                              parameters=get_params(),
                                              reduction=25000,
                                              n_jobs=op.n_jobs)
    elif op.mode == 'dci':
        print('Learning Distributional Correspondence Indexing with Linear Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(base_learner=get_learner(),
                                              dcf='linear',
                                              z_parameters=get_params(dense=True),
                                              n_jobs=op.n_jobs)
    elif op.mode == 'clesa':
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(),
                                                lW=lW,
                                                z_parameters=get_params(dense=True),
                                                n_jobs=op.n_jobs)
    elif op.mode == 'upper':
        assert data.langs()==['en'], 'only English is expected in the upper bound call'
        print('Learning Upper bound as the English-only Classifier')
        # this is just to match the multilingual dataset format (despite there are only English documents)
        classifier = NaivePolylingualClassifier(base_learner=get_learner(),
                                                parameters=get_params(),
                                                n_jobs=op.n_jobs)
    elif op.mode == 'fun-mono':
        assert data.langs()==['en'], 'only English is expected in the monolingual class embedding call'
        print('Learning Monolingual Class-Embedding in the English-only corpus')
        classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=True),
                                                     meta_learner=get_learner(calibrate=False),
                                                     first_tier_parameters=None,
                                                     meta_parameters=get_params(dense=True),
                                                     n_jobs=op.n_jobs)

    elif op.mode == 'ple':
        print('Learning Poly-lingual Embedding (PLE) Classifier')
        classifier = PolylingualEmbeddingsClassifier(wordembeddings_path=op.we_path,
                                                     learner=get_learner(calibrate=False),
                                                     c_parameters=get_params(dense=False),
                                                     n_jobs=op.n_jobs)

    elif op.mode == 'kcca':
        lW = pickle.load(open(op.dataset.replace('.pickle', '.wiki.pickle'), 'rb'))

        print('Learning KCCA-based Classifier')
        classifier = KCCAPolylingualClassifier(base_learner=get_learner(kernel='rbf'),
                                               lW=lW,
                                               z_parameters=get_params(dense=True),
                                               numCC=1000,
                                               reg=op.kccareg,
                                               max_wiki=2000,
                                               n_jobs=op.n_jobs)


    # train the classifier
    classifier.fit(lXtr, lytr)
    tr_time = classifier.time

    # test and evaluate
    l_eval, te_time = evaluate_method(classifier, lXte, lyte, return_time=True)
    grand_totals = average_results(l_eval, show=True)

    # write result
    if op.calmode!='cal': # the default value does not add a postfix to the method name
        op.mode += op.calmode

    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        notes=op.note + ('C='+str(op.set_c) if op.set_c!=1 else '') + str(classifier.best_params() if op.optimc else '') + (' te_time: {:.1f}'.format(te_time))
        results.add_row(result_id, op.mode, 'svm', op.optimc, data.dataset_name, op.binary, op.lang_ablation, tr_time, lang, macrof1, microf1, macrok, microk, notes=notes)

