import util.disable_sklearn_warnings
import os,sys
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.results import PolylingualClassificationResults
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from util.file import exists
import pickle

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")
parser.add_option("-m", "--mode", dest="mode",
                  help="Training documents are allowed to have parallel versions of it", type=str, default=None)
parser.add_option("-l", "--learner", dest="learner",
                  help="Learner method for classification", type=str, default='svm')
parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./binary_results.csv')
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

def get_learner(calibrate=False):
    if op.learner == 'svm':
        learner = SVC(kernel='linear', probability=calibrate, cache_size=1000, C=op.set_c)
    elif op.learner == 'nb':
        learner = MultinomialNB()
    elif op.learner == 'lr':
        learner = LogisticRegression(C=op.set_c)
    return learner

def get_params(z_space=False):
    if not op.optimc:
        return None

    c_range = [1e4, 1e3, 1e2, 1e1, 1]
    if op.learner == 'svm':
        params = [{'kernel': ['linear'], 'C': c_range}] if not z_space else [{'kernel': ['rbf'], 'C': c_range}] # [{'kernel': ['poly'], 'C': c_range, 'coef0':[0., 1.], 'gamma':['auto', 2.], 'degree':[3,4]}]
        #, 'gamma' : [0.001, 0.01, 0.1, 1]
    elif op.learner == 'nb':
        params = [{'alpha': [1.0, .1, .05, .01, .001, 0.0]}]
    elif op.learner == 'lr':
        params = [{'C': c_range}]
    return params

if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert op.learner in ['svm', 'lr', 'nb'], 'unexpected learner'
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'
    assert op.mode in ['class','class-10'], 'unexpected mode'

    results = PolylingualClassificationResults(op.output)

    dataset_file = os.path.basename(op.dataset)
    result_id = dataset_file+'_'+op.mode+op.learner+('_optimC' if op.optimc else '')

    if not op.force and results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()
    #data.show_category_prevalences()

    if op.mode == 'class':
        print('Learning Class-Embedding Poly-lingual Classifier')
        classifier = FunnelingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                    final_learner=get_learner(calibrate=False),
                                                    base_parameters=None, meta_parameters=get_params(z_space=True),
                                                    n_jobs=op.n_jobs)
    elif op.mode == 'class-10':
        print('Learning 10-Fold CV Class-Embedding Poly-lingual Classifier')
        classifier = FunnelingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                    final_learner=get_learner(calibrate=False),
                                                    base_parameters=None, meta_parameters=get_params(z_space=True),
                                                    folded_projections=10,
                                                    n_jobs=op.n_jobs)
    nC = data.num_categories()
    for c in range(nC):
        print('binary: ' + str(c))
        data.set_view(categories=c)
        classifier.fit(data.lXtr(), data.lYtr())
        l_eval = get_binary_counters(classifier, data.lXte(), data.lYte())

        for lang in data.langs():
            tp, tn, fp, fn = l_eval[lang]
            print('Lang %s: tp=%d tn=%d fp=%d fn=%d' % (lang, tp, tn, fp, fn))
            notes = 'TP,TN,FP,FN'+op.note + ('C=' + str(op.set_c) if op.set_c != 1 else '') + str(classifier.best_params() if op.optimc else '')
            results.add_row(result_id, op.mode, op.learner, op.optimc, data.dataset_name, c, -1, classifier.time, lang, tp, tn, fp, fn, notes=notes)



