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

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")
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
    result_id = dataset_file+'_fun-tat'+('_optimC' if op.optimc else '')

    if not op.force and results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()

    classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=True),
                                                 meta_learner=get_learner(calibrate=False),
                                                 first_tier_parameters=None,
                                                 meta_parameters=get_params(dense=True),
                                                 folded_projections=1,
                                                 calmode='cal',
                                                 n_jobs=op.n_jobs)

    nC = data.num_categories()
    for c in range(nC):
        print('binary: ' + str(c))
        data.set_view(categories=c)
        """
        if you get the error "TypeError: No loop matching the specified signature and casting was found for ufunc true_divide"
        try replacing this line: 
        Y /= np.sum(Y, axis=1)[:, np.newaxis]
        with this line
        Y = np.true_divide(Y, np.sum(Y, axis=1)[:, np.newaxis])
        in <your-env-path>/site-packages/sklearn/multiclass.py line 352
        """
        classifier.fit(data.lXtr(), data.lYtr())
        l_eval = get_binary_counters(classifier, data.lXte(), data.lYte())

        for lang in data.langs():
            tp, tn, fp, fn = l_eval[lang]
            print('Lang %s: tp=%d tn=%d fp=%d fn=%d' % (lang, tp, tn, fp, fn))
            notes = 'TP,TN,FP,FN'+op.note + ('C=' + str(op.set_c) if op.set_c != 1 else '') + str(classifier.best_params() if op.optimc else '')
            results.add_row(result_id, 'fun-tat', 'svm', op.optimc, data.dataset_name, c, -1, classifier.time, lang, tp, tn, fp, fn, notes=notes)



