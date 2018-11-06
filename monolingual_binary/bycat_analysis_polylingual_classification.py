import util.disable_sklearn_warnings
import os,sys
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser

from util.metrics import __check_consistency_and_adapt, f1
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
                  help="Result file", type=str,  default='./by_cat_analysis.csv')
parser.add_option("-n", "--note", dest="note",
                  help="A description note to be added to the result file", type=str,  default='')
parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimices hyperparameters", default=False)
parser.add_option("-b", "--binary", dest="binary",type=int,
                  help="Run experiments on a single category specified with this parameter", default=-1)
parser.add_option("-L", "--languages", dest="languages",type=int,
                  help="Chooses the maximum number of random languages to consider", default=-1)
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


def evaluation_metrics_by_cat(y, y_, metric=f1):
    true_labels, predicted_labels, nC = __check_consistency_and_adapt(y, y_)
    return [metric(hard_single_metric_statistics(true_labels[:, c], predicted_labels[:, c])) for c in range(nC)]

def evaluate_by_cat(polylingual_method, lX, ly, predictor=None):
    print('prediction for test')
    assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
    n_jobs = polylingual_method.n_jobs
    if predictor is None:
        predictor = polylingual_method.predict
    ly_ = predictor(lX)
    print('evaluation (n_jobs={})'.format(n_jobs))
    langs = list(ly.keys())
    evals = Parallel(n_jobs=n_jobs)(delayed(evaluation_metrics_by_cat)(ly[lang], ly_[lang]) for lang in langs)
    return {lang: evals[i] for i, lang in enumerate(langs)}

if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert op.learner in ['svm', 'lr', 'nb'], 'unexpected learner'
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'
    # assert op.mode in ['class','class-lang','class-10', 'class-10-nocal', 'naive', 'juxta', 'lri', 'lri-25k',
    #                    'dci-lin', 'dci-pmi', 'clesa', 'upper', 'monoclass', 'juxtaclass'], 'unexpected mode'

    dataset_file = os.path.basename(op.dataset)

    data = MultilingualDataset.load(op.dataset)
    if op.binary != -1:
        assert op.binary < data.num_categories(), 'category not in scope'
        data.set_view(categories=np.array([op.binary]))
    if op.languages != -1:
        assert op.languages < len(data.langs()), 'too many languages'
        languages = ['en'] + [l for l in data.langs() if l != 'en']
        data.set_view(languages=languages[:op.languages])
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
    elif op.mode == 'naive':
        print('Learning Naive Poly-lingual Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs)
    elif op.mode == 'juxta':
        print('Learning Juxtaposed Poly-lingual Classifier')
        classifier = JuxtaposedPolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs)
    elif op.mode == 'lri':
        assert op.learner != 'nb', 'nb operates only on positive matrices'
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs)
    elif op.mode == 'lri-25k':
        assert op.learner != 'nb', 'nb operates only on positive matrices'
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(base_learner=get_learner(), parameters=get_params(), reduction=25000, n_jobs=op.n_jobs)
    elif op.mode == 'dci-lin':
        assert op.learner!='nb', 'nb operates only on positive matrices'
        print('Learning Distributional Correspondence Indexing with Linear Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(base_learner=get_learner(), dcf='linear', z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'dci-pmi':
        assert op.learner != 'nb', 'nb operates only on positive matrices'
        print('Learning Distributional Correspondence Indexing with PMI Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(base_learner=get_learner(), dcf='pmi', z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'clesa':
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(), lW=lW, z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'upper':
        assert data.langs()==['en'], 'only English is expected in the upper bound call'
        print('Learning Upper bound as the English-only Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs) #this is just to match the multilingual dataset format (despite there are only English documents)
    elif op.mode == 'monoclass':
        assert data.langs()==['en'], 'only English is expected in the monolingual class embedding call'
        print('Learning Monolingual Class-Embedding in the English-only corpus')
        classifier = FunnelingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                    final_learner=get_learner(calibrate=False),
                                                    base_parameters=None, meta_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'juxtaclass':
        print('Learning Juxtaposed-Class-Embeddings Poly-lingual Classifier')
        classifier = ClassJuxtaEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                              final_learner=get_learner(calibrate=False),
                                                              alpha=0.5,
                                                              c_parameters=get_params(), y_parameters=get_params(), n_jobs=op.n_jobs)

    classifier.fit(data.lXtr(), data.lYtr())
    l_eval = evaluate_by_cat(classifier, data.lXte(), data.lYte())

    with open(op.output, 'a') as fout:
        for lang in data.langs():
            by_cat_f1 = l_eval[lang]
            print('Lang %s: %s' % (lang, str(by_cat_f1)))
            fout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(op.mode, op.learner, op.optimc, data.dataset_name, lang, '\t'.join([str(x) for x in by_cat_f1])))

    print('Done')