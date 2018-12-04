import util.disable_sklearn_warnings
from sklearn.svm import SVC
import os,sys
from dataset_builder import MultilingualDataset
from feature_selection.round_robin import RoundRobin
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
                  help="Model to apply", type=str, default=None)
parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results.csv')
parser.add_option("-n", "--note", dest="note",
                  help="A description note to be added to the result file", type=str,  default='')
parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimices hyperparameters", default=False)
parser.add_option("-b", "--binary", dest="binary",type=int,
                  help="Run experiments on a single category specified with this parameter", default=-1)
parser.add_option("-L", "--lang_ablation", dest="lang_ablation",type=str,
                  help="Removes the language from the training", default=None)
parser.add_option("-f", "--force", dest="force", action='store_true',
                  help="Run even if the result was already computed", default=False)
parser.add_option("-j", "--n_jobs", dest="n_jobs",type=int,
                  help="Number of parallel jobs (default is -1, all)", default=-1)
parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=1)

parser.add_option("-w", "--we-path", dest="we_path",
                  help="Path to the polylingual word embeddings (required only if --mode polyembeddings)")
parser.add_option("-W", "--wiki", dest="wiki",
                  help="Path to Wikipedia raw documents", type=str, default=None)
parser.add_option("--calmode", dest="calmode",
                  help="Calibration mode for the base classifiers (only for class-based models). Valid ones are"
                       "'cal' (default, calibrates the base classifiers and use predict_proba to project), "
                       "'nocal' (does not calibrate, use the decision_function to project)"
                       "'sigmoid' (does not calibrate, use the sigmoid of the decision function to project)",
                  default='cal')

"""
Last changes:
- poly as the default kernel
- language trace activated (value 1)
"""

#TODO: think about the neural-net extension
#TODO: redo the juxtaclass, according to "Discriminative Methods for Multi-labeled Classification" and rename properly

#note: Multinomial Naive-Bayes descartado: no está calibrado, no funciona con valores negativos, la adaptación a valores
#reales es artificial
#note: really make_scorer(macroF1) seems to be better with the actual loss [tough not significantly]

def get_learner(calibrate=False):
    return SVC(kernel='linear', probability=calibrate, cache_size=1000, C=op.set_c, random_state=1)

def get_params(dense=False):
    if not op.optimc:
        return None

    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    if not dense:
        return [{'kernel': ['linear'], 'C': c_range}]
    else:
        return [{'kernel': ['rbf'], 'C': c_range}]


if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'

    results = PolylingualClassificationResults(op.output)

    dataset_file = os.path.basename(op.dataset)
    result_id = dataset_file+'_'+op.mode+'_svm'+('_optimC' if op.optimc else ('_setc'+str(op.set_c) if op.set_c!=1. else ''))+\
                ('_bin'+str(op.binary) if op.binary != -1 else '')+\
                ('_langablation_'+str(op.lang_ablation) if op.lang_ablation else '')

    if not op.force and results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

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
    #data.show_category_prevalences()

    lXtr = data.lXtr()
    lytr = data.lYtr()
    lXte = data.lXte()
    lyte = data.lYte()

    calibrate = (op.calmode == 'cal')
    if op.mode == 'class':
        print('Learning Class-Embedding Poly-lingual Classifier')
        classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=calibrate),
                                                     meta_learner=get_learner(calibrate=False),
                                                     first_tier_parameters=None, meta_parameters=get_params(dense=True),
                                                     calmode=op.calmode,
                                                     n_jobs=op.n_jobs)
    elif op.mode == 'class-10':
        print('Learning 10-Fold CV Class-Embedding Poly-lingual Classifier')
        classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=calibrate),
                                                     meta_learner=get_learner(calibrate=False),
                                                     first_tier_parameters=None, meta_parameters=get_params(dense=True),
                                                     folded_projections=10,
                                                     calmode=op.calmode,
                                                     n_jobs=op.n_jobs)
    elif op.mode == 'naive':
        print('Learning Naive Poly-lingual Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs)
    elif op.mode == 'juxta':
        print('Learning Juxtaposed Poly-lingual Classifier')
        classifier = JuxtaposedPolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs)
    elif op.mode == 'lri':
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs)
    elif op.mode == 'lri-25k':
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(base_learner=get_learner(), parameters=get_params(), reduction=25000, n_jobs=op.n_jobs)
    elif op.mode == 'dci-lin':
        print('Learning Distributional Correspondence Indexing with Linear Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(base_learner=get_learner(), dcf='linear', z_parameters=get_params(dense=True), n_jobs=op.n_jobs)
    elif op.mode == 'dci-pmi':
        print('Learning Distributional Correspondence Indexing with PMI Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(base_learner=get_learner(), dcf='pmi', z_parameters=get_params(dense=True), n_jobs=op.n_jobs)
    elif op.mode == 'clesa':
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(), lW=lW, z_parameters=get_params(dense=True), n_jobs=op.n_jobs)
    elif op.mode == 'upper':
        assert data.langs()==['en'], 'only English is expected in the upper bound call'
        print('Learning Upper bound as the English-only Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs) #this is just to match the multilingual dataset format (despite there are only English documents)
    elif op.mode == 'monoclass':
        assert data.langs()==['en'], 'only English is expected in the monolingual class embedding call'
        print('Learning Monolingual Class-Embedding in the English-only corpus')
        classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=True),
                                                     meta_learner=get_learner(calibrate=False),
                                                     first_tier_parameters=None, meta_parameters=get_params(dense=True), n_jobs=op.n_jobs)
    elif op.mode == 'juxtaclass':
        print('Learning Juxtaposed-Class-Embeddings Poly-lingual Classifier')
        classifier = ClassJuxtaEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                              final_learner=get_learner(calibrate=False),
                                                              alpha=0.5,
                                                              c_parameters=get_params(), y_parameters=get_params(), n_jobs=op.n_jobs)
    elif op.mode == 'polyembeddings':
        print('Learning Poly-lingual Word Embedding based Classifier')
        classifier = PolylingualEmbeddingsClassifier(wordembeddings_path=op.we_path, learner=get_learner(calibrate=False),
                                                     c_parameters=get_params(dense=False), n_jobs=op.n_jobs)
    elif op.mode == 'plda':
        print('Learning Poly-lingual Classifier based on Poly-LDA')
        lW = pickle.load(open(op.wiki, 'rb')) # to be preprocessed
        classifier = PLDAPolylingualClassifier(base_learner=get_learner(), lW=lW, z_parameters=get_params(dense=True),
                                               nTopics=400, iterations=2000, max_wiki=2500, n_jobs=-1)

    elif op.mode == 'kcca':
        lW = pickle.load(open(op.dataset.replace('.pickle', '.wiki.pickle'), 'rb'))

        print('Learning KCCA-based Classifier')
        classifier = KCCAPolylingualClassifier(base_learner=get_learner(), lW=lW, z_parameters=get_params(dense=True),
                                               numCC=1000, reg=100, max_wiki=2000, n_jobs=op.n_jobs)


    else:
        raise ValueError('Unknown mode {}'.format(op.mode))



    classifier.fit(lXtr, lytr)
    l_eval = evaluate_method(classifier, lXte, lyte)
    grand_totals = average_results(l_eval, show=True)

    if op.calmode!='cal': # the default value does not add a postfix to the method name
        op.mode += op.calmode

    for lang in lXte.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        notes=op.note + ('C='+str(op.set_c) if op.set_c!=1 else '') + str(classifier.best_params() if op.optimc else '')
        results.add_row(result_id, op.mode, 'svm', op.optimc, data.dataset_name, op.binary, op.lang_ablation, classifier.time, lang, macrof1, microf1, macrok, microk, notes=notes)

