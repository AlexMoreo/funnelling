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
                  help="Result file", type=str,  default='./results.csv')
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



#TODO: more baselines
#TODO: think about the neural-net extension
#TODO: make CLESA work
#TODO: redo the juxtaclass, according to "Discriminative Methods for Multi-labeled Classification" and rename properly
#TODO: calibration single-label
#TODO: experiment with varying number of categories
#TODO: arreglar la calibración y la búsqueda de parámetros
#TODO: fisher scores
#TODO: finish singlelabel-fragment
#TODO: probar feature selection?
#TODO: learners: lasso?
#TODO: dejo el single-fragment a False, o sea que genero un falso multilabel
#TODO: class-10 creo que el problema esta en como vienen representadas las categorias raras, los unicos ejemplos positivos
#      son clasificados por clasificadores provenientes de un fold donde no habia ningun ejemplo positivo

#note: Multinomial Naive-Bayes descargado: no está calibrado, no funciona con valores negativos, la adaptación a valores
#reales es artificial
#note: really make_scorer(macroF1) seems to be better with the actual loss [tough not significantly]

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
    # assert op.mode in ['class','class-lang','class-10', 'class-10-nocal', 'naive', 'juxta', 'lri', 'lri-25k',
    #                    'dci-lin', 'dci-pmi', 'clesa', 'upper', 'monoclass', 'juxtaclass'], 'unexpected mode'

    results = PolylingualClassificationResults(op.output)

    dataset_file = os.path.basename(op.dataset)
    result_id = dataset_file+'_'+op.mode+op.learner+('_optimC' if op.optimc else '')+\
                ('_bin'+str(op.binary) if op.binary != -1 else '')+\
                ('_langs'+str(op.languages) if op.languages != -1 else '')

    if not op.force and results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

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
        classifier = ClassEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                         final_learner=get_learner(calibrate=False),
                                                         parameters=None, z_parameters=get_params(z_space=True),
                                                         n_jobs=op.n_jobs)
    elif op.mode == 'class-10':
        print('Learning 10-Fold CV Class-Embedding Poly-lingual Classifier')
        classifier = ClassEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                         final_learner=get_learner(calibrate=False),
                                                         parameters=None, z_parameters=get_params(z_space=True),
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
    elif op.mode == 'clesa-1':
        assert not op.optimc, 'optim c is not allowed for method ' + op.mode
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(defaultC=1.), lW=lW, z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'clesa-10':
        assert not op.optimc, 'optim c is not allowed for method ' + op.mode
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(defaultC=10.), lW=lW, z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'clesa-100':
        assert not op.optimc, 'optim c is not allowed for method ' + op.mode
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(defaultC=100.), lW=lW, z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'clesa-1000':
        assert not op.optimc, 'optim c is not allowed for method ' + op.mode
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(defaultC=1000.), lW=lW, z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'clesa-10000':
        assert not op.optimc, 'optim c is not allowed for method ' + op.mode
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(defaultC=10000.), lW=lW, z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)

    elif op.mode == 'upper':
        assert data.langs()==['en'], 'only English is expected in the upper bound call'
        print('Learning Upper bound as the English-only Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs) #this is just to match the multilingual dataset format (despite there are only English documents)
    elif op.mode == 'monoclass':
        assert data.langs()==['en'], 'only English is expected in the monolingual class embedding call'
        print('Learning Monolingual Class-Embedding in the English-only corpus')
        classifier = ClassEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                         final_learner=get_learner(calibrate=False),
                                                         parameters=None, z_parameters=get_params(z_space=True), n_jobs=op.n_jobs)
    elif op.mode == 'juxtaclass':
        print('Learning Juxtaposed-Class-Embeddings Poly-lingual Classifier')
        classifier = ClassJuxtaEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                              final_learner=get_learner(calibrate=False),
                                                              alpha=0.5,
                                                              c_parameters=get_params(), y_parameters=get_params(), n_jobs=op.n_jobs)

    classifier.fit(data.lXtr(), data.lYtr())
    l_eval = evaluate(classifier, data.lXte(), data.lYte())

    for lang in data.langs():
        macrof1, microf1, macrok, microk = l_eval[lang]
        #macrof1, microf1 = l_eval[lang]
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
        #results.add_row(result_id, op.mode, op.optimc, dataset_name, classifier.time, lang, macrof1, microf1, macrok, microk, notes=op.note)
        notes=op.note + ('C='+str(op.set_c) if op.set_c!=1 else '') + str(classifier.best_params() if op.optimc else '')
        results.add_row(result_id, op.mode, op.learner, op.optimc, data.dataset_name, op.binary, op.languages, classifier.time, lang, macrof1, microf1, macrok, microk, notes=notes)



