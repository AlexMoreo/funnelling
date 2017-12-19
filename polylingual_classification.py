import util.disable_sklearn_warnings
from data.dataset_builder import *
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.results import PolylingualClassificationResults
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


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
parser.add_option("-f", "--force", dest="force", action='store_true',
                  help="Run even if the result was already computed", default=False)


#TODO: more baselines
#TODO: think about the neural-net extension
#TODO: make CLESA work
#TODO: redo the juxtaclass, according to "Discriminative Methods for Multi-labeled Classification" and rename properly
#TODO: calibration single-label
#TODO: experiment with varying number of categories
#TODO: arreglar la calibración y la búsqueda de parámetros
#TODO: fisher scores
#TODO: finish singlelabel-fragment
#TODO: really make_scorer(macroF1) is the best choice?
#TODO: probar feature selection?

#note: Multinomial Naive-Bayes descargado: no está calibrado, no funciona con valores negativos, la adaptación a valores
#reales es artificial

def get_learner(calibrate=False):
    if op.learner == 'svm':
        learner = SVC(kernel='linear', probability=calibrate, cache_size=1000, C=1)
    elif op.learner == 'nb':
        learner = MultinomialNB()
    elif op.learner == 'lr':
        learner = LogisticRegression()
    return learner

def get_params(z_space=False):
    if not op.optimc:
        return None

    c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
    if op.learner == 'svm':
        params = [{'C': c_range}] if not z_space else \
            [{'kernel': ['linear'], 'C': c_range},
             {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': c_range}]
    elif op.learner == 'nb':
        params = [{'alpha': [1.0, .1, .05, .01, .001, 0.0]}]
    elif op.learner == 'lr':
        params = [{'C': c_range}]
    return params

if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert op.learner in ['svm', 'lr', 'nb'], 'unexpected learner'
    assert op.mode in ['class', 'naive', 'juxta', 'lri', 'lri-half', 'dci-lin', 'dci-pmi', 'clesa', 'upper', 'monoclass', 'juxtaclass'], 'unexpected mode'

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
        classifier = ClassEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                         final_learner=get_learner(calibrate=False),
                                                         parameters=get_params(), z_parameters=get_params(z_space=True)) #optimize only for z_params
    elif op.mode == 'naive':
        print('Learning Naive Poly-lingual Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(), parameters=get_params())
    elif op.mode == 'juxta':
        print('Learning Juxtaposed Poly-lingual Classifier')
        classifier = JuxtaposedPolylingualClassifier(base_learner=get_learner(), parameters=get_params())
    elif op.mode == 'lri':
        assert op.learner != 'nb', 'nb operates only on positive matrices'
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(base_learner=get_learner(), parameters=get_params())
    elif op.mode == 'lri-half':
        assert op.learner != 'nb', 'nb operates only on positive matrices'
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(base_learner=get_learner(), parameters=get_params(), reduction=0.5)
    elif op.mode == 'dci-lin':
        assert op.learner!='nb', 'nb operates only on positive matrices'
        print('Learning Distributional Correspondence Indexing with Linear Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(base_learner=get_learner(), dcf='linear', z_parameters=get_params(z_space=True))
    elif op.mode == 'dci-pmi':
        assert op.learner != 'nb', 'nb operates only on positive matrices'
        print('Learning Distributional Correspondence Indexing with PMI Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(base_learner=get_learner(), dcf='pmi', z_parameters=get_params(z_space=True))
    elif op.mode == 'clesa':
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(base_learner=get_learner(), lW=lW, z_parameters=get_params(z_space=True))
    elif op.mode == 'upper':
        assert data.langs()==['en'], 'only English is expected in the upper bound call'
        print('Learning Upper bound as the English-only Classifier')
        classifier = NaivePolylingualClassifier(base_learner=get_learner(), parameters=get_params()) #this is just to match the multilingual dataset format (despite there are only English documents)
    elif op.mode == 'monoclass':
        assert data.langs()==['en'], 'only English is expected in the monolingual class embedding call'
        print('Learning Monolingual Class-Embedding in the English-only corpus')
        classifier = ClassEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                         final_learner=get_learner(calibrate=False),
                                                         parameters=None, z_parameters=get_params(z_space=True))
    elif op.mode == 'juxtaclass':
        print('Learning Juxtaposed-Class-Embeddings Poly-lingual Classifier')
        classifier = ClassJuxtaEmbeddingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                              final_learner=get_learner(calibrate=False),
                                                              alpha=0.5,
                                                              c_parameters=get_params(), y_parameters=get_params())

    classifier.fit(data.lXtr(), data.lYtr())
    l_eval = evaluate(classifier, data.lXte(), data.lYte())

    for lang in data.langs():
        #macrof1, microf1, macrok, microk = l_eval[lang]
        macrof1, microf1 = l_eval[lang]
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
        #results.add_row(result_id, op.mode, op.optimc, dataset_name, classifier.time, lang, macrof1, microf1, macrok, microk, notes=op.note)
        notes=op.note # + classifier.best_params
        results.add_row(result_id, op.mode, op.learner, op.optimc, data.dataset_name, classifier.time, lang, macrof1, microf1, notes=op.note)



