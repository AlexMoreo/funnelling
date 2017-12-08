import util.disable_sklearn_warnings
from data.dataset_builder import *
from optparse import OptionParser
from util.results import PolylingualClassificationResults


parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")
parser.add_option("-m", "--mode", dest="mode",
                  help="Training documents are allowed to have parallel versions of it", type=str, default=None)
parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results.csv')
parser.add_option("-n", "--note", dest="note",
                  help="A description note to be added to the result file", type=str,  default='')
parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimices the soft-marging C parameter by 5-fold cv", default=False)


# TODO: the embedded space is dense and low-dimensional, maybe other kernels would be preferable
# TODO: rcv1-v2
# TODO: use lighter auxiliar classifiers (naive Bayes?)
# TODO: more baselines
# TODO: receive the wiki matrix as an additional parameter
# TODO: try with other dcf (PMI might be better)
# note: looks like a linearsvm classifier on top of a linearsvm is merely countering back the effect of C w/o optimization
#       try with gaussian kernel, think of a neural net doing the same thing as a way to do away (or diminish) the effect of meta-parameters
if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert op.mode in ['class', 'naive', 'juxta', 'lri', 'lri-half', 'dci', 'clesa', 'upper', 'monoclass', 'juxtaclass'], 'Error: unknown mode'

    results = PolylingualClassificationResults(op.output)

    dataset_name = os.path.basename(op.dataset)
    result_id = dataset_name+'_'+op.mode+('_optimC' if op.optimc else '')
    if results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()

    params, z_params = None, None
    if op.optimc:
        c_range = [1e4, 1e3, 1e2, 1e1, 1, 1e-1]
        params = [{'C': c_range}]
        z_params = [{'kernel':['linear'], 'C': c_range}, {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': c_range}]

    if op.mode == 'class':
        [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1e4, 1e3, 1e2, 1e1, 1, 1e-1]},
         {'kernel': ['linear'], 'C': [1e4, 1e3, 1e2, 1e1, 1, 1e-1]}]
        print('Learning Class-Embedding Poly-lingual Classifier')
        classifier = ClassEmbeddingPolylingualClassifier(None, z_params) #optimize only for z_params
    elif op.mode == 'naive':
        print('Learning Naive Poly-lingual Classifier')
        classifier = NaivePolylingualClassifier(params)
    elif op.mode == 'juxta':
        print('Learning Juxtaposed Poly-lingual Classifier')
        classifier = JuxtaposedPolylingualClassifier(params)
    elif op.mode == 'juxtaclass':
        print('Learning Juxtaposed-Class-Embeddings Poly-lingual Classifier')
        classifier = ClassJuxtaEmbeddingPolylingualClassifier(params, params)
    elif op.mode == 'lri':
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(params)
    elif op.mode == 'lri-half':
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(params, reduction=0.5)
    elif op.mode == 'dci':
        print('Learning Distributional Correspondence Indexing Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(z_params)
    elif op.mode == 'clesa':
        lW = pickle.load(open(op.dataset.replace('.pickle','.wiki.pickle'), 'rb'))
        print('Learning Cross-Lingual Explicit Semantic Analysis Poly-lingual Classifier')
        classifier = CLESAPolylingualClassifier(lW, z_params)
    elif op.mode == 'upper':
        assert data.langs()==['en'], 'only English is expected in the upper bound call'
        print('Learning Upper bound as the English-only Classifier')
        classifier = NaivePolylingualClassifier(params) #this is just to match the multilingual dataset format (despite there are only English documents)
    elif op.mode == 'monoclass':
        assert data.langs()==['en'], 'only English is expected in the monolingual class embedding call'
        print('Learning Monolingual Class-Embedding in the English-only corpus')
        classifier = ClassEmbeddingPolylingualClassifier(None, z_params)

    classifier.fit(data.lXtr(), data.lYtr())
    l_eval = classifier.evaluate(data.lXte(), data.lYte())

    for lang in data.langs():
        macrof1, microf1, macrok, microk = l_eval[lang]
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
        results.add_row(result_id, op.mode, op.optimc, dataset_name, classifier.time, lang, macrof1, microf1, macrok, microk, notes=op.note)



