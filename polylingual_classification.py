import util.disable_sklearn_warnings
from data.jrc_dataset_builder import *
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
# TODO: baselines: CL-ESA, others?
# TODO: possible improvement: cross-validation in C inside the auxiliar projectors in ClassEmbeddingPolylingualClassifier (the problem is that it is already very slow)
# TODO: DCI embedding, but instead of dcf(f,pi) with dcf(f,ci), i.e., pivots are the categories! then, the doc embedding is the weighted sum of feat embeddings
# TODO: upper bound: same documents but only in English, and MonolingualClassifier
if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert op.mode in ['class', 'naive', 'yuxta', 'lri', 'lri-half', 'dci'], 'Error: unknown mode'

    results = PolylingualClassificationResults(op.output)

    dataset_name = os.path.basename(op.dataset)
    result_id = dataset_name+'_'+op.mode
    if results.already_calculated(result_id):
        print('Experiment <'+result_id+'> already computed. Exit.')
        sys.exit()

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()

    params = {'C': [1e2, 1e1, 1, 1e-1]} if op.optimc else None
    if op.mode == 'class':
        print('Learning Class-Embedding Poly-lingual Classifier')
        classifier = ClassEmbeddingPolylingualClassifier(params)
    elif op.mode == 'naive':
        print('Learning Naive Poly-lingual Classifier')
        classifier = NaivePolylingualClassifier(params)
    elif op.mode == 'yuxta':
        print('Learning Yuxtaposed Poly-lingual Classifier')
        classifier = YuxtaposedPolylingualClassifier(params)
    elif op.mode == 'lri':
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(params)
    elif op.mode == 'lri-half':
        print('Learning Lightweight Random Indexing Poly-lingual Classifier')
        classifier = LRIPolylingualClassifier(params, reduction=0.5)
    elif op.mode == 'dci':
        print('Learning Distributional Correspondence Indexing Poly-lingual Classifier')
        classifier = DCIPolylingualClassifier(params)

    classifier.fit(data.lXtr(), data.lYtr())
    l_eval = classifier.evaluate(data.lXte(), data.lYte())

    for lang in data.langs():
        macrof1, microf1, macrok, microk = l_eval[lang]
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
        results.add_row(result_id, op.mode, dataset_name, classifier.time, lang, macrof1, microf1, macrok, microk, notes=op.note)



