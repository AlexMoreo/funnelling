from sklearn.svm import SVC

import util.disable_sklearn_warnings
import os,sys
from dataset_builder import MultilingualDataset
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.results import PolylingualClassificationResults
from util.file import exists

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")
parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results.csv')
parser.add_option("-w", "--we-path", dest="we_path",
                  help="Path to the polylingual word embeddings (required only if --mode polyembeddings)")
parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=-1)


if __name__=='__main__':

    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    results = PolylingualClassificationResults(op.output)

    dataset_file = os.path.basename(op.dataset)

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()

    lang_order = sorted(data.langs())
    for nlangs in range(1,len(lang_order)+1):
        print('Training languages {}'.format(lang_order[:nlangs]))

        result_id = dataset_file+'_'+str(nlangs)
        if results.already_calculated(result_id):
            print('Experiment <' + result_id + '> already computed. Exit.')
            continue

        lXtr,lytr  = data.training()
        lXte, lyte = data.test()

        if op.set_c != -1:
            meta_parameters = None
        else:
            meta_parameters = [{'C': [1e3, 1e2, 1e1, 1, 1e-1]}]
        print('Learning Fun(Embedding) Poly-lingual Classifier')
        classifier = FunnellingEmbedding(first_tier_learner= SVC(kernel='linear', probability=True, cache_size=1000, random_state=1),
                                         embed_learner = SVC(kernel='linear', probability=True, cache_size=1000, C=10, random_state=1),
                                         meta_learner = SVC(kernel='rbf', probability=False, cache_size=1000, C=1000, random_state=1),
                                         wordembeddings_path = op.we_path,
                                         training_languages = lang_order[:nlangs],
                                         first_tier_parameters=None,
                                         embed_parameters = None,
                                         meta_parameters = meta_parameters,
                                         n_jobs=-1)

        classifier.fit(lXtr, lytr)
        l_eval = evaluate_method(classifier, lXte, lyte)

        metrics  = []
        for lang in lXte.keys():
            macrof1, microf1, macrok, microk = l_eval[lang]
            metrics.append([macrof1, microf1, macrok, microk])
            print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))
            results.add_row(result_id, 'L{}'.format(nlangs), 'svm', False, data.dataset_name, -1, -1, classifier.time, lang, macrof1, microf1, macrok, microk)

        print('Averages: MF1, mF1, MK, mK', np.mean(np.array(metrics), axis=0))
