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
from sklearn.utils import shuffle

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")
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
parser.add_option("-j", "--n_jobs", dest="n_jobs",type=int,
                  help="Number of parallel jobs (default is -1, all)", default=-1)
parser.add_option("-s", "--set_c", dest="set_c",type=float,
                  help="Set the C parameter", default=1)


# in this experiment we apply an ablation to a given training language, and check the performance on a training
# language for which only 10% of the training data is retained; the idea is to better understand the contrib/benefit

def get_learner(calibrate=False):
    return SVC(kernel='linear', probability=calibrate, cache_size=1000, C=op.set_c, random_state=1)


def get_params(z_space=False):
    if not op.optimc:
        return None
    c_range = [1e4, 1e3, 1e2, 1e1, 1]
    return [{'kernel': ['linear'], 'C': c_range}] if not z_space else [{'kernel': ['rbf'], 'C': c_range}]

def funnelling_classify_and_test(variant_name, ablated):
    print('Learning Class-Embedding Poly-lingual Classifier')
    classifier = FunnelingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                final_learner=get_learner(calibrate=False),
                                                parameters=None, z_parameters=get_params(z_space=True),
                                                n_jobs=op.n_jobs)
    classifier.fit(data.lXtr(), data.lYtr())

    # hard eval metrics
    Xte = data.lXte()[train_lang]
    yte = data.lYte()[train_lang]
    macrof1, microf1, macrok, microk = evaluate_single_lang(classifier, Xte, yte, lang=train_lang)

    print('Lang %s: macro-F1=%.3f micro-F1=%.3f macro-K=%.3f micro-K=%.3f' %
          (train_lang, macrof1, microf1, macrok, microk))

    notes = op.note + ('C=' + str(op.set_c) if op.set_c != 1 else '') + str(classifier.best_params() if op.optimc else '')
    results.add_row(result_id, variant_name, op.learner, op.optimc, data.dataset_name, ablated, tr_proportion,
                    classifier.time, train_lang, macrof1, microf1, macrok, microk, notes=notes)

def monolingual_classify_and_test():
    #comparison with monolingual classifier
    print('Learning Monolingual Classifier for tr_lang ' + train_lang)

    classifier = MonolingualClassifier(base_learner=get_learner(), parameters=get_params(), n_jobs=op.n_jobs)

    Xtr = data.lXtr()[train_lang]
    Ytr = data.lYtr()[train_lang]
    classifier.fit(Xtr, Ytr)

    Xte = data.lXte()[train_lang]
    Yte = data.lYte()[train_lang]
    Yte_ = classifier.predict(Xte)
    macrof1, microf1, macrok, microk = evaluation_metrics(Yte, Yte_)

    print('Lang %s: macro-F1=%.3f micro-F1=%.3f macro-K=%.3f micro-K=%.3f' %
          (train_lang, macrof1, microf1, macrok, microk))

    notes = op.note + ('C=' + str(op.set_c) if op.set_c != 1 else '') + str(classifier.best_params() if op.optimc else '')
    results.add_row(result_id, 'monolingual', op.learner, op.optimc, data.dataset_name, 'none', tr_proportion,
                    classifier.time, train_lang, macrof1, microf1, macrok, microk, notes=notes)


def evaluate_posterior_probs(variant_name):
    print('Learning Class-Embedding Poly-lingual Classifier')
    classifier = FunnelingPolylingualClassifier(auxiliar_learner=get_learner(calibrate=True),
                                                final_learner=get_learner(calibrate=False),
                                                parameters=None, z_parameters=get_params(z_space=True),
                                                n_jobs=op.n_jobs)

    classifier.fit(data.lXtr(), data.lYtr())

    # soft eval metrics
    for lang in data.langs():
        X = data.lXtr()[lang]
        Y = data.lYtr()[lang]

        Z = classifier.doc_projector.predict_proba({lang:X})[lang]

        macrof1, microf1, macrok, microk = soft_evaluation_metrics(Y, Z)

        print('Lang %s: [soft] macro-F1=%.3f micro-F1=%.3f macro-K=%.3f micro-K=%.3f' %
              (lang, macrof1, microf1, macrok, microk))

        notes = op.note + ('C=' + str(op.set_c) if op.set_c != 1 else '') + str(classifier.best_params() if op.optimc else '')

        results.add_row(result_id, variant_name+'trsoft', op.learner, op.optimc, data.dataset_name, 'none', 1,
                        classifier.time, lang, macrof1, microf1, macrok, microk, notes=notes+" 100%examples in train")


if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)
    assert op.learner in ['svm', 'lr', 'nb'], 'unexpected learner'
    assert not (op.set_c != 1. and op.optimc), 'Parameter C cannot be defined along with optim_c option'

    results = PolylingualClassificationResults(op.output)
    dataset_file = os.path.basename(op.dataset)
    result_id = dataset_file + 'ablation_tr1%' + ('_optimC' if op.optimc else ('_setc' + str(op.set_c) if op.set_c != 1. else ''))

    data = MultilingualDataset.load(op.dataset)
    langs = list(data.langs())

    evaluate_posterior_probs('Fun(tat)-PostPr')

    for i,train_lang in enumerate(langs):
        print('Starting with language {} ({}/{})'.format(train_lang,i+1,len(langs)))
        data = MultilingualDataset.load(op.dataset)

        # reduce the training set of the selected language to 10% of its content
        tr_proportion = 0.1
        ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids)) = data.multiling_dataset[train_lang]
        Xtr, Ytr = shuffle(Xtr, Ytr, random_state=0)
        Xtr.sort_indices()
        nD = Xtr.shape[0]
        ndocs = int(nD * tr_proportion)
        Xtr = Xtr[:ndocs]
        Ytr = Ytr[:ndocs]
        data.multiling_dataset[train_lang] = ((Xtr, Ytr, tr_ids), (Xte, Yte, te_ids))

        funnelling_classify_and_test('fun(tat)-all-langs', ablated='none')

        monolingual_classify_and_test()

        for ablation_lang in langs:
            if ablation_lang == train_lang: continue

            # ablation and classify & test
            languages = list(data.langs())
            languages.remove(ablation_lang)
            data.set_view(languages=languages)
            print('Ablation {} Tr-Language {} reduction {} shape {}:'.format(ablation_lang, train_lang, tr_proportion, Xtr.shape))
            data.show_dimensions()
            data.show_category_prevalences()

            funnelling_classify_and_test('fun(tat)-ablated', ablated=ablation_lang)

