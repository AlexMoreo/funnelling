from sklearn.externals.joblib import Parallel, delayed
from util.metrics import macroF1,microF1,macroK,microK, hard_single_metric_statistics
from sklearn.metrics import f1_score
import numpy as np

def evaluate(polylingual_method, lX, ly, predictor=None):
    print('prediction for test')
    assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
    n_jobs = polylingual_method.n_jobs
    if predictor is None:
        predictor = polylingual_method.predict
    ly_ = predictor(lX)
    print('evaluation (n_jobs={})'.format(n_jobs))
    if n_jobs == 1:
        return {lang: evaluation_metrics(ly[lang], ly_[lang]) for lang in ly.keys()}
    else:
        langs = list(ly.keys())
        evals = Parallel(n_jobs=n_jobs)(delayed(evaluation_metrics)(ly[lang], ly_[lang]) for lang in langs)
        return {lang: evals[i] for i, lang in enumerate(langs)}

def get_binary_counters(polylingual_method, lX, ly, predictor=None):
    print('prediction for test')
    assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
    n_jobs = polylingual_method.n_jobs
    if predictor is None:
        predictor = polylingual_method.predict
    ly_ = predictor(lX)
    print('evaluation (n_jobs={})'.format(n_jobs))
    if n_jobs == 1:
        return {lang: binary_counters(ly[lang], ly_[lang]) for lang in ly.keys()}
    else:
        langs = list(ly.keys())
        evals = Parallel(n_jobs=n_jobs)(delayed(binary_counters)(ly[lang], ly_[lang]) for lang in langs)
        return {lang: evals[i] for i, lang in enumerate(langs)}

def binary_counters(y, y_):
    y = np.reshape(y, (-1))
    assert y.shape==y_.shape and len(y.shape)==1, 'error, binary vector expected'
    counters = hard_single_metric_statistics(y, y_)
    return counters.tp, counters.tn, counters.fp, counters.fn


def evaluation_metrics(y, y_):
    if len(y.shape)==len(y_.shape)==1 and len(np.unique(y))>2: #single-label
        return f1_score(y,y_,average='macro'), f1_score(y,y_,average='micro')
    else: #the metrics I implemented assume multiclass multilabel classification as binary classifiers
        return macroF1(y, y_), microF1(y, y_), macroK(y, y_), microK(y, y_)