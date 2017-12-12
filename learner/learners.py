import numpy as np
import scipy
import time
from sklearn.feature_extraction.text import TfidfTransformer
from model.clesa import CLESA
from util.metrics import macroF1, microF1, macroK, microK
from scipy.sparse import issparse, csr_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from model.riboc import RandomIndexingBoC
from model.dci import DistributionalCorrespondenceIndexing
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals.joblib import Parallel, delayed


class ClassJuxtaEmbeddingPolylingualClassifier:
    """
    This classifier combines the juxtaposed space with the class embeddings before training the final classifier
    """
    def __init__(self, c_parameters=None, y_parameters=None, n_jobs=-1):
        """
        :param c_parameters: parameters for the previous class-embedding projector
        :param y_parameters: parameters for the final combined learner
        """
        self.c_parameters=c_parameters
        self.y_parameters = y_parameters
        self.class_projector = NaivePolylingualClassifier(self.c_parameters)
        self.model = None
        self.n_jobs = n_jobs

    def fit(self, lX, ly, n_jobs=-1):
        tinit = time.time()
        print('fitting the projectors...')
        self.class_projector.fit(lX,ly,n_jobs)

        print('projecting the documents')
        langs = list(lX.keys())
        lZ = self.class_projector.decision_function(lX)

        print('joining X and Z spaces')
        Z = np.vstack([lZ[lang] for lang in langs])  # Z is the language independent space
        Z = Z/np.linalg.norm(Z,axis=1, keepdims=True)
        X = scipy.sparse.vstack([lX[lang] for lang in langs])
        XZ = self._XZhstack(X, Z)
        Y = np.vstack([ly[lang] for lang in langs])

        print('fitting the XZ-space of shape={}'.format(XZ.shape))
        self.model = MonolingualClassifier(self.y_parameters, n_jobs=self.n_jobs)
        self.model.fit(XZ,Y)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        lZ = self.class_projector.decision_function(lX)
        return {lang:self.model.predict(self._XZhstack(lX[lang], lZ[lang])) for lang in lZ.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        ly_ = self.predict(lX)
        return {lang:_evaluate(ly[lang],ly_[lang]) for lang in lX.keys()}

    def _XZhstack(self, X, Z):
        assert isinstance(X, csr_matrix), 'expected csr_matrix in X-space'
        assert isinstance(Z, np.ndarray), 'expected np.ndarray in Z-space'
        return csr_matrix(scipy.sparse.hstack([X, csr_matrix(Z)]))


class ClassEmbeddingPolylingualClassifier:
    """
    This classifier projects each document d into a language-independent feature space where each dimension fi is the
    decision score phi_l(d,ci) of an auxiliar classifier phi_l trained on category ci for documents in language l;
    then trains one single classifier for all documents in this space, irrespective of their originary language
    """
    def __init__(self, auxiliar_learner=SVC(kernel='linear'), parameters=None, z_parameters=None, folded_projections=1, n_jobs=-1):
        """
        :param parameters: parameters for the learner in the doc_projector
        :param z_parameters: parameters for the learner in the z-space
        :param folded_predictions: if 1 then the model trains the auxiliar classifiers with all training data and
        projects the data before training the final classifier; if greater than one, the training set is split in as
        many folds as indicated, and the projected space is composed by concatenating each fold prediction based on
        models trained on the remaining folds. This should increase the generality of the space to unseen data.
        """
        assert folded_projections>0, "positive number of folds expected"
        self.auxiliar_learner = auxiliar_learner
        self.parameters=parameters
        self.z_parameters = z_parameters
        self.doc_projector = NaivePolylingualClassifier(self.auxiliar_learner, self.parameters, n_jobs=n_jobs)
        self.folded_projections = folded_projections
        self.n_jobs = n_jobs

    def _get_zspace(self, lXtr, lYtr, lXproj=None, lYproj=None):
        """
        :param lXtr: {lang:matrix} to train
        :param lYtr: {lang:labels} to train
        :param lXproj: {lang:matrix} to project (if None, then projects the lXtr)
        :param lYproj: {lang:labels} to stack in the same order (if None, then lYtr will be stacked)
        :param n_jobs: number of jobs
        :return: the projection of lXproj documents into the Z-space defined by the confidence scores of language-specific
        models trained on lXtr, and the lYproj labels stacked consistently
        """
        if lXproj is None and lYproj is None:
            lXproj, lYproj = lXtr, lYtr

        print('fitting the projectors...')
        self.doc_projector.fit(lXtr, lYtr)

        print('projecting the documents')
        langs = list(lXtr.keys())
        lZ = self.doc_projector.decision_function(lXproj)
        Z = np.vstack([lZ[lang] for lang in langs])  # Z is the language independent space
        zy = np.vstack([lYproj[lang] for lang in langs])

        return Z, zy

    def fit(self, lX, ly):
        tinit = time.time()

        if self.folded_projections == 1:
            Z, zy = self._get_zspace(lX, ly)
        else:
            print('stratified split of {} folds'.format(self.folded_projections))
            skf = KFold(n_splits=self.folded_projections, shuffle=True)

            Z, zy = [], []
            lfold = {lang:list(skf.split(lX[lang],ly[lang])) for lang in lX.keys()}
            for fold in range(self.folded_projections):
                print('fitting the projectors ({}/{})...'.format(fold+1,self.folded_projections))
                lfoldXtr, lfoldYtr = {}, {}
                lfoldXte, lfoldYte = {}, {}
                for lang in lX.keys():
                    train, test = lfold[lang][fold]
                    lfoldXtr[lang] = lX[lang][train]
                    lfoldYtr[lang] = ly[lang][train]
                    lfoldXte[lang] = lX[lang][test]
                    lfoldYte[lang] = ly[lang][test]
                Zfold, zYfold = self._get_zspace(lfoldXtr, lfoldYtr, lfoldXte, lfoldYte)
                Z.append(Zfold)
                zy.append(zYfold)
            # compose the Z-space as the union of all folded predictions
            Z = np.vstack(Z)
            zy = np.vstack(zy)
            # refit the document projector with all examples to have a more reliable projector for test data
            self.doc_projector.fit(lX, ly)

        print('fitting the Z-space of shape={}'.format(Z.shape))
        self.model = MonolingualClassifier(parameters=self.z_parameters, n_jobs=self.n_jobs)
        self.model.fit(Z,zy)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        lZ = self.doc_projector.decision_function(lX)
        return _joblib_transform_multiling(self.model.predict, lZ, n_jobs=self.n_jobs)

    def evaluate(self, lX, ly):
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.doc_projector.decision_function(lX)
        return _joblib_eval_multiling(self.model.evaluate, lZ, ly, self.n_jobs)


class NaivePolylingualClassifier:
    """
    Is a mere set of independet MonolingualClassifiers
    """
    def __init__(self, base_learner=SVC(kernel='linear'), parameters=None, n_jobs=-1):
        self.base_learner = base_learner
        self.parameters = parameters
        self.model = None
        self.n_jobs = n_jobs

    def fit(self, lX, ly):
        """
        trains the independent monolingual classifiers
        :param lX: a dictionary {language_label: X csr-matrix}
        :param ly: a dictionary {language_label: y np.array}
        :return: self
        """
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent language mappings in fit'
        if self.n_jobs==1:
            self.model = {lang:MonolingualClassifier(self.base_learner, self.parameters).fit(lX[lang],ly[lang]) for lang in lX.keys()}
        else:
            langs = list(lX.keys())
            models = Parallel(n_jobs=self.n_jobs)(delayed(MonolingualClassifier(self.base_learner, self.parameters).fit)(lX[lang],ly[lang]) for lang in langs)
            self.model = {lang: models[i] for i, lang in enumerate(langs)}
        self.time = time.time() - tinit
        return self

    def decision_function(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of classification scores for each class
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in decision function'
        if self.n_jobs==1:
            return {lang: self.model[lang].decision_function(lX[lang]) for lang in lX.keys()}
        else:
            langs=list(lX.keys())
            scores = Parallel(n_jobs=self.n_jobs)(delayed(self.model[lang].decision_function)(lX[lang]) for lang in langs)
            return {lang:scores[i] for i,lang in enumerate(langs)}

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in predict'
        if self.n_jobs == 1:
            return {lang:self.model[lang].predict(lX[lang]) for lang in lX.keys()}
        else:
            langs = list(lX.keys())
            scores = Parallel(n_jobs=self.n_jobs)(delayed(self.model[lang].predict)(lX[lang]) for lang in langs)
            return {lang: scores[i] for i, lang in enumerate(langs)}

    def evaluate(self, lX, ly):
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in evaluate'
        print('evaluation-parallel')
        if self.n_jobs==1:
            return {lang:self.model[lang].evaluate(lX[lang],ly[lang]) for lang in lX.keys()}
        langs = list(lX.keys())
        evals = Parallel(n_jobs=self.n_jobs)(delayed(self.model[lang].evaluate)(lX[lang],ly[lang]) for lang in langs)
        return {lang:evals[i] for i,lang in enumerate(langs)}


class CLESAPolylingualClassifier:
    """
    A polylingual classifier based on the cross-lingual ESA method
    """
    def __init__(self, lW, z_parameters=None, similarity='dot', post=False, n_jobs=-1):
        """
        :param lW: a dictionary {lang : wikipedia doc-by-term matrix}
        :param z_parameters: the parameters of the learner to optimize for via 5-fold cv in the z-space
        """
        self.z_parameters=z_parameters
        self.doc_projector = CLESA(similarity=similarity, post=post).fit(lW)
        self.n_jobs = n_jobs

    def fit(self, lX, ly):
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in fit'

        print('projecting the documents')
        langs = list(lX.keys())
        lZ = self.doc_projector.transform(lX)
        Z = np.vstack([lZ[lang] for lang in langs]) # Z is the language independent space
        zy = np.vstack([ly[lang] for lang in langs])

        print('fitting the Z-space of shape={}'.format(Z.shape))
        self.model = MonolingualClassifier(self.z_parameters, n_jobs=self.n_jobs)
        self.model.fit(Z, zy)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        lZ = self.doc_projector.transform(lX)
        return _joblib_transform_multiling(self.model.predict, lZ, n_jobs=self.n_jobs)

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.doc_projector.transform(lX)
        return _joblib_eval_multiling(self.model.evaluate, lZ, ly, n_jobs=self.n_jobs)


class DCIPolylingualClassifier:
    """
    An instantiation of DCI in polylingual documents using categories as pivots
    """
    def __init__(self, z_parameters=None):
        self.z_parameters=z_parameters
        self.doc_projector = DistributionalCorrespondenceIndexing(dcf='linear', post='normal')

    def fit(self, lX, ly, n_jobs=-1):
        tinit = time.time()
        print('fitting the projectors...')
        self.doc_projector.fit(lX,ly)

        print('projecting the documents')
        langs = list(lX.keys())
        lZ = self.doc_projector.transform(lX)
        Z = np.vstack([lZ[lang] for lang in langs]) # Z is the language independent space
        zy = np.vstack([ly[lang] for lang in langs])

        print('fitting the Z-space of shape={}'.format(Z.shape))
        self.model = MonolingualClassifier(self.z_parameters)
        self.model.fit(Z, zy, n_jobs=n_jobs)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        lZ = self.doc_projector.transform(lX)
        return _joblib_transform_multiling(self.model.predict, lZ, n_jobs=self.n_jobs)

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.doc_projector.transform(lX)
        return _joblib_eval_multiling(self.model.evaluate, lZ, ly, n_jobs=self.n_jobs)


class LRIPolylingualClassifier:
    """
    Performs Random Indexing (Bag-of-Concepts) in the juxtaposed representation, see: Moreo Fernández, A., Esuli, A.,
    & Sebastiani, F. (2016). Lightweight Random Indexing for Polylingual Text Classification. Journal of Artificial
    Intelligence Research, 57, 151-185.
    """
    def __init__(self, parameters=None, reduction=0., n_jobs=-1):
        """
        :param parameters: the parameters of the learner to optimize for via 5-fold cv
        :param reduction: the ratio of reduction of the dimensionality
        :param reweight: indicates whether to reweight the ri-matrix using tfidf
        """
        assert 0 <= reduction < 1, 'reduction ratio should be in range [0,1)'
        self.model = MonolingualClassifier(parameters=parameters, n_jobs=n_jobs)
        self.reduction = reduction
        self.n_jobs = n_jobs

    def fit(self, lX, ly):
        """
        trains one classifiers in the juxtaposed random indexed feature space
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be juxtaposed
        :param ly: a dictionary {language_label: y np.array}
        :return: self
        """
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent language mappings in fit'
        assert len(np.unique([X.shape[1] for X in lX.values()]))==1, \
            'feature-spaces in the juxtaposed representation should be equal'
        langs = list(lX.keys())
        Xtr = scipy.sparse.vstack([lX[lang] for lang in langs])
        Ytr = np.vstack([ly[lang] for lang in langs])
        nF = Xtr.shape[1]
        print('random projection with LRI')
        dimensions = int(nF * (1.-self.reduction))
        self.BoC = RandomIndexingBoC(latent_dimensions=dimensions, non_zeros=2) # extremely sparse
        Xtr = self.BoC.fit_transform(Xtr)
        print('model fit')
        self.model.fit(Xtr, Ytr)
        self.time = time.time() - tinit
        return self

    def transform(self, lX):
        return _joblib_transform_multiling(self.BoC.transform, lX, n_jobs=self.n_jobs)

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be juxtaposed
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        lZ = self.transform(lX)
        return _joblib_transform_multiling(self.model.predict, lZ, n_jobs=self.n_jobs)

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.transform(lX)
        return _joblib_eval_multiling(self.model.evaluate, lZ, ly, n_jobs=self.n_jobs)


class JuxtaposedPolylingualClassifier:

    def __init__(self, parameters=None, n_jobs=-1):
        self.model = MonolingualClassifier(parameters, n_jobs=n_jobs)
        self.n_jobs = n_jobs

    def fit(self, lX, ly):
        """
        trains one classifiers in the juxtaposed feature space
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be juxtaposed
        :param ly: a dictionary {language_label: y np.array}
        :return: self
        """
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent language mappings in fit'
        assert len(np.unique([X.shape[1] for X in lX.values()]))==1, \
            'feature-spaces in the juxtaposed representation should be equal'
        langs = list(lX.keys())
        Xtr = scipy.sparse.vstack([lX[lang] for lang in langs])
        Ytr = np.vstack([ly[lang] for lang in langs])
        self.model.fit(Xtr, Ytr)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be juxtaposed
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        return _joblib_transform_multiling(self.model.predict, lX, n_jobs=self.n_jobs)

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        return _joblib_eval_multiling(self.model.evaluate, lX, ly, n_jobs=self.n_jobs)


class MonolingualClassifier:

    def __init__(self, base_learner=SVC(kernel='linear'), parameters=None, n_jobs=-1):
        self.learner = base_learner
        self.parameters = parameters
        self.model = None
        self.n_jobs = n_jobs

    def fit(self, X, y):
        tinit = time.time()
        _sort_if_sparse(X)

        # multiclass?
        if len(y.shape) == 2:
            if self.parameters is not None:
                self.parameters = [{'estimator__' + key: params[key] for key in params.keys()}
                                   for params in self.parameters]
            self.model = OneVsRestClassifier(self.learner, n_jobs=self.n_jobs)
        else:
            #not debugged
            self.model = self.learner

        # parameter optimization?
        if self.parameters:
            print('debug: optimizing parameters:', self.parameters)
            self.model = GridSearchCV(self.model, param_grid=self.parameters, refit=True, cv=5, n_jobs=self.n_jobs,
                                      scoring=make_scorer(macroF1), error_score=0)

        print('fitting:',self.model)
        self.model.fit(X,y)
        if isinstance(self.model, GridSearchCV):
            print('best parameters: ', self.model.best_params_)
        self.time=time.time()-tinit
        return self

    def decision_function(self, X):
        assert self.model is not None, 'predict called before fit'
        _sort_if_sparse(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return self.model.decision_function(X)

    def predict(self, X):
        assert self.model is not None, 'predict called before fit'
        _sort_if_sparse(X)
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_ = self.predict(X)
        return _evaluate(y,y_)


def _evaluate(y,y_):
    return macroF1(y, y_), microF1(y, y_) #, macroK(y, y_), microK(y, y_)


def _sort_if_sparse(X):
    if issparse(X) and not X.has_sorted_indices:
        X.sort_indices()

def _joblib_eval_multiling(monolingual_eval, lX, ly, n_jobs=-1):
    print('evaluation (n_jobs={})'.format(n_jobs))
    if n_jobs == 1:
        return {lang: monolingual_eval(lX[lang], ly[lang]) for lang in lX.keys()}
    else:
        langs = list(lX.keys())
        evals = Parallel(n_jobs=n_jobs)(delayed(monolingual_eval)(lX[lang], ly[lang]) for lang in langs)
        return {lang: evals[i] for i, lang in enumerate(langs)}

def _joblib_transform_multiling(transformer, lX, n_jobs=-1):
    if n_jobs == 1:
        return {lang:transformer(lX[lang]) for lang in lX.keys()}
    else:
        langs = list(lX.keys())
        transformations = Parallel(n_jobs=n_jobs)(delayed(transformer)(lX[lang]) for lang in langs)
        return {lang: transformations[i] for i, lang in enumerate(langs)}


def calibrated_multilabel_svm(params=None, calibration='sigmoid', cv=5, n_jobs=-1):
    assert calibration in ['sigmoid', 'isotonic'], 'calibration should either be "sigmoid" or "isotonic"'

    multilabel_svm = OneVsRestClassifier(SVC, n_jobs=n_jobs)
    calibrated_svm = CalibratedClassifierCV(multilabel_svm, method=calibration, cv=cv)
    if params:
        params = [{'estimator__' + key: params[key] for key in params.keys()} for params in params]
        calibrated_svm = GridSearchCV(calibrated_svm, param_grid=params, refit=True, cv=cv, n_jobs=n_jobs,
                     scoring=make_scorer(macroF1), error_score=0)
    return calibrated_svm