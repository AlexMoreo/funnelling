import numpy as np
import scipy
import time

from model.clesa import CLESA
from util.metrics import macroF1, microF1, macroK, microK
from scipy.sparse import issparse
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from model.riboc import RandomIndexingBoC
from model.dci import DistributionalCorrespondenceIndexing

class ClassEmbeddingPolylingualClassifier:
    """
    This classifier projects each document d into a language-independent feature space where each dimension fi is the
    decision score phi_l(d,ci) of an auxiliar classifier phi_l trained on category ci for documents in language l;
    then trains one single classifier for all documents in this space, irrespective of their originary language
    """
    def __init__(self, parameters={'C': [1e2, 1e1, 1, 1e-1]}):
        self.parameters=parameters
        self.doc_projector = NaivePolylingualClassifier(parameters=parameters)

    def fit(self, lX, ly, n_jobs=-1):
        tinit = time.time()
        print('fitting the projectors...')
        self.doc_projector.fit(lX,ly,n_jobs)

        print('projecting the documents')
        langs = list(lX.keys())
        lZ = self.doc_projector.decision_function(lX)
        Z = np.vstack([lZ[lang] for lang in langs]) # Z is the language independent space
        zy = np.vstack([ly[lang] for lang in langs])

        print('fitting the Z-space of shape={}'.format(Z.shape))
        self.model = MonolingualClassifier(self.parameters)
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
        return {lang:self.model.predict(lZ[lang]) for lang in lZ.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.doc_projector.decision_function(lX)
        return {lang:self.model.evaluate(lZ[lang],ly[lang]) for lang in lX.keys()}


class NaivePolylingualClassifier:
    """
    Is a mere set of independet MonolingualClassifiers
    """
    def __init__(self, parameters={'C': [1e2, 1e1, 1, 1e-1]}):
        self.parameters = parameters
        self.model = None

    def fit(self, lX, ly, n_jobs=-1):
        """
        trains the independent monolingual classifiers
        :param lX: a dictionary {language_label: X csr-matrix}
        :param ly: a dictionary {language_label: y np.array}
        :return: self
        """
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent language mappings in fit'
        self.model = {lang:MonolingualClassifier(self.parameters).fit(lX[lang],ly[lang],n_jobs) for lang in lX.keys()}
        self.time = time.time() - tinit
        return self

    def decision_function(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of classification scores for each class
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in decision function'
        return {lang: self.model[lang].decision_function(lX[lang]) for lang in lX.keys()}

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in predict'
        return {lang:self.model[lang].predict(lX[lang]) for lang in lX.keys()}

    def evaluate(self, lX, ly):
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        assert set(lX.keys()).issubset(set(self.model.keys())), 'unknown languages requested in evaluate'
        return {lang:self.model[lang].evaluate(lX[lang],ly[lang]) for lang in lX.keys()}


class CLESAPolylingualClassifier:
    """
    A polylingual classifier based on the cross-lingual ESA method
    """
    def __init__(self, lW, parameters={'C': [1e2, 1e1, 1, 1e-1]}, similarity='dot', post=False):
        """
        :param lW: a dictionary {lang : wikipedia doc-by-term matrix}
        :param parameters: the parameters of the learner to optimize for via 5-fold cv
        """
        self.parameters=parameters
        self.doc_projector = CLESA(similarity=similarity, post=post).fit(lW)

    def fit(self, lX, ly, n_jobs=-1):
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in fit'

        print('projecting the documents')
        langs = list(lX.keys())
        lZ = self.doc_projector.transform(lX)
        Z = np.vstack([lZ[lang] for lang in langs]) # Z is the language independent space
        zy = np.vstack([ly[lang] for lang in langs])

        print('fitting the Z-space of shape={}'.format(Z.shape))
        self.model = MonolingualClassifier(self.parameters)
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
        return {lang:self.model.predict(lZ[lang]) for lang in lZ.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.doc_projector.transform(lX)
        return {lang:self.model.evaluate(lZ[lang],ly[lang]) for lang in lX.keys()}


class DCIPolylingualClassifier:
    """
    An instantiation of DCI in polylingual documents using categories as pivots
    """
    def __init__(self, parameters={'C': [1e2, 1e1, 1, 1e-1]}):
        self.parameters=parameters
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
        self.model = MonolingualClassifier(self.parameters)
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
        return {lang:self.model.predict(lZ[lang]) for lang in lZ.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.doc_projector.transform(lX)
        return {lang:self.model.evaluate(lZ[lang],ly[lang]) for lang in lX.keys()}


class LRIPolylingualClassifier:

    """
    Performs Random Indexing (Bag-of-Concepts) in the yuxtaposed representation, see: Moreo Fernández, A., Esuli, A.,
    & Sebastiani, F. (2016). Lightweight Random Indexing for Polylingual Text Classification. Journal of Artificial
    Intelligence Research, 57, 151-185.
    """
    def __init__(self, parameters={'C': [1e2, 1e1, 1, 1e-1]}, reduction=0.):
        """
        :param parameters: the parameters of the learner to optimize for via 5-fold cv
        :param reduction: the ratio of reduction of the dimensionality
        """
        assert 0 <= reduction < 1, 'reduction ratio should be in range [0,1)'
        self.model = MonolingualClassifier(parameters)
        self.reduction = reduction

    def fit(self, lX, ly, n_jobs=-1):
        """
        trains one classifiers in the yuxtaposed random indexed feature space
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be yuxtaposed
        :param ly: a dictionary {language_label: y np.array}
        :return: self
        """
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent language mappings in fit'
        assert len(np.unique([X.shape[1] for X in lX.values()]))==1, \
            'feature-spaces in the yuxtaposed representation should be equal'
        langs = list(lX.keys())
        Xtr = scipy.sparse.vstack([lX[lang] for lang in langs])
        Ytr = np.vstack([ly[lang] for lang in langs])
        nF = Xtr.shape[1]
        print('random projection with LRI')
        dimensions = int(nF * (1.-self.reduction))
        self.BoC = RandomIndexingBoC(latent_dimensions=dimensions, non_zeros=2) # extremely sparse
        Xtr = self.BoC.fit_transform(Xtr)
        print('model fit')
        self.model.fit(Xtr, Ytr, n_jobs)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be yuxtaposed
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        model = self.model
        boc = self.BoC
        return {lang:model.predict(boc.transform(lX[lang])) for lang in lX.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        model = self.model
        boc = self.BoC
        return {lang:model.evaluate(boc.transform(lX[lang]),ly[lang]) for lang in lX.keys()}


class YuxtaposedPolylingualClassifier:

    def __init__(self, parameters={'C': [1e2, 1e1, 1, 1e-1]}):
        self.model = MonolingualClassifier(parameters)

    def fit(self, lX, ly, n_jobs=-1):
        """
        trains one classifiers in the yuxtaposed feature space
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be yuxtaposed
        :param ly: a dictionary {language_label: y np.array}
        :return: self
        """
        tinit = time.time()
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent language mappings in fit'
        assert len(np.unique([X.shape[1] for X in lX.values()]))==1, \
            'feature-spaces in the yuxtaposed representation should be equal'
        langs = list(lX.keys())
        Xtr = scipy.sparse.vstack([lX[lang] for lang in langs])
        Ytr = np.vstack([ly[lang] for lang in langs])
        self.model.fit(Xtr, Ytr, n_jobs)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be yuxtaposed
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        return {lang:self.model.predict(lX[lang]) for lang in lX.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        return {lang:self.model.evaluate(lX[lang],ly[lang]) for lang in lX.keys()}


class MonolingualClassifier:
    def __init__(self, parameters={'C': [1e2, 1e1, 1, 1e-1]}):
        self.learner = LinearSVC()
        self.parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()} if parameters else None
        self.model = None

    def fit(self, X, y, n_jobs=-1):
        tinit = time.time()
        if issparse(X) and not X.has_sorted_indices:
            X.sort_indices()

        # multiclass?
        if len(y.shape) == 2:
            self.model = OneVsRestClassifier(self.learner, n_jobs=n_jobs)
        else:
            self.model = self.learner

        # parameter optimization?
        if self.parameters:
            print('debug: optimizing parameters...')
            self.model = GridSearchCV(self.model, param_grid=self.parameters, refit=True, cv=5, n_jobs=n_jobs, scoring=make_scorer(macroF1), error_score=0)

        self.model.fit(X,y)
        self.time=time.time()-tinit
        return self

    def decision_function(self, X):
        assert self.model is not None, 'predict called before fit'
        if issparse(X) and not X.has_sorted_indices:
            X.sort_indices()
        return self.model.decision_function(X)

    def predict(self, X):
        assert self.model is not None, 'predict called before fit'
        if issparse(X) and not X.has_sorted_indices:
            X.sort_indices()
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_ = self.predict(X)
        return macroF1(y, y_), microF1(y, y_), macroK(y, y_), microK(y, y_)


