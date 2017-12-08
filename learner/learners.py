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

#TODO: class hierarchy inheritin from PolylingualClassifier
#TODO: abstract evaluate as a function receiving a Polylingual Classifier and an array of metrics
#TODO: Juxta+ClassEmbedding, LRI+ClassEmbedding, DCI+ClassEmbedding
#TODO: check if ClassEmbedding improves in monolingual (the upper bound)
#TODO: LRI: reweight!
#TODO: PLTC take an unprocessed version of the dataset (no stem), and do "frustatingly easy dom-adaptation" with LRI
#TODO: read about svm-nets (extreme learner machines has some pointers)
#TODO: class-embedding with fold-validation, embed only on unseen documents (the left-out fold)
#TODO: think about the neural-net extension
#TODO: fix the evaluation in LRI and juxtaposed -- takes too long


class ClassJuxtaEmbeddingPolylingualClassifier:
    """
    This classifier combines the juxtaposed space with the class embeddings before training the final classifier
    """
    def __init__(self, c_parameters=None, y_parameters=None):
        """
        :param c_parameters: parameters for the previous class-embedding projector
        :param y_parameters: parameters for the final combined learner
        """
        self.c_parameters=c_parameters
        self.y_parameters = y_parameters
        self.class_projector = NaivePolylingualClassifier(self.c_parameters)
        self.model = None

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
        self.model = MonolingualClassifier(self.y_parameters)
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
    def __init__(self, parameters=None, z_parameters=None):
        """
        :param parameters: parameters for the learner in the doc_projector
        :param z_parameters: parameters for the learner in the z-space
        """
        self.parameters=parameters
        self.z_parameters = z_parameters
        self.doc_projector = NaivePolylingualClassifier(self.parameters)

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
        self.model = MonolingualClassifier(self.z_parameters)
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
    def __init__(self, parameters=None):
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
    def __init__(self, lW, z_parameters=None, similarity='dot', post=False):
        """
        :param lW: a dictionary {lang : wikipedia doc-by-term matrix}
        :param z_parameters: the parameters of the learner to optimize for via 5-fold cv in the z-space
        """
        self.z_parameters=z_parameters
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
        return {lang:self.model.predict(lZ[lang]) for lang in lZ.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        lZ = self.doc_projector.transform(lX)
        return {lang:self.model.evaluate(lZ[lang],ly[lang]) for lang in lX.keys()}


class LRIPolylingualClassifier:
    """
    Performs Random Indexing (Bag-of-Concepts) in the juxtaposed representation, see: Moreo Fernández, A., Esuli, A.,
    & Sebastiani, F. (2016). Lightweight Random Indexing for Polylingual Text Classification. Journal of Artificial
    Intelligence Research, 57, 151-185.
    """
    def __init__(self, parameters=None, reduction=0.):
        """
        :param parameters: the parameters of the learner to optimize for via 5-fold cv
        :param reduction: the ratio of reduction of the dimensionality
        :param reweight: indicates whether to reweight the ri-matrix using tfidf
        """
        assert 0 <= reduction < 1, 'reduction ratio should be in range [0,1)'
        self.model = MonolingualClassifier(parameters)
        self.reduction = reduction

    def fit(self, lX, ly, n_jobs=-1):
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
        self.model.fit(Xtr, Ytr, n_jobs)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be juxtaposed
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


class JuxtaposedPolylingualClassifier:

    def __init__(self, parameters=None):
        self.model = MonolingualClassifier(parameters)

    def fit(self, lX, ly, n_jobs=-1):
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
        self.model.fit(Xtr, Ytr, n_jobs)
        self.time = time.time() - tinit
        return self

    def predict(self, lX):
        """
        :param lX: a dictionary {language_label: X csr-matrix}; the feature space is assumed to be juxtaposed
        :return: a dictionary of predictions
        """
        assert self.model is not None, 'predict called before fit'
        return {lang:self.model.predict(lX[lang]) for lang in lX.keys()}

    def evaluate(self, lX, ly):
        print('evaluation')
        assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
        return {lang:self.model.evaluate(lX[lang],ly[lang]) for lang in lX.keys()}


class MonolingualClassifier:

    def __init__(self, parameters=None):
        self.learner = SVC(kernel='linear')
        self.parameters = parameters
        self.model = None

    def fit(self, X, y, n_jobs=-1):
        tinit = time.time()
        _sort_if_sparse(X)

        # multiclass?
        if len(y.shape) == 2:
            if self.parameters is not None:
                self.parameters = [{'estimator__' + key: params[key] for key in params.keys()}
                                   for params in self.parameters]
            self.model = OneVsRestClassifier(self.learner, n_jobs=n_jobs)
        else:
            #not debugged
            self.model = self.learner

        # parameter optimization?
        if self.parameters:
            print('debug: optimizing parameters:', self.parameters)
            self.model = GridSearchCV(self.model, param_grid=self.parameters, refit=True, cv=5, n_jobs=n_jobs,
                                      scoring=make_scorer(macroF1), error_score=0)

        self.model.fit(X,y)
        if isinstance(self.model, GridSearchCV):
            print('best parameters: ', self.model.best_params_)
        self.time=time.time()-tinit
        return self

    def decision_function(self, X):
        assert self.model is not None, 'predict called before fit'
        _sort_if_sparse(X)
        return self.model.decision_function(X)

    def predict(self, X):
        assert self.model is not None, 'predict called before fit'
        _sort_if_sparse(X)
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_ = self.predict(X)
        return _evaluate(y,y_)

def _evaluate(y,y_):
    return macroF1(y, y_), microF1(y, y_), macroK(y, y_), microK(y, y_)

def _sort_if_sparse(X):
    if issparse(X) and not X.has_sorted_indices:
        X.sort_indices()