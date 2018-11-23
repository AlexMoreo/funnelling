import os
from os.path import exists
import sys
import util.disable_sklearn_warnings
from dataset_builder import *
import numpy as np
import matplotlib.pyplot as plt
from dataset_builder import MultilingualDataset
from feature_selection.tsr_function import fisher_score_binary
from learning.learners import *
from util.evaluation import *
from optparse import OptionParser
from util.results import PolylingualClassificationResults
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset",
                  help="Path to the multilingual dataset processed and stored in .pickle format")
parser.add_option("-l", "--learner", dest="learner",
                  help="Learner method for classification", type=str, default='svm')
parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='./results.csv')
parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimices hyperparameters", default=False)
parser.add_option("-f", "--force", dest="force", action='store_true',
                  help="Run even if the result was already computed", default=False)

def get_learner(calibrate=False, defaultC=1):
    if op.learner == 'svm':
        learner = SVC(kernel='linear', probability=calibrate, cache_size=1000, C=defaultC)
    elif op.learner == 'nb':
        learner = MultinomialNB()
    elif op.learner == 'lr':
        learner = LogisticRegression(C=defaultC)
    return learner

def get_params(z_space=False):
    if not op.optimc:
        return None

    c_range = [1e4, 1e3, 1e2, 1e1, 1]
    if op.learner == 'svm':
        params = [{'kernel': ['linear'], 'C': c_range}] if not z_space else [{'kernel': ['rbf'], 'C': c_range}]
    elif op.learner == 'nb':
        params = [{'alpha': [1.0, .1, .05, .01, .001, 0.0]}]
    elif op.learner == 'lr':
        params = [{'C': c_range}]
    return params

if __name__=='__main__':
    (op, args) = parser.parse_args()

    assert exists(op.dataset), 'Unable to find file '+str(op.dataset)

    data = MultilingualDataset.load(op.dataset)
    data.show_dimensions()

    #data.set_view(categories=range(10), languages=['en','es','it'])
    folds = 1

    print('Learning 10-Fold CV Class-Embedding Poly-lingual Classifier')
    classifier = FunnellingPolylingualClassifier(first_tier_learner=get_learner(calibrate=True),
                                                 meta_learner=get_learner(calibrate=False),
                                                 first_tier_parameters=None, meta_parameters=get_params(z_space=True),
                                                 folded_projections=folds)

    print('Obtaining the z-space')
    if folds>1:
        Z, zy = classifier._get_zspace_folds(data.lXtr(), data.lYtr())
    else:
        Z, zy = classifier._get_zspace(data.lXtr(), data.lYtr())

    print('Computing the Fisher-scores')
    nC = zy.shape[1]
    if hasattr(data, 'labels'):
        labels = data.labels
    else:
        labels = ['C'+str(i) for i in range(nC)]
    fs = np.zeros((nC,nC), dtype=np.float)
    l1_l2_fs = []

    for c in range(nC):
        #the simetric values are not recomputed
        # for f in range(0,c):
        #     fs[c,f]=fs[f,c]
        # the case f==c is not computed
        for f in range(c+1,nC):
            fs[c,f] = fisher_score_binary(Z[:,f], zy[:,c])
            if np.isnan(fs[c,f]):
                fs[c, f] = 0
            l1_l2_fs.append(('{0}->{1} ({2:.3f})'.format(labels[f],labels[c],fs[c,f]), fs[c, f]))
            #print("{0:.2f}".format(fs[c,f]), end=' ')
        #print()

    print('Done')


    high_predictors = 1000
    l1_l2_fs.sort(key=lambda x:x[1], reverse=True)
    for h in range(high_predictors):
        print(h, l1_l2_fs[h][0])

    plt.imshow(fs, cmap='hot', interpolation='nearest')
    plt.show()



