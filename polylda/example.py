from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from data.languages import NLTK_LANGMAP
from data.text_preprocessor import NLTKLemmaTokenizer
from nltk.corpus import stopwords
from dataset_builder import MultilingualDataset
from learning.learners import MonolingualClassifier
from polylda import PolyLDA
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import itertools

from util.evaluation import evaluation_metrics


def _preprocess(documents, lang):
    tokens = NLTKLemmaTokenizer(lang, verbose=True)
    sw = stopwords.words(NLTK_LANGMAP[lang])
    return [' '.join([w for w in tokens(doc) if w not in sw]) for doc in documents]


data = MultilingualDataset.load('/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle')

data.set_view(languages=['en','es','it'])




lW = pickle.load(open('/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processed.wiki.raw.pickle', 'rb'))
langs = data.langs()
lXtr = data.lXtr()
lXte = data.lXte()
lYtr = data.lYtr()
lYte = data.lYte()

ndocs = 100

print('preprocessing')
for l in langs:
    print('\n\tlang {}'.format(l))
    lW[l] = lW[l][:ndocs]
    lXtr[l] = lXtr[l][:ndocs] # ya preprocesado
    lYtr[l] = lYtr[l][:ndocs]
    lXte[l] = lXte[l][:ndocs] # ya preprocesado
    lYte[l] = lYte[l][:ndocs]
    lW[l] = _preprocess(lW[l], l)

for l in list(lW.keys()):
    if l not in langs:
        print('removing language {}'.format(l))
        del lW[l]
print()

wiki_pool = list(itertools.chain.from_iterable(lW.values())) # contains all documents, since the matrix should account for all terms

counter = CountVectorizer(min_df=10)

print('fit')
counter.fit(wiki_pool)

print('transform')
wiki_matrices = [counter.transform(lW[l]) for l in langs]

print('dense')
wiki_matrices = [m.toarray() for m in wiki_matrices]


print('fit poly-lda')
# polylda = PolyLDA(n_topics=1000, n_iter=2000, languages=len(langs))
polylda = PolyLDA(n_topics=30, n_iter=100, languages=len(langs))
polylda.fit(wiki_matrices)

print('transform poly-lda')
lXtr = {l:counter.transform(lXtr[l]).toarray() for i,l in enumerate(langs)}
lXtr = {l:polylda.transform(lXtr[l], which_language=i) for i,l in enumerate(langs)}


Z = np.vstack([lXtr[lang] for lang in langs])  # Z is the language independent space
zy = np.vstack([lYtr[lang] for lang in langs])
del lXtr

print('fitting the Z-space of shape={}'.format(Z.shape))
model = MonolingualClassifier(base_learner=LinearSVC(), parameters=[{'C':[1e4, 1e3, 1e2, 1e1, 1, 1e-1]}], n_jobs=-1)
    # parameters=None, n_jobs=-1)
model.fit(Z, zy)

print('eval')

with open('resultados_polyLDA.txt', 'w') as foo:
    for i,lang in enumerate(langs):
        Xte = counter.transform(lXte[lang]).toarray()
        y_ = model.predict(polylda.transform(Xte, which_language=i))
        y  = lYte[lang]
        print('{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(lang, *evaluation_metrics(y,y_)))
        foo.write('{} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(lang, *evaluation_metrics(y, y_)))

# print('only 500 documents!!')
#
# lXtr = data.lXtr()
# lytr = data.lYtr()




sys.exit()
l1 = ['futbol jugador futbol porteria comercio dinero euros jugador porteria balon gato gato perro',
      'futbol porteria porteria balon',
      'dinero dinero dinero euros euros comercio',
      'gato perro perro perro',
      'futbol dinero gato']
l2 = ['soccer player soccer goal commerce money euros player goal ball cat cat dog',
      'soccer goal goal ball',
      'money money money euros euros commerce',
      'cat dog dog dog',
      'soccer money cat']

counter = CountVectorizer()
counter.fit(l1 + l2)
C1 = counter.transform(l1).toarray()
C2 = counter.transform(l2).toarray()

print(C1)
print(C2)


polylda = PolyLDA(n_topics=3, n_iter=100, languages=2)

polylda.fit([C1, C2])

c1_ = polylda.transform(C1, which_language=0)
c2_ = polylda.transform(C2, which_language=1)

plt.matshow(c1_)
plt.show()
plt.matshow(c2_)
plt.show()



