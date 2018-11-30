import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from torch.autograd import Variable
from data.embeddings import WordEmbeddings
import torch.nn.functional as F
import numpy as np

from data.text_preprocessor import preprocess_documents, NLTKLemmaTokenizer
from dataset_builder import MultilingualDataset
from util.evaluation import evaluation_metrics
import matplotlib.pyplot as plt
import sys

USE_CUDA = True
INTERACTIVE = True

class LSTMTextClassificationNet(nn.Module):

    def __init__(self, vocabulary_size, embedding_size, nclasses, hidden_size, nlayers, fflayer_sizes, dropout,
                 pretrained_embeddings=None, add_pad_zero_embedding=True, train_embeddings=True, max_length=-1):
        super().__init__()

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 1
        self.max_length = max_length

        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocabulary_size, embedding_size)
            if add_pad_zero_embedding:
                self.PAD = pretrained_embeddings.shape[0]
                pretrained_embeddings=np.vstack([pretrained_embeddings,np.zeros((1,embedding_size))]) # PAD null embedding
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings.astype(np.float32)), freeze=not train_embeddings)
        else:
            self.PAD = vocabulary_size
            self.embedding = nn.Embedding(vocabulary_size + 1, embedding_size, padding_idx=self.PAD)

        self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout)

        prev_size = hidden_size
        self.ff = nn.ModuleList()
        for lin_size in fflayer_sizes:
            self.ff.append(nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.hidden2classes = nn.Linear(prev_size, nclasses)

    def init_hidden(self, batch_size):
        var_hidden = Variable(torch.zeros(self.nlayers * self.num_directions, batch_size, self.hidden_size))
        var_cell   = Variable(torch.zeros(self.nlayers * self.num_directions, batch_size, self.hidden_size))
        if USE_CUDA:
            return (var_hidden.cuda(), var_cell.cuda())
        else:
            return (var_hidden, var_cell)


    # x is already padded
    def forward(self, x, aslogits=True):
        # Xi = rnn_utils.pack_sequence([torch.Tensor(xi) for xi in Xi])
        batch_size=x.shape[0]
        x = x.transpose(0, 1)
        embedded = self.embedding(x)

        # Xi = rnn_utils.pad_packed_sequence(Xi)
        rnn_output, (h_n,c_n) = self.lstm(embedded, self.init_hidden(batch_size))

        assert self.num_directions==1 and self.nlayers==1, \
            'various directions and layers not yet supported (see view in the next line!)'
        # h_n = h_n.view(batch_size, self.hidden_size)
        # abstracted = F.dropout(h_n, self.dropout)
        # abstracted = self.dropout(rnn_output[-1])
        abstracted = rnn_output[-1]
        for linear in self.ff:
            abstracted = self.dropout(F.relu(linear(abstracted)))

        logits = self.hidden2classes(abstracted)

        if aslogits:
            output = logits
        else:
            probabilities = torch.sigmoid(logits)
            output = probabilities

        return output

basedir = '/media/moreo/1TB Volume/Datasets/PolylingualEmbeddings'
dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run1.pickle'
data = MultilingualDataset.load(dataset)
langsel='es'
data.set_view(languages=[langsel])
langs = data.langs()

lX, lY = data.training()
lXte, lYte = data.test()

X=lX[langsel]
y=lY[langsel]
Xte=lXte[langsel]
yte=lYte[langsel]

some_label = y.sum(axis=1)>0
X=(np.array(X)[some_label]).tolist()
y=y[some_label]

some_positive = y.sum(axis=0)>5
y   = y[:,some_positive]
yte = yte[:,some_positive]

print(len(X))
print(y.shape)
print(len(Xte))
print(yte.shape)


svmMf1, svmmf1 = -1, -1
# svm = GridSearchCV(OneVsRestClassifier(LinearSVC(), n_jobs=-1), param_grid=[{'estimator__C': [1e5, 1e4, 1e3, 1e2, 1e1, 1, 1e-1]}], cv=5,  n_jobs=-1)
# vec = TfidfVectorizer(sublinear_tf=True)
# svm.fit(vec.fit_transform(X),y)
# print(svm.best_params_)
# yte_=svm.predict(vec.transform(Xte))
# svmMf1, svmmf1, _, _= evaluation_metrics(yte, yte_)
# print('\nLinearSVM {:.3f} {:.3f}'.format(svmMf1,svmmf1))


counter = CountVectorizer()  # text is already processed
counter.fit(X+Xte)
vocabulary = frozenset(counter.vocabulary_.keys())
#
# #cheat the dataset adding a word that indicates the presence of the category
# def cheat(lX,lY):
#     for l in lX.keys():
#         X = lX[l]
#         y = lY[l]
#         nclasses=len(y[0])
#         class_words = np.array(sorted(vocabularies[l])[5000:5000+nclasses])
#         lX[l] = [(' '.join(class_words[y[i]==1].tolist())) for i,xi in enumerate(X)]
#     print('done')
#     return lX
# lX = cheat(lX,lY)
# lXte = cheat(lXte,lYte)
# for lang in lX.keys():
#     counter = CountVectorizer()  # text is already processed
#     counter.fit(lX[lang]+lXte[lang])
#     vocabularies[lang] = frozenset(counter.vocabulary_.keys())
#
#
#
#
strip_accents=CountVectorizer(strip_accents='unicode').build_analyzer()
# pwe = WordEmbeddings.load_poly(basedir, langs, vocabularies, word_preprocessor=strip_accents)
pwe = WordEmbeddings.load(basedir, langsel, word_preprocessor=strip_accents)

vocabulary = sorted(vocabulary.intersection(pwe.vocabulary()))
we = pwe.get_vectors(vocabulary)

# del vocabularies

worddim = {w:i for i,w in enumerate(vocabulary)}

def index_doc(doc):
    return [worddim[w] for w in doc.split() if w in worddim]

def index_collection(X):
    return np.array([index_doc(d) for d in X])

def collection_stats(X):
    lens = np.array([len(x) for x in X])
    print('mean(len)={:.3f}'.format(lens.mean()))
    print('std(len) ={:.3f}'.format(lens.std()))
    print('max(len) ={:.3f}'.format(lens.max()))
    print('min(len) ={:.3f}'.format(lens.min()))
    return X

max_length=200

lr = 0.01
batch_size = 10
ndocs = len(X)
epochs = 5000
vocabulary_size, embedding_size = we.shape
nclasses = y.shape[1]
hidden_size=300
nlayers=1
fflayer_sizes=[]
dropout=0.5
lstm = LSTMTextClassificationNet(vocabulary_size, embedding_size, nclasses, hidden_size, nlayers, fflayer_sizes, dropout,
                 pretrained_embeddings=None, add_pad_zero_embedding=True, train_embeddings=False, max_length=max_length)

X   = collection_stats(index_collection(X))
Xte = collection_stats(index_collection(Xte))


loss_function = nn.BCEWithLogitsLoss()
# loss_function = nn.CrossEntropyLoss()
# loss_function = nn.MSELoss()
# loss_function = nn.L1Loss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, lstm.parameters()), lr=lr)
# optimizer = optim.Adam(lstm.parameters(), lr=lr)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, lstm.parameters()), lr=lr)
# optimizer = optim.SGD(lstm.parameters(), lr=lr)

print(lstm)

if USE_CUDA:
    lstm = lstm.cuda()
    loss_function = loss_function.cuda()

if INTERACTIVE:
    from inntt import InteractiveNeuralTrainer
    from inntt import *
    innt = InteractiveNeuralTrainer()
    innt.add_optim_param_adapt('ws', optimizer, 'lr', inc_factor=10.)
    innt.add_optim_param_adapt('da', optimizer, 'weight_decay', inc_factor=2.)
    innt.start()


def pad(x, PAD_TOKEN, max_length=-1):
    order = np.argsort([len(xi) for xi in x])[::-1]
    x = x[order]
    longest = len(x[0])
    if max_length > 0 and max_length < longest:
        longest = max_length
    padded = np.full((x.shape[0],longest), PAD_TOKEN, dtype=np.int)
    for i,xi in enumerate(x):
        len_i = min(len(xi),longest)
        padded[i,:len_i] = xi[:len_i]
    return padded

def batch(X, Y, batch_size):
    ndocs = len(X)
    nbatches = ndocs // batch_size
    nbatches = (nbatches+1) if (ndocs % batch_size)>0 else nbatches
    for ibatch in range(nbatches):
        Xi = X[ibatch * batch_size:(ibatch + 1) * batch_size]
        Xi = pad(Xi, lstm.PAD, max_length)
        Xi = Variable(torch.from_numpy(Xi.astype(np.int)))
        Yi = Variable(torch.from_numpy(Y[ibatch * batch_size:(ibatch + 1) * batch_size].astype(np.float32)))
        if USE_CUDA:
            Xi = Xi.cuda()
            Yi = Yi.cuda()
        yield Xi,Yi

def validation(Xval, yval, batch_size):
    lstm.eval()
    # evals = []

    # probs_ = []
    predictions = []
    for i, (Xi, Yi) in enumerate(batch(Xval, yval, batch_size)):
        probs = lstm.forward(Xi, aslogits=False).data
        probs = probs.cpu().numpy() if USE_CUDA else probs.numpy()
        # probs_.append(probs)
        predictions.append((probs > 0.5) * 1)
    predictions = np.vstack(predictions)
    # probs_ = np.vstack(probs_)

    # plt.matshow(np.hstack((probs_,predictions,lYte[l])))
    # if not showed:
    #     plt.show()
    #     showed=True

    Mf1, mf1, Mk, mK = evaluation_metrics(yval, predictions)
    # print('Lang %s: macro-F1=%.3f micro-F1=%.3f macro-K=%.3f micro-K=%.3f' % (l, Mf1, mf1, Mk, mK))
    # evals.append([Mf1, mf1, Mk, mK])
    # Mf1, mf1, Mk, mK = np.array(evals).mean(axis=0).tolist()
    return Mf1, mf1, Mk, mK
    # print('AVERAGE: macro-F1=%.3f micro-F1=%.3f macro-K=%.3f micro-K=%.3f' % (Mf1,mf1,Mk,mK))

losses = []
iter=0
average_throught = 20 # iterations that affect the loss
showed=False
Mf1, mf1, Mk, mK = -1, -1, -1, -1
Mf1tr, mf1tr, Mktr, mKtr = -1, -1, -1, -1
for epoch in range(epochs):
    #shuffle
    rand_order = np.random.permutation(ndocs)
    X,y = X[rand_order],y[rand_order]

    lstm.train()

    for i,(Xi,Yi) in enumerate(batch(X,y,batch_size)):
    # for i in range(100):
        optimizer.zero_grad()
        aslogits = True if isinstance(loss_function, nn.BCEWithLogitsLoss) else False
        logits = lstm.forward(Xi, aslogits=aslogits)
        loss = loss_function(logits, Yi)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i%10==0:
            print('epoch={}:it={}: loss = {:.5f} (+-{:.5f}) [last Tr={:.3f} {:.3f} Te={:.3f} {:.3f} SVM={:.3f} {:.3f}]'.format(
                epoch, iter, np.mean(losses[-average_throught:]), np.std(losses[-average_throught:]),
                Mf1tr, mf1tr, Mf1, mf1, svmMf1, svmmf1)
            )
        iter+=1

    if epoch % 2 == 0:
        Mf1tr, mf1tr, Mktr, mKtr = validation(X,y,batch_size)
    else:
        Mf1, mf1, Mk, mK = validation(Xte, yte, batch_size)



print('done')

