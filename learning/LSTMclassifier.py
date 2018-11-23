import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from torch.autograd import Variable
from data.embeddings import WordEmbeddings
import torch.nn.functional as F
import numpy as np
from dataset_builder import MultilingualDataset

USE_CUDA = True
INTERACTIVE = False

class LSTMTextClassificationNet(nn.Module):

    def __init__(self, vocabulary_size, embedding_size, nclasses, hidden_size, nlayers, fflayer_sizes, dropout,
                 pretrained_embeddings=None, train_embeddings=True):
        super().__init__()

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.PAD = vocabulary_size # adding one dimension for the PAD symbol
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocabulary_size, embedding_size)
            pretrained_embeddings=np.vstack([pretrained_embeddings,np.zeros((1,embedding_size))]) # PAD null embedding
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings.astype(np.float32)), freeze=not train_embeddings)
        else:
            self.embedding = nn.Embedding(vocabulary_size + 1, embedding_size, padding_idx=self.PAD)

        self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=self.dropout)
        prev_size = hidden_size
        self.ff = nn.ModuleList()
        for lin_size in fflayer_sizes:
            self.ff.append(nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.hidden2classes = nn.Linear(prev_size, nclasses)

    def init_hidden(self, set_size):
        var_hidden = Variable(torch.zeros(self.nlayers, set_size, self.hidden_size))
        var_cell   = Variable(torch.zeros(self.nlayers, set_size, self.hidden_size))
        if USE_CUDA:
            return (var_hidden.cuda(), var_cell.cuda())
        else:
            return (var_hidden, var_cell)

    def _pad(self, x):
        x = x[np.argsort([len(xi) for xi in x])[::-1]]
        longest = len(x[0])
        pad = np.zeros((x.shape[0],longest), dtype=np.int)
        for i,xi in enumerate(x):
            len_i = len(xi)
            pad[i,:len_i]=xi
            pad[i,len_i:]=self.PAD
        pad = torch.from_numpy(pad).transpose(0,1)
        if USE_CUDA: pad = pad.cuda()
        return pad


    def forward(self, x, return_abstraction=False):
        # Xi = rnn_utils.pack_sequence([torch.Tensor(xi) for xi in Xi])
        x = self._pad(x)
        embedded = self.embedding(x)

        # Xi = rnn_utils.pad_packed_sequence(Xi)
        rnn_output, rnn_hidden = self.lstm(embedded, self.init_hidden(embedded.shape[0]))
        abstracted = F.dropout(rnn_hidden[0][-1], self.dropout)
        for linear in self.ff:
            abstracted = F.dropout(F.relu(linear(abstracted)))
        output = self.hidden2classes(abstracted)
        class_output = F.softmax(output,dim=1)
        if return_abstraction:
            return class_output,abstracted
        else:
            return class_output

basedir = '/media/moreo/1TB Volume/Datasets/PolylingualEmbeddings'
dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle'
langs = ['en']
# data = MultilingualDataset.load(dataset)
# data.set_view(languages=langs)
#
#
# lX, lY = data.training()
# lXte, lYte = data.test()

lX = {'en':['this is a document referring to houses','this is about cats','this is about document']}
#       'es': ['este documento habla sobre casas', 'este habla sobre gatos', 'este trata sobre documento']}
lY = {'en':[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]}
lXte=lX



vocabularies = {}

for lang in lX.keys():
    print('preparing language {}'.format(lang))
    print('\tloading vocabulary')
    counter = CountVectorizer()  # text is already processed
    counter.fit(lX[lang]+lXte[lang])
    vocabularies[lang] = frozenset(counter.vocabulary_.keys())

strip_accents=CountVectorizer(strip_accents='unicode').build_analyzer()
pwe = WordEmbeddings.load_poly(basedir, langs, vocabularies, word_preprocessor=strip_accents)


def index_doclist(lX,lY,worddim):
    X, Y = [], []
    for l in lX.keys():
        X.extend([[worddim[l+'::'+w] for w in d.split() if l+'::'+w in worddim] for d in lX[l]])
        Y.extend(lY[l])
    return np.array(X),np.array(Y)

X,Y = index_doclist(lX, lY, pwe.worddim)

lr = 0.01
batch_size = 1
ndocs = len(X)
epochs = 500
vocabulary_size, embedding_size = pwe.we.shape
nclasses = len(Y[0])
hidden_size=300
nlayers=1
fflayer_sizes=[100,100]
dropout=0.1
lstm = LSTMTextClassificationNet(vocabulary_size, embedding_size, nclasses, hidden_size, nlayers, fflayer_sizes, dropout,
                 pretrained_embeddings=pwe.we, train_embeddings=False)

loss_function = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, lstm.parameters()), lr=lr)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, lstm.parameters()), lr=lr)

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


losses = []
iter=0
for epoch in range(epochs):
    #shuffle
    rand_order = np.random.permutation(ndocs)
    X,Y = X[rand_order],Y[rand_order]

    nbatches = ndocs // batch_size
    for ibatch in range(nbatches):
        Xi = X[ibatch * batch_size:(ibatch + 1) * batch_size]
        Yi = torch.from_numpy(Y[ibatch * batch_size:(ibatch + 1) * batch_size].astype(np.float32))
        if USE_CUDA:
            Yi=Yi.cuda()

        lstm.zero_grad()
        probs = lstm(Xi)
        loss = loss_function(probs, Yi)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if ibatch%10==0:
            ave = np.mean(losses[-10:])
            std = np.std(losses[-10:])
            print('epoch={}:it={}: loss = {:.5f} (+-{:.5f})'.format(epoch, iter, ave, std))
            if not USE_CUDA:
                print('NOT USING CUDA!')
        iter+=1

print('done')

