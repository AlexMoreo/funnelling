import numpy as np
from keras import Input
from keras import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from data.embeddings import WordEmbeddings
from dataset_builder import MultilingualDataset
import sys

# import tensorflow as tf
# print(tf.test.gpu_device_name())

# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())

# import tensorflow
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# sys.exit()

# basedir = '/media/moreo/1TB Volume/Datasets/PolylingualEmbeddings'
# dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run1.pickle'
basedir = '/home/moreo/CLESA/PolylingualEmbeddings'
dataset = '/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run1.pickle'
data = MultilingualDataset.load(dataset)
langsel='en'
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
nclasses = y.shape[1]

nTr = len(X)
nTe = len(Xte)

alltexts = X+Xte
tokenizer = Tokenizer()
tokenizer.fit_on_texts(alltexts)
sequences = tokenizer.texts_to_sequences(alltexts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH=1000
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_train = data[:nTr]
y_train = y
x_test = data[nTr:]
y_test = yte

strip_accents=CountVectorizer(strip_accents='unicode').build_analyzer()
# pwe = WordEmbeddings.load_poly(basedir, langs, vocabularies, word_preprocessor=strip_accents)
pwe = WordEmbeddings.load(basedir, langsel, word_preprocessor=strip_accents)
embeddings_index = pwe.worddim

EMBEDDING_DIM = pwe.dim()
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(nclasses, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc','f1'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=1000, batch_size=128)