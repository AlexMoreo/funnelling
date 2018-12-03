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
from util.evaluation import evaluation_metrics

# basedir = '/media/moreo/1TB Volume/Datasets/PolylingualEmbeddings'
# dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle'
basedir = '/home/moreo/CLESA/PolylingualEmbeddings'
dataset = '/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run1.pickle'
data = MultilingualDataset.load(dataset)
langs = data.langs()

lX, lY = data.training()
lXte, lYte = data.test()

nclasses = data.num_categories()

import itertools

def add_lang_prefix(docs, lang):
    return [' '.join([lang+'-'+word for word in doc.split()]) for doc in docs]

alltexts = list(itertools.chain.from_iterable([add_lang_prefix(lX[l], l) for l in langs]))
trdocs = len(alltexts)
tr_labels = list(itertools.chain.from_iterable([lY[l] for l in langs]))

alltexts.extend(list(itertools.chain.from_iterable([add_lang_prefix(lXte[l],l) for l in langs])))
te_labels = list(itertools.chain.from_iterable([lYte[l] for l in langs]))

tokenizer = Tokenizer(filters='!"#$%&()*+,:./;<=>?@[\]^_`{|}~') # free the '-'
tokenizer.fit_on_texts(alltexts)
sequences = tokenizer.texts_to_sequences(alltexts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH=200
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_train = data[:trdocs]
y_train = np.vstack(tr_labels)
x_test = data[trdocs:]
y_test = np.vstack(te_labels)


strip_accents=CountVectorizer(strip_accents='unicode').build_analyzer()
EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
empty=0
for lang in langs:
    pwe = WordEmbeddings.load(basedir, lang, word_preprocessor=strip_accents)
    embeddings_index = pwe.worddim

    for word, i in word_index.items():
        lang_prefix,word=word.split('-')
        if lang_prefix==lang:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                empty+=1

print('empty vectors={}'.format(empty))

from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout

trainable=True
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=trainable)

# lo lance con trainable a True

def CNN():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(nclasses, activation='sigmoid')(x)
    model = Model(inputs=sequence_input, outputs=preds)
    return model

def RNN():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    layer = embedding_layer(sequence_input)
    layer = LSTM(64)(layer)
    layer = Dense(256,activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(nclasses, activation='sigmoid')(layer)
    model = Model(inputs=sequence_input,outputs=layer)
    return model


from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = RNN()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              # optimizer='adam',
              metrics=['acc', f1])

batch_size = 500
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=1000, batch_size=batch_size, shuffle=True)


probs = model.predict(x_test, batch_size=batch_size)
yte_ = 1*(probs>0.5)
Mf1, mf1, Mk, mK = evaluation_metrics(y_test, yte_)
print('Eval: {:.3f} {:.3f} {:.3f} {:.3f}'.format(Mf1, mf1, Mk, mK))

print('trainable={}'.format(trainable))