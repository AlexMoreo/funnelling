from optparse import OptionParser
import numpy as np
import time,os,sys
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras import optimizers
from sklearn.feature_extraction.text import CountVectorizer
from data.embeddings import WordEmbeddings
from dataset_builder import MultilingualDataset
from util.evaluation_keras import batchf1_keras
from util.evaluation import evaluate, average_results
import itertools

from util.results import PolylingualClassificationResults



parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset", help="Path to the multilingual dataset preprocessed and stored in .pickle format")
parser.add_option("-w", "--we-path", dest="we_path", help="Path to the polylingual word embeddings")
parser.add_option("-o", "--output", dest="output", help="Result file", type=str, default='./lstm_results.csv')
parser.add_option("-b", "--batchsize", dest="batchsize", help="Batch size", type=int, default=500)
parser.add_option("-l", "--lstmsize", dest="lstmsize", help="LSTM hidden size", type=int, default=512)
parser.add_option("-f", "--densesize", dest="densesize", help="Dense layer size", type=int, default=512)
parser.add_option("-L", "--maxlength", dest="maxlength", help="Max sentence length", type=int, default=200)
parser.add_option("-e", "--trainembedding", dest="trainembedding", help="Trainable embeddings (1=train(default) 0=fixed)", type=int, default=1)
parser.add_option("-m", "--maskunknown", dest="maskunknown", help="Mask words w/o pretrained embedding as unknown (1=True 0=False(default))", type=int, default=0)
parser.add_option("-E", "--nepochs", dest="nepochs", help="Number of epochs (default=100)", type=int, default=100)
parser.add_option("-p", "--patience", dest="patience", help="Patience for Early-Stop (default=10)", type=int, default=10)



def DataLoad(dataset_path, embeddings_path):
    print('Loading dataset from ' + dataset_path)
    data = MultilingualDataset.load(dataset_path)  # this dataset is assumed to be preprocessed

    data.set_view(languages=data.langs()[:2])

    langs = data.langs()
    lXtr, lYTr = data.training()
    lXte, lYte = data.test()

    print('Loading Polylingual Word Embeddings from ' + embeddings_path)
    pwe = {}
    strip_accents = CountVectorizer(strip_accents='unicode').build_analyzer()
    for lang in langs:
        pwe[lang] = WordEmbeddings.load(embeddings_path, lang, word_preprocessor=strip_accents)

    dataset_name = 'unknown'
    if 'rcv' in dataset_path: dataset_name='rcv1-2'
    elif 'jrc' in dataset_path: dataset_name='jrc-acquis'

    return langs, (lXtr, lYTr), (lXte, lYte), pwe, dataset_name


def PrepareData(lXtr, lYtr, lXte, lYte, mask_unknown=False):

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False


    def add_lang_prefix(docs, lang):
        vocab = pwe[lang].vocabulary()

        def add_prefix(word):
            if is_number(word):
                return lang + LANG_PREFIX + 'wordtoken'
            elif mask_unknown == False or word in vocab:
                return lang + LANG_PREFIX + word
            else:
                return lang + LANG_PREFIX + 'unktoken'  # works much better without grouping unknown words

        return [' '.join([add_prefix(word) for word in doc.split()]) for doc in docs]


    print('')
    alltexts = list(itertools.chain.from_iterable([add_lang_prefix(lXtr[l], l) for l in langs]))
    trdocs = len(alltexts)
    tr_labels = list(itertools.chain.from_iterable([lYtr[l] for l in langs]))

    alltexts.extend(list(itertools.chain.from_iterable([add_lang_prefix(lXte[l], l) for l in langs])))
    te_labels = list(itertools.chain.from_iterable([lYte[l] for l in langs]))

    docwords = [doc.split() for doc in alltexts]

    lengths = [len(doc_i_words) for doc_i_words in docwords]
    print('ave(length)={:.2f}'.format(np.mean(lengths)))
    print('std(length)={:.2f}'.format(np.std(lengths)))
    print('max(length)={:.0f}'.format(np.max(lengths)))
    print('min(length)={:.0f}'.format(np.min(lengths)))

    # we cut sequences longer than MAX_SEQUENCE_LENGTH *before* calling the tokenizer to prevent if from assigning
    # ids to unused tokens
    alltexts=[' '.join(doc_i_words[-MAX_SEQUENCE_LENGTH:]) for doc_i_words in docwords]


    tokenizer = Tokenizer(filters='')  # text is already preprocessed, so we want to tokenize words exclusively on spaces
    tokenizer.fit_on_texts(alltexts)
    sequences = tokenizer.texts_to_sequences(alltexts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    x_train = data[:trdocs]
    y_train = np.vstack(tr_labels)
    x_test = data[trdocs:]
    y_test = np.vstack(te_labels)

    nlabels_by_doc = y_train.sum(axis=1)
    print('ave(labels)={:.2f}'.format(np.mean(nlabels_by_doc)))
    print('std(labels)={:.2f}'.format(np.std(nlabels_by_doc)))
    print('max(labels)={:.0f}'.format(np.max(nlabels_by_doc)))
    print('min(labels)={:.0f}'.format(np.min(nlabels_by_doc)))

    return (x_train, y_train), (x_test, y_test), word_index


def PrepareEmbeddingMatrix(pwe, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    empty = 0
    for lang in langs:
        embeddings_index = pwe[lang].worddim
        for word, i in word_index.items():
            assert i != 0, '0 index is reserved for pad'
            pre_word = word.split(LANG_PREFIX)
            if len(pre_word)==2:
                lang_prefix, word = pre_word
                if lang_prefix == lang:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[i] = embedding_vector
                    else:
                        print('empty={}'.format(word))
                        empty += 1
            else:
                print('unparsable token {}'.format(word))
    print('empty vectors={}'.format(empty))

    return embedding_matrix

def get_loffset(lX):
    loffsets = {}
    offset=0
    for l in langs:
        end=offset + len(lX[l])
        loffsets[l]=(offset,end)
        offset=end
    return loffsets

def pack_by_lang(X,loffsets):
    lX = {}
    for l in loffsets.keys():
        start,end = loffsets[l]
        lX[l] = X[start:end]
    return lX

def RNN(embedding_matrix, trainable, lstmsize, densesize):

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=trainable)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    layer = embedding_layer(sequence_input)
    layer = LSTM(lstmsize, dropout=0.5, recurrent_dropout=0.5)(layer)
    layer = Dense(densesize, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(nclasses, activation='sigmoid')(layer)
    model = Model(inputs=sequence_input, outputs=layer)

    return model


if __name__=='__main__':
    (op, args) = parser.parse_args()

    dataset_path = op.dataset
    embeddings_path = op.we_path

    maskunknown=op.maskunknown==1
    trainable=op.trainembedding==1
    MAX_SEQUENCE_LENGTH=op.maxlength
    method_config = 'LSTM_b{}_h{}_ff{}_e{}_L{}_m{}_E{}_p{}'.format(
        op.batchsize, op.lstmsize, op.densesize, trainable, MAX_SEQUENCE_LENGTH, maskunknown, op.nepochs, op.patience
    )
    print('Running ',method_config)

    LANG_PREFIX='-'
    EMBEDDING_DIM = 300

    results = PolylingualClassificationResults(op.output)

    langs, (lXtr, lY), (lXte, lYte), pwe, datasetname = DataLoad(dataset_path, embeddings_path)
    (x_train, y_train), (x_test, y_test), word_index = PrepareData(lXtr, lY, lXte, lYte, mask_unknown=maskunknown)
    embedding_matrix = PrepareEmbeddingMatrix(pwe, word_index)
    nclasses = y_train.shape[1]
    del pwe

    optimizer='rmsprop'
    model = RNN(embedding_matrix, trainable, op.lstmsize, op.densesize)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc',batchf1_keras])

    rand_order = np.random.permutation(x_train.shape[0])
    x_train=x_train[rand_order]
    y_train=y_train[rand_order]
    tinit = time.time()
    # reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    earlystop = EarlyStopping(monitor='val_'+batchf1_keras.__name__, patience=op.patience, restore_best_weights=True, mode='max')
    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=op.nepochs,
                        batch_size=op.batchsize,
                        shuffle=True,
                        callbacks=[earlystop])

    tr_time = time.time()-tinit

    loffsets_test = get_loffset(lXte)

    print('LSTM-evaluation')
    tinit = time.time()
    probs = model.predict(x_test, batch_size=op.batchsize)
    y_lstm = 1 * (probs > 0.5)
    te_time = time.time()-tinit
    y_lstm = pack_by_lang(y_lstm, loffsets_test)
    l_eval = evaluate(lYte, y_lstm)
    grand_totals = average_results(l_eval, show=True)
    print('epochs={} batch={} lstmsize={} densesize={}'.format(op.nepochs, op.batchsize, op.lstmsize, op.densesize))


    dataset_file = os.path.basename(op.dataset)
    result_id = dataset_file+'__'+method_config
    for lang in l_eval.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        # using the "notest" to show the test time
        results.add_row(result_id, method_config, 'keras-lstm', optimizer, datasetname, '', '', tr_time, lang,
                        macrof1, microf1, macrok, microk, notes=te_time)

    # import matplotlib.pyplot as plt
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    # plt.plot(history.history[batchf1_keras.__name__])
    # plt.plot(history.history['val_'+batchf1_keras.__name__])
    # plt.title('Classification performance')
    # plt.ylabel('F1')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    model.summary()

