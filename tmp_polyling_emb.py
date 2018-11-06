import os, sys
import numpy as np
import pickle

from sklearn.multiclass import OneVsRestClassifier

from data.reader.rcv_reader import fetch_RCV1, fetch_RCV2
from dataset_builder import *
from sklearn.svm import SVC

from util.evaluation import evaluation_metrics







if __name__ == '__main__':


    def get_rcv_documents(rcv1_data_home, rcv2_data_home, tr_ids, te_ids):
        assert tr_ids.keys()==te_ids.keys(),'inconsistent keys tr vs te'
        langs = list(tr_ids.keys())

        print('fetching the datasets')
        rcv1_documents, labels_rcv1 = fetch_RCV1(rcv1_data_home, split='train')
        rcv2_documents, labels_rcv2 = fetch_RCV2(rcv2_data_home, [l for l in langs if l!='en'])

        filter_by_categories(rcv1_documents, labels_rcv2)
        filter_by_categories(rcv2_documents, labels_rcv1)

        label_names = get_active_labels(rcv1_documents + rcv2_documents)
        print('Active labels in RCV1/2 {}'.format(len(label_names)))

        print('rcv1: {} train, {} test, {} categories'.format(len(rcv1_documents), 0, len(label_names)))
        print('rcv2: {} documents'.format(len(rcv2_documents)), Counter([doc.lang for doc in rcv2_documents]))

        all_docs = rcv1_documents + rcv2_documents
        tr_docs = {lang: [(d.text,d.categories) for d in all_docs if d.lang == lang and d.id in tr_ids[lang]] for lang in langs}
        te_docs = {lang: [(d.text,d.categories) for d in all_docs if d.lang == lang and d.id in te_ids[lang]] for lang in langs}

        return tr_docs, te_docs, label_names

    def embed_docs(docs, lang, word_embeddings_path, tfidf, method='tfidf', normalize=False):

        analyzer = tfidf.build_analyzer()
        V = tfidf.vocabulary_
        X = tfidf.transform(docs)
        # X.sort_indices()
        # return X
        idf = tfidf.idf_

        print('processing text')
        docs = [analyzer(d) for d in docs]

        print('loading word embeddings for '+lang)
        we = WordEmbeddings(word_embeddings_path, lang)

        nD = len(docs)
        doc_vecs = np.zeros((nD, we.dim()))

        print('averaging documents')
        for i,doc in enumerate(docs):
            print('\r\tcomplete {}%'.format(100*(i+1)/nD), end='')

            # media simple
            if method=='mean':
                wordcount = 0
                for w in doc:
                    if w in we:
                        doc_vecs[i] += we[w]
                        wordcount += 1
                doc_vecs /= wordcount

            #media con tfidf
            elif method=='tfidf':
                added=set()
                for w in doc:
                    if w in added: continue
                    if w in we and w in V:
                        doc_vecs[i] += (we[w] * X[i, V[w]])
                        added.add(w)

            # media con idf
            elif method == 'idf':
                for w in doc:
                    if w in we and w in V:
                        doc_vecs[i] += (we[w] * idf[V[w]])
        print()
        if normalize:
            doc_vecs = doc_vecs/np.linalg.norm(doc_vecs,axis=1,keepdims=True)

        return doc_vecs


    def get_lX_lY(tr_doc_dict, te_doc_dict, label_names, method, normalize):
        mlb = MultiLabelBinarizer()
        mlb.fit([label_names])

        lXtr,lYtr = {}, {}
        lXte, lYte = {}, {}
        langs = (tr_doc_dict.keys())
        for lang in langs:
            tr_docs, tr_labels = list(zip(*tr_doc_dict[lang]))
            te_docs, te_labels = list(zip(*te_doc_dict[lang]))

            tfidf = TfidfVectorizer(strip_accents='unicode', min_df=3, sublinear_tf=True,
                                    stop_words=stopwords.words(NLTK_LANGMAP[lang]))

            tfidf.fit(tr_docs)

            Xtr = embed_docs(tr_docs, lang, WORDEMBEDDINGS_PATH, tfidf, method, normalize)
            Ytr = mlb.transform(tr_labels)
            Xte = embed_docs(te_docs, lang, WORDEMBEDDINGS_PATH, tfidf, method, normalize)
            Yte = mlb.transform(te_labels)

            lXtr[lang] = Xtr
            lYtr[lang] = Ytr
            lXte[lang] = Xte
            lYte[lang] = Yte

        return lXtr,lYtr,lXte,lYte






    RCV1_PATH = '/media/moreo/1TB Volume/Datasets/RCV1-v2/unprocessed_corpus'
    RCV2_PATH = '/media/moreo/1TB Volume/Datasets/RCV2'
    WORDEMBEDDINGS_PATH = '/media/moreo/1TB Volume/Datasets/PolylingualEmbeddings'


    # we_es = WordEmbeddings(WORDEMBEDDINGS_PATH, 'es')
    # we_it = WordEmbeddings(WORDEMBEDDINGS_PATH, 'it')
    # wordvec_es = we_es.get_vectors(['gato','españa','madre'])
    # words, sim = we_it.most_similar(wordvec_es, 10)
    # print(words)
    #print(sim)
    # sys.exit(0)

    dataset = "/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processed_run0.pickle"
    tr_ids, te_ids = MultilingualDataset.load_ids(dataset)




    # tr_ids = {'es':tr_ids['es'], 'it':tr_ids['it']}
    # te_ids = {'es': te_ids['es'], 'it': te_ids['it']}




    tr_docs, te_docs, label_names = get_rcv_documents(RCV1_PATH,RCV2_PATH, tr_ids, te_ids)

    lXtr, lYtr, lXte, lYte = get_lX_lY(tr_docs, te_docs, label_names, method='idf', normalize=False)


    langs = list(lXtr.keys())

    from scipy.sparse import issparse,vstack
    if issparse(lXtr[langs[0]]):
        print('sparse stack')
        X = vstack([lXtr[l] for l in langs])
    else:
        print('dense stack')
        X = np.vstack([lXtr[l] for l in langs])
    Y = np.vstack([lYtr[l] for l in langs])

    print('shapes')
    print(X.shape)
    print(Y.shape)
    svm = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=-1)
    #Lang es: macro-F1=0.450 micro-F1=0.825 macro-K=0.422 micro-K=0.737 kernel linear, X sparse
    #Lang es: macro-F1=0.451 micro-F1=0.793 macro-K=0.416 micro-K=0.707 X ave we*tfidf
    #Lang es: macro-F1=0.345 micro-F1=0.710 macro-K=0.323 micro-K=0.569 X ave we*tfidf, l2
    # svm = GridSearchCV(svm, param_grid=['estim_'], refit=True, cv=5, n_jobs=self.n_jobs,
    #              error_score=0, verbose=0)
    svm.fit(X,Y)

    averages = []
    for lang in langs:
        y_ = svm.predict(lXte[lang])
        macrof1, microf1, macrok, microk = evaluation_metrics(lYte[lang], y_)
        print('Lang %s: macro-F1=%.3f micro-F1=%.3f macro-K=%.3f micro-K=%.3f' %
              (lang, macrof1, microf1, macrok, microk))
        averages.append((macrof1, microf1, macrok, microk))

    averages = np.array(averages)
    averages = np.mean(averages, axis=0)
    print(averages)

    print('------> with l2 and idf-scaling')

    #with l2 [0.31681716  0.62227614  0.28567076  0.48061795]
    #without l2 [ 0.41073421  0.6914574   0.36936868  0.58681635]
    #with l2 and idf-scaling [ 0.39424158  0.65372597  0.46140288  0.6151079 ]

    print('done')

