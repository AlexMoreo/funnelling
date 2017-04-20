from data.reader.jrcacquis_reader import fetch_jrcacquis, JRCAcquis_Document
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import cPickle as pickle


def split_data(data, train_years, test_years):
    train = []
    test  = []
    for doc in data:
        if doc.year in train_years:
            train.append((doc.text, doc.categories))
        elif doc.year in test_years:
            test.append((doc.text, doc.categories))
    return train, test

def show_classification_scheme(Y):
    class_count = {}
    for y in Y:
        for c in y:
            if c not in class_count:
                class_count[c] = 0
            class_count[c] += 1
    for c in class_count.keys():
        if class_count[c]<=20:
            del class_count[c]
    print "num classes", len(class_count)
    print class_count


train_years = [2005]
test_years  = [2006]
langs = ['es','it']
data='./storage'

for lang in langs:
    pickle_name = os.path.join(data, 'preprocessed_' + lang
                               + '_tr_' + '_'.join([str(y) for y in train_years])
                               + '_te_' + '_'.join([str(y) for y in test_years ]) + '.pickle')
    if os.path.exists(pickle_name):
        print("unpickling %s" % pickle_name)
        ((trX, trY), (teX, teY)) = pickle.load(open(pickle_name, 'rb'))
    else:
        raw_data = fetch_jrcacquis(langs=lang, data_dir=data, years=train_years + test_years)
        train, test = split_data(raw_data, train_years, test_years)
        print("processing %d documents for language <%s>" % (len(train) + len(test), lang))

        tr_data, tr_labels = zip(*train)
        te_data, te_labels = zip(*test)
        tfidf = TfidfVectorizer(strip_accents='unicode', stop_words=None, max_df=0.9, min_df=3, sublinear_tf=True)
        trX = tfidf.fit_transform(tr_data)
        teX = tfidf.transform(te_data)
        trY=tr_labels
        teY=te_labels
        pickle.dump(((trX, trY), (teX, teY)), open(pickle_name,'wb'),pickle.HIGHEST_PROTOCOL)
    print("load train=%s and test=%s matrices" % (str(trX.shape), str(teX.shape)))
    show_classification_scheme(trY)

print "Done"