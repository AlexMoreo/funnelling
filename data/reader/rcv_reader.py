from zipfile import ZipFile
import xml.etree.ElementTree as ET

from data.languages import RCV2_LANGS_WITH_NLTK_STEMMING
from util.file import list_files
from sklearn.datasets import get_data_home
import gzip
from os.path import join, exists
from util.file import download_file_if_not_exists

"""
RCV2's Nomenclature:
ru = Russian
da = Danish
de = German
es = Spanish
lat = Spanish Latin-American (actually is also 'es' in the collection)
fr = French
it = Italian
nl = Dutch
pt = Portuguese
sv = Swedish
ja = Japanese
htw = Chinese
no = Norwegian
"""

#RCV1_BASE_URL= 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files'
RCV1_BASE_URL = "http://www.daviddlewis.com/resources/testcollections/rcv1/"
RCV2_BASE_URL = "http://trec.nist.gov/data/reuters/reuters.html"

rcv1_test_data_gz = ['lyrl2004_tokens_test_pt0.dat.gz',
             'lyrl2004_tokens_test_pt1.dat.gz',
             'lyrl2004_tokens_test_pt2.dat.gz',
             'lyrl2004_tokens_test_pt3.dat.gz']

rcv1_train_data_gz = ['lyrl2004_tokens_train.dat.gz']

rcv1_doc_cats_data_gz = 'rcv1-v2.topics.qrels.gz'

RCV2_LANG_DIR = {'ru':'REUTE000',
                 'de':'REUTE00A',
                 'fr':'REUTE00B',
                 'sv':'REUTE001',
                 'no':'REUTE002',
                 'da':'REUTE003',
                 'pt':'REUTE004',
                 'it':'REUTE005',
                 'es':'REUTE006',
                 'lat':'REUTE007',
                 'jp':'REUTE008',
                 'htw':'REUTE009',
                 'nl':'REUTERS_'}


class RCV_Document:

    def __init__(self, id, text, categories, date='', lang=None):
        self.id = id
        self.date = date
        self.lang = lang
        self.text = text
        self.categories = categories


# def fetch_RCV1(data_path=None, split='all'):
#
#     def load_doc_cat_file(path):
#         doc_labels = {}
#         labels = set()
#         for line in gzip.open(join(path, rcv1_doc_cats_data_gz), 'rt'):
#             cat, id, _ = line.split()
#             if id not in doc_labels:
#                 doc_labels[id] = []
#             doc_labels[id].append(cat)
#             labels.add(cat)
#         return doc_labels, list(labels)
#
#     def parse_documents(path, files):
#         if isinstance(files, str):
#             files = [files]
#         request = []
#         doc_buffer = []
#         for file in files:
#             for line in gzip.open(join(path, file), 'rt'):
#                 line = line.strip()
#                 if line.startswith('.I'):
#                     id = line.split()[1]
#                 elif line.startswith('.W'):
#                     doc_buffer = []
#                 elif line:
#                     doc_buffer.append(line)
#                 else:
#                     if doc_buffer:
#                         request.append(RCV_Document(id, text=' '.join(doc_buffer), categories=None, lang='en'))
#                         print('\r[{}] documents read {}'.format(path + '/' + file, len(request)), end='')
#         print('\n')
#         return request
#
#     assert split in ['train','test','test0','test1','test2','test3','all'], 'unexpected <split> request'
#
#     if not data_path:
#         data_path = get_data_home()
#
#     if split=='train':
#         target_files = rcv1_train_data_gz
#     elif split.startswith('test'):
#         if split == 'test':
#             target_files = rcv1_test_data_gz
#         else:
#             target_files = [rcv1_test_data_gz[int(split[-1])]]
#     else:
#         target_files = rcv1_train_data_gz + rcv1_test_data_gz
#
#     for data_file in target_files + [rcv1_doc_cats_data_gz]:
#         download_file_if_not_exists(url=join(RCV1_BASE_URL,data_file), archive_path=join(data_path, data_file))
#
#     print('parsing document-labels assignment')
#     doc_labels, labels = load_doc_cat_file(data_path)
#
#     print('parsing documents')
#     request = parse_documents(data_path, target_files)
#
#     print('assigning labels to documents')
#     for doc in request:
#         doc.categories = doc_labels[doc.id]
#
#     return request, labels


def parse_document(xml_content, assert_lang=None, valid_id_range=None):
    root = ET.fromstring(xml_content)
    if assert_lang:
        if assert_lang not in root.attrib.values():
            if assert_lang != 'jp' or 'ja' not in root.attrib.values():  # some documents are attributed to 'ja', others to 'jp'
                raise ValueError('error: document of a different language')

    doc_id = root.attrib['itemid']
    if valid_id_range is not None:
        if not valid_id_range[0] <= int(doc_id) <= valid_id_range[1]:
            raise IndexError

    doc_date = root.attrib['date']
    doc_title = root.find('.//title').text
    doc_headline = root.find('.//headline').text
    doc_body = '\n'.join([p.text for p in root.findall('.//text/p')])

    doc_categories = [cat.attrib['code'] for cat in
                      root.findall('.//metadata/codes[@class="bip:topics:1.0"]/code')]

    if not doc_body:
        raise ValueError('Empty document')

    if doc_title is None: doc_title = ''
    if doc_headline is None: doc_headline = ''
    text = '\n'.join([doc_title, doc_headline, doc_body]).strip()

    return RCV_Document(id=doc_id, text=text, categories=doc_categories, date=doc_date, lang=assert_lang)

def fetch_RCV1(data_path, split='all'):
    assert split in ['train', 'test', 'all'], 'split should be "train", "test", or "all"'

    request = []
    labels = set()
    read_documents = 0
    lang = 'en'

    training_documents = 23149
    test_documents = 781265
    train_range = (2286, 26150)
    test_range = (26151, 810596)

    if split == 'all':
        split_range = train_range + test_range
        expected = training_documents+test_documents
    elif split == 'train':
        split_range = train_range
        expected = training_documents
    else:
        split_range = test_range
        expected = test_documents

    for part in list_files(data_path):
        target_file = join(data_path, part)
        assert exists(target_file), \
            "You don't seem to have the file "+part+" in " + data_path + ", and the RCV1 corpus can not be downloaded"+\
            " w/o a formal permission. Please, refer to " + RCV1_BASE_URL + " for more information."
        zipfile = ZipFile(target_file)
        for xmlfile in zipfile.namelist():
            xmlcontent = zipfile.open(xmlfile).read()
            try:
                doc = parse_document(xmlcontent, assert_lang=lang, valid_id_range=split_range)
                labels.update(doc.categories)
                request.append(doc)
                read_documents += 1
                if read_documents == expected: break
            except ValueError:
                print('\n\tskipping document {} with inconsistent language label: expected language {}'.format(part+'/'+xmlfile, lang))
            except IndexError:
                pass
            print('\r[{}] read {} documents'.format(part, len(request)), end='')
        if read_documents == expected: break
    print()
    return request, list(labels)


def fetch_RCV2(data_path, languages=None):

    if not languages:
        languages = list(RCV2_LANG_DIR.keys())
    else:
        assert set(languages).issubset(set(RCV2_LANG_DIR.keys())), 'languages not in scope'

    request = []
    labels = set()
    for lang in languages:
        path = join(data_path, RCV2_LANG_DIR[lang])
        lang_docs_read = 0
        for part in list_files(path):
            target_file = join(path, part)
            assert exists(target_file), \
                "You don't seem to have the file "+part+" in " + path + ", and the RCV2 corpus can not be downloaded"+\
                " w/o a formal permission. Please, refer to " + RCV2_BASE_URL + " for more information."
            zipfile = ZipFile(target_file)
            for xmlfile in zipfile.namelist():
                xmlcontent = zipfile.open(xmlfile).read()
                try:
                    doc = parse_document(xmlcontent, assert_lang=lang)
                    labels.update(doc.categories)
                    request.append(doc)
                    lang_docs_read += 1
                except ValueError:
                    print('\n\tskipping document {} with inconsistent language label: expected language {}'.format(RCV2_LANG_DIR[lang]+'/'+part+'/'+xmlfile, lang))
                print('\r[{}] read {} documents, {} for language {}'.format(RCV2_LANG_DIR[lang]+'/'+part, len(request), lang_docs_read, lang), end='')
        print()
    return request, list(labels)


if __name__=='__main__':

    RCV1_PATH = '/media/moreo/1TB Volume/Datasets/RCV1-v2/unprocessed_corpus'
    RCV2_PATH = '/media/moreo/1TB Volume/Datasets/RCV2'

    rcv1_train, labels1 = fetch_RCV1(RCV1_PATH, split='train')
    rcv1_test, labels2 = fetch_RCV1(RCV1_PATH, split='test')
    #rcv2_documents, labels2 = fetch_RCV2(RCV2_PATH, RCV2_LANGS_WITH_NLTK_STEMMING)

    print('read {} documents in rcv1-train, and {} labels'.format(len(rcv1_train), len(labels1)))
    print('read {} documents in rcv1-test, and {} labels'.format(len(rcv1_test), len(labels2)))
    #print('read {} documents in rcv2, and {} labels'.format(len(rcv2_documents), len(labels2)))