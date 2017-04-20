from __future__ import print_function
import os
from os.path import join
import tarfile
import xml.etree.ElementTree as ET
from sklearn.datasets import get_data_home
import cPickle as pickle
from util.file import download_file, list_dirs, list_files

LANGS = ['bg','cs','da','de','el','en','es','et','fi','fr','hu','it','lt','lv','mt','nl','pl','pt','ro','sk','sl','sv']

class JRCAcquis_Document:
    def __init__(self, id, name, lang, year, head, body, categories):
        self.id = id
        self.parallel_id = name
        self.lang = lang
        self.year = year
        self.text = body if not head else head + "\n" + body
        self.categories = categories

# this is a workaround... for some reason, acutes are codified in a non-standard manner in titles
# however, it seems that the title is often appearing as the first paragraph in the text/body (with
# standard codification), so it might be preferable not to read the header after all
def proc_acute(text):
    for ch in ['a','e','i','o','u']:
        text = text.replace('%'+ch+'acute%',ch)
    return text

def raise_if_empty(field, from_file):
    if isinstance(field, str):
        if not field.strip():
            raise ValueError("Error, field empty in file %s" % from_file)

def parse_document(file, year, head=False):
    root = ET.parse(file).getroot()

    doc_name = root.attrib['n'] # e.g., '22006A0211(01)'
    doc_lang = root.attrib['lang'] # e.g., 'es'
    doc_id   = root.attrib['id'] # e.g., 'jrc22006A0211_01-es'
    doc_categories = [int(cat.text) for cat in root.findall('.//teiHeader/profileDesc/textClass/classCode[@scheme="eurovoc"]')]
    doc_head = proc_acute(root.find('.//text/body/head').text) if head else ''
    doc_body = '\n'.join([p.text for p in root.findall('.//text/body/div[@type="body"]/p')])

    raise_if_empty(doc_name, file)
    raise_if_empty(doc_lang, file)
    raise_if_empty(doc_id, file)
    if head: raise_if_empty(doc_head, file)
    raise_if_empty(doc_body, file)

    return JRCAcquis_Document(id=doc_id, name=doc_name, lang=doc_lang, year=year, head=doc_head, body=doc_body, categories=doc_categories)

def fetch_jrcacquis(langs=None, data_dir=None, years=None, ignore_unclassified=True):
    if not langs:
        langs = LANGS
    else:
        if isinstance(langs, str): langs = [langs]
        for l in langs:
            if l not in LANGS:
                raise ValueError('Language %s is not among the valid languages in JRC-Acquis v3' % l)

    if not data_dir:
        data_dir = get_data_home()

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    request = []
    DOWNLOAD_URL_BASE = 'http://optima.jrc.it/Acquis/JRC-Acquis.3.0/corpus/'
    total_read = 0
    for l in langs:

        file_name = 'jrc-'+l+'.tgz'
        archive_path = join(data_dir, file_name)

        if not os.path.exists(archive_path):
            print("downloading language-specific dataset (once and for all) into %s" % data_dir)
            DOWNLOAD_URL = join(DOWNLOAD_URL_BASE, file_name)
            download_file(DOWNLOAD_URL, archive_path)
            print("\nuntarring dataset...")
            tarfile.open(archive_path, 'r:gz').extractall(data_dir)

        documents_dir = join(data_dir, l)

        print("Reading documents...")
        read = 0
        for dir in list_dirs(documents_dir):
            year = int(dir)
            if years==None or year in years:
                year_dir = join(documents_dir,dir)
                pickle_name = join(data_dir, 'jrc_' + l + '_' + dir + '.pickle')
                if os.path.exists(pickle_name):
                    print("loading from file %s" % pickle_name)
                    l_y_documents = pickle.load(open(pickle_name, "rb"))
                    read += len(l_y_documents)
                else:
                    l_y_documents = []
                    all_documents = list_files(year_dir)
                    empty = 0
                    for i,doc_file in enumerate(all_documents):
                        jrc_doc = parse_document(join(year_dir, doc_file), year)
                        if not ignore_unclassified or jrc_doc.categories:
                            l_y_documents.append(jrc_doc)
                        else: empty += 1
                        if (i+1) % (len(all_documents)/50) == 0:
                            print('\r\tfrom %s: completed %d%%' % (year_dir, int((i+1)*100.0/len(all_documents))), end='')
                        read+=1
                    print('\r\tfrom %s: completed 100%% read %d documents (discarded %d without categories)\n' % (year_dir, i+1, empty), end='')
                    print("Pickling object for future runs in %s" % pickle_name)
                    pickle.dump(l_y_documents, open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)
                request += l_y_documents
        print("Read %d documents for language %s" % (read, l))
        total_read += read
    print("Read %d documents in total" % (total_read))
    return request

def __main__():

    storage_path = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"

    fetch_jrcacquis(langs=['it', 'en'], data_dir=storage_path, years=[2004, 2005, 2006])


