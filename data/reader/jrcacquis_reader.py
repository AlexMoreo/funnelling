from __future__ import print_function
import os, sys
from os.path import join
import tarfile
import xml.etree.ElementTree as ET
from sklearn.datasets import get_data_home
import cPickle as pickle
from util.file import download_file, list_dirs, list_files
import rdflib
from rdflib.namespace import RDF, SKOS
from rdflib import URIRef
import zipfile

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
# standard codification), so it might be preferable not to read the header after all (as here by default)
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
    doc_categories = [cat.text for cat in root.findall('.//teiHeader/profileDesc/textClass/classCode[@scheme="eurovoc"]')]
    doc_head = proc_acute(root.find('.//text/body/head').text) if head else ''
    doc_body = '\n'.join([p.text for p in root.findall('.//text/body/div[@type="body"]/p')])

    raise_if_empty(doc_name, file)
    raise_if_empty(doc_lang, file)
    raise_if_empty(doc_id, file)
    if head: raise_if_empty(doc_head, file)
    raise_if_empty(doc_body, file)

    return JRCAcquis_Document(id=doc_id, name=doc_name, lang=doc_lang, year=year, head=doc_head, body=doc_body, categories=doc_categories)

#filters out documents which do not contain any category in the cat_filter list
def filter_by_category(doclist, cat_filter):
    if not isinstance(cat_filter, frozenset):
        cat_filter = frozenset(cat_filter)
    filtered = []
    for doc in doclist:
        doc.categories = list(cat_filter & set(doc.categories))
        if doc.categories:
            doc.categories.sort()
            filtered.append(doc)
            #print("Categories %s: %s" % (doc.categories, doc.text))
    print("filtered %d documents out without categories in the filter list" % (len(doclist) - len(filtered)))
    return filtered

#filters out categories with less than cat_threshold documents (and filters documents containing those categories)
def filter_by_frequency(doclist, cat_threshold):
    cat_count = {}
    for d in doclist:
        for c in d.categories:
            if c not in cat_count:
                cat_count[c] = 0
            cat_count[c] += 1

    freq_categories = [cat for cat,count in cat_count.items() if count>cat_threshold]
    return filter_by_category(doclist, freq_categories)


def fetch_jrcacquis(langs=None, data_dir=None, years=None, ignore_unclassified=True, cat_filter=None, cat_threshold=0):
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
    if cat_filter:
        request = filter_by_category(request, cat_filter)
        request = filter_by_frequency(request, cat_threshold)
    return request

def inspect_eurovoc(path, filename, pickle_name=None,
                    eurovoc_url="http://publications.europa.eu/mdr/resource/thesaurus/eurovoc-20160630-0/skos/eurovoc_in_skos_core_concepts.zip",
                    select="broadest"):
    if pickle_name:
        fullpath_pickle = join(path, pickle_name)
        if os.path.exists(fullpath_pickle):
            print("Pickled object found in %s. Loading it." % fullpath_pickle)
            return pickle.load(open(fullpath_pickle,'rb'))


    fullpath = join(path, filename)
    if not os.path.exists(fullpath):
        print("Path %s does not exist. Trying to download the skos EuroVoc file from %s" % (path, eurovoc_url))
        download_file(eurovoc_url, path+'.zip')
        print("Unzipping file...")
        zipped = zipfile.ZipFile(path+'.zip', 'r')
        zipped.extract("eurovoc_in_skos_core_concepts.rdf", path)
        zipped.close()

    print("Parsing %s" %fullpath)
    g = rdflib.Graph()
    g.parse(location=fullpath, format="application/rdf+xml")

    if select=="broadest":
        print("Selecting broadest concepts (those without any other broader concept linked to it)")
        all_concepts = set(g.subjects(RDF.type, SKOS.Concept))
        narrower_concepts = set(g.subjects(SKOS.broader, None))
        broadest_concepts = [c.toPython().split('/')[-1] for c in (all_concepts - narrower_concepts)]
        broadest_concepts.sort()

        print("%d broad concepts found" % len(broadest_concepts))
        if pickle_name:
            print("Pickling concept list for faster further requests in %s" % fullpath_pickle)
            pickle.dump(broadest_concepts, open(fullpath_pickle,'wb'), pickle.HIGHEST_PROTOCOL)
        return broadest_concepts
    else:
        raise ValueError("Selection policy %s is not currently supported" % select)


if __name__ == "__main__":
    storage_path = "/media/moreo/1TB Volume/Datasets/Multilingual/JRC_Acquis_v3"

    cat_list = inspect_eurovoc(storage_path, 'eurovoc_in_skos_core_concepts.rdf', pickle_name="broadest_concepts.pickle")
    request = fetch_jrcacquis(langs=['es', 'it', 'en'], data_dir=storage_path, years=[2004, 2005, 2006], cat_filter=cat_list, cat_threshold=30)

    print("request length: %d" % len(request))

    #number of effective categories
    categories = set()
    for d in request:
        categories = categories | set(d.categories)

    print("Effective categories: %d" % len(categories))




