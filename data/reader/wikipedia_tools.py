from __future__ import print_function
import ijson
import os, sys
sys.path.append("/home/moreo/CLESA/cl-esa-p")
from os.path import join
from bz2 import BZ2File
from StringIO import StringIO
from ijson.common import parse
from ijson.common import ObjectBuilder
from ijson.common import items
import cPickle as pickle
from util.file import list_dirs, list_files, makedirs_if_not_exist
from itertools import islice
import re
from xml.sax.saxutils import escape

policies = ["IN_ALL_LANGS", "IN_ANY_LANG"]

"""
This file contains a set of tools for processing the Wikipedia multilingual documents.
In what follows, it is assumed that you have already downloaded a Wikipedia dump (https://dumps.wikimedia.org/)
and have processed each document to clean their texts with one of the tools:
    - https://github.com/aesuli/wikipediatools (Python 2)
    - https://github.com/aesuli/wikipedia-extractor (Python 3)
It is also assumed you have dowloaded the all-entities json file (e.g., https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2)

This tools help you in:
    - Process the huge json file as a stream, and create a multilingual map of corresponding titles for each language.
    Set the policy = "IN_ALL_LANGS" will extract only titles which appear in all (AND) languages, whereas "IN_ANY_LANG"
    extract all titles appearing in at least one (OR) language (warning: this will creates a huge dictionary).
    Note: This version is quite slow. Although it is run once for all, you might be interested in taking a look at "Wikidata in BigQuery".
    - Use the multilingual map to extract, from the clean text versions, individual xml documents containing all
    language-specific versions from the document.
    - Fetch the multilingual documents to create, for each of the specified languages, a list containing all documents,
    in a way that the i-th element from any list refers to the same element in the respective language.
"""

def _doc_generator(text_path, langs):
    dotspace = re.compile(r'\.(?!\s)')
    for l,lang in enumerate(langs):
        print("Processing language <%s> (%d/%d)" % (lang, l, len(langs)))
        lang_dir = join(text_path, lang)
        split_dirs = list_dirs(lang_dir)
        for sd,split_dir in enumerate(split_dirs):
            print("\tprocessing split_dir <%s> (%d/%d)" % (split_dir, sd, len(split_dirs)))
            split_files = list_files(join(lang_dir, split_dir))
            for sf,split_file in enumerate(split_files):
                print("\t\tprocessing split_file <%s> (%d/%d)" % (split_file, sf, len(split_files)))
                with BZ2File(join(lang_dir, split_dir, split_file), 'r', buffering=1024*1024) as fi:
                    while True:
                        doc_lines = list(islice(fi, 3))
                        if doc_lines:
                            # some sentences are not followed by a space after the dot
                            doc_lines[1] = dotspace.sub('. ', doc_lines[1])
                            # [workaround] I found &nbsp; html symbol was not treated, and unescaping it now might not help...
                            doc_lines[1] = escape(doc_lines[1].replace("&nbsp;", " "))
                            yield doc_lines, lang
                        else: break

def _extract_title(doc_lines):
    m = re.search('title="(.+?)"', doc_lines[0])
    if m: return m.group(1).decode('utf-8')
    else: raise ValueError("Error in xml format: document head is %s" % doc_lines[0])

def _create_doc(target_file, id, doc, lang):
    doc[0] = doc[0][:-2] + (' lang="%s">\n'%lang)
    with open(target_file, 'w') as fo:
        fo.write('<multidoc id="%s">\n'%id)
        [fo.write(line) for line in doc]
        fo.write('</multidoc>')

def _append_doc(target_file, doc, lang):
    doc[0] = doc[0][:-2] + (' lang="%s">\n' % lang)
    with open(target_file, 'r', buffering=1024*1024) as fi:
        lines = fi.readlines()
    if doc[0] in lines[1::3]:
        return
    lines[-1:-1]=doc
    with open(target_file, 'w', buffering=1024*1024) as fo:
        [fo.write(line) for line in lines]

def extract_multilingual_documents(inv_dict, langs, text_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for lang in langs:
        if lang not in inv_dict:
            raise ValueError("Lang %s is not in the dictionary" % lang)

    docs_created = len(list_files(out_path))
    print("%d multilingual documents found." % docs_created)
    for doc,lang in _doc_generator(text_path, langs):
        title = _extract_title(doc)

        if title in inv_dict[lang]:
            ids = inv_dict[lang][title]
            for id in ids:
                target_file = join(out_path, id) + ".xml"
                if os.path.exists(target_file):
                    _append_doc(target_file, doc, lang)
                else:
                    _create_doc(target_file, id, doc, lang)
                    docs_created+=1
    print("Multilingual documents %d" % docs_created)

def extract_multilingual_titles(data_dir, langs, policy="IN_ALL_LANGS", return_both=True):
    json_file = "latest-all.json.bz2"  # in https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
    latest_all_json_file = join(data_dir,json_file)

    if policy not in policies:
        raise ValueError("Policy %s not supported." % policy)
    else:
        print("extracting multilingual titles with policy %s (%s)" % (policy,' '.join(langs)))

    lang_prefix = list(langs)
    lang_prefix.sort()
    pickle_prefix = "extraction_" + "_".join(lang_prefix) + "." + policy
    pickle_dict = join(data_dir, pickle_prefix+".multi_dict.pickle")
    pickle_invdict = join(data_dir, pickle_prefix+".multi_invdict.pickle")
    if os.path.exists(pickle_invdict):
        if return_both and os.path.exists(pickle_dict):
            print("Pickled files found in %s. Loading both (direct and inverse dictionaries)." % data_dir)
            return pickle.load(open(pickle_dict, 'rb')), pickle.load(open(pickle_invdict, 'rb'))
        elif return_both==False:
            print("Pickled file found in %s. Loading inverse dictionary only." % pickle_invdict)
            return pickle.load(open(pickle_invdict, 'rb'))

    else:
        multiling_titles = {}
        inv_dict = {lang:{} for lang in langs}

        def process_entry(last, fo):
            id = last["id"]
            if id in multiling_titles:
                raise ValueError("id <%s> already indexed" % id)

            titles = None
            if policy == "IN_ALL_LANGS" and langs.issubset(last["labels"].keys()):
                titles = {lang: last["labels"][lang]["value"] for lang in langs}
            elif policy == "IN_ANY_LANG":
                titles = {lang: last["labels"][lang]["value"] for lang in langs if lang in last["labels"]}

            if titles:
                multiling_titles[id] = titles
                for lang, title in titles.items():
                    if title in inv_dict[lang]:
                        inv_dict[lang][title].append(id)
                    inv_dict[lang][title] = [id]
                fo.write((id+'\t'+'\t'.join([lang+':'+multiling_titles[id][lang] for lang in titles.keys()])+'\n').encode('utf-8'))

        _open = BZ2File if latest_all_json_file.endswith(".bz2") else open
        with _open(latest_all_json_file, 'r', buffering=1024*1024*16) as fi, \
                open(join(data_dir,pickle_prefix+".simple.txt"),'w') as fo:
            builder = ObjectBuilder()
            completed = 0
            for event, value in ijson.basic_parse(fi, buf_size=1024*1024*16):
                 builder.event(event, value)
                 if len(builder.value)>1:
                    process_entry(builder.value.pop(0), fo)
                    completed += 1
                    print("\rCompleted %d\ttitles %d" % (completed,len(multiling_titles)), end="")
            print("")

            #process the last entry
            process_entry(builder.value.pop(0))

            print("Pickling dictionaries in %s" % data_dir)
            pickle.dump(multiling_titles, open(pickle_dict,'wb'), pickle.HIGHEST_PROTOCOL)
            pickle.dump(inv_dict, open(pickle_invdict, 'wb'), pickle.HIGHEST_PROTOCOL)
            print("Done")

        return (multiling_titles, inv_dict) if return_both else inv_dict

def simplify_multilingual_titles(data_dir, langs, policy="IN_ALL_LANGS"):
    json_file = "latest-all.json.bz2"  # in https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
    latest_all_json_file = join(data_dir,json_file)

    if policy not in policies:
        raise ValueError("Policy %s not supported." % policy)
    else:
        print("extracting multilingual titles with policy %s (%s)" % (policy,' '.join(langs)))

    lang_prefix = list(langs)
    lang_prefix.sort()
    simple_titles_path = join(data_dir, "extraction_" + "_".join(lang_prefix) + "." + policy)


    def process_entry(last, fo):
        global written
        id = last["id"]
        titles = None
        if policy == "IN_ALL_LANGS" and langs.issubset(last["labels"].keys()):
            titles = {lang: last["labels"][lang]["value"] for lang in langs}
        elif policy == "IN_ANY_LANG":
            titles = {lang: last["labels"][lang]["value"] for lang in langs if lang in last["labels"]}

        if titles:
            fo.write((id+'\t'+'\t'.join([lang+':'+titles[lang] for lang in titles.keys()])+'\n').encode('utf-8'))
            return True
        else:
            return False

    written = 0
    _open = BZ2File if latest_all_json_file.endswith(".bz2") else open
    with _open(latest_all_json_file, 'r', buffering=1024*1024*16) as fi, \
            BZ2File(join(data_dir,simple_titles_path+".simple.bz2"),'w') as fo:
        builder = ObjectBuilder()
        completed = 0
        for event, value in ijson.basic_parse(fi, buf_size=1024*1024*16):
             builder.event(event, value)
             if len(builder.value)>1:
                if process_entry(builder.value.pop(0), fo): written += 1
                completed += 1
                print("\rCompleted %d\ttitles %d" % (completed,written), end="")
        print("")

        #process the last entry
        process_entry(builder.value.pop(0))

"""
Reads all multi-lingual documents in a folder (see wikipedia_tools.py to generate them) and generates, for each of the
specified languages, a list contanining all its documents, so that the i-th element of any list refers to the language-
specific version of the same document. Documents are forced to contain version in all specified languages and to contain
a minimum number of words; otherwise it is discarded.
"""
def _load_multilang_doc(path, langs, min_words=100):
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import Element
    root = ET.parse(path).getroot()
    doc = {}
    for lang in langs:
        doc_body = root.find('.//doc[@lang="' + lang + '"]')
        if isinstance(doc_body, Element):
            n_words = len(doc_body.text.split(' '))
            if n_words >= min_words:
                doc[lang] = doc_body.text
            else:
                return None
        else:
            return None
    return doc

#returns the multilingual documents mapped by language, and a counter with the number of documents readed
def fetch_wikipedia_multilingual(wiki_multi_path, langs, min_words=100):
    multi_docs = list_files(wiki_multi_path)
    mling_documents = {l:[] for l in langs}
    valid_documents = 0
    for d,multi_doc in enumerate(multi_docs):
        if d % 100 == 0:
            print("\rProcessed %d/%d documents, valid %d/%d" % (d, len(multi_docs), valid_documents, len(multi_docs)), end="")
        m_doc = _load_multilang_doc(join(wiki_multi_path, multi_doc), langs, min_words)
        if m_doc:
            valid_documents += 1
            for l in langs:
                mling_documents[l].append(m_doc[l])
    print("\rProcessed %d/%d documents, valid %d/%d" % (d, len(multi_docs), valid_documents, len(multi_docs)), end="\n")

    return mling_documents, valid_documents

if __name__ == "__main__":

    #storage_path = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia"
    storage_path = "/home/data/wikipedia/dumps"
    #storage_path = "/Users/moreo/cl-esa-p/storage"


    from data.clesa_data_generator import LANGS_WITH_NLTK_STEMMING as langs
    #from jrcacquis_reader import LANGS as langs
    langs = frozenset(langs)

    #simplify_multilingual_titles(storage_path, langs, policy="IN_ANY_LANG")
    multi_dict, inv_dict = extract_multilingual_titles(storage_path, langs, policy='IN_ALL_LANGS')

    #extract_multilingual_documents(inv_dict, ['es','it','en'], join(storage_path,'text'), join(storage_path,'multilingual_docs'))

    #langs = ["en", "it", "es"]
    #wiki_multi_path = join(storage_path, "multilingual_docs")

    #m_docs = fetch_wikipedia_multilingual(wiki_multi_path, langs, min_words=100)

    #print("Read % documents" % len(m_docs[langs[0]]))
