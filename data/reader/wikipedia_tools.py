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
from util.file import list_dirs, list_files
from itertools import islice
import re


policies = ["IN_ALL_LANGS", "IN_ANY_LANG"]

def doc_generator(text_path, langs):
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
                            yield doc_lines, lang
                        else: break

def extract_title(doc_lines):
    m = re.search('title="(.+?)"', doc_lines[0])
    if m: return m.group(1).decode('utf-8')
    else: raise ValueError("Error in xml format: document head is %s" % doc_lines[0])

def create_doc(target_file, id, doc, lang):
    doc[0] = doc[0][:-2] + (' lang="%s">\n'%lang)
    with open(target_file, 'w') as fo:
        fo.write('<multidoc id="%s">\n'%id)
        [fo.write(line) for line in doc]
        fo.write('</multidoc>')

def append_doc(target_file, doc, lang):
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
    print docs_created, "multilingual documents found."
    for doc,lang in doc_generator(text_path, langs):
        title = extract_title(doc)

        if title in inv_dict[lang]:
            ids = inv_dict[lang][title]
            for id in ids:
                target_file = join(out_path, id)
                if os.path.exists(target_file):
                    append_doc(target_file, doc, lang)
                else:
                    create_doc(target_file, id, doc, lang)
                    docs_created+=1
    print "Multilingual documents",docs_created

def extract_multilingual_titles(latest_all_json_file, pickle_dir, langs, policy="IN_ALL_LANGS", return_both=True):
    if policy not in policies:
        raise ValueError("Policy %s not supported." % policy)
    else:
        print("extracting multilingual titles with policy %s (%s)" % (policy,' '.join(langs)))

    pickle_dict = join(pickle_dir,"multi_dict.pickle")
    pickle_invdict = join(pickle_dir, "multi_invdict.pickle")
    if os.path.exists(pickle_invdict):
        if return_both and os.path.exists(pickle_dict):
            print("Pickled files found in %s. Loading both (direct and inverse dictionaries)." % pickle_dir)
            return pickle.load(open(pickle_dict, 'rb')), pickle.load(open(pickle_invdict, 'rb'))
        elif return_both==False:
            print("Pickled file found in %s. Loading inverse dictionary only." % pickle_dir)
            return pickle.load(open(pickle_invdict, 'rb'))

    else:
        multiling_titles = {}
        inv_dict = {lang:{} for lang in langs}

        def process_entry(last):
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

        with BZ2File(latest_all_json_file, 'r', buffering=1024*1024*16) as fi:
            builder = ObjectBuilder()
            completed = 0
            for event, value in ijson.basic_parse(fi, buf_size=1024*1024*16):
                builder.event(event, value)
                if len(builder.value)>1:
                    process_entry(builder.value.pop(0))

                    completed += 1
                    if completed % 500==0:
                        print("Completed %d" % completed)

            #process the last entry
            process_entry(builder.value.pop(0))

            print("Pickling dictionaries in %s" % pickle_dir)
            pickle.dump(multiling_titles, open(pickle_dict,'wb'), pickle.HIGHEST_PROTOCOL)
            pickle.dump(inv_dict, open(pickle_invdict, 'wb'), pickle.HIGHEST_PROTOCOL)
            print "Done"

        return (multiling_titles, inv_dict) if return_both else inv_dict

def split_dictionaries(pickle_path):
    print "Loading dictionaries..."
    multi_dict, inv_dict = pickle.load(open(pickle_path, 'rb'))
    print "Done."
    print "Writting multi_dict"
    pickle.dump(multi_dict, open(pickle_path.replace('.pickle','.dict.pickle'),'wb'), pickle.HIGHEST_PROTOCOL)
    multi_dict = []
    print "Writting inv_dict"
    pickle.dump(inv_dict, open(pickle_path.replace('.pickle', '.inv_dict.pickle'), 'wb'), pickle.HIGHEST_PROTOCOL)
    print "Done."


if __name__ == "__main__":

    #storage_path = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia"
    storage_path = "/home/data/wikipedia/dumps"
    #storage_path = "/Users/moreo/cl-esa-p/storage"
    json_file = "latest-all.json.bz2"
    latest_all_json_file = join(storage_path, json_file)
    policy = "IN_ALL_LANGS"

    from jrcacquis_reader import LANGS_WITH_NLTK_STEMMING as langs
    langs = frozenset(langs)
    lang_prefix = list(langs)
    lang_prefix.sort()
    out_dir = "extraction_" + "_".join(lang_prefix) + "." + policy

    pickle_dir = join(storage_path, out_dir)
    multi_dict, inv_dict = extract_multilingual_titles(latest_all_json_file, pickle_dir, langs, policy)

    extract_multilingual_documents(inv_dict, ['es','it','en'], join(storage_path,'text'), join(storage_path,'multilingual_docs'))
