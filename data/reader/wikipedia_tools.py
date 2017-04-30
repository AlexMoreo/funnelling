import ijson
import os, sys
from os.path import join
from bz2 import BZ2File
from StringIO import StringIO
from ijson.common import parse
from ijson.common import ObjectBuilder
from ijson.common import items
import cPickle as pickle

policies = ["IN_ALL_LANGS", "IN_ANY_LANG"]

def extract_multilingual_titles(latest_all_json_file, fullpath_pickle, langs, policy="IN_ALL_LANGS"):
    if policy not in policies:
        raise ValueError("Policy %s not supported." % policy)

    if os.path.exists(fullpath_pickle):
        print("Pickled file found in %s. Loading it." % fullpath_pickle)
        return pickle.load(open(fullpath_pickle,'rb'))

    else:
        multiling_titles = {}
        inv_dict = {lang:{} for lang in langs}

        def process_entry(last):
            id = last["id"]
            if id in multiling_titles:
                raise ValueError("id <%s> already indexed" % id)

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

        with BZ2File(latest_all_json_file, 'r', buffering=1024*1024) as fi:
            builder = ObjectBuilder()
            completed = 0
            for event, value in ijson.basic_parse(fi):
                builder.event(event, value)
                if len(builder.value)>1:
                    process_entry(builder.value.pop(0))

                    completed += 1
                    if completed % 500==0:
                        print("Completed %d" % completed)

            #process the last entry
            process_entry(builder.value.pop(0))

            print("Pickling dictionaries in %s" % fullpath_pickle)
            pickle.dump((multiling_titles,inv_dict), open(join(storage_path,out_file),'wb'), pickle.HIGHEST_PROTOCOL)
            print "Done"


if __name__ == "__main__":
    #storage_path = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia"
    storage_path = "/Users/moreo/cl-esa-p/storage"
    json_file = "latest-all.json.bz2"
    latest_all_json_file = join(storage_path, json_file)
    policy = "IN_ANY_LANG"

    from jrcacquis_reader import LANGS
    langs = frozenset(LANGS)
    lang_prefix = list(langs)
    lang_prefix.sort()
    out_file = "extraction_" + "_".join(lang_prefix) + "." + policy + ".pickle"

    fullpath_pickle = join(storage_path, out_file)

    (multiling_titles, inv_dict) = extract_multilingual_titles(latest_all_json_file, fullpath_pickle, langs, policy)
    print "done"