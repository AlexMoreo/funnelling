import ijson
import os, sys
from os.path import join
from bz2 import BZ2File
from StringIO import StringIO
from ijson.common import parse
from ijson.common import ObjectBuilder
from ijson.common import items
import cPickle as pickle


storage_path = "/media/moreo/1TB Volume/Datasets/Multilingual/Wikipedia"

json_file = "latest-all.json.bz2"

fullpath = join(storage_path,json_file)


#import json
#j = json.load(BZ2File(fullpath, 'r', buffering=1024*1024))

#print "End"
#sys.exit()

#langs = frozenset(["es", "en", "it"])

from jrcacquis_reader import LANGS
langs = frozenset(LANGS)

lang_prefix = list(langs)
lang_prefix.sort()
out_file  = "extraction"+"_".join(lang_prefix)+".pickle"
fullpath_pickle = join(storage_path, out_file)



if os.path.exists(fullpath_pickle):
    print("Pickled file found in %s. Loading it." % fullpath_pickle)
    (multiling_titles, inv_dict) = pickle.load(open(fullpath_pickle,'rb'))
    print "Loaded"

else:
    multiling_titles = {}
    inv_dict = {lang:{} for lang in langs}

    def process_entry(last):
        id = last["id"]
        if langs.issubset(last["labels"].keys()):
            titles = {lang: last["labels"][lang]["value"] for lang in langs}
            if id in multiling_titles:
                raise ValueError("id <%s> already indexed" % id)

            multiling_titles[id] = titles
            for lang, title in titles.items():
                if title in inv_dict[lang]:
                    inv_dict[lang][title].append(id)
                inv_dict[lang][title] = [id]

    with BZ2File(fullpath, 'r',buffering=1024*1024) as fi:
        parser = ijson.parse(fi)
        builder = ObjectBuilder()
        completed = 0
        for event, value in ijson.basic_parse(fi):
            builder.event(event, value)
            if len(builder.value)>1:
                process_entry(builder.value.pop(0))

                completed += 1
                if completed % 500==0:
                    print("Completed %d" % completed)
                #if completed == 100:
                #
                    #     break

        #process the last entry
        #process_entry(builder.value.pop(0))

        print("Pickling dictionaries in %s" % fullpath_pickle)
        pickle.dump((multiling_titles,inv_dict), open(join(storage_path,out_file),'wb'), pickle.HIGHEST_PROTOCOL)
        print "Done"

