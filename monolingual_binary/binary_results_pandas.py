import pandas as pd
import numpy as np
from util.metrics import ContTable, f1, K

csv_path = './monolingualbinary_results.csv'
# dataset = 'JRCacquis'
dataset = 'RCV1/2'

assert dataset in ['RCV1/2', 'JRCacquis'], 'wrong dataset'

languages = ['da', 'de', 'en', 'es', 'fr', 'it', 'nl', 'pt', 'sv'] if dataset == 'RCV1/2' else ['da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'nl', 'pt', 'sv']


csv = pd.read_csv(csv_path, sep='\t')

def get_contingency_table(counter_names, counter_values):
    c = ContTable()
    for i, cn in enumerate(counter_names):
        if cn == 'TP':
            c.tp = counter_values[i]
        elif cn == 'TN':
            c.tn = counter_values[i]
        elif cn == 'FP':
            c.fp = counter_values[i]
        elif cn == 'FN':
            c.fn = counter_values[i]
    return c

def compute_micro(dataset):
    table = pd.pivot_table(csv, values=['TP', 'TN', 'FP', 'FN'],
                           index=['dataset', 'run', 'lang'],
                           columns=[], aggfunc=np.sum)
    table = table.ix[dataset]

    #print(csv)
    #print(table)

    global_micro_f1_ave = []
    global_micro_K_ave = []
    for run in range(10):
        microf1_aveByLang = []
        microK_aveByLang = []
        for lang in languages:
            counter_names = table.ix[run, lang].index.values
            counter_values = table.ix[run, lang].values
            c = get_contingency_table(counter_names, counter_values)

            microf1 = f1(c)
            microk = K(c)
            microf1_aveByLang.append(microf1)
            microK_aveByLang.append(microk)
            #print(run, lang, microf1, microk)
        microf1_aveByLang = np.mean(microf1_aveByLang)
        microK_aveByLang = np.mean(microK_aveByLang)
        global_micro_f1_ave.append(microf1_aveByLang)
        global_micro_K_ave.append(microK_aveByLang)
        print(run, microf1_aveByLang, microK_aveByLang)
    print('MicroAVE {:.3f}+-{:.3f}\t{:.3f}+-{:.3f}'.format(np.mean(global_micro_f1_ave),np.std(global_micro_f1_ave), np.mean(global_micro_K_ave),np.std(global_micro_K_ave)))

def compute_macro(dataset):
    table = pd.pivot_table(csv, values=['TP', 'TN', 'FP', 'FN'],
                           index=['dataset', 'run', 'lang', 'binary'],
                           columns=[], aggfunc=np.sum)
    table = table.ix[dataset]

    #print(csv)
    #print(table)

    global_macro_f1_ave = []
    global_macro_K_ave = []
    for run in range(10):
        macrof1_aveByLang = []
        macroK_aveByLang = []
        for lang in languages:
            for cat in table.ix[run, lang].index.values:
                counter_names = table.ix[run, lang, cat].index.values
                counter_values = table.ix[run, lang, cat].values
                c = get_contingency_table(counter_names, counter_values)

                microf1 = f1(c)
                microk = K(c)
                macrof1_aveByLang.append(microf1)
                macroK_aveByLang.append(microk)
            #print(run, lang, microf1, microk)
        macrof1_aveByLang = np.mean(macrof1_aveByLang)
        macroK_aveByLang = np.mean(macroK_aveByLang)
        global_macro_f1_ave.append(macrof1_aveByLang)
        global_macro_K_ave.append(macroK_aveByLang)
        print(run, macrof1_aveByLang, macroK_aveByLang)
    print('MacroAVE {:.3f}+-{:.3f}\t{:.3f}+-{:.3f}'.format(np.mean(global_macro_f1_ave),np.std(global_macro_f1_ave), np.mean(global_macro_K_ave),np.std(global_macro_K_ave)))

compute_macro(dataset)
compute_micro(dataset)
print('DATASET {}'.format(dataset))