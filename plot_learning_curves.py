import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

filename='/home/moreo/CLESA/cl-esa-p/results_crosslingual_rcv_optimc.csv'
#filename='/home/moreo/CLESA/cl-esa-p/crosslingual_optimc_results_jrc.csv' # <-- este es el fichero que genera
#filename='/home/moreo/CLESA/cl-esa-p/results_crosslingual_jrc_optimc.csv' # <-- aqui he mezclado lo de local (R0) + ilona (R9)

filename__= filename+'__'

#preprocess the file to get the run values from the ids
with open(filename__,'w') as fo:
    header = True
    for line in open(filename, 'r').readlines():
        print(line)
        if not line.strip(): continue
        if not header:
            lineparts = line.split('\t')
            id = lineparts[0]
            runsplit = id.find('_run') + len('_run')
            run = id[runsplit:runsplit+1]
            print(run)
            line = '\t'.join([run]+lineparts[1:]) + '\n'
            print(line)
        else:
            line=line.replace('id\t','run\t').replace('languages\t','tr_prop\t')
        fo.write(line)
        header = False
    fo.write('\n')

df = pd.read_csv(filename__, sep='\t')
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=500, formatter={'float':lambda x:'%.3f'%x})

print(df)


eval = 'macrof1'


table_mu = pd.pivot_table(df, values=[eval], fill_value=np.nan,
                            index=['method', 'lang'], columns=['tr_prop'],aggfunc=np.mean)
table_std = pd.pivot_table(df, values=[eval], fill_value=np.nan,
                           index=['method', 'lang'], columns=['tr_prop'], aggfunc=np.std)

naive = table_mu[eval].loc['naive']
funtat = table_mu[eval].loc['class']

print(naive)

method = 'class'

mu = table_mu[eval].loc[method]
std = table_std[eval].loc[method]

paper_order = ['en', 'it', 'es', 'fr', 'de', 'sv', 'da', 'pt', 'nl', 'fi', 'hu']
df_langs = frozenset(table_mu.index.get_level_values('lang').values.tolist())
tr_props = table_mu.columns.get_level_values('tr_prop').values
print(df_langs)
print(tr_props)
# plt.figure()
# for lang in paper_order:
#     if lang in df_langs:
#         plt.errorbar(tr_props, mu.loc[lang], yerr=std.loc[lang], fmt='--o', label=lang)

# plt.title(method)
# plt.xlabel('proportion of training examples')
# plt.ylabel(eval)
# plt.legend()
# plt.grid(True)
# plt.show()

#sys.exit()




#this part makes the plot for rel_improvement
table = pd.pivot_table(df, values=[eval], fill_value=np.nan,
                             index=['method', 'lang', 'run'], columns=['tr_prop'], aggfunc=np.mean)

#for run in

naive = table.loc['naive']
fun = table.loc['class']

rel_impr = 100*(fun- naive)/(fun+1e-6)

print(rel_impr)
from scipy.interpolate import spline

means = rel_impr.mean(level=0)
std = rel_impr.std(level=0)

paper_order = ['en', 'it', 'es', 'fr', 'de', 'sv', 'da', 'pt', 'nl', 'fi', 'hu']
df_langs = frozenset(table.index.get_level_values('lang').values.tolist())
tr_props = table.columns.get_level_values('tr_prop').values
print(df_langs)
print(tr_props)
plt.figure(figsize=(10, 5))
for lang in paper_order:
    if lang in df_langs:
        #plt.errorbar(tr_props, means.loc[lang], yerr=std.loc[lang],fmt='--o', label=lang)
        #plt.errorbar(tr_props, means.loc[lang], fmt='-o', label=lang)
        xnew = np.linspace(0, 1, 301)  # 300 represents number of points to make between T.min and T.max
        smooth = spline(tr_props, means.loc[lang], xnew)
        plt.errorbar(xnew, smooth, fmt='-', label=lang, markevery=30, marker='o')
        #plt.errorbar(tr_props, means.loc[lang], fmt='o')

# print(means)
# print(std)

#plt.title('learning curves')
plt.xlabel('proportion of training examples')
plt.ylabel('% relative improvement in '+eval.replace('acro','acro-').title())
plt.legend()
plt.grid(True)

plt.show()
