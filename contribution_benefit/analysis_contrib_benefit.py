import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

#filename = './ablation_tr10perc_jrc_results.csv'
filename = './ablation_tr10perc_rcv_results.csv'

def rel_performance(m1, relative_to):
    return 100*(relative_to-m1)/relative_to

def standardize(m):
    return (m-np.min(m))/(np.max(m)-np.min(m))

df = pd.read_csv(filename, sep='\t', index_col=0)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=500, formatter={'float':lambda x:'%.3f'%x})

print(df)

def report_line(averages):
    # print(table.columns.levels[1].values.tolist())
    # print(averages)
    lang_order_pandas = {l: i for i, l in enumerate(table.columns.levels[1].values.tolist())}
    paper_order = ['en', 'it', 'es', 'fr', 'de', 'sv', 'da', 'pt', 'nl', 'fi', 'hu']
    reordered_contrib = np.array([averages[lang_order_pandas[l]] for l in paper_order if l in lang_order_pandas])
    # print(paper_order)
    return ' & '.join([eval] + [
        '%s%s\%%' % ('+' if x > 0 else '', '\\cellcolor[gray]{.8}\\textbf{%.2f}' % x if x == reordered_contrib.max() else '%.2f' % x) for x
        in reordered_contrib.tolist()])

contrib_report=[]
benefit_report=[]
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, squeeze=False)
ax = [ax1,ax2,ax3,ax4]
for axi,eval in enumerate(['microf1', 'macrof1','microk','macrok']):
    #the field binary has been used to show the ablated lang ...
    table = pd.pivot_table(df, values=[eval], fill_value=np.nan,
                            index=['method', 'binary'], columns=['lang'], aggfunc=np.mean)

    # print(table)
    monolingual10 = table[eval].loc['monolingual'].values # at 10% to measure its benefit
    monolingual100 = table[eval].loc['naive'].values # at 100% to measure its contribution
    funtat = table[eval].loc['fun(tat)-all-langs'].values
    postprob100 = table[eval].loc['Fun(tat)-PostPrtrsoft'].values
    postprob10 = table[eval].loc['Fun(tat)-PostPr10trsoft'].values
    ablated = table[eval].loc['fun(tat)-ablated'].values

    ablated_rel = rel_performance(ablated, funtat)

    monoabs10 = monolingual10
    monoabs100 = monolingual100
    #mono = rel_performance(monolingual10, funtat).squeeze()
    postprob100 = rel_performance(postprob100, funtat).squeeze()
    postprob10 = rel_performance(postprob10, funtat).squeeze()
    contrib = np.nanmean(ablated_rel,axis=1)
    benefit = np.nanmean(ablated_rel,axis=0)

    measurements = OrderedDict()
    measurements['Mono10%'] = monoabs10.squeeze()
    measurements['Mono100%'] = monoabs100.squeeze()
    measurements['PostPr10%'] = postprob10.squeeze()
    measurements['PostPr100%'] = postprob100.squeeze()
    #measurements['Impr(Fun-mon)']=mono
    #measurements['Impr(Fun-PPr)']=postprob
    measurements['Contribution']=contrib
    measurements['Benefit']=benefit

    measures = list(measurements.keys())
    print(eval+'\n'+'-'*80)
    for i,m1 in enumerate(measures):
        for m2 in measures[i+1:]:
            if m1.replace('10%','')==m2.replace('100%',''): continue
            r,pval = pearsonr(measurements[m1], measurements[m2])
            if abs(r)>0.5:
                print('%s vs. %s\tR=%.3f (p-val %.5f)\tas %s increases %s %s'%(m1,m2,r,pval, m1, m2, 'increases' if r>0 else 'decreases'))
                ax[axi].plot(standardize(measurements[m1]),standardize(measurements[m2]),'o',label='%s vs. %s R=%.3f'%(m1,m2,r))
    print()
    ax[axi].set_title(eval)
    ax[axi].legend()

    contrib_report.append(report_line(contrib))
    benefit_report.append(report_line(benefit))
#plt.show()

print('CONTRIBUTION')
for r in contrib_report:
    print(r)

print('BENEFIT')
for r in benefit_report:
    print(r)

