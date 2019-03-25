import matplotlib.pyplot as plt
from util.results import PolylingualClassificationResults
import pandas as pd

def min_max(evalmetrics, dataset_prefixes):
    min_,max_=1.,0.
    for d in dataset_prefixes:
        df = r.df.loc[r.df['id'].isin(['{}_run{}.pickle_{}'.format(d, run, nlang) for run in range(10) for nlang in range(12)])]
        for e in evalmetrics:
            values = pd.pivot_table(df, index=['method'], columns=['lang'], values=[e]).values
            min_=min(values.min(),min_)
            max_ = max(values.max(), max_)
    return min_,max_
    # return 0,1

def matrix_metric(ax, evalmetric, showyticks, dataset_prefix):
    evalmetricname = {'macrof1':'$F_{1}^M$', 'microf1':'$F_{1}^{\mu}$', 'macrok':'$K^M$', 'microk':'$K^{\mu}$'}

    df = r.df.loc[r.df['id'].isin(['{}_run{}.pickle_{}'.format(dataset_prefix,run,nlang) for run in range(10) for nlang in range(12)])]
    df['method']=df['method'].apply(lambda x:int(x[1:]))
    piv = pd.pivot_table(df, index=['method'], columns=['lang'], values=[evalmetric])
    print('eval={} dataset={} min={:.3f} max={:.3f}'.format(evalmetric,dataset_prefix[:3], pd.np.min(piv.values), pd.np.max(piv.values)))
    print(piv)
    cax = ax.matshow(piv, vmin=min_,vmax=max_,  cmap='RdYlGn')#cmap='RdYlGn') # check cmaps in https://matplotlib.org/examples/color/colormaps_reference.html
    # some nice ones: gist_gray, inferno o hot, and the diverging colormaps, e.g., RdYlGn
    lang_list = piv.columns.get_level_values(1).values.tolist()
    print(lang_list)
    ax.set_xticks(pd.np.arange(len(lang_list)))
    ax.set_xticklabels(lang_list)
    experiments = piv.index.values.tolist()
    if showyticks:
        # yticks = ['Tr={'+'+'.join(lang_list[:i+1])+'}' for i in range(len(lang_list))]
        def trSetRepr(i): return '\mathcal{L}_{'+str(i)+'}' if i>0 else ''
        yticks = ['$' + trSetRepr(i+1)+'='+trSetRepr(i)+('\cup' if i>0 else '')+'\{' + l + '\}$' for i,l in enumerate(lang_list)]
        ax.set_yticks(pd.np.arange(len(yticks)))
        ax.set_yticklabels( yticks)
    else:
        ax.set_yticks(pd.np.arange(len(lang_list)))
        ax.set_yticklabels(['']*(len(lang_list)))
        pass
    ax.set_title(evalmetricname[evalmetric]+'\n')
    return cax


r = PolylingualClassificationResults('../results_funneling_embedding.csv')
datasets = ['jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed','rcv1-2_doclist_trByLang1000_teByLang1000_processed']
# datasets = ['rcv1-2_doclist_trByLang1000_teByLang1000_processed']
# datasets = ['jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed']
metrics = ['microf1','macrof1','microk','macrok']
# metrics = ['microf1']

for dataset in datasets:
    for metric in metrics:
# dataset = datasets[0]
# metric = metrics[0]

        min_,max_=min_max([metric],[dataset])
        # min_=0
        # max_=1.

        print('min={:.3f} max={:.3f}'.format(min_,max_))

        # fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
        fig, axes = plt.subplots(1, 1, figsize=(8, 4), subplot_kw={'xticks': [], 'yticks': []})


        # for dataset_prefix in datasets:

        showticks = (metric=='microf1')
        cax = matrix_metric(axes, metric, showticks, dataset)

        # for i,(ax,metric,dataset_prefix) in enumerate(zip(axes.flat,metrics*len(datasets), datasets[:1]*len(metrics)+datasets[1:]*len(metrics))):
        #     showticks = (i%len(metrics)==0)
        #     cax = matrix_metric(ax, metric, showticks, dataset_prefix)


        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.5, right=0.8,
                             wspace=0.02, hspace=0.02)

        cb_ax = fig.add_axes([0.83, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(cax, cax=cb_ax)

        # set the colorbar ticks and tick labels
        xticks = pd.np.arange(min_, max_, 0.05)
        cbar.set_ticks(xticks)
        cbar.set_ticklabels(list(map('{:.3f}'.format,xticks.tolist())))


        # plt.show()
        plt.savefig('{}_{}.pdf'.format(dataset[:3],metric))