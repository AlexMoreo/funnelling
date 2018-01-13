from dataset_builder import MultilingualDataset
import numpy as np
import scipy

dataset = '/media/moreo/1TB Volume/Datasets/JRC_Acquis_v3/jrc_nltk_1958-2005vs2006_all_top300_noparallel_processed_run1.pickle'
#dataset = '/media/moreo/1TB Volume/Datasets/JRC_Acquis_v3/jrc_nltk_1958-2005vs2006_leaves_top300_noparallel_processed_singlefragment_run1.pickle'
#for run in range(10):
#dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processed_run0.pickle'
#dataset = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processedsinglefragment_run0.pickle'
data = MultilingualDataset.load(dataset)
juxta = MultilingualDataset.load(dataset.replace('.pickle', '_yuxtaposed.pickle'))
#data.set_view(languages=['nl'])

lXtr = data.lXtr()
lXte = data.lXte()
lYtr = data.lYtr()
lYte = data.lYte()


langs = data.langs()
nFjux =juxta.lXtr()[langs[0]].shape[1]

nL = len(langs)
nC = data.num_categories()
nD = sum([data.lXtr()[lang].shape[0] for lang in langs])
nDtest = sum([data.lXte()[lang].shape[0] for lang in langs])

print(data.dataset_name)
print('nL='+str(nL)+' languages: ' + ', '.join(langs))
print('nD={} nDte={}'.format(nD, nDtest))
nd_ave, nf_ave = 0., 0.
for l in langs:
    print('\t{}: tr={} te={}'.format(l, lXtr[l].shape, lXte[l].shape))
    nd_ave += lXtr[l].shape[0]
    nf_ave += lXtr[l].shape[1]
print('ave: shape={0:.2f}, {1:.2f}, non-unique-features={2:.0f}, juxta={3:.0f}'.format(nd_ave/nL, nf_ave/nL, nf_ave - nFjux, nFjux))
print('nC={}'.format(nC))

lX, ly = data.lXtr(), data.lYtr()

#Xtr = scipy.sparse.vstack([lX[lang] for lang in langs])
Ytr = np.vstack([ly[lang] for lang in langs])
ncats = np.sum(Ytr, axis=1)
print('#cats/doc: min={0:.0f} max={1:.0f} ave={2:.2f}'.format(np.min(ncats), np.max(ncats), np.mean(ncats)))
ncats = np.sum(Ytr, axis=0)
print('#docs/cat: min={0:.0f} max={1:.0f} ave={2:.2f}'.format(np.min(ncats), np.max(ncats), np.mean(ncats)))

data.show_category_prevalences()