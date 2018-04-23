from dataset_builder import MultilingualDataset
import numpy as np
import matplotlib.pyplot as plt



def get_visualization(dataset, binary=False, mode='tr'):
    data = MultilingualDataset.load(dataset)


    lytr = data.lYtr() if mode=='tr' else data.lYte()
    langs = data.langs()
    nL = len(langs)
    nC = data.num_categories()
    print(nC)

    prev = np.zeros((nC, nL))

    for c in range(nC):
        for l,li in enumerate(langs):
            prev[c, l] = np.sum(lytr[li][:, c])
    if binary:
        threshold = 1
        prev[prev < threshold] = 0
        prev[prev >= threshold] = 1


    return prev

rcv_path = '/media/moreo/1TB Volume/Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processed_run1.pickle'
jrc_path = '/media/moreo/1TB Volume/Datasets/JRC_Acquis_v3/jrc_nltk_1958-2005vs2006_all_top300_noparallel_processed_run1.pickle'

prev_rcv = get_visualization(rcv_path, binary=True, mode='tr')
prev_rcv_te = get_visualization(rcv_path, binary=True, mode='te')
prev_jrc = get_visualization(jrc_path, binary=True, mode='tr')
prev_jrc_te = get_visualization(jrc_path, binary=True, mode='te')

from matplotlib import gridspec
fig, (ax1, ax2) = plt.subplots(2,1)#, gridspec_kw=gridspec.GridSpec(1, 2, width_ratios=[3, 1]) )
ax1.imshow(prev_rcv.T, cmap='hot', interpolation='nearest')
ax1.set_title('RCV1/2')

ax2.imshow(prev_jrc.T, cmap='hot', interpolation='nearest')
ax2.set_title('JRC-Acquis')

plt.show()





