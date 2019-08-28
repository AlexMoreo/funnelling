# Funnelling Code
Funnelling is a new _ensemble_ method for _heterogeneous transfer learning_. 
The present Python implementation concerns the application of Funnelling to Polylingual Text Classification (PLC). 

The two variants of Funnelling, Fun(KFCV) and Fun(TAT), are implemented by 
the [FunnellingPolylingualClassifier](./learning/learners.py) class, and instantiated by setting 
_folded_projections=k_ for Fun(KFCV) (with _k>1_ the number of folds) or _folded_projections=1_ for Fun(TAT).


This code has been used to produce all experimental results reported in the article "Esuli, A., Moreo, A., & Sebastiani, F. (2019). [Funnelling: A New Ensemble Method for Heterogeneous Transfer Learning and Its Application to Cross-Lingual Text Classification](https://dl.acm.org/citation.cfm?id=3326065). ACM Transactions on Information Systems (TOIS), 37(3), 37.". 


## Baselines

This package also contains the code implementing all baselines involved in the experimental evaluation. 
Some of these baselines may require external resources.
All baselines are implemented in the [learners.py](./learning/learners.py) script.
The list of baselines include: 
Naive, 
Lightweight Random Indexing (LRI), 
Cross-Lingual Explicit Semantic Analysis (CLESA),
Kernel Canonical Correlation Analysis (KCCA), 
Distributional Correspondence Indexing (DCI),
Poly-lingual Embeddings (MLE and MLE-LSTM), 
and UpperBound.
Among those, CLESA, KCCA, MLE, and MLE-LSTM require the following additional resources:

* CLESA: the class [CLESAPolylingualClassifier](./learning/learners.py) requires a processed version of a [Wikipedia dump](https://dumps.wikimedia.org/); see section **Datasets** for more information. 
* KCCA: the class [KCCAPolylingualClassifier](./learning/learners.py) also requires a processed version of Wikipedia. KCCA is built on top of a wrapper of [pyrcca](https://github.com/gallantlab/pyrcca) 
from the article [Regularized kernel canonical correlation analysis in Python](https://www.frontiersin.org/articles/10.3389/fninf.2016.00049/full).
If you intend to run KCCA you might first fork the aforementioned project and make it accessible at the root of this project. 
* MLE: the class [PolylingualEmbeddingsClassifier](./learning/learners.py) uses the multilingual embeddings from the article 
[Word Translation without Parallel Data](https://arxiv.org/abs/1710.04087) which can be downloaded from the [MUSE](https://github.com/facebookresearch/MUSE) repo.
* MLE-LSTM: is implemented in [LSTMclassifierKeras.py](./learning/LSTMclassifierKeras.py) and requires:
    * The availability of the polylingual embeddings (as in MLE).
    * A [Keras](https://keras.io/) installation.
    

## Datasets

The datasets we used to run our experiments include:
* RCV1/RCV2: a _comparable_ corpus of Reuters newstories
* JRC-Acquis: a _parallel_ corpus of legislative texts of the European Union

The datasets need to be built before running any experiment.
This process requires _downloading_, _parsing_, _preprocessing_, _splitting_, and _vectorizing_.
The datasets we generated and used in our experiments can be directly downloaded (in vector form) from [here](http://hlt.isti.cnr.it/funnelling/).
Note that some methods (e.g., the PLE and PLE-LSTM methods) might require the original documents in raw form, which we are not allowed to distribute.
The tools we used in order to build the datasets are also available in this repo, and are explained below 
(feel free to skip reading if you are ok with the pre-built version).

The dataset generation relies on [NLTK](http://www.nltk.org/) for text preprocessing. 
Make sure you have NLTK installed and you have downloaded the packages needed for enabling stopword removal 
and stemming (via SnowballStemmer) before building the datasets.

A multilingual dump of the _Wikipedia_ is required during the generation of the datasets for the CLESA and KCCA 
baselines (see section **Baselines**). 
If you are not interested in running CLESA or KCCA, you can simply omit this requirement by setting max_wiki=0 before
running the script.
If otherwise, you would have to go through the [documentation](./data/reader/wikipedia_tools.py) which 
contains some tools and explanations on how to prepare the Wikipedia dump (you might require external tools).

We adapted the [Wikipedia_Extractor](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor) to extract a comparable
set of documents for all of the 11 languages involved in our experiments. 
Technical details and ad-hoc tools might be found in [wikipedia_tools.py](./data/reader/wikipedia_tools.py) (in this repo).
The toolkit allows:
* Simpliying the (huge) json dump file
* Processing the json file as a stream and filter out documents not satisfying certain conditions (e.g., do not have a view for all of the specified languages).
* Extract clean versions of documents (see the Wikipedia_Extractor for more information)
* Create multilingual maps of comparable documents, and pickle them for faster usage. 
 
The dataset splits are built once for all using the [dataset_builder.py](./dataset_builder.py) script and
then pickled for fast subsequent runs.
JRC-Acquis is automatically donwloaded the first time.
RCV1/RCV2, despite being public, cannot be downloaded without a formal permission. 
Please, refer to [RCV1's site](http://www.daviddlewis.com/resources/testcollections/rcv1/) and 
[RCV2's site](http://trec.nist.gov/data/reuters/reuters.html) before proceeding.

Once locally available, this script preprocesses the documents, and vectorizes them. 
10 random splits are generated for experimental purposes. 
The list of ids we ended up using are accessible (in pickle format) [here](http://hlt.isti.cnr.it/funnelling/).


---
## Reproducing the Experiments

Most of the experiments were run using the script [polylingual_classification.py](polylingual_classification.py).
This script can be run with different command line arguments to reproduce all multilabel experiments (with the exception of PLE-LSTM, see below). 

Run it with _-h_ or _--help_ to show this help.

```
Usage: polylingual_classification.py [options]

Options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset=DATASET
                        Path to the multilingual dataset processed and stored
                        in .pickle format
  -m MODE, --mode=MODE  Model code of the polylingual classifier, valid ones
                        include ['fun-kfcv', 'fun-tat', 'naive', 'lri',
                        'clesa', 'kcca', 'dci', 'ple', 'upper', 'fun-mono']
  -o OUTPUT, --output=OUTPUT
                        Result file
  -n NOTE, --note=NOTE  A description note to be added to the result file
  -c, --optimc          Optimice hyperparameters
  -b BINARY, --binary=BINARY
                        Run experiments on a single category specified with
                        this parameter
  -L LANG_ABLATION, --lang_ablation=LANG_ABLATION
                        Removes the language from the training
  -f, --force           Run even if the result was already computed
  -j N_JOBS, --n_jobs=N_JOBS
                        Number of parallel jobs (default is -1, all)
  -s SET_C, --set_c=SET_C
                        Set the C parameter
  -r KCCAREG, --kccareg=KCCAREG
                        Set the regularization parameter for KCCA
  -w WE_PATH, --we-path=WE_PATH
                        Path to the polylingual word embeddings (required only
                        if --mode polyembeddings)
  -W WIKI, --wiki=WIKI  Path to Wikipedia raw documents
  --calmode=CALMODE     Calibration mode for the base classifiers (only for
                        class-based models). Valid ones are'cal' (default,
                        calibrates the base classifiers and use predict_proba
                        to project), 'nocal' (does not calibrate, use the
                        decision_function to project)'sigmoid' (does not
                        calibrate, use the sigmoid of the decision function to
                        project)
```

For example, the following command will produce the results for Fun(TAT) on the first random split 
of the RCV1/RCV2 dataset optimizing the _C_ parameter of the first-tier SVM classifiers.

```
$> python polylingual_classification.py -d "../Datasets/RCV2/rcv1-2_nltk_trByLang1000_teByLang1000_processed_run0.pickle" -o ./results.csv --mode fun-tat --optimc
```

Once the experiment is over, some results will be displayed in the standard output:

```
evaluation (n_jobs=-1)
Lang nl: macro-F1=0.540 micro-F1=0.829
Lang es: macro-F1=0.582 micro-F1=0.843
Lang fr: macro-F1=0.499 micro-F1=0.765
Lang en: macro-F1=0.528 micro-F1=0.764
Lang sv: macro-F1=0.540 micro-F1=0.775
Lang it: macro-F1=0.511 micro-F1=0.789
Lang da: macro-F1=0.490 micro-F1=0.797
Lang pt: macro-F1=0.706 micro-F1=0.879
Lang de: macro-F1=0.416 micro-F1=0.741
Averages: MF1, mF1, MK, mK [0.53464632 0.79803785 0.5088316  0.75633335]
```

The complete record of the experiment is saved in the result file, which can be consulted with _Pandas_.
For example, the following snippet will display the results for all languages:

```
import pandas as pd
results = pd.read_csv('results.csv', sep='\t')
pd.pivot_table(results, index = ['method', 'lang'], values=['microf1','macrof1','microk','macrok'])

Out[11]: 
               macrof1    macrok   microf1    microk
method  lang                                        
fun-tat da    0.490002  0.455626  0.796524  0.742877
        de    0.415858  0.394820  0.741391  0.698547
        en    0.528280  0.488883  0.764349  0.716628
        es    0.581849  0.577447  0.842697  0.823296
        fr    0.499307  0.477912  0.764876  0.704686
        it    0.510546  0.471447  0.788944  0.751368
        nl    0.540213  0.510137  0.828782  0.789040
        pt    0.705507  0.698201  0.879412  0.850810
        sv    0.540255  0.505011  0.775367  0.729749
```  

The code to run PLE-LSTM is implemented in [LSTMclassifierKeras.py](./learning/LSTMclassifierKeras.py).
Note that you need the raw version of the documents to run it (see the **Datasets** section).

Other scripts used include:
* [monolingual_classification.py](./monolingual_classification.py) runs the multilabel monolingual experiments. 
* [binary_classification.py](./binary_classification.py) runs the binary polylingual experiments.
* [crosslingual_classification.py](./crosslingual_classification.py) generates the learning curves simulating under-resourced languages
* [funemb_classification.py](./funemb_classification.py) runs experiments using Fun(TAT)-PLE


