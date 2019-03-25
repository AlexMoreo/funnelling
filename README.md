# Funnelling Code
Funnelling is a new _ensemble_ method for _heterogeneous transfer learning_. 
The present Python implementation concerns the application of Funnelling to Polylingual Text Classification (PLC). 
This code has been used to produce all experimental results reported in the (_forthcoming_) article. 
Please, access (_forthcoming link_) for a description of the method, which is not described here.

* Fun(KFCV)
* Fun(TAT)

## Baselines

This package not only contains the code implementing Funnelling but also all baselines involved in the
experimental evaluation. Some of them may require additional resources, or are actually wrappers of external implementations.
All baselines are implemented in the [learners.py](./learning/learners.py) script.
The list of baselines include:
* Naive
* Lightweight Random Indexing (LRI)
* Cross-Lingual Explicit Semantic Analysis (CLESA)
* Kernel Canonical Correlation Analysis (KCCA): 
the x is a wrapper of the project [pyrcca](https://github.com/gallantlab/pyrcca) 
from the article [Regularized kernel canonical correlation analysis in Python](https://www.frontiersin.org/articles/10.3389/fninf.2016.00049/full)
* Distributional Correspondence Indexing (DCI)
* Poly-lingual Embeddings
    * Averaged (PLE)
    * LSTM (PLE-LSTM)
* UpperBound
    

## Datasets

The datasets we used to run our experiments include:
* RCV1/RCV2: a _comparable_ corpus of Reuters newstories
* JRC-Acquis: a _parallel_ corpus of legislative texts of the European Union

The dataset generation relies on [NLTK](http://www.nltk.org/) for text preprocessing. 
Make sure you have NLTK installed and you have downloaded the packages needed for enabling stopword removal 
and stemming (via SnowballStemmer) before building the datasets.

A multilingual dump of the _Wikipedia_ is required during the generation of the datasets for the CLESA 
baseline (see section **Baselines**). 
If you are not interested in running CLESA, you can simply omit this requirement by setting max_wiki=0 before
running the script.
If otherwise, you would have to go through the [documentation](./data/reader/wikipedia_tools.py) which 
contains some tools and explanations on how to prepare the Wikipedia dump (you might require external tools). 
 
These dataset splits are built once for all using the [dataset_builder.py](./dataset_builder.py) script and
then pickled for fast subsequent runs.
JRC-Acquis is automatically donwloaded the first time.
RCV1/RCV2, despite being public, cannot be downloaded without a formal permission. 
Please, refer to [RCV1's site](http://www.daviddlewis.com/resources/testcollections/rcv1/) and 
[RCV2's site](http://trec.nist.gov/data/reuters/reuters.html) before proceeding.

Once locally available, this script preprocesses the documents, and creates the matrix versions. 
10 random splits are generated for experimental purposes. 
The list of ids we ended up using are accessible (in pickle format) [here](./doc_ids.zip).


---
## Reproducing the Experiments
### Multilabel PLC
### Multilabel monolingual and binary polylingual
### Learning curves for under-resourced languages
### Contribution vs. Benefit
### Importance of calibration
### Funnelling in Cross-Lingual Zero-Shot mode

