import os
import pandas as pd
import numpy as np
import scipy

class PolylingualClassificationResults:
    def __init__(self, file, autoflush=True, verbose=False):
        self.file = file
        self.columns = ['id', 'method', 'learner', 'optimp', 'dataset', 'binary', 'languages', 'time', 'lang', 'macrof1', 'microf1', 'macrok', 'microk', 'notes']
        self.autoflush = autoflush
        self.verbose = verbose
        if os.path.exists(file):
            self.tell('Loading existing file from {}'.format(file))
            self.df = pd.read_csv(file, sep='\t')
        else:
            self.tell('File {} does not exist. Creating new frame.'.format(file))
            dir = os.path.dirname(self.file)
            if dir and not os.path.exists(dir): os.makedirs(dir)
            self.df = pd.DataFrame(columns=self.columns)

    def already_calculated(self, id):
        return (self.df['id'] == id).any()

    def add_row(self, id, method, learner, optimp, dataset, binary, ablation_lang, time, lang, macrof1, microf1, macrok=np.nan, microk=np.nan, notes=''):
        s = pd.Series([id, method, learner, optimp, dataset, binary, ablation_lang, time, lang, macrof1, microf1, macrok, microk, notes], index=self.columns)
        self.df = self.df.append(s, ignore_index=True)
        if self.autoflush: self.flush()
        self.tell(s.to_string())

    def flush(self):
        self.df.to_csv(self.file, index=False, sep='\t')

    def tell(self, msg):
        if self.verbose: print(msg)
