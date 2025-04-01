import csv
from pathlib import Path
import numpy as np
import scanpy as sc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from h5 import h5_reader
     

class Data:
    def __init__(self, file_X, file_y, name, train_size, seed, cv, top_n_genes=None):
        self.file_X = str(Path(file_X))
        self.file_y = str(Path(file_y))
        self.name = name
        self.top_n_genes = top_n_genes
        self.train_size = train_size
        self.seed = seed
        
        if self.top_n_genes is None:
            self.scaler = self._log_transformation
        else:
            self.scaler = self._normalise

        self.folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    def _log_transformation(self, X):
        return np.log1p(X)

    def _normalise(self, X):
        '''
        This code is modefied based on the impelemtation by 
        https://github.com/xuebaliang/scziDesk
        '''
        adata = sc.AnnData(X)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=self.top_n_genes, subset=True)
        sc.pp.scale(adata)
        X = adata.X.astype(np.float32)

        return X 

    def _csv_reader(self, data, labels=False):
        rows = []
        with open(data, 'r') as file:
            reader = csv.reader(file)
            _ = reader.__next__()
            if labels:
                for row in reader:
                    rows.append(row)

                return np.array(rows, dtype='>30U')

            else:
                for row in reader:
                    rows.append(row[1:])

                return np.array(rows, dtype=np.float32)

    def _read(self):
        if self.file_X.endswith('.csv'):
            X = self.scaler(self._csv_reader(self.file_X))

        elif self.file_X.endswith('.h5'):
            X = self.scaler(h5_reader(self.file_X))

        else:
            raise NotImplementedError(
                f'File X type is not supported. Please use one of the following types: '\
                '.h5 or .csv'
            )
        
        if self.file_y.endswith('.csv'):
            self.labels = self._csv_reader(self.file_y, labels=True)

        else:
            raise NotImplementedError(
                f'File y type is not supported. Please use the type .csv'
            )

        assert X.shape[0] == self.labels.shape[0], f'The number of samples in file_X ({X.shape[0]}) '\
            f'does not match the number of labels in file_y ({self.labels.shape[0]})'

        self.class_dict = {c: i for i, c in enumerate(np.unique(self.labels))}
        self.y = np.array([self.class_dict[c[0]] for c in self.labels]).reshape(-1, 1)

        self.dim = X.shape[1]
        self.n = X.shape[0]
        self.n_classes = len(self.class_dict)

        return X

    def split(self):
        X = self._read()
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X, self.y, 
                                                                              train_size=self.train_size, 
                                                                              random_state=self.seed, 
                                                                              shuffle=True, 
                                                                              stratify=self.y)

