from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import torch
import torch.nn.functional as F

from networks import Encoder


def get_assigned_labels_kmeans(train_X, test_X, n_clusters):
    train_X = F.normalize(train_X, dim=1).detach().numpy()
    test_X = F.normalize(test_X, dim=1).detach().numpy()
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100).fit(train_X)

    return kmeans.predict(test_X).flatten()


def get_assigned_labels_centroids(train_X, train_y, test_X):
    train_centroids = []
    train_X = train_X.detach().numpy()
    test_X = test_X.detach().numpy()
    
    for y in np.unique(train_y):
        cluster = train_X[train_y == y]
        train_centroids.append(cluster.mean(axis=0))

    train_centroids = np.vstack(train_centroids)
    dist = pairwise_distances(test_X, train_centroids)

    return dist.argmin(axis=1).flatten()


def load_pretrained_encoder(fold, method, metric, dataset, dim, enc_path):
    enc_kwargs = {
            'enc_in_dim': dim,
            'enc_dim': 1024,
            'enc_out_dim': 512,
            'proj_dim': 256,
            'proj_out_dim': 128,
            'dropout': 0.9 if 'RGMRCL' in method else None
        }
    enc = Encoder(**enc_kwargs)
    checkpoint = torch.load(
        Path(
            enc_path, f'{method}--{dataset}--{fold}--{metric}_checkpoint.pt'
        ), map_location=torch.device('cpu')
    )
    enc.load_state_dict(checkpoint)
    enc.eval()

    return enc