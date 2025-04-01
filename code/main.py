import argparse
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from data import Data
from utils import get_assigned_labels_kmeans, get_assigned_labels_centroids
from utils import load_pretrained_encoder


def main(args):
    if args.method not in ['Sup-GsRCL', 'Sup-RGMRCL-5000', 'Self-GsRCL', 'Self-RGMRCL-3000']:
        raise NotImplementedError('Please use one of the following methods: Sup-GsRCL, Sup-RGMRCL-5000, Self-GsRCL, or Self-RGMRCL-3000')

    torch.manual_seed(args.seed)
    top_n_genes = 3000 if args.method == 'Self-RGMRCL-3000' else 5000 if args.method == 'Sup-RGMRCL-5000' else None
    data = Data(args.X, args.y, args.name, args.train_size, args.seed, args.cv, top_n_genes)
    data.split()

    if args.eval == 'distributions':
        results = pd.DataFrame(columns=['Dataset', 'Method', 'metric', 'fold', 'ari', 'nmi', 's', 'ch'])
        results = results.astype({'ari': 'float32', 'nmi': 'float32', 's': 'float32', 'ch': 'float32'})

    elif args.eval == 'centroids':
        results = pd.DataFrame(columns=['Dataset', 'Method', 'metric', 'fold', 'mcc', 'f1', 'acc'])
        results = results.astype({'mcc': 'float32', 'f1': 'float32', 'acc': 'float32'})

    else:
        raise NotImplementedError('The evaluation approach is not implemented. Please select on of the following options: distributions or centroids')
    
    idx = 0
    for fold, (train_fold, test_fold) in enumerate(data.folds.split(data.train_X, data.train_y)):
        train_X = torch.tensor(data.train_X[train_fold], dtype=torch.float32)
        train_y = data.train_y[train_fold].flatten()
        test_X = torch.tensor(data.train_X[test_fold], dtype=torch.float32)
        test_y = data.train_y[test_fold].flatten()

        encoder = load_pretrained_encoder(fold, args.method, args.metric, args.name, data.dim, args.enc_path)

        with torch.no_grad():
            train_h, _ = encoder(train_X)
            test_h, _ = encoder(test_X)

        if args.eval == 'distributions':
            assigned_labels = get_assigned_labels_kmeans(train_h, test_h, data.n_classes)
            ari = adjusted_rand_score(test_y, assigned_labels)
            nmi = normalized_mutual_info_score(test_y, assigned_labels)
            s = silhouette_score(test_h.detach().numpy(), assigned_labels)
            ch = calinski_harabasz_score(test_h.detach().numpy(), assigned_labels)
            results.loc[idx, :] = [args.name, args.method, args.metric, fold, ari, nmi, s, ch]
            idx += 1

        elif args.eval == 'centroids':
            assigned_labels = get_assigned_labels_centroids(train_h, train_y, test_h)
            mcc = matthews_corrcoef(test_y, assigned_labels)
            f1 = f1_score(test_y, assigned_labels, average='macro')
            acc = accuracy_score(test_y, assigned_labels)
            results.loc[idx, :] = [args.name, args.method, args.metric, fold, mcc, f1, acc]
            idx += 1

    results.to_csv(Path(args.res_path, 'results.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--X', help='the full path to file X (i.e. genes expression matrix')
    parser.add_argument('--y', help='the full path to file y (i.e. cell types annotations)')
    parser.add_argument('--name', help='the dataset name')
    parser.add_argument('--eval', help='the evaluation approach. Please enter one of the following options: distributions or centroids')
    parser.add_argument('--method', help='the method name. Please enter one of the following methods: Sup-GsRCL, Sup-RGMRCL-5000, Self-GsRCL, or Self-RGMRCL-3000')
    parser.add_argument('--metric', help='the evaluation metric used for model selection. Please enter one of the following metrics: ari or nmi')
    parser.add_argument('--train_size', type=float, default=0.8, help='the training set size when splitting the data')
    parser.add_argument('--cv', type=int, default=5, help='the number of cross validation folds')
    parser.add_argument('--seed', type=int, default=1111, help='seed')
    parser.add_argument('--res_path', help='the path where the results are saved')
    parser.add_argument('--enc_path', help='the path where the pretrained encoder is saved')

    args = parser.parse_args()
    main(args)