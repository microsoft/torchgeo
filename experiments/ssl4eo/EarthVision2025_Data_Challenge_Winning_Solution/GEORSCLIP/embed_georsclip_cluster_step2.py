import argparse
import itertools

import pandas as pd
from torch.optim import AdamW
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import torch.utils.data as tud
from scipy.cluster.hierarchy import linkage, fcluster

from dataloader import L2AZarrDataset


def cluster_by_distances(X_t: np.ndarray, distances: list[int] = [28, 11, 7, 3]) -> np.ndarray:
    """
    Clusters using hierarchical linkage at specific cut distances.

    Parameters:
        X_t: E.g., (5149, 256) numpy array (same-season embeddings)
        distances: list of cut distances from dendrogram

    Returns:
        clusters: (5149, len(distances)) matrix of cluster labels
    """
    Z = linkage(X_t, method='ward')
    clusters = np.column_stack([
        fcluster(Z, t=dist, criterion='distance') for dist in distances
    ])

    return clusters


def main():
    parser = argparse.ArgumentParser(description="SVD compression and self-supervised discriminativeness-tuning")
    
    # Required args
    parser.add_argument("--embedding-path", "-i", type=str, required=True, help="Path to the GeoRSCLIP embeddings")
    parser.add_argument("--dataset-path", "-d", type=str, required=True, help="Path to the root of the Zarr dataset, e.g., /data/embed2scale/data_eval")
    
    # Optional args
    parser.add_argument("--output-path", "-o", type=str, default="submission.csv", help="Path to the output CSV file")
    parser.add_argument("--svd-dim", "-hdim", type=int, default=128, help="Number of dimensions to keep in SVD compression")
    parser.add_argument("--num-epochs", "-e", type=int, default=100, help="Number of epochs to train the linear classifier. 340 works well for eval.")
    parser.add_argument(
        "--cluster-cutoffs",
        "-c",
        type=int,
        nargs="+",
        default=[28, 11, 7, 3],
        help="Cutoffs to use for different cluster labels. Suggested values for dev are 28, 11, 7, 3; for eval are 40, 20, 10, 5.",
    )
    args = parser.parse_args()

    # Load the embeddings
    ds = L2AZarrDataset(args.dataset_path)
    embeddings = torch.load(args.embedding_path, weights_only=False)

    # SVD compression for each season
    low_dim_embs = []
    hid_sz = args.svd_dim

    with torch.no_grad():
        for idx in range(4):
            x = torch.from_numpy(embeddings[:, idx, :])
            u, s, v = x.svd()
            print(f"Variance kept for season {idx + 1}: {(s / s.sum())[:hid_sz].sum():.4f}")
            low_dim_emb = u[:, :hid_sz] @ torch.diag(s[:hid_sz])
            low_dim_embs.append(low_dim_emb)
    
    X = torch.cat(low_dim_embs, dim=1).cpu().numpy()
    
    # Cluster to generate pseudolabels at multiple levels of granularity
    labels = cluster_by_distances(X, args.cluster_cutoffs) - 1
    Xy = [(X[i], labels[i]) for i in range(len(X))]
    train_ds, dev_ds = torch.utils.data.random_split(Xy, [0.99, 0.01])
    
    train_dl = tud.DataLoader(train_ds, batch_size=4096, shuffle=True)
    dev_dl = tud.DataLoader(dev_ds, batch_size=4096, shuffle=False)

    # Increase the discriminability of the embeddings
    ## We jointly fine-tune the linear map M and the linear probes Ws, all to maximize the accuracy
    ## of the linear probes to discriminate between clusters..

    ## M is the linear map that increases discriminability between clusters
    M = nn.Parameter(torch.randn(hid_sz, hid_sz) * 0.1)

    ## Ws are the linear probes without a bias term, as specified in the competition
    Ws = [nn.Linear(hid_sz * 4, len(set(l)), bias=False) for l in labels.T]

    ## All our parameters to optimize
    params = [M] + list(itertools.chain(*(W.parameters() for W in Ws)))
    loss = nn.CrossEntropyLoss()
    optimizer = AdamW(params, lr=0.001)
    pbar = trange(args.num_epochs, desc="Training")

    for idx in pbar:
        for x in train_dl:
            optimizer.zero_grad()
            H1 = (M @ x[0][:, :hid_sz].T).T
            H2 = (M @ x[0][:, hid_sz:2*hid_sz].T).T
            H3 = (M @ x[0][:, 2*hid_sz:3*hid_sz].T).T
            H4 = (M @ x[0][:, 3*hid_sz:].T).T
            H = torch.cat([H1, H2, H3, H4], dim=1)
            l = x[1].T
            loss_ = 0.0

            for w, l_ in zip(Ws, l):
                loss_ += loss(w(H), l_.long())
            
            loss_.backward()
            optimizer.step()
        
        loss_val = 0.0

        with torch.no_grad():
            for x in dev_dl:
                optimizer.zero_grad()
                H1 = (M @ x[0][:, :hid_sz].T).T
                H2 = (M @ x[0][:, hid_sz:2*hid_sz].T).T
                H3 = (M @ x[0][:, 2*hid_sz:3*hid_sz].T).T
                H4 = (M @ x[0][:, 3*hid_sz:].T).T
                H = torch.cat([H1, H2, H3, H4], dim=1)
                l = x[1].T

                for w, l_ in zip(Ws, l):
                    loss_val += loss(w(H), l_.long())

            pbar.set_postfix({"loss": f"{loss_val / len(dev_dl) / 4:.4f}"})

    # Project the embeddings to the new space and save them
    M = M.detach()
    h1 = (M @ X[:, :hid_sz].T).T
    h2 = (M @ X[:, hid_sz:2*hid_sz].T).T
    h3 = (M @ X[:, 2*hid_sz:3*hid_sz].T).T
    h4 = (M @ X[:, 3*hid_sz:].T).T
    X2 = torch.cat([h1, h2, h3, h4], dim=1)
    X2 = X2.cpu().numpy()

    df = pd.DataFrame(X2)
    df.columns = [f"x{i + 1}" for i in range(df.shape[1])]
    df["id"] = ds.keys

    # df.to_csv(args.output_path, index=False)

    # Move the last column to the first
    last_col = df.columns[-1]
    df = df[[last_col] + list(df.columns[:-1])]

    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
