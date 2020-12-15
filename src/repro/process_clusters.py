import argparse
import pandas as pd
from sklearn.cluster import KMeans
from train_utils import read_data


def fit_kmeans(data, num_clusters):
    # clustering the data by the address
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data[["addr"]])
    return kmeans


def calc_deltas(input_data):
    output_data = pd.DataFrame()
    output_data["id"] = input_data["id"]
    output_data["pc"] = input_data["pc"]
    output_data["cluster"] = input_data["cluster"]

    deltas = (input_data["addr"].shift(-1)) - input_data["addr"]
    output_data["delta_out"] = deltas
    output_data["delta_in"] = deltas.shift(1)
    output_data = output_data.dropna()

    output_data["delta_in"] = output_data["delta_in"].astype("long")
    output_data["delta_out"] = output_data["delta_out"].astype("long")
    return output_data


def process_data(data, kmeans, num_clusters):
    data["id"] = range(len(data))
    data["cluster"] = kmeans.predict(data[["addr"]])

    cluster_dfs = [
        calc_deltas(data[data.cluster == cluster_id])
        for cluster_id in range(num_clusters)
    ]

    return pd.concat(cluster_dfs).sort_values(by=["id"]).drop(["id"], axis=1)


def main(args):
    # Only fit KMeans on the training data
    data, train_data = read_data(
        args.infile,
        args.train_size,
        args.val_size,
        args.batch_size,
        args.val_freq,
        parse_hex=True,
    )

    # Only fit the KMeans estimator on the training dataset, but assign clusters
    # to ALL data points
    num_clusters = 6
    kmeans = fit_kmeans(train_data, num_clusters)
    clustered = process_data(data, kmeans, num_clusters)
    clustered.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Input data to process", type=str)
    parser.add_argument("outfile", help="File to output processed results", type=str)
    parser.add_argument(
        "--train_size", help="Size of training set", default=5000, type=int
    )
    parser.add_argument(
        "--val_size", help="Size of validation set", default=1500, type=int
    )
    parser.add_argument(
        "--batch_size", help="Batch size for training", default=50, type=int
    )
    parser.add_argument(
        "--val_freq",
        help="Frequency to use batches for evaluation purposes",
        default=4,  # one in every four batches will be used for eval
        type=int,
    )

    args = parser.parse_args()
    main(args)