import argparse
import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans(data):
    #clustering the data by the address
    kmeans6 = KMeans(n_clusters = 6)
    kmeans6.fit(data[["addr"]])
    return kmeans6


def calc_deltas(input_data):
    output_data = pd.DataFrame()

    output_data["id"] = input_data["id"]
    output_data["pc"] = input_data["pc"]
    output_data["cluster"] = input_data["cluster"]

    output_data["delta_out"] = (input_data["addr"] - input_data["addr"].shift(-1))
    output_data["delta_in"] = output_data["delta_out"].shift(1)
    output_data = output_data.dropna()

    return output_data


def read_data(infile, nrows, skip=None):
    if skip == None:
        data = pd.read_csv(infile, nrows=nrows)
    else:
        data = pd.read_csv(infile, nrows=nrows, skiprows=range(1, skip + 1))

    convert_hex_to_dec = lambda x: int(x, 16)
    data["pc"] = data["pc"].apply(convert_hex_to_dec)
    data["addr"] = data["addr"].apply(convert_hex_to_dec)
    return data


def process_data(data, kmeans):
    data["cluster"] = kmeans.predict(data[["addr"]])
    data["id"] = range(len(data))

    cluster_dfs = [
        calc_deltas(data[data.cluster == cluster_id]) 
        for cluster_id in range(6)
    ]

    return pd.concat(cluster_dfs).sort_values(by=["id"]).drop(["id"], axis=1)


def main(args):
    # Only fit KMeans on the training data
    train_data = read_data(args.infile, args.train_size)
    kmeans = fit_kmeans(train_data)

    all_data = read_data(args.infile, args.train_size + args.val_size)
    df = process_data(all_data, kmeans)
    df.to_csv(args.outfile, index=False)


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

    args = parser.parse_args()
    main(args)