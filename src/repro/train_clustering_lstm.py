import argparse
import pandas as pd
import torch
from sklearn.cluster import KMeans
from vocab import build_vocabs
from clustering_lstm import ClusteringLSTM


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


# Load data from input file
def process_data(data, kmeans, batch_size=2):
    data["cluster"] = kmeans.predict(data[["addr"]])
    data["id"] = range(len(data))

    cluster_dfs = [
        calc_deltas(data[data.cluster == cluster_id]) 
        for cluster_id in range(6)
    ]

    data = pd.concat(cluster_dfs).sort_values(by=["id"]).drop(["id"], axis=1)
    print(data)

    # Convert data to PyTorch tensors
    pc = torch.tensor(data["pc"].to_numpy())
    delta_in = torch.tensor(data["delta_in"].to_numpy())
    clusters = torch.tensor(data["cluster"].to_numpy())
    targets = torch.tensor(data["delta_out"].to_numpy())

    # Wrap tensors in a DataLoader object for convenience
    dataset = torch.utils.data.TensorDataset(pc, delta_in, clusters, targets)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    return data_iter


def main(args):
    # Reproducibility
    torch.manual_seed(0)

    # Training and validation data setup
    train_data = read_data(args.datafile, args.train_size)

    kmeans = fit_kmeans(train_data)
    
    train_iter = process_data(train_data, kmeans, batch_size=args.batch_size)

    eval_data = read_data(args.datafile, args.val_size, skip=args.train_size)
    eval_iter = process_data(eval_data, kmeans, batch_size=args.val_size)

    # Check for cuda usage
    device = torch.device("cuda:0") if args.cuda else "cpu"

    # Input/Output dimensions (add 1 for deltas that we're not training on)

    # Tunable hyperparameters
    num_pred = 10
    embed_dim = 256
    hidden_dim = 256
    num_layers = 2
    dropout = 0.1
    lr = 1e-3

    model = ClusteringLSTM(
        num_pc,
        num_input_delta,
        num_output_delta,
        embed_dim,
        hidden_dim,
        num_pred=num_pred,
        num_layers=num_layers,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check for existing model
    if args.model_file != None:
        model.load_state_dict(torch.load(args.model_file))

    # Train
    if not args.e:
        loss_list = train_net(
            model,
            train_iter,
            args.epochs,
            optimizer,
            device=device,
            print_interval=args.print_interval,
        )

    # Eval
    if args.e:
        model = model.to(device)

    state = eval_net(model, train_iter, device=device)
    state = eval_net(model, eval_iter, device=device, state=state)

    # Save model parameters
    if args.model_file != None:
        torch.save(model.cpu().state_dict(), args.model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="Input data set to train/test on", type=str)
    parser.add_argument(
        "--train_size", help="Size of training set", default=5000, type=int
    )
    parser.add_argument(
        "--batch_size", help="Batch size for training", default=50, type=int
    )
    parser.add_argument(
        "--val_size", help="Size of training set", default=1500, type=int
    )
    parser.add_argument(
        "--epochs", help="Number of epochs to train", default=1, type=int
    )
    parser.add_argument(
        "--print_interval", help="Print loss during training", default=10, type=int
    )
    parser.add_argument(
        "--cuda", help="Use cuda or not", action="store_true", default=False
    )
    parser.add_argument(
        "--model_file",
        help="File to load/save model parameters to continue training",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-e", help="Load and evaluate only", action="store_true", default=False
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    kmeans = fit_kmeans()
    load_data("data/raw/pi_noprefetch_raw.csv", nrows = 15000)
