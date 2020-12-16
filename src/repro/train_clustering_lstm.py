import os
import torch
from vocab import build_vocabs
from clustering_lstm import ClusteringLSTM
from train_utils import train_net, eval_net, read_data, parse_args


# Load data from input file
def load_data(data, vocabs, batch_size=2):
    pc_vocab, delta_vocab, target_vocab = vocabs

    # Convert data to PyTorch tensors
    pc = torch.tensor(data["pc"].map(pc_vocab.get_val).to_numpy())
    delta_in = torch.tensor(data["delta_in"].map(delta_vocab.get_val).to_numpy())
    clusters = torch.tensor(data["cluster"].to_numpy())

    # Map targets for each cluster independently to improve parallelization
    for cluster in range(len(target_vocab)):
        data.loc[data.cluster == cluster, "delta_out"] = (
            data.loc[data.cluster == cluster, "delta_out"]
            .map(target_vocab[cluster].get_val)
        )

    targets = torch.tensor(data["delta_out"].to_numpy())

    # Wrap tensors in a DataLoader object for convenience
    dataset = torch.utils.data.TensorDataset(pc, delta_in, clusters, targets)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)


def main(args):
    # Reproducibility
    torch.manual_seed(0)

    data, train_data = read_data(
        args.datafile, args.train_size, args.val_size, args.batch_size, args.val_freq
    )

    # Training and validation data setup (datafile is the processed version)
    vocabs = build_vocabs(train_data, num_output_deltas=20000)
    batch_iter = load_data(data, vocabs, batch_size=args.batch_size)

    # Input/Output dimensions (add 1 for deltas that we're not training on)
    pc_vocab, delta_vocab, target_vocab = vocabs
    num_pc = len(pc_vocab) + 1
    num_input_delta = len(delta_vocab) + 1
    num_output_delta = [len(vocab) + 1 for vocab in target_vocab]

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
        if os.path.exists(args.model_file):
            print("Loading model from file: {}".format(args.model_file))
            model.load_state_dict(torch.load(args.model_file))
        else:
            print("Model file does not exist, training from scratch...")

    # Check for cuda usage
    device = torch.device("cuda:0") if args.cuda else "cpu"

    if not args.e:
        train_net(
            model,
            batch_iter,
            args.epochs,
            optimizer,
            args.val_freq,
            device=device,
            print_interval=args.print_interval,
        )

        if args.model_file != None:
            torch.save(model.cpu().state_dict(), args.model_file)
    else:
        model = model.to(device)
        eval_net(model, batch_iter, args.val_freq, target_vocab, device=device)


if __name__ == "__main__":
    main(parse_args())
