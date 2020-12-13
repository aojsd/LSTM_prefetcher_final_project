import argparse
import pandas as pd
import torch
from vocab import build_vocabs
from embedding_lstm import EmbeddingLSTM
from train_utils import train_net, eval_net


# Load data from input file
def load_data(infile, nrows, vocabs, skip=None, batch_size=2):
    # Same function is used to read training, validation, and test datasets,
    # so we need to skip ahead in the file to avoid using the same data for
    # training and testing.
    if skip == None:
        data = pd.read_csv(infile, nrows=nrows)
    else:
        data = pd.read_csv(infile, nrows=nrows, skiprows=range(1, skip + 1))

    pc_vocab, delta_vocab, target_vocab = vocabs

    # Need to map PCs, input deltas, and output deltas to indices as inputs
    # to the embedding layers or as outputs of the model
    pc = torch.tensor(data["pc"].map(pc_vocab.get_val).to_numpy())
    delta_in = torch.tensor(data["delta_in"].map(delta_vocab.get_val).to_numpy())
    targets = torch.tensor(data["delta_out"].map(target_vocab.get_val).to_numpy())

    # Wrap tensors in a DataLoader object for convenience
    dataset = torch.utils.data.TensorDataset(pc, delta_in, targets)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    return data_iter


def main(args):
    # Reproducibility
    torch.manual_seed(0)

    # Training and validation data setup
    vocabs = build_vocabs(args.datafile, args.train_size)
    
    train_iter = load_data(
        args.datafile, args.train_size, vocabs, batch_size=args.batch_size
    )

    eval_iter = load_data(
        args.datafile,
        args.val_size,
        vocabs,
        skip=args.train_size,
        batch_size=args.val_size,
    )

    # Check for cuda usage
    device = torch.device("cuda:0") if args.cuda else "cpu"

    # Input/Output dimensions (add 1 for deltas that we're not training on)
    pc_vocab, delta_vocab, target_vocab = vocabs
    num_pc = len(pc_vocab) + 1
    num_input_delta = len(delta_vocab) + 1
    num_output_delta = len(target_vocab) + 1

    # Tunable hyperparameters
    num_pred = 10
    embed_dim = 256
    hidden_dim = 256
    num_layers = 2
    dropout = 0.1
    lr = 1e-3

    model = EmbeddingLSTM(
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