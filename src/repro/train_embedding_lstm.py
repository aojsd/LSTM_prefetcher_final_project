import torch
from vocab import build_vocabs
from embedding_lstm import EmbeddingLSTM
from train_utils import train_net, eval_net, read_data, parse_args


# Load data from input file
def load_data(data, vocabs, batch_size=2):
    pc_vocab, delta_vocab, target_vocab = vocabs

    # Need to map PCs, input deltas, and output deltas to indices as inputs
    # to the embedding layers or as outputs of the model
    pc = torch.tensor(data["pc"].map(pc_vocab.get_val).to_numpy())
    delta_in = torch.tensor(data["delta_in"].map(delta_vocab.get_val).to_numpy())
    targets = torch.tensor(data["delta_out"].map(target_vocab.get_val).to_numpy())

    # Wrap tensors in a DataLoader object for convenience
    dataset = torch.utils.data.TensorDataset(pc, delta_in, targets)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)


def main(args):
    # Reproducibility
    torch.manual_seed(0)

    # Training and validation data setup
    data, train_data = read_data(
        args.datafile, args.train_size, args.val_size, args.batch_size, args.val_freq
    )

    # Training and validation data setup (datafile is the processed version)
    vocabs = build_vocabs(train_data)
    batch_iter = load_data(data, vocabs, batch_size=args.batch_size)

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

    # Check for cuda usage
    device = torch.device("cuda:0") if args.cuda else "cpu"

    # Train
    if not args.e:
        loss_list = train_net(
            model,
            batch_iter,
            args.epochs,
            optimizer,
            args.val_freq,
            device=device,
            print_interval=args.print_interval,
        )

    # Eval
    if args.e:
        model = model.to(device)

    eval_net(model, batch_iter, args.val_freq, device=device)

    # Save model parameters
    if args.model_file != None:
        torch.save(model.cpu().state_dict(), args.model_file)


if __name__ == "__main__":
    main(parse_args())