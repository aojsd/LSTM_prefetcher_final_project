import argparse
import pandas as pd
import torch
import torch.nn as nn


# Load data from input file
def load_data(infile, nrows, skip=None, batch_size=2):
    # Same function is used to read training, validation, and test datasets, 
    # so we need to skip ahead in the file to avoid using the same data for
    # training and testing.
    if skip == None:
        data = pd.read_csv(infile, nrows=nrows)
    else:
        data = pd.read_csv(infile, nrows=nrows, skiprows=range(1, skip + 1))

    # Convert data to PyTorch tensors
    pc = torch.tensor(data["pc"].to_numpy())
    delta_in = torch.tensor(data["delta_in"].to_numpy())
    types = torch.tensor(data["type"].to_numpy())
    targets = torch.tensor(data["delta_out"].to_numpy())

    # Wrap tensors in a DataLoader object for convenience
    dataset = torch.utils.data.TensorDataset(pc, delta_in, types, targets)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    return data_iter


# Train the network
def train_net(
    net, train_iter, epochs, optimizer, device="cpu", scheduler=None, print_interval=10
):
    loss_list = []
    net = net.to(device)

    print("Train Start:")
    
    for e in range(epochs):
        # Certain layers behave differently depending on whether we're training
        # or testing, so tell the model to run in training mode
        net.train()
        state = None
        
        for train_data in train_iter:
            train_data = [ds.to(device) for ds in train_data]

            # Because of how the data is wrapped up into the DataLoader in `load_data`,
            # the first three things are pc, delta_in, and types, and the last thing
            # is the targets (delta_out).
            X = train_data[:-1]
            target = train_data[-1]

            loss, out, state = net(X, state, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss)

            # Detach state gradients to avoid autograd errors
            state = tuple([s.detach() for s in list(state)])

        if scheduler != None:
            scheduler.step()

        if (e + 1) % print_interval == 0:
            print(f"\tEpoch {e+1}\tLoss:\t{loss_list[-1]:.8f}")
    
    return loss_list


# Evaluate the network on a labeled dataset
def eval_net(net, eval_iter, device="cpu", state=None):
    comp_acc = Accuracy()
    train_acc = state == None
    
    net.eval()
    prob_acc_list = []
    block_acc_list = []
    
    for i, eval_data in enumerate(eval_iter):
        eval_data = [ds.to(device) for ds in eval_data]
        
        X = eval_data[:-1]
        target = eval_data[-1]

        preds, state = net.predict(X, state)

        prob_acc = comp_acc.prob_acc(preds.cpu(), target.cpu())
        prob_acc_list.append(prob_acc)

        block_acc = comp_acc.block_acc(preds.cpu(), target.cpu())
        block_acc_list.append(block_acc)

    if train_acc:
        print("Training Prob Acc.: {:.4f}".format(torch.tensor(prob_acc_list).mean()))
        print("Training Block Acc.: {:.4f}".format(torch.tensor(block_acc_list).mean()))
    else:
        print("Val Prob Acc.: {:.4f}".format(torch.tensor(prob_acc_list).mean()))
        print("Val Acc.: {:.4f}".format(torch.tensor(block_acc_list).mean()))
    
    return state


def main(args):
    # Reproducibility
    torch.manual_seed(0)

    # Training and validation data setup
    train_iter = load_data(args.datafile, args.train_size, batch_size=args.batch_size)
    eval_iter = load_data(args.datafile, args.val_size, skip=args.train_size, batch_size=args.val_size)

    # Check for cuda usage
    device = torch.device("cuda:0") if args.cuda else "cpu"

    # Tunable hyperparameters
    num_bits = 64
    embed_dim = 256
    type_embed_dim = 4
    hidden_dim = 256
    num_layers = 2
    dropout = 0.1
    linear_end = args.lin
    lr = 1e-3

    # Prefetching Model
    prefetch_net = PrefetchBinary(
        num_bits,
        embed_dim,
        type_embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        linear_end=linear_end,
    )

    optimizer = torch.optim.Adam(prefetch_net.parameters(), lr=lr)

    # Check for existing model
    if args.model_file != None:
        prefetch_net.load_state_dict(torch.load(args.model_file))

    # Train
    if not args.e:
        loss_list = train_net(
            prefetch_net,
            train_iter,
            args.epochs,
            optimizer,
            device=device,
            print_interval=args.print_interval,
        )

    # Eval
    if args.e:
        prefetch_net = prefetch_net.to(device)
    
    state = eval_net(prefetch_net, train_iter, device=device)
    state = eval_net(prefetch_net, eval_iter, device=device, state=state)

    # Save model parameters
    if args.model_file != None:
        torch.save(prefetch_net.cpu().state_dict(), args.model_file)


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
        "--lin",
        help="Use a linear layer at the end or not",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--cuda", help="Use cuda or not", action="store_true", default=True
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