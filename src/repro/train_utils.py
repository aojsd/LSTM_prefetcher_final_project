import argparse
import torch
import pandas as pd


def read_data(infile, train_size, val_size, batch_size, val_freq, parse_hex=False):
    data = pd.read_csv(infile, nrows=(train_size + val_size))

    if parse_hex:
        convert_hex_to_dec = lambda x: int(x, 16)
        data["pc"] = data["pc"].apply(convert_hex_to_dec)
        data["addr"] = data["addr"].apply(convert_hex_to_dec)

    # If val_freq = 4, every 4th batch is used for validation. Everything else
    # is the training dataset.
    train_data = data[(data.index // batch_size + 1) % val_freq != 0]
    return data, train_data


# Train the network
def train_net(
    net,
    batch_iter,
    epochs,
    optimizer,
    val_freq,
    device="cpu",
    scheduler=None,
    print_interval=10,
):
    loss_list = []
    net = net.to(device)

    print("Train Start:")

    for e in range(epochs):
        # Certain layers behave differently depending on whether we're training
        # or evaluating, so tell the model to run in training mode
        net.train()
        state = None

        for i, data in enumerate(batch_iter):
            data = [ds.to(device) for ds in data]
            X = data[:-1]
            target = data[-1]

            loss, _, state = net(X, state, target)

            # Interleave training and validation
            if (i + 1) % val_freq != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(float(loss.detach()))

            if (i + 1) % print_interval == 0:
                print(f"Epoch {e + 1}, Batch {i + 1}, Loss:\t{loss_list[-1]:.8f}")

            # Detach state gradients to avoid autograd errors
            state = tuple([s.detach() for s in state])

        if scheduler != None:
            scheduler.step()

    return loss_list


def prob_acc(pred, target, target_vocab, clusters=None):
    # pred shape: (N, K)
    # target: (N, 1)
    num_correct = 0

    if clusters is None:
        clusters = [None] * len(pred)

    for expected, topk, cluster in zip(target, pred, clusters):
        # Automatically mark prediction as wrong if we're not
        # training on the expected output delta (be sure to use
        # correct output vocab for clustering LSTM)
        trainable = (
            expected.item() in target_vocab[cluster].val_to_key
            if isinstance(target_vocab, list)
            else expected.item() in target_vocab.val_to_key
        )

        if trainable and expected in topk:
            num_correct += 1

    return num_correct / len(target)


# Evaluate the network on a labeled dataset
def eval_net(net, batch_iter, val_freq, target_vocab, device="cpu", state=None):
    net.eval()
    train_acc_list = []
    eval_acc_list = []

    for i, data in enumerate(batch_iter):
        print(f"Evaluating batch {i}")

        data = [ds.to(device) for ds in data]
        X = data[:-1]
        clusters = X[-1] if len(X) == 3 else None
        target = data[-1]

        preds, state = net.predict(X, state)
        acc = prob_acc(preds.cpu(), target.cpu(), target_vocab, clusters)

        # Interleave training and validation
        if (i + 1) % val_freq != 0:
            train_acc_list.append(acc)
        else:
            eval_acc_list.append(acc)

    print("Train Acc.: {:.4f}".format(torch.tensor(train_acc_list).mean()))
    print("Val Acc.: {:.4f}".format(torch.tensor(eval_acc_list).mean()))


def parse_args():
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
        "--val_freq",
        help="Frequency to use batches for evaluation purposes",
        default=4,  # one in every four batches will be used for eval
        type=int,
    )
    parser.add_argument(
        "--epochs", help="Number of epochs to train", default=1, type=int
    )
    parser.add_argument(
        "--lr", help="Learning rate", default=1e-3, type=float
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

    return parser.parse_args()
