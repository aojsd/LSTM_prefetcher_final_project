import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from binary_nn import PrefetchBinary

def load_data(infile, nrows, skip=None):
    if skip == None:
        data = pd.read_csv(infile, nrows=nrows)
    else:
        data = pd.read_csv(infile, nrows=nrows, skiprows=range(1, skip+1))
    pc = torch.tensor(data['pc'].to_numpy())
    delta_in = torch.tensor(data['delta_in'].to_numpy())
    types = torch.tensor(data['type'].to_numpy())
    targets = torch.tensor(data['delta_out'].to_numpy())
    return pc, delta_in, types, targets

def setup_data(pc, delta, types, target, batch_size=2):
    dataset = torch.utils.data.TensorDataset(
                    pc, delta, types, target)
    data_iter = torch.utils.data.DataLoader(
                    dataset, batch_size, shuffle=False)
    return data_iter

def train_net(net, train_iter, epochs, optimizer, device='cpu', scheduler=None,
                print_interval=10):
    loss_list = []
    net = net.to(device)
    print("Train Start:")
    for e in range(epochs):
        net.train()
        state = None
        for i, train_data in enumerate(train_iter):
            train_data = [ds.to(device) for ds in train_data]
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
        
        if (e+1) % print_interval == 0:
            print(f"\tEpoch {e+1}\tLoss:\t{loss_list[-1]:.8f}")
    return loss_list

def eval_net(net, eval_iter, device='cpu', line_size=64, state=None):
    def comp_acc(preds, target, line_size=64):
        diff = preds - target
        correct = torch.lt(diff, line_size) * torch.ge(diff, 0)
        acc = correct.sum() / correct.numel()
        return acc
    train_acc = state == None
    net.eval()
    acc_list = []
    for i, eval_data in enumerate(eval_iter):
        eval_data = [ds.to(device) for ds in eval_data]
        X = eval_data[:-1]
        target = eval_data[-1]

        preds, state = net.predict(X, state)
        pred_acc = comp_acc(preds, target, line_size=line_size)
        acc_list.append(pred_acc)
    
    if train_acc:
        print('Training Acc.: {:.4f}'.format(torch.tensor(acc_list).mean()))
    else:
        print('Val Acc.: {:.4f}'.format(torch.tensor(acc_list).mean()))
    return state
        

def main(args):    
    # Load training examples
    datafile = args.datafile
    train_size = args.train_size
    pc, delta, types, target = load_data(datafile, train_size)
    num_bits = 64

    # Training and validation data setup
    batch_size = args.batch_size
    train_iter = setup_data(pc, delta, types, target, batch_size=batch_size)
    pc, delta, types, target = load_data(datafile, args.val_size, skip=train_size)
    eval_iter = setup_data(pc, delta, types, target, batch_size=args.val_size)

    # Check for cuda usage
    if args.cuda:
        device = torch.device('cuda:0')
    else: device = 'cpu'

    # Tunable hyperparameters
    embed_dim = 128
    type_embed_dim = 4
    hidden_dim = 128
    num_layers = 1
    dropout = 0
    linear_end = args.lin
    lr = 5e-4
    
    # Prefetching Model
    prefetch_net = PrefetchBinary(num_bits, embed_dim, type_embed_dim, hidden_dim,
                        num_layers, dropout, linear_end=linear_end)
    optimizer = torch.optim.Adam(prefetch_net.parameters(), lr=lr)

    # Check for existing model
    if args.load_file != None:
        prefetch_net.load_state_dict(torch.load(args.load_file))
    elif args.continue_file != None:
        prefetch_net.load_state_dict(torch.load(args.continue_file))

    # Train
    epochs = args.epochs
    print_interval = args.print_interval
    if not args.e:
        loss_list = train_net(prefetch_net, train_iter, epochs, optimizer, device=device,
                                print_interval=print_interval)

    # Eval
    if args.e:
        prefetch_net = prefetch_net.to(device)
    state = eval_net(prefetch_net, train_iter, device=device)
    state = eval_net(prefetch_net, eval_iter, device=device, state=state)

    # Save model parameters
    if args.save_file != None:
        torch.save(prefetch_net.cpu().state_dict(), args.save_file)
    elif args.continue_file != None:
        torch.save(prefetch_net.cpu().state_dict(), args.continue_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="Input data set to train/test on", type=str)
    parser.add_argument("--train_size", help="Size of training set", default=1000, type=int)
    parser.add_argument("--batch_size", help="Batch size for training", default=200, type=int)
    parser.add_argument("--val_size", help="Size of training set", default=1000, type=int)
    parser.add_argument("--epochs", help="Number of epochs to train", default=1, type=int)
    parser.add_argument("--print_interval", help="Print loss during training", default=10, type=int)
    parser.add_argument("--lin", help="Use a linear layer at the end or not", action="store_true", default=True)
    parser.add_argument("--cuda", help="Use cuda or not", action="store_true", default=True)
    parser.add_argument("--save_file", help="File to save model parameters in", default=None, type=str)
    parser.add_argument("--load_file", help="File to load model parameters from", default=None, type=str)
    parser.add_argument("--continue_file", help="File containing model parameters to continue training", default=None, type=str)
    parser.add_argument("-e", help="Load and evaluate only", action="store_true", default=False)

    args = parser.parse_args()
    main(args)