import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from binary_nn import PrefetchBinary

def load_data(infile, nrows):
    data = pd.read_csv(infile, nrows=nrows)
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
                print_interval=1):
    loss_list = []
    net = net.to(device)
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
            print(f"Epoch {e+1}\tLoss:\t{loss_list[-1]:.8f}")
    return loss_list

def eval_net(net, eval_iter, device='cpu', line_size=64):
    def comp_acc(preds, target, line_size=64):
        diff = preds - target
        correct = torch.lt(diff, line_size) * torch.ge(diff, 0)
        acc = correct.sum() / correct.numel()
        return acc
    net.eval()
    acc_list = []
    for i, eval_data in enumerate(eval_iter):
        eval_data = [ds.to(device) for ds in eval_data]
        X = eval_data[:-1]
        target = eval_data[-1]

        preds = net.module.predict(X)
        pred_acc = comp_acc(preds, target, line_size=line_size)
        acc_list.append(pred_acc)
    
    print('Prediction Acc.: {:.4f}'.format(torch.tensor(acc_list).mean()))
        

def main(args):    
    # Load training examples
    datafile = args.datafile
    train_size = args.train_size
    pc, delta, types, target = load_data(datafile, train_size)
    num_bits = 64

    # Training data setup
    batch_size = args.batch_size
    train_iter = setup_data(pc, delta, types, target, batch_size=batch_size)
    eval_iter = setup_data(pc, delta, types, target, batch_size=train_size)

    # Check for cuda usage
    if args.cuda:
        device = torch.device('cuda:0')
    else: device = 'cpu'

    # Tunable hyperparameters
    embed_dim = 64
    type_embed_dim = 4
    hidden_dim = 64
    num_layers = 1
    dropout = 0
    linear_end = args.lin
    lr = 2e-3
    
    # Prefetching Model
    prefetch_net = PrefetchBinary(num_bits, embed_dim, type_embed_dim, hidden_dim,
                        num_layers, dropout, linear_end=linear_end)
    optimizer = torch.optim.Adam(prefetch_net.parameters(), lr=lr)

    # Check for using multiple GPUs
    if args.cuda_parallel:
        prefetch_net = nn.DataParallel(prefetch_net, range(args.cuda_parallel))

    # Print parameters for debugging purposes
    # for name, param in prefetch_net.named_parameters():
    #     if param.requires_grad:
    #         print(name + "\t" + str(param.shape))

    # Train
    epochs = args.epochs
    print_interval = args.print_interval
    print("Train Start:")
    loss_list = train_net(prefetch_net, train_iter, epochs, optimizer, device=device,
                            print_interval=print_interval)

    # Eval
    eval_net(prefetch_net, eval_iter, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="Input data set to train/test on", type=str)
    parser.add_argument("--train_size", help="Size of training set", default=1000, type=int)
    parser.add_argument("--batch_size", help="Batch size for training", default=50, type=int)
    parser.add_argument("--epochs", help="Number of epochs to train", default=1)
    parser.add_argument("--print_interval", help="Print loss during training", default=1)
    parser.add_argument("--lin", help="Use a linear layer at the end or not", action="store_true", default=True)
    parser.add_argument("--cuda", help="Use cuda or not", action="store_true", default=True)
    parser.add_argument("--cuda_parallel", help="Use multiple GPUs for computation", default=1)
    args = parser.parse_args()
    main(args)