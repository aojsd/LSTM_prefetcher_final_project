import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from binary_nn import PrefetchBinary
from bits_module import un_binarize

# Load data from input file
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

# Wrap tensors into a dataloader object
def setup_data(pc, delta, types, target, batch_size=2):
    dataset = torch.utils.data.TensorDataset(
                    pc, delta, types, target)
    data_iter = torch.utils.data.DataLoader(
                    dataset, batch_size, shuffle=False)
    return data_iter

# Train the network
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

class Accuracy(nn.Module):
    def __init__(self, num_bits=64, line_size=64, margin=2):
        super(Accuracy, self).__init__()
        self.num_bits = num_bits
        self.line_size = line_size
        self.margin = margin

        # Used later for generating multiple predictions
        num_preds = 2**margin
        section = num_preds >> 1
        self.T = torch.zeros((2**margin, margin))
        for b in range(margin):
            for i in range(0, int(num_preds/section), 2):
                self.T[i*section:(i+1)*section, b] = 1
            section >>= 1
        self.T = self.T.byte()
        
        self.mask = 2**torch.arange(num_bits - 1, -1, -1)

    # Fetches lines from multiple sources based on least confident bits
    def prob_acc(self, preds, target):
        # First select the correct half based on the sign bit
        signs = torch.ge(preds[..., -1], 0).byte().unsqueeze(-1)
        reduced_preds = signs * preds[..., :self.num_bits]
        reduced_preds += (1-signs) * preds[..., self.num_bits:-1]

        # Identify the least confident bits in the result
        conf = torch.abs(reduced_preds)
        c, inds = torch.topk(conf, self.margin, dim=-1, largest=False)

        # Generate bits prediction list
        bits = torch.ge(reduced_preds, 0).byte()
        N = bits.shape[0]
        bit_pred_list = []
        for tensor in list(self.T):
            bits.scatter_(-1, inds, tensor.repeat((N,1)))
            bit_pred_list.append(bits.clone())

        # Exponentiate to produce final delta predictions
        delta_list = []
        for bit_pred in bit_pred_list:
            pos = bit_pred.mul(self.mask).sum(dim=-1).unsqueeze(-1)
            del_pred = pos.mul(signs) - pos.mul(1-signs)
            delta_list.append(del_pred)
        delta_preds = torch.cat(delta_list, dim=-1)

        # Check if the target deltas have been predicted
        diff = delta_preds - target.unsqueeze(-1)
        check = torch.le(torch.abs(diff), 2*self.line_size).sum(dim=-1)
        check = torch.gt(check, 0).byte()
        acc = torch.sum(check) / N
        return acc

    def block_acc(self, preds, target):
        diff = un_binarize(preds, self.num_bits, signed=True) - target
        correct = torch.le(torch.abs(diff), 4*self.line_size)
        acc = correct.sum() / correct.numel()
        return acc

# Evaluate the network on a labeled dataset
def eval_net(net, eval_iter, device='cpu', state=None):
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
        print('Training Prob Acc.: {:.4f}'.format(torch.tensor(prob_acc_list).mean()))
        print('Training Block Acc.: {:.4f}'.format(torch.tensor(block_acc_list).mean()))
    else:
        print('Val Prob Acc.: {:.4f}'.format(torch.tensor(prob_acc_list).mean()))
        print('Val Acc.: {:.4f}'.format(torch.tensor(block_acc_list).mean()))
    return state
        

def main(args):
    # Reproducibility
    torch.manual_seed(0)

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
    embed_dim = 256
    type_embed_dim = 4
    hidden_dim = 256
    num_layers = 2
    dropout = 0.1
    linear_end = args.lin
    lr = 1e-3
    
    # Prefetching Model
    prefetch_net = PrefetchBinary(num_bits, embed_dim, type_embed_dim, hidden_dim,
                        num_layers, dropout, linear_end=linear_end)
    optimizer = torch.optim.Adam(prefetch_net.parameters(), lr=lr)

    # Check for existing model
    if args.model_file != None:
        prefetch_net.load_state_dict(torch.load(args.model_file))

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
    if args.model_file != None:
        torch.save(prefetch_net.cpu().state_dict(), args.model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="Input data set to train/test on", type=str)
    parser.add_argument("--train_size", help="Size of training set", default=5000, type=int)
    parser.add_argument("--batch_size", help="Batch size for training", default=50, type=int)
    parser.add_argument("--val_size", help="Size of training set", default=1500, type=int)
    parser.add_argument("--epochs", help="Number of epochs to train", default=1, type=int)
    parser.add_argument("--print_interval", help="Print loss during training", default=10, type=int)
    parser.add_argument("--lin", help="Use a linear layer at the end or not", action="store_true", default=True)
    parser.add_argument("--cuda", help="Use cuda or not", action="store_true", default=True)
    parser.add_argument("--model_file", help="File to load/save model parameters to continue training", default=None, type=str)
    parser.add_argument("-e", help="Load and evaluate only", action="store_true", default=False)

    args = parser.parse_args()
    main(args)