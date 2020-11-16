import sys
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
                print_interval=5):
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
            print(f"Epoch {e+1}\tLoss:\t{loss_list[-1]}")
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

        preds = net.predict(X)
        pred_acc = comp_acc(preds, target, line_size=line_size)
        acc_list.append(pred_acc)
    
    print('Prediction Acc.: {:.4f}'.format(torch.tensor(acc_list).mean()))
        

def main(argv):
    if len(argv) != 2:
        print("Invalid Arguments, need:")
        print("\t1) Input data file")
        return
    
    # Load training examples
    train_size = 100
    pc, delta, types, target = load_data(argv[1], train_size)
    num_bits = 64

    # Training data setup
    batch_size = 10
    train_iter = setup_data(pc, delta, types, target, batch_size=batch_size)
    eval_iter = setup_data(pc, delta, types, target, batch_size=train_size)

    # Tunable hyperparameters
    embed_dim = 128
    type_embed_dim = 4
    hidden_dim = 128
    num_layers = 1
    dropout = 0
    linear_end=True
    lr = 2e-3
    epochs = 200

    # Prefetching Model
    prefetch_net = PrefetchBinary(num_bits, embed_dim, type_embed_dim, hidden_dim,
                        num_layers, dropout, linear_end=linear_end)
    optimizer = torch.optim.Adam(prefetch_net.parameters(), lr=lr)

    # Print parameters for debugging purposes
    # for name, param in prefetch_net.named_parameters():
    #     if param.requires_grad:
    #         print(name + "\t" + str(param.shape))

    # Train
    loss_list = train_net(prefetch_net, train_iter, epochs, optimizer)

    # Eval
    eval_net(prefetch_net, eval_iter)


if __name__ == "__main__":
    main(sys.argv)