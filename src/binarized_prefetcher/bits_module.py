import sys
import numpy as np
import torch
import torch.nn as nn

def binarize(X, num_bits, signed=False):
    # X can be of any size, must have a integer dtype
    mask = 2**torch.arange(num_bits - 1, -1, -1, device=X.device)
    if signed:
        signs = torch.ge(X, 0).byte().unsqueeze(-1)
        X = torch.abs(X)
        bits = X.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        retval = torch.cat((bits * signs, bits * (1 - signs), signs), dim=-1)
    else:
        retval = X.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    return retval.long()

def un_binarize(X, num_bits, signed=False):
    # If signed=False, X has shape (..., num_bits)
    # Otherwise, X should have shape (..., 2*num_bits + 1)
    #       +1 is for a sign indicator
    mask = 2**torch.arange(num_bits - 1, -1, -1)
    X = torch.ge(X, 0).byte()
    n_dims = len(X.shape)
    if signed:
        signs = X[..., -1]
        pos = X[..., :num_bits].mul(mask).sum(dim=-1)
        neg = X[..., num_bits:-1].mul(mask).sum(dim=-1)
        retval = pos.mul(signs) - neg.mul(1 - signs)
    else:
        retval = torch.sum(torch.mul(X, mask), dim=-1)
    return retval

def main(argv):
    X = torch.arange(8)
    X = torch.cat((X, -X), dim=-1)
    embed = nn.EmbeddingBag(17, 5, mode='sum')
    lstm = nn.LSTM(5, 17, batch_first=True)
    loss_func = nn.BCEWithLogitsLoss()

    res1 = binarize(X, 8, signed=True).long()
    res2 = embed(res1)
    res3, state = lstm(res2.unsqueeze(0), None)
    loss = loss_func(res3.squeeze(), res1.float())
    res4 = res3
    out = un_binarize(res4, 8, signed=True)

    print(res1.shape)
    print(res2.shape)
    print(res3.shape)
    print(loss)
    print(out)

if __name__ == "__main__":
    main(sys.argv)