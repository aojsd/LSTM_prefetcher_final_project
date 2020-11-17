import sys
import torch
import torch.nn as nn
import bits_module as bits

class PrefetchBinary(nn.Module):
    def __init__(self, num_bits, embed_dim, type_embed_dim, hidden_dim, num_layers=1,
                 dropout=0, linear_end=False):
        super(PrefetchBinary, self).__init__()
        self.num_bits = num_bits
        self.pc_embed = nn.EmbeddingBag(num_bits, embed_dim)
        self.delta_embed = nn.EmbeddingBag(2*num_bits, embed_dim)
        self.type_embed = nn.Embedding(3, type_embed_dim)

        self.linear_end = linear_end
        if linear_end:
            self.lstm = nn.LSTM(2*embed_dim + type_embed_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout)
            self.out_lin = nn.Linear(hidden_dim, 2*num_bits+1)
        else:
            self.lstm = nn.LSTM(2*embed_dim + type_embed_dim, 2*num_bits+1, num_layers,
                                batch_first=True, dropout=dropout)

        # Automatically sigmoids and calculates cross entropy loss with given weights
        # weights = torch.ones(2*num_bits+1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, X, lstm_state, target):
        # X is the tuple (pc's, deltas, types) where:
        #       pc's, deltas, and types have shape (T,)
        # target is a tensor of the target deltas, has shape (T,)
        #       target deltas are not binarized
        # Returns loss, lstm output, and lstm state
        pc, delta, types = X     
        pc = self.pc_embed(bits.binarize(pc, self.num_bits, signed=False))
        delta = self.delta_embed(bits.binarize(delta, self.num_bits, signed=True))
        types = self.type_embed(types)

        if len(pc.shape) < 3:
            lstm_in = torch.cat((pc, delta, types), dim=-1).unsqueeze(0)
        else:
            lstm_in = torch.cat((pc, delta, types), dim=-1)

        if self.linear_end:
            out, state = self.lstm(lstm_in, lstm_state)
            out = self.out_lin(out)
        else:
            out, state = self.lstm(lstm_in, lstm_state)

        # Calculate loss
        out = out.squeeze()
        target = bits.binarize(target, self.num_bits, signed=True).float()
        loss = self.loss_func(out, target)
        return loss, out, state

    def predict(self, X, state):
        pc, delta, types = X
        pc = self.pc_embed(bits.binarize(pc, self.num_bits, signed=False))
        delta = self.delta_embed(bits.binarize(delta, self.num_bits, signed=True))
        types = self.type_embed(types)

        if len(pc.shape) < 3:
            lstm_in = torch.cat((pc, delta, types), dim=-1).unsqueeze(0)
        else:
            lstm_in = torch.cat((pc, delta, types), dim=-1)

        if self.linear_end:
            out, state = self.lstm(lstm_in, state)
            out = self.out_lin(out)
        else:
            out, state = self.lstm(lstm_in, state)

        # Calculate predictions
        out = out.squeeze()
        preds = bits.un_binarize(out, self.num_bits, signed=True)
        return preds, state

def main(argv):
    pc = torch.arange(4)
    delta = torch.arange(3, -1, -1)
    types = torch.randint(0, 3, (4,)).long()
    target = torch.arange(4)
    net = PrefetchBinary(4, 8, 2, 3, num_layers=1, linear_end=False)

    l, o, s = net((pc, delta, types), None, target)
    l, o, s = net((pc, delta, types), s, target)
    print(l)
    print(o.shape)
    print(s[0].shape)
    print(s[1].shape)

    preds = net.predict((pc, delta, types))
    print(preds)

if __name__ == "__main__":
    main(sys.argv)