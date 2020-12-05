import sys
import torch
import torch.nn as nn
import bits_module as bits

def bit_split(X, splits, len_split, signed=True):
    # Separate splits in input based on bitwise values
    # Output will have shape (N, 2*splits)
    #   Lower order bits will have lower index in the splits dimension
    #   Output has a positive and negative section, if original is positive,
    #   the negative section will be all zeroes, and vice versa
    T = []
    signs = torch.ge(X, 0).byte().unsqueeze(-1)
    X = torch.abs(X)
    mask = (1 << len_split) - 1
    for i in range(splits):
        t_c = torch.bitwise_and(X, mask)
        T.append(t_c.unsqueeze(1))
        X >>= len_split
    out = torch.cat(T, dim=1)
    if signed:
        out = torch.cat([out * signs, out * (1-signs)], dim=1)
    return out

class MultibitSoftmax(nn.Module):
    def __init__(self, num_bits, splits):
        super(MultibitSoftmax, self).__init__()
        # Assumes splits divides evenly with num_bits
        self.splits = splits
        self.len_split = int(num_bits/splits)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, X, target):
        # X holds inputs of shape (N, 2 * splits * (2^len_split))
        # target has shape (N, ), dtype = long
        # Output:
        #   preds --> shape (N, splits)
        #   loss ---> type float
        N, _ = X.shape

        # Reshape X and to separate splits for positive and negative cases
        ce_in = X.reshape(N, -1, 2*self.splits)

        # Separate splits in target based on bitwise values
        # ce_target will have shape (N, 2*splits)
        # Lower order bits will have lower index in the splits dimension
        ce_target = bit_split(target, self.splits, self.len_split)

        # Calculate multi-dimensional cross-entropy loss and class predictions
        loss = self.CE(ce_in, ce_target)
        preds = ce_in.argmax(1)
        return preds, loss

    def predict(self, X):
        N, _ = X.shape
        x_splits = X.reshape(N, -1, 2*self.splits)
        preds = x_splits.argmax(1)
        return preds

class BitsplitEmbedding(nn.Module):
    def __init__(self, num_bits, splits, embedding_dim, signed=True):
        super(BitsplitEmbedding, self).__init__()

        # Assumes splits divides evenly into both num_bits and embedding_dim
        self.splits = splits
        self.len_split = int(num_bits/splits)
        self.split_embed = int(embedding_dim/splits)
        self.signed = signed

        num_embedding = 1 << self.len_split
        if signed:
            num_embed = 2*splits
        else:
            num_embed = splits
        self.embeds = nn.ModuleList(
                        [nn.Embedding(num_embedding, self.split_embed)
                            for _ in range(num_embed)] )
    
    def forward(self, X):
        # X holds inputs of shape (N, )
        # Converts X to a tensor of shape (N, 2*splits) representing
        #   the bits in each split for positive and negative cases
        # Returns tensor of shape (N, 2*embedding_dim), if signed
        # else shape (N, embedding_dim)
        N = X.shape[0]

        # Separate splits in X based on bitwise values
        X = bit_split(X, self.splits, self.len_split, self.signed)

        # Perform multiple embeddings for each row in the batch
        embed_list = []
        for i, E in enumerate(self.embeds):
            embed_list.append( E(X[:,i]) )
        out = torch.cat(embed_list, dim=-1)
        return out

class MESoftNet(nn.Module):
    def __init__(self, num_bits, embed_dim, type_dim, hidden_dim, num_layers=1,
                 dropout=0.1, splits=8, sign_weight=1):
        super(MESoftNet, self).__init__()
        self.num_bits = num_bits
        self.splits = splits
        self.len_split = int(num_bits/splits)
        self.num_classes = 1 << self.len_split
        self.sign_weight = sign_weight

        self.pc_embed = BitsplitEmbedding(num_bits, splits, embed_dim, signed=False)
        self.delta_embed = BitsplitEmbedding(num_bits, splits, embed_dim)
        self.type_embed = nn.Embedding(3, type_dim)
        self.lstm = nn.LSTM(3*embed_dim + type_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.lin_magnitude = nn.Linear(hidden_dim, 2*splits*self.num_classes)
        self.lin_sign = nn.Linear(hidden_dim, 2)
        self.m_soft = MultibitSoftmax(num_bits, splits)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, X, lstm_state, target):
        # X is the tuple (pc's, deltas, types) where:
        #       pc's, deltas, and types have shape (T,)
        # target is a tensor of the target deltas, has shape (T,)
        #       target deltas are not binarized
        # Returns loss, predictions, and lstm state
        pc, delta, types = X
        pc = self.pc_embed(pc)
        delta = self.delta_embed(delta)
        types = self.type_embed(types)

        # Concatenate and feed into LSTM
        lstm_in = torch.cat((pc, delta, types), dim=-1).unsqueeze(0)
        lstm_out, state = self.lstm(lstm_in, lstm_state)

        # Separately calculate magnitude values and signs
        mag = self.lin_magnitude(lstm_out).squeeze()
        sign_probs = self.lin_sign(lstm_out).squeeze()

        # Loss and prediction calculation
        mag_preds, mag_loss = self.m_soft(mag, target)
        sign_preds = sign_probs.argmax(-1).unsqueeze(-1)
        target_signs = torch.ge(target, 0).long()
        sign_loss = self.CE(sign_probs, target_signs)

        # Final weighted loss and predictions
        loss = mag_loss + self.sign_weight * sign_loss
        preds = torch.cat([mag_preds, sign_preds], dim=-1)
        return loss, preds, state

    def predict(self, X, lstm_state):
        pc, delta, types = X
        pc = self.pc_embed(pc)
        delta = self.delta_embed(delta)
        types = self.type_embed(types)

        lstm_in = torch.cat((pc, delta, types), dim=-1).unsqueeze(0)
        lstm_out, state = self.lstm(lstm_in, lstm_state)

        mag = self.lin_magnitude(lstm_out).squeeze()
        sign_probs = self.lin_sign(lstm_out).squeeze()

        mag_preds = self.m_soft.predict(mag)
        sign_preds = sign_probs.argmax(-1).unsqueeze(-1)
        preds = torch.cat([mag_preds, sign_preds], dim=-1)
        return preds, state

def main(argv):
    N = 3
    num_bits = 16
    splits = 4
    len_split = int(num_bits/splits)
    d = 2 * splits * (2**len_split)
    e_dim = 16

    net = MESoftNet(num_bits, e_dim, 5, 128, 2)
    pc = torch.randint(2**num_bits, (N,))
    delta = torch.randint(-(2**num_bits), 2**num_bits - 1, (N,))
    types = torch.randint(3, (N,))
    X = (pc, delta, types)
    target = torch.randint(-(2**num_bits), 2**num_bits - 1, (N,))

    loss, preds, state = net(X, None, target)
    print(loss)
    print(preds.shape)
    loss.backward()

    state = (state[0].detach(), state[1].detach())
    loss, preds, state = net(X, state, target)
    loss.backward()

if __name__ == "__main__":
    main(sys.argv)