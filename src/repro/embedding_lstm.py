import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLSTM(nn.Module):
    def __init__(
        self,
        num_pc,
        num_input_delta,
        num_output_delta,
        embed_dim,
        hidden_dim,
        num_pred=10,  # how many predictions to return
        num_layers=2,  # number of LSTM layers
        dropout=0,  # probability with which to apply dropout
    ):
        super(EmbeddingLSTM, self).__init__()
        self.num_pred = num_pred

        # The concatenation of these two things will be the input to the LSTM
        self.pc_embed = nn.Embedding(num_pc, embed_dim)
        self.delta_embed = nn.Embedding(num_input_delta, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim * 2,
            hidden_dim,
            num_layers,
            dropout=dropout,
        )

        # Need to be able to map LSTM outputs to the output vocab
        self.fc = nn.Linear(hidden_dim, num_output_delta)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X, lstm_state, target=None):
        pc, delta = X
        pc_embed = self.pc_embed(pc)
        delta_embed = self.delta_embed(delta)
        lstm_in = torch.cat([pc_embed, delta_embed], dim=-1)

        # Unsqueeze a `batch` dimension (necessary for LSTM input)
        if len(lstm_in.shape) < 3:
            lstm_in = lstm_in.unsqueeze(dim=1)

        # Run the embeddings through the LSTM and get the top-K predictions
        # (`topk` returns tuple of values and indices, the indices represent
        #  the deltas themselves and the values are their probabilities)
        lstm_out, state = self.lstm(lstm_in, lstm_state)
        outputs = self.dropout(self.fc(lstm_out))

        # Squeeze to remove `batch` dim
        delta_probabilities = F.log_softmax(outputs, dim=-1).squeeze(dim=1)
        _, preds = torch.topk(delta_probabilities, self.num_pred, sorted=False)

        # Cross entropy loss (log softmax part was already performed)
        loss = F.nll_loss(delta_probabilities, target) if target is not None else None
        return loss, preds, state

    def predict(self, X, lstm_state):
        with torch.no_grad():
            _, preds, state = self.forward(X, lstm_state)
            return preds, state


def test_net():
    pc = torch.arange(0, 4)  # [0, 1, 2, 3]
    delta = torch.arange(3, -1, -1)  # [3, 2, 1, 0]
    target = torch.arange(0, 4)

    net = EmbeddingLSTM(4, 4, 4, 10, 30, num_pred=2)

    print("Testing forward pass of embedding LSTM")
    loss, preds, state = net((pc, delta), None, target)
    loss, preds, state = net((pc, delta), state, target)

    print(loss)
    print(preds.shape)
    print(state[0].shape)  # hidden state
    print(state[1].shape)  # cell state

    print("\nTesting prediction of embedding LSTM")
    preds, state = net.predict((pc, delta), None)
    preds, state = net.predict((pc, delta), state)

    print(preds.shape)
    print(state[0].shape)  # hidden state
    print(state[1].shape)  # cell state


if __name__ == "__main__":
    test_net()
