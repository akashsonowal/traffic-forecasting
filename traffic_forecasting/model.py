import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ST_GAT(torch.nn.Module):
    """
    Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
    """

    def __init__(self, in_channels, out_channels, n_nodes, heads=8, dropout=0.0):
        """
        Initialize the ST-GAT model
        :param in_channels number of input channels
        :param out_channels number of output channels
        :param num_nodes number of nodes in the graph
        :param heads number of attention heads to use in graph
        :param droput Dropout probability on output of Graph Attention Network
        """
        super(ST_GAT, self).__init__()
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 9
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # single graph attention layer with 8 attention heads
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=heads,
            dropout=0,
            concat=False,
        )

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(
            input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1
        )
        for name, param in self.lstm1.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(
            input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1
        )
        for name, param in self.lstm2.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)

        # fully connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes * self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, data, device):
        """
        Forward pass of the ST-GAT model
        :param data Data to make a pass on
        :param device Device to operate on
        """
        x, edge_index = data.x, data.edge_index
        if device == "cpu":
            x = torch.FloatTensor(x)
        else:
            x = torch.cuda.FloatTensor(x)

        # gat layer: output of gat: (11400, 12) where 50 x 228 = 11400 where batch size is 50
        x = self.gat(x, edge_index)  # 11400, 12) -> (11400, 12)
        # apply dropout
        x = F.dropout(x, self.dropout, training=self.training)

        # RNN: 2 LSTMs
        # [batchsize*num_nodes, seq_length] -> [batchsize, num_nodes, seq_length]
        batch_size = data.num_graphs  # 50
        n_node = int(data.num_nodes / batch_size)  # 11400 / 50 = 228
        x = torch.reshape(x, (batch_size, n_node, data.num_features))  # (50, 228, 12)

        # for lstm: x should be (seq_length, batch_size, n_nodes)
        # sequence_length: 12, batch_size: 50, n_nodes: 228
        x = torch.movedim(x, 2, 0)  # (12, 50, 228) # time series sequence
        x, _ = self.lstm1(x)  # (12, 50, 228) -> (12, 50, 32)
        x, _ = self.lstm2(x)  # (12, 50, 32) -> (12, 50, 128)

        # Output contains h_t for each time_step, only the last one has all inputs accounted for.
        x = torch.squeeze(x[-1, :, :])  # (12, 50, 128) -> (50, 128)
        x = self.linear(x)  # (50, 128) -> (50, 228*9)

        # Now reshape into final output
        s = x.shape
        x = torch.reshape(
            x, (s[0], self.n_nodes, self.n_pred)
        )  # (50, 228*9) -> (50, 228, 9)
        x = torch.reshape(
            x, (s[0] * self.n_nodes, self.n_pred)
        )  # (50, 228, 9) -> (11400, 9)
        return x