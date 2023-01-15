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
    self.gat = GATConv(in_channels=in_channels, out_channels=in_channels, heads=heads, dropout=0, concat=False)
    
    # add two LSTM layers
    self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
    for name, param in self.lstm1.named_parameters():
      if 'bias' in name:
        torch.nn.init.constant_(param, 0.0)
      elif 'weight' in name:
        torch.nn.xavier_uniform_(param)
    self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
    for name, param in self.lstm2.named_paramters():
      if 'bias' in name:
        torch.nn.init_constant_(param, 0.0)
      elif 'weight' in name:
        torch.nn.xavier_uniform_(param)
    
    # fully connected neural network
    self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_pred)
    torch.nn.xavier_uniform_(self.linear.weight)
    
 def forward(self, data, device):
  """
  Forward pass of the ST-GAT model
  :param data Data to make a pass on
  :param device Device to operate on
  """
  x, edge_index = data.x, data.edge_index
  # apply dropout
  if device == "cpu":
    x = torch.FloatTensor(x)
  else:
    x = torch.cuda.FloatTensor(x)
  
  # gat layer: output of gat: [11400, 12]
  x = self.gat(x, edge_index)
      
    
