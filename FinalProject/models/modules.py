import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input, adj):
        """
        Forward computation assumes graphs have the same number of images
        in a given batch. Otherwise we should implement some masking tricks.

        Parameters
        ----------
        input : torch.Tensor
            Tensor shape should be [batch, n_imgs, ch, height, width].
        adj : torch.Tensor
            Adjacency matrix (including self-loops) of the graph.
            Shape should be [batch, n_imgs, n_imgs]
        """
        T = torch.sqrt(1 / adj.sum(dim=1))
        D = torch.stack([torch.diag(x) for x in T])
        A_bar = (D @ adj @ D)

        shape = input.shape
        # print("!!!input shape for one GCN layer", shape)
        input = input.reshape(-1, shape[2], shape[3])
        # input = input.reshape(-1, 1, shape[2])
        # input = input.repeat(1, self.out_channels, 1) # [batch, 77, clip_dim]
        # print("!!!input shape for one GCN layer after", input.shape)
        h = self.conv(input)
        h = h.reshape(shape[0], shape[1], -1)
        output = (A_bar @ h).reshape(shape[0], shape[1], -1, shape[3])

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GraphConvolution(in_channels, out_channels, bias))
            else:
                self.layers.append(GraphConvolution(out_channels, out_channels, bias))
        self.num_layers = num_layers

    def forward(self, x, A):
        for i in range(self.num_layers):
            x = self.layers[i](x, A)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5)
        return x

# class CLIPAdapter(nn.Module):
#     def __init__(self, clip_dim=768, hidden_dim=768, seq_len=77, rank=64):
#         super().__init__()
#         self.seq_len = seq_len
#         self.hidden_dim = hidden_dim

#         self.mlp = nn.Sequential(
#             nn.Linear(clip_dim, rank),
#             nn.ReLU(),
#             nn.Linear(rank, hidden_dim * seq_len)
#         )
    
#     def forward(self, image_embed):
#         """
#         image_embed: (batch_size, clip_dim)
#         returns: (batch_size, seq_len, hidden_dim)
#         """
#         x = self.mlp(image_embed)  # (batch_size, seq_len * hidden_dim)
#         x = x.view(-1, self.seq_len, self.hidden_dim)
#         return x

class CLIPAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)