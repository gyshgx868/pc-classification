import torch
import torch.nn as nn

from point_gcn.tools import utils


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Conv1d(
            in_features, out_features, kernel_size=1, bias=bias
        )

    def forward(self, adj, x):
        x = torch.bmm(x, adj)
        x = self.conv(x)
        return x


class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x0 = self.max_pool(x).view(batch_size, -1)
        x1 = self.avg_pool(x).view(batch_size, -1)
        x = torch.cat((x0, x1), dim=-1)
        return x


def main():
    layer = GraphConvolution(in_features=3, out_features=64)
    print('Parameters:', utils.get_total_parameters(layer))
    x = torch.rand(4, 3, 1024)
    adj = torch.rand(4, 1024, 1024)
    y = layer(adj, x)
    print(y.size())

    pool = GlobalPooling()
    y = pool(y)
    print(y.size())


if __name__ == '__main__':
    main()
