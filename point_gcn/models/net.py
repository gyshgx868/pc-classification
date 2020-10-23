import torch

import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from point_gcn.models.layers import GlobalPooling
from point_gcn.models.layers import GraphConvolution
from point_gcn.tools import utils


class MultiLayerGCN(nn.Module):
    def __init__(self, dropout=0.5, num_classes=40):
        super(MultiLayerGCN, self).__init__()
        self.conv0 = GraphConvolution(3, 64, bias=False)
        self.conv1 = GraphConvolution(64, 64, bias=False)
        self.conv2 = GraphConvolution(64, 128, bias=False)
        self.conv3 = GraphConvolution(128, 256, bias=False)
        self.conv4 = GraphConvolution(512, 1024, bias=False)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)
        self.pool = GlobalPooling()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(2048, 512, bias=False)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),
            ('bn0', nn.BatchNorm1d(512)),
            ('drop0', nn.Dropout(p=dropout)),
            ('fc1', nn.Linear(512, 256, bias=False)),
            ('relu1', nn.LeakyReLU(negative_slope=0.2)),
            ('bn1', nn.BatchNorm1d(256)),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(256, num_classes)),
        ]))

    def forward(self, adj, x):
        x0 = F.leaky_relu(self.bn0(self.conv0(adj, x)), negative_slope=0.2)
        x1 = F.leaky_relu(self.bn1(self.conv1(adj, x0)), negative_slope=0.2)
        x2 = F.leaky_relu(self.bn2(self.conv2(adj, x1)), negative_slope=0.2)
        x3 = F.leaky_relu(self.bn3(self.conv3(adj, x2)), negative_slope=0.2)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = F.leaky_relu(self.bn4(self.conv4(adj, x)), negative_slope=0.2)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def main():
    features = torch.rand(4, 3, 1024)
    adj = torch.rand(4, 1024, 1024)
    model = MultiLayerGCN()
    print('Model:', utils.get_total_parameters(model))
    score = model(adj, features)
    print('Classification:', score.size())


if __name__ == '__main__':
    main()
