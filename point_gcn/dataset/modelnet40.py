import os
import pickle

import numpy as np

from torch.utils.data import Dataset

from point_gcn.tools.utils import build_graph
from point_gcn.tools.utils import load_h5


class ModelNet40(Dataset):
    def __init__(self, data_path, num_points=1024, k=20, phase='train'):
        self.data_path = os.path.join(data_path, 'modelnet40_ply_hdf5_2048')
        self.num_points = num_points
        self.num_classes = 40

        # store data
        shape_name_file = os.path.join(self.data_path, 'shape_names.txt')
        self.shape_names = [line.rstrip() for line in open(shape_name_file)]
        self.coordinates = []
        self.labels = []
        try:
            files = os.path.join(self.data_path, '{}_files.txt'.format(phase))
            files = [line.rstrip() for line in open(files)]
            for index, file in enumerate(files):
                file_name = file.split('/')[-1]
                files[index] = os.path.join(self.data_path, file_name)
        except FileNotFoundError:
            raise ValueError('Unknown phase or invalid data path.')
        for file in files:
            current_data, current_label = load_h5(file)
            current_data = current_data[:, 0:self.num_points, :]
            self.coordinates.append(current_data)
            self.labels.append(current_label)
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)

        # build graph
        graph_path = os.path.join(data_path, 'graph_{}_{}.pkl'.format(k, phase))
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as handle:
                self.graphs = pickle.load(handle)
        else:
            self.graphs = []
            for i in range(self.coordinates.shape[0]):
                adj = build_graph(self.coordinates[i], k)
                self.graphs.append(adj)
            with open(graph_path, 'wb') as handle:
                pickle.dump(self.graphs, handle)

    def __len__(self):
        return self.coordinates.shape[0]

    def __getitem__(self, index):
        coord = self.coordinates[index].T  # 3 * N
        graph = np.array(self.graphs[index].todense())
        label = self.labels[index]
        return coord, graph, label


def main():
    loader = ModelNet40('../../data', phase='test')
    print(len(loader))
    print(len(loader.shape_names))
    print(loader.shape_names)
    for i in range(10):
        x, adj, label = loader.__getitem__(i)
        print(x.shape, adj.shape, label)


if __name__ == '__main__':
    main()
