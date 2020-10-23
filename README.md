# GCN for 3D Point Cloud Classification

This repository is a toy PyTorch implementation of GCN-based 3D point cloud classification model.

## Requirements

- Python3==3.7
- pytorch==1.4.0
- tensorboardX==2.0
- hdf5==1.10.4

## Dataset

To evaluate the model, `ModelNet40` dataset in HDF5 format are required to be downloaded and unzipped to the `data` folder.

Download `ModelNet40` dataset for classification task by running the following commands:

```shell script
cd ./data
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip
```

## Usage

To train the model, run

```shell script
python main.py --phase train --device 0 --train-batch-size 32
``` 

The log files, network parameters, and TensorBoard logs will be saved to `results` folder by default. We can use TensorBoard to view the training progress:

```shell script
tensorboard --logdir ./results/tensorboard
```

For more hyper-parameters, please refer to `point_gcn/tools/configuration.py`.

To evaluate the model, run

```shell script
python main.py --phase test --device 0 --test-batch-size 32 --weights [checkpoints] 
```

You should specify the `[checkpoints]`. For instance:

```shell script
python main.py --phase test --device 0 --test-batch-size 32 --weights ./results/models/model1.pt
```

## Acknowledgement

Our code is released under MIT License (see `LICENSE` for details).
