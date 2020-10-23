import argparse

from point_gcn.tools.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        description='GCN for 3D Point Cloud Classification'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='path to the configuration file'
    )

    # model hyper-parameters
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='dropout for classifier'
    )
    parser.add_argument(
        '--knn',
        type=int,
        default=20,
        help='number of k nearest neighbors'
    )

    # runner
    parser.add_argument(
        '--use-cuda',
        type=str2bool,
        default='true',
        help='whether to use GPUs'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=[0, 1, 2, 3],
        nargs='+',
        help='the indices of GPUs for training or testing'
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='train',
        help='it must be \'train\', or \'test\''
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=32,
        help='training batch size'
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=32,
        help='testing batch size'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=512,
        help='maximum number of training epochs'
    )
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for model initialization'
    )
    parser.add_argument(
        '--ignore',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored during initialization'
    )
    parser.add_argument(
        '--eval-model',
        type=str2bool,
        default=True,
        help='if true, the model will be evaluated during training'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#epoch)'
    )

    # dataset
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data',
        help='dataset path'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='point_gcn.dataset.ModelNet40',
        help='dataset provider full class name'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=1024,
        help='number of points to use for classification'
    )

    # optimizer
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help='optimizer to use'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='initial learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum (default: 0.9)'
    )

    # logging
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='path to save results'
    )
    parser.add_argument(
        '--show-details',
        type=str2bool,
        default=True,
        help='whether to show the main classification metrics'
    )
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logs or not'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        help='the interval for printing logs (#iteration)'
    )
    parser.add_argument(
        '--save-model',
        type=str2bool,
        default=True,
        help='if true, the model will be stored'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#epoch)'
    )
    parser.add_argument(
        '--use-tensorboard',
        type=str2bool,
        default='true',
        help='whether to use TensorBoard to visualize results'
    )

    return parser


def main():
    import json
    p = get_parser()
    js = json.dumps(vars(p), indent=2)
    print(js)


if __name__ == '__main__':
    main()
