import yaml

from point_gcn.runner import TestRunner
from point_gcn.runner import TrainRunner
from point_gcn.tools.configuration import get_parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    args = parser.parse_args()

    if args.phase == 'train':
        runner = TrainRunner(args)
    elif args.phase == 'test':
        runner = TestRunner(args)
    else:
        raise ValueError('Unknown phase.')

    runner.run()


if __name__ == '__main__':
    main()
