from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()

        # network saving and loading parameters
        self.parser.add_argument('--val_freq', type=int, default=5000, help='frequency of validation(also saving model)')
        self.parser.add_argument('--save_by_iter', type=bool, default=False, help='whether saves model by iteration')
        self.parser.add_argument('--continue_train', type=bool, default=False, help='continue training: load the latest model')
        self.parser.add_argument('--ckpt_path', type=str, default='./exps/ckpt', help='checkpoint path to start training')

        # training parameters
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--visualize', type=bool, default=False, help='whether visualize in validation')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        self.parser.add_argument('--max_iters', type=int, default=500000, help='maximum iterations of training (if trained by iterations, not epochs)')
        self.parser.add_argument('--train_batch_size', type=int, default=128)
        self.parser.add_argument('--val_batch_size', type=int, default=32)

        # wandb parameters
        self.parser.add_argument('--use_wandb', type=bool, default=False, help='if specified, then init wandb logging')
        self.parser.add_argument('--wandb_entity', type=str, default='ray_park', help='user name of wandb')
        self.parser.add_argument('--wandb_project_name', type=str, default='G2S', help='specify wandb project name')

        self.isTrain = True
