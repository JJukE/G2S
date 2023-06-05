from .base_options import TrainOptions, TestOptions
from .base_options import int2tuple, str2tuple, int2list, str2list, str2bool


class DiffusionAETrainOptions(TrainOptions):
    """This class includes model-specific options for training.

    It also includes shared options defined in BaseOptions and TrainOptions.
    """

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)
        
        # Model arguments
        parser.add_argument('--latent_dim', type=int, default=128)
        parser.add_argument('--num_steps', type=int, default=200)
        parser.add_argument('--num_points', type=int, default=2048)
        parser.add_argument('--beta_1', type=float, default=1e-4)
        parser.add_argument('--beta_T', type=float, default=0.05)
        parser.add_argument('--sched_mode', type=str, default='linear')
        parser.add_argument('--flexibility', type=float, default=0.0)
        parser.add_argument('--residual', type=str2bool, default=True)
        parser.add_argument('--resume', type=str, default=None)

        # Datasets and loaders
        parser.add_argument('--categories', type=str2list, default=['all'])
        parser.add_argument('--scale_mode', type=str, default='shape_unit')
        parser.add_argument('--rotate', type=str2bool, default=False)

        # Optimizer and scheduler
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--max_grad_norm', type=float, default=10)
        parser.add_argument('--end_lr', type=float, default=1e-4)
        parser.add_argument('--sched_start_epoch', type=int, default=150000)
        parser.add_argument('--sched_end_epoch', type=int, default=300000)

        # Training
        parser.add_argument('--num_val_batches', type=int, default=-1)
        
        return parser

class DiffusionAETestOptions(TestOptions):
    """This class includes model-specific options for evaluation.

    It also includes shared options defined in BaseOptions and TestOptions.
    """

    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)
        
        parser.add_argument('--categories', type=str2list, default=['all'])
        parser.add_argument('--num_points', type=int, default=2048)
        parser.add_argument('--scale_mode', type=str, default='shape_unit')
        parser.add_argument('--num_vis', type=int, default=50, help='Number of objects to visualize') 
        
        return parser