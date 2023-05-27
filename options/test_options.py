from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()
        
        # result parameters
        self.parser.add_argument('--visualize', type=bool, default=False, help='whether visualize in evaluation')
        self.parser.add_argument('--results_name', type=str, default='results', help='saves results here. It will be subfolder of "exps_dir/name".')
        
        # evaluation parameters
        self.parser.add_argument('--ckpt_path', type=str, default='./exps/ckpt.pt', help='checkpoint path to start evaluation')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--test_batch_size', type=int, default=128)

        # wandb parameters
        self.parser.add_argument('--use_wandb', type=bool, default=False, help='if specified, then init wandb logging')
        self.parser.add_argument('--wandb_entity', type=str, default='ray_park', help='user name of wandb')
        self.parser.add_argument('--wandb_project_name', type=str, default='G2S', help='specify wandb project name')

        self.isTrain = False