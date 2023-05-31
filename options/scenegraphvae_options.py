from .base_options import TrainOptions, TestOptions
from .base_options import int2tuple, str2tuple, int2list, str2list, str2bool


class SGVAETrainOptions(TrainOptions):
    """This class includes model-specific options for training.

    It also includes shared options defined in BaseOptions and TrainOptions.
    """

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)
        
        # Model arguments
        parser.add_argument('--gconv_dim', type=int, default=64, help='embedding dimension of graph convolution layer')
        parser.add_argument('--residual', type=str2bool, default=True, help="residual in GCN")
        parser.add_argument('--pooling', type=str, default='avg', help="pooling method in GCN")

        # Datasets and loaders
        parser.add_argument('--categories', type=str2list, default=['all'])
        parser.add_argument('--crop_useless', type=str2bool, default=False, help='whether to crop the useless objects')
        parser.add_argument('--center_scene_to_floor', type=str2bool, default=False, help='if crop_useless, this arg should be True too')
        parser.add_argument('--use_large_dataset', type=str2bool, default=True, help='whether to use large set of shape class labels')
        parser.add_argument('--use_canonical', type=str2bool, default=True)
        parser.add_argument('--use_scene_rels', type=str2bool, default=True, help='connect all nodes to a root scene node')
        parser.add_argument('--use_scene_splits', type=str2bool, default=False, help='True if you want to use splitted training data')
        parser.add_argument('--label_fname', type=str, default='labels.instances.align.annotated.ply')
        parser.add_argument('--shuffle_objs', type=str2bool, default=True, help='shuffle objs of a scene')
        parser.add_argument('--use_rio27', type=str2bool, default=False)

        # Loss, optimizer and scheduler
        parser.add_argument('--max_grad_norm', type=float, default=5)
        parser.add_argument('--kl_weight', type=float, default=0.1, help='weight of KL loss term')

        # Training
        
        return parser

class SGVAETestOptions(TestOptions):
    """This class includes model-specific options for evaluation.

    It also includes shared options defined in BaseOptions and TestOptions.
    """

    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)
        
        # Datasets and loaders
        parser.add_argument('--categories', type=str2list, default=['all'])
        parser.add_argument('--crop_useless', type=str2bool, default=False, help='whether to crop the useless objects')
        parser.add_argument('--center_scene_to_floor', type=str2bool, default=False, help='if crop_useless, this arg should be True too')
        parser.add_argument('--use_large_dataset', type=str2bool, default=True, help='whether to use large set of shape class labels')
        parser.add_argument('--use_canonical', type=str2bool, default=True)
        parser.add_argument('--use_scene_rels', type=str2bool, default=True, help='connect all nodes to a root scene node')
        parser.add_argument('--use_scene_splits', type=str2bool, default=False, help='True if you want to use splitted training data')
        parser.add_argument('--label_fname', type=str, default='labels.instances.align.annotated.ply')
        parser.add_argument('--shuffle_objs', type=str2bool, default=True, help='shuffle objs of a scene')
        parser.add_argument('--use_rio27', type=str2bool, default=False)
        
        return parser