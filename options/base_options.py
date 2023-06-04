import argparse
import os
from utils import util
import torch

"""BaseOptions, TrainOptions, TestOptions"""


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options.
    """

    def __init__(self):

        self.initialized = False
        self.device = None
    
    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        
        # basic parameters # TODO: DDP 추가, debug 모드 추가
        parser.add_argument('--data_dir', type=str, default='./dataset', help='path or directory to dataset')
        parser.add_argument('--data_name', type=str, default='datafile', help='filename of the dataset')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--exps_dir', type=str, default='./exps', help='models and logs are saved here')

        # dataset parameters
        parser.add_argument('--num_threads', default=4, type=int, help='# threads(workers) for loading data')
        parser.add_argument('--shuffle', type=str2bool, default=True, help='data shuffling when training')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', type=str2bool, default=False, help='if specified, print more debugging information')
        
        # debugging mode
        parser.add_argument('--debug', type=str2bool, default=True, help='debugging mode or not')
        
        # setting seed
        parser.add_argument('--use_randomseed', type=str2bool, default=False, help='if True, use specified seed, else, use random seed')
        parser.add_argument('--seed', type=int, default=2023)
        
        self.initialized = True
        return parser

    def print_args(self):
        """Print arguments
        It will print both current args and default values(if different).
        """
        message = 'Arguments info\n'
        message += '--------------- Arguments ---------------\n'
        for k, v in sorted(vars(self.args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        message += ''
        
        return message

    
    def print_device(self):
        """Print device (if not using DDP)
        It will print the device to use
        """
        message = 'Device info\n'
        message += '--------------- Device ---------------\n'
        message += 'Device: {} \n'.format(self.device)
        message += 'Current cuda device: {} \n'.format(torch.cuda.current_device())
        message += '----------------- End -------------------'
        message += ''

        return message

    def parse(self, parser=None):
        """Parse our arguments, create checkpoints directory suffix, and set up gpu device."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        self.parser = parser
        args = parser.parse_args()

        # set gpu ids
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) # TODO: DDP 활용 시 변경
            torch.cuda.set_device(self.device)
        else:
            raise ValueError('Invalid gpu ID specified. Please provide at least one valid GPU ID.')
        
        args.device = self.device
        args.isTrain = self.isTrain
        self.args = args
        
        arg_message = self.print_args()
        device_message = self.print_device()
        return self.args, arg_message, device_message


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # network saving and loading parameters
        parser.add_argument('--val_freq', type=int, default=10, help='frequency of validation when training by iteration')
        parser.add_argument('--vis_freq', type=int, default=10, help='frequency of visualization in validation')
        parser.add_argument('--save_freq', type=int, default=20, help='frequency of saving model.')
        parser.add_argument('--save_by_iter', type=str2bool, default=False, help='whether saves model by iteration')
        parser.add_argument('--continue_train', type=str2bool, default=False, help='continue training: load the latest model')
        parser.add_argument('--ckpt_path', type=str, default='./exps/ckpt', help='checkpoint path to start training')

        # training parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--visualize', type=str2bool, default=False, help='whether visualize in validation')
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--max_iters', type=int, default=500000, help='maximum iterations of training (if trained by iterations, not epochs)')
        parser.add_argument('--train_batch_size', type=int, default=128)
        parser.add_argument('--val_batch_size', type=int, default=32)

        # wandb parameters
        parser.add_argument('--use_wandb', type=str2bool, default=False, help='if specified, then init wandb logging')
        parser.add_argument('--wandb_entity', type=str, default='ray_park', help='user name of wandb')
        parser.add_argument('--wandb_project_name', type=str, default='G2S', help='specify wandb project name')

        self.isTrain = True
        return parser


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # result parameters
        parser.add_argument('--visualize', type=str2bool, default=False, help='whether visualize in evaluation')
        
        # evaluation parameters
        parser.add_argument('--ckpt_name', type=str, default='ckpt.pt', help='checkpoint file name to start evaluation')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--test_batch_size', type=int, default=128)

        # wandb parameters
        parser.add_argument('--use_wandb', type=str2bool, default=False, help='if specified, then init wandb logging')
        parser.add_argument('--wandb_entity', type=str, default='ray_park', help='user name of wandb')
        parser.add_argument('--wandb_project_name', type=str, default='G2S', help='specify wandb project name')

        self.isTrain = False
        return parser

#============================================================
# for type setting
#============================================================

def int2tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str2tuple(argstr):
    return tuple(argstr.split(','))


def int2list(argstr):
    return list(map(int, argstr.split(',')))


def str2list(argstr):
    return list(argstr.split(','))

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")