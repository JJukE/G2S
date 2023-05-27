import argparse
import os
from utils import util
import torch


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options.
    """

    def __init__(self):
        
        parser = argparse.ArgumentParser()

        # basic parameters # TODO: DDP 추가, debug 모드 추가
        parser.add_argument('--data_dir', type=str, default='./dataset', help='path or directory to dataset')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--exps_dir', type=str, default='./exps', help='models and logs are saved here')

        # dataset parameters
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--shuffle', type=bool, default=True, help='data shuffling when training')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', type=bool, default=False, help='if specified, print more debugging information')
        parser.add_argument('--suffix', type=str, default='', help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        
        # debugging mode
        parser.add_argument('--debug', type=bool, default=True, help='debugging mode or not')

        self.parser = parser
        self.device = None

    def print_args(self, flag=False):
        """Print arguments
        It will print both current args and default values(if different).
        """
        message = ''
        message += '--------------- Arguments ---------------\n'
        for k, v in sorted(vars(self.args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        message += ''
        if flag:
            print(message)
        return message

    
    def print_device(self, flag=False):
        """Print device (if not using DDP)
        It will print the device to use
        """
        message = ''
        message += '--------------- Device ---------------\n'
        message += 'Device: '.format(self.device)
        message += 'Current cuda device: '.format(torch.cuda.current_device())
        message += '----------------- End -------------------'
        message += ''

        if flag:
            print(message)
        return message

    def parse(self, parser=None):
        """Parse our arguments, create checkpoints directory suffix, and set up gpu device."""
        if parser == None:
            args = self.parser.parse_args()
        else:
            self.parser = parser
            args = self.parser.parse_args()

        # process args.suffix
        if args.suffix:
            suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
            args.name = args.name + suffix

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
        
        args.device = self.device
        self.args = args
        return self.args
