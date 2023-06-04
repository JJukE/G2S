import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.sg_dataset import SceneGraphDataset, collate_fn_sgvae
from models.diffusionae import DiffusionAE
from models.scenegraphvae import LayoutVAE
from options.diffusionae_options import DiffusionAETestOptions
from options.scenegraphvae_options import SGVAETestOptions
from utils.util import seed_all, CheckpointManager, print_model
from utils.viz_util import params_to_8points
from jjuke.logger import CustomLogger
from utils.visualizer import SceneVisualizer
from utils.graph_visualizer import vis_graph

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))


#============================================================
# Scene Composition
#============================================================

if __name__ == "__main__":
    # Arguments
    args_dpmpc, args_msg_dpmpc, _ = DiffusionAETestOptions().parse()
    args_sgvae, arg_msg_sgvae, _ = SGVAETestOptions().parse()
    
    if args_sgvae.debug:
        args_dpmpc.data_dir = "/root/hdd1/DPMPC"
        args_dpmpc.data_name = "3rlabel_shapenetdata.hdf5"
        args_dpmpc.name = 'G2S_DPMPC_practice_230527'
        args_dpmpc.gpu_ids = '0' # only 0 is available while debugging
        args_dpmpc.exps_dir = '/root/hdd1/G2S/practice'
        args_dpmpc.ckpt_name = 'ckpt_100000.pt'
        args_dpmpc.test_batch_size = 128
        args_dpmpc.use_randomseed = False
        args_dpmpc.visualize = True