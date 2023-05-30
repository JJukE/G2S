import os
import random

import wandb
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dpmpc_dataset import ShapeNetCore
from models.diffusionae import DiffusionAE
from utils.util import seed_all, CheckpointManager

from options.diffusionae_options import DiffusionAETestOptions
from jjuke.logger import CustomLogger
from jjuke.metrics import EMD_CD

from utils.visualizer import ObjectVisualizer

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))

#============================================================
# Evaluation
#============================================================

@torch.no_grad()
def eval(model, test_loader, args, ckpt_args, res_dir):
    if args.visualize:
        visualizer = ObjectVisualizer()
    
    all_ref = []
    all_recons = []
    all_labels = []
    for i, batch in enumerate(tqdm(test_loader)):
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        label = batch['cate']
        model.eval()
        
        code = model.encode(ref) # (B, z_dim)
        recons = model.decode(code, flexibility=ckpt_args.flexibility) # (B, num_points, 3)
        
        ref = ref * scale + shift
        recons = recons * scale + shift
        
        all_ref.append(ref.detach().cpu())
        all_recons.append(recons.detach().cpu())
        all_labels.append(label)
    
    all_ref = torch.cat(all_ref, dim=0) # (num_all_objects, num_points, 3)
    all_recons = torch.cat(all_recons, dim=0) # (num_all_objects, num_points, 3)
    all_labels = np.concatenate(all_labels, axis=0) # (num_all_objects)
    
    if args.visualize:
        logger.info('Saving point clouds...')
        ref_to_save = all_ref[:args.num_vis]
        recon_to_save = all_recons[:args.num_vis]
        label_to_save = all_labels[:args.num_vis]
        
        visualizer.save(ref_to_save, os.path.join(res_dir, "references.ply"))
        visualizer.save(recon_to_save, os.path.join(res_dir, "recons.ply"))
        np.save(os.path.join(res_dir, "labels.npy"), label_to_save)
    
    logger.info('Computing metrics...')
    metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.test_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    
    # if args.visualize: # visualize on the window
    #     visualizer.visualize(all_ref[:args.num_vis], num_in_row=8)
    logger.info('[Eval] CD {:.12f} | EMD {:.12f}'.format(cd, emd))

#============================================================
# Additional Arguments
#============================================================

def get_local_parser(parser):
    """Get training and additional arguments for evaluation
    Args:
        parser (TestOptions.parser)
    Returns:
        parser : New parser with additional arguments
    """
    # Additional arguments
    parser.add_argument('--categories', type=str_list, default=['all'])
    parser.add_argument('--scale_mode', type=str, default='shape_unit')
    parser.add_argument('--num_vis', type=int, default=50, help='Number of objects to visualize')
    
    return parser

#============================================================
# Main
#============================================================

if __name__ == '__main__':
    # Arguments for training
    args, arg_msg, device_msg = DiffusionAETestOptions().parse()

    if args.debug:
        args.data_dir = '/root/hdd1/DPMPC'
        args.name = 'G2S_DPMPC_practice_230527'
        args.gpu_ids = '0'
        args.exps_dir = '/root/hdd1/G2S/practice'
        args.ckpt_name = 'ckpt_100000.pt'
        args.test_batch_size = 128
        args.use_randomseed = False
        args.visualize = True

    # get logger, heckpoint manager and visualizer
    exp_dir = os.path.join(args.exps_dir, args.name, "ckpts")
    ckpt_path = os.path.join(exp_dir, args.ckpt_name)
    res_dir = os.path.join(args.exps_dir, args.name, "results")
    
    logger = CustomLogger(res_dir, isTrain=args.isTrain)
    ckpt_mgr = CheckpointManager(res_dir, isTrain=args.isTrain, logger=logger)
    
    logger.info(arg_msg)
    logger.info(device_msg)
    
    
    # set seed
    if not args.use_randomseed:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)

    # Datasets and loaders
    logger.info('Loading datasets...')
    dataset_path = os.path.join(args.data_dir, 'shapenet.hdf5')
    test_dataset = ShapeNetCore(
        path=dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.scale_mode,
    )

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.test_batch_size,
                            num_workers=0)

    # Model
    logger.info('Loading model...')
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = DiffusionAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])

    # Main loop
    logger.info('Start evaluation...')
    try:
        eval(model, test_loader, args, ckpt_args=ckpt['args'], res_dir=res_dir)

    except KeyboardInterrupt:
        logger.info('Terminating...')
        logger.flush()
        if args.use_wandb:
            wandb.finish()