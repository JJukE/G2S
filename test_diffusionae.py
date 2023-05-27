import os
import random

import wandb
import trimesh
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from data.dpmpc_dataset import ShapeNetCore
from models.diffusionae import DiffusionAE
from models.dpmpc import get_linear_scheduler
from utils.data import *
from utils.util import str_list, seed_all, CheckpointManager, print_model
from options.test_options import TestOptions
from jjuke.logger import CustomLogger
from jjuke.metrics import EMD_CD
from jjuke.pointcloud.transform import *

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))

#============================================================
# Evaluation
#============================================================

@torch.no_grad()
def eval(model, test_loader, args, ckpt_args): # TODO: comment shapes
    all_ref = []
    all_recons = []
    for i, batch in enumerate(tqdm(test_loader)):
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        model.eval()
        
        code = model.encode(ref) # ()
        recons = model.decode(code, flexibility=ckpt_args.flexibility) # ()
        
        ref = ref * scale + shift
        recons = recons * scale + shift
        
        all_ref.append(ref.detach().cpu())
        all_recons.append(recons.detach().cpu())
    
    all_ref = torch.cat(all_ref, dim=0) # ()
    all_recons = torch.cat(all_recons, dim=0) # ()
    
    logger.info('Saving point clouds...') # TODO: save the point clouds with visualizer
    
    logger.info('Computing metrics...')
    metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.test_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
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
    
    return parser

#============================================================
# Main
#============================================================

if __name__ == '__main__':
    # Arguments for training
    options = TestOptions()
    parser = options.parser
    parser = get_local_parser(parser)
    options.parser = parser
    args = options.parse()

    if args.debug:
        args.data_dir = '/root/hdd1/DPMPC'
        args.name = 'G2S_DPMPC_practice_230527'
        args.gpu_ids = '0'
        args.exps_dir = '/root/hdd1/G2S/practice'
        args.train_batch_size = 128
        args.use_wandb = True

    # get logger and checkpoint manager
    exp_dir = os.path.join(args.exps_dir, args.name)
    res_dir = os.path.join(exp_dir, args.results_dir)
    logger = CustomLogger(res_dir, isTrain=options.isTrain)
    ckpt_mgr = CheckpointManager(res_dir, logger=logger)
    logger.info(options.print_device())
    logger.info(options.print_args())
    
    # set seed
    if not args.use_seed:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)
    
    # set wandb
    if args.use_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=args.name + "_eval"
        )

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
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model = DiffusionAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])

    logger.info(print_model(model))


    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr
    )

    # Main loop
    logger.info('Start training...')
    try:
        # wandb setting
        if args.use_wandb:
            wandb.watch(model, model.diffusion.get_loss, log="all")

        it = 1
        while it <= args.max_iters:
            # Training
            train(it)
            
            # Validation
            if it % args.val_freq == 0 or it == args.max_iters:
                with torch.no_grad():
                    cd_loss = validate_loss(it)
                    
                    if args.visualize:
                        validate_inspect(it)

                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                # save model
                save_fname = "ckpt_{}.pt".format(int(it))
                ckpt_mgr.save(model, args, score=cd_loss, others=opt_states, step=it) # misc.py
            it += 1
        logger.info("Training completed!")
        logger.flush()
        if args.use_wandb:
            wandb.finish()

    except KeyboardInterrupt:
        logger.info('Terminating...')
        logger.flush()
        wandb.finish()