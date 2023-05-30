import os
import random
import time

import wandb
import trimesh
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data.sg_dataset import SGDataset
from models.scenegraphvae import SceneGraphVAE
from models.dpmpc import get_linear_scheduler
from utils.util import seed_all, CheckpointManager, print_model
from options.diffusionae_options import DiffusionAETrainOptions
from jjuke.logger import CustomLogger
from jjuke.metrics import EMD_CD
from jjuke.pointcloud.transform import RandomRotate

from utils.visualizer import ObjectVisualizer

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))

#============================================================
# Training and Validation
#============================================================

def train_one_epoch():
    for i, data in enumerate(train_loader):
        # skip invalid data
        if data == -1:
            continue
        
        try:
            enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'],\
                        data['encoder']['triplets'], data['encoder']['boxes'], data['encoder']['obj_to_scene'], data['encoder']['tiple_to_scene']

            enc_points = data['encoder']['points']
            enc_points = enc_points.cuda()

            dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'],\
                        data['decoder']['triplets'], data['decoder']['boxes'], data['decoder']['obj_to_scene'], data['decoder']['tiple_to_scene']

            if 'points' in data['decoder']:
                dec_points = data['decoder']['points']
                dec_points = dec_points.cuda()

        except Exception as e:
            print('Exception', str(e))
            continue
        

def train(iter):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model(x)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    if args.use_wandb:
        wandb.log({"train_iter": iter, "train_loss": loss})

    if iter % 1000 == 0:
        logger.info('[Train] Iter {:04d} | Loss {:.6f} | Grad {:.4f} '.format(it, loss.item(), orig_grad_norm))

def validate_loss(iter):

    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code, flexibility=args.flexibility)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    
    if args.use_wandb:
        wandb.log({"val_iter": iter, "CD_loss_val": cd, "EMD_loss_val": emd})
    logger.info('[Val] Iter {:04d} | CD {:.6f} | EMD {:.6f}  '.format(iter, cd, emd))

    return cd


def validate_inspect(iter):
    visualizer = ObjectVisualizer()
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch['pointcloud'].to(args.device)
        if i == 0:
            logger.info('refence category: {} in {}th iteration'.format(batch['cate'][i], iter))
        model.eval()
        code = model.encode(x)
        recons = model.decode(code, flexibility=args.flexibility).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch
    
    arr_pc = recons.cpu().detach().numpy().reshape(-1,3) # only visualize the first batch(first category)

    visualizer.visualize(arr_pc)

#============================================================
# Main
#============================================================

if __name__ == '__main__':
    # Arguments for training
    args, arg_msg, device_msg = DiffusionAETrainOptions().parse()

    if args.debug:
        args.data_dir = '/root/hdd1/G3D/GT'
        # args.data_dir_3RScan = '/root/hdd1/G3D/3RScan'
        args.name = 'G2S_SGVAE_practice_230529'
        args.gpu_ids = '0'
        args.exps_dir = '/root/hdd1/G2S/practice'
        args.train_batch_size = 48
        args.num_treads = 8
        args.lr = 0.0001
        args.num_epochs = 200
        
        args.path2ae = '/root/hdd1/G2S/practice/G2S_DPMPC_practice_230529/ckpts/ckpt_100000.pt'
        
        args.use_wandb = True
        args.visualize = False

    # get logger and checkpoint manager
    exp_dir = os.path.join(args.exps_dir, args.name, "ckpts")
    logger = CustomLogger(exp_dir, isTrain=args.isTrain)
    ckpt_mgr = CheckpointManager(exp_dir, isTrain=args.isTrain, logger=logger)
    logger.info(arg_msg)
    logger.info(device_msg)
    
    # set seed
    if args.use_randomseed:
        args.seed = random.randint(1, 10000)
    logger.info("Seed: {}".format(args.seed))
    seed_all(args.seed)

    # set wandb
    if args.use_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=args.name + "_train"
        )
    
    # prepare pre-trained autoencoder used to shape generation
    logger.info("Loading pre-trained autoencoder... ")
    ckpt = torch.load(args.path2ae, map_location=args.device)
    model = DiffusionAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])

    # Datasets and loaders : TODO
    logger.info("Loading datasets...")
    train_dataset = SGDataset(
        path=args.data_dir
        # TODO
    )
    val_dataset = SGDataset(
        path=args.data_dir,
        # TODO
    )

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.train_batch_size,
                            shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_batch_size,
                            shuffle=False)

    # Model
    logger.info('Building model...')
    if args.continue_train:
        logger.info("Continue training from checkpoint...")
        ckpt = torch.load(args.ckpt_path)
        model = SceneGraphVAE(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
    else:
        model = SceneGraphVAE(args).to(args.device)

    if args.use_wandb:
        wandb.watch(model, log="all")
    logger.info(print_model(model))


    # Optimizer and scheduler
    params = filter(lambda x: x.requires_grad, list(model.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Main loop
    logger.info("Start training...")
    try:
        for epoch in range(args.num_epochs):
            epoch_start_time = time.time()
            train_one_epoch()
            logger.info("End of epoch {}/{} \t Time taken: {} sec".format(epoch, args.num_epochs, time.time() - epoch_start_time))

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
                        logger.info("Continue training...")

                opt_states = {
                    'optimizer': optimizer.state_dict()
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
        logger.info("Terminating...")
        logger.flush()
        if args.use_wandb:
            wandb.finish()