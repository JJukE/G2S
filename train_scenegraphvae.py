import os
import random
import time

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data.sg_dataset import SceneGraphDataset, collate_fn_sgvae
from models.scenegraphvae import LayoutVAE
from utils.util import seed_all, CheckpointManager, print_model
from utils.viz_util import params_to_8points, batch_torch_denormalize_box_params
from options.scenegraphvae_options import SGVAETrainOptions
from jjuke.logger import CustomLogger

from utils.visualizer import SceneVisualizer
from utils.graph_visualizer import vis_graph

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))

#============================================================
# Training and Validation
#============================================================

def train_one_epoch(args, model, train_loader, optimizer):
    model.train()
    train_losses = {'recon_loss': [], 'angle_loss': [], 'KL_Gauss_loss': [], 'total_loss': []}
    for i, data in enumerate(tqdm(train_loader, desc='Train')):
        # skip invalid data
        if data == -1:
            continue
        
        try:
            objs = data['objs'].to(args.device)
            triples = data['triples'].to(args.device)
            tight_boxes = data['boxes'].to(args.device)

        except Exception as e:
            print('Exception', str(e))
            continue

        # TODO: avoid batches with insufficient number of instances with valid shape classes
        # mask = [ob in dataset.point_classes_idx for ob in dec_objs] # indices after eliminating underrepresented classes
        # if sum(mask) <= 1:
        #     continue
        
        optimizer.zero_grad()
        
        # separate boxes and angles from tight_boxes
        # limit the angle bin range from 0 to 24
        boxes = tight_boxes[:, :6]
        angles = tight_boxes[:, 6].long() - 1 
        angles = torch.where(angles > 0, angles, torch.zeros_like(angles))
        angles = torch.where(angles < 24, angles, torch.zeros_like(angles))
        
        boxes = boxes.to(args.device)
        angles = angles.to(args.device)
        
        attributes = None
        
        mu, logvar, boxes_pred, angles_pred = model(objs, triples, boxes,
                                                    angles=angles, attributes=attributes)
        
        # loss calculation
        total_loss = 0  
        recon_loss = F.l1_loss(boxes_pred, boxes)
        angle_loss = F.nll_loss(angles_pred, angles)
        try:
            gauss_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

        except:
            logger.warn("blowup!!!")
            logger.warn("logvar", torch.sum(logvar.data), torch.sum(torch.abs(logvar.data)), torch.max(logvar.data),
                torch.min(logvar.data))
            logger.warn("mu", torch.sum(mu.data), torch.sum(torch.abs(mu.data)), torch.max(mu.data), torch.min(mu.data))
        
        train_losses['recon_loss'].append(recon_loss.item())
        train_losses['angle_loss'].append(angle_loss.item())
        train_losses['KL_Gauss_loss'].append(gauss_loss.item())
        
        loss = args.box_weight * recon_loss + args.angle_weight * angle_loss + \
            args.kl_weight * gauss_loss
        
        train_losses['total_loss'].append(loss.item())
        
        loss.backward()

        # Cap the occasional super mutant gradient spikes
        # Do now a gradient step and plot the losses
        clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and p.requires_grad and torch.isnan(p.grad).any():
                    logger.info("NaN grad in {}th iteration.".format(i))
                    p.grad[torch.isnan(p.grad)] = 0
        
        optimizer.step()

    for key in train_losses.keys():
        train_losses[key] = np.asarray(train_losses[key])
        train_losses[key] = np.mean(train_losses[key])
    
    if args.use_wandb:
        wandb.log({"[Train] recon_loss": train_losses['recon_loss'],
                   "[Train] angle_loss": train_losses['angle_loss'],
                   "[Train] KL_Gauss_loss": train_losses['KL_Gauss_loss'],
                   "[Train] total_loss": train_losses['total_loss']})
    
    return train_losses

        

@torch.no_grad()
def val_one_epoch(args, model, val_loader, res_dir, epoch, logger, dataset):
    model.eval()
    val_losses = {'recon_loss': [], 'angle_loss': [], 'KL_Gauss_loss': [], 'total_loss': []}
    for i, data in enumerate(tqdm(val_loader, desc='Validate')):
        # skip invalid data
        if data == -1:
            continue
        
        try:
            objs = data['objs'].to(args.device)
            triples = data['triples'].to(args.device)
            tight_boxes = data['boxes'].to(args.device)

            instances = data['instance_id']
            scan = data['scan_id']
            split = data['split_id']

        except Exception as e:
            print('Exception', str(e))
            continue
        
        # separate boxes and angles from tight_boxes
        # limit the angle bin range from 0 to 24
        boxes = tight_boxes[:, :6]
        angles = tight_boxes[:, 6].long() - 1 
        angles = torch.where(angles > 0, angles, torch.zeros_like(angles))
        angles = torch.where(angles < 24, angles, torch.zeros_like(angles))

        boxes = boxes.to(args.device)
        angles = angles.to(args.device)
        
        attributes = None
        
        mu, logvar = model.vae_box.encoder(objs, triples, boxes,
                                                  angles_gt=angles, attributes=attributes)
        mu = mu.detach().cpu()
        mean_est = torch.mean(mu, dim=0, keepdim=True)
        mu = mu - mean_est
        cov_est = np.cov(mu.numpy().T)
        
        n, d = mu.size(0), mu.size(1)
        cov_est_ = np.zeros((d, d))
        for i in range(n):
            x = mu[i].numpy()
            cov_est_ += 1.0 / (n - 1.0) * np.outer(x, x)
        
        mean_est = mean_est[0]
        
        boxes_pred, angles_pred = model.sample_box(mean_est, cov_est_, objs, triples) # cov_est? cov_est_?
        
        boxes_denorm = batch_torch_denormalize_box_params(boxes)
        boxes_pred_denorm = batch_torch_denormalize_box_params(boxes_pred)
        
        # loss calculation
        total_loss = 0
        recon_loss = F.l1_loss(boxes_pred, boxes)
        angle_loss = F.nll_loss(angles_pred, angles)

        val_losses['recon_loss'].append(recon_loss.item())
        val_losses['angle_loss'].append(angle_loss.item())
        
        total_loss = args.box_weight * recon_loss + args.angle_weight * angle_loss
        
        val_losses['total_loss'].append(total_loss.item())

    for key in val_losses.keys():
        val_losses[key] = np.asarray(val_losses[key])
        val_losses[key] = np.mean(val_losses[key])
    
    if args.use_wandb:
        wandb.log({"[Val] recon_loss": val_losses['recon_loss'],
                   "[Val] angle_loss": val_losses['angle_loss'],
                   "[Val] total_loss": val_losses['total_loss']})
    
    if args.visualize and epoch % args.vis_freq == 0:
        angles_pred = torch.argmax(angles_pred, dim=1, keepdim=True) * 15.0 # 24 * 15 = 360

        save_path = os.path.join(res_dir, scan + "_" + split)
        classes = {}
        for k, v in dataset.classes_to_vis.items(): # {label: idx}
            classes[v] = k # {idx: label}
        
        objs_in_scene = []
        for global_id in objs.tolist():
            if global_id in classes.keys():
                objs_in_scene.append(classes[global_id])

        # only visualize the scene which contains more than 4 overlapped object labels with ShapeNet
        if len(objs_in_scene) >= 4:
            logger.info("Number of objects to visualize: {} in scan '{}'".format(len(objs_in_scene), scan))
            # visualize the scene graph
            vis_graph(use_sampled_graphs=False, scan_id=scan, split=str(split), data_dir=args.data_dir,
                    outfolder=res_dir, train_or_test='train')

            with open(save_path + "_info.txt", "w") as write_file:
                write_file.write("objects in the scene: \n{}".format(objs_in_scene))
            
            angles = angles.unsqueeze(1) # (9) -> (9, 1)
            box_points = np.zeros((len(objs_in_scene), 8, 3))
            box_points_pred = np.zeros((len(objs_in_scene), 8, 3))
            for i in range(len(objs_in_scene)):
                box_and_angle = torch.cat([boxes_denorm[i].float(), angles[i].float()])
                box_points[i] = params_to_8points(box_and_angle, degrees=False)
                box_and_angle_pred = torch.cat([boxes_pred_denorm[i].float(), angles_pred[i].float()])
                box_points_pred[i] = params_to_8points(box_and_angle_pred, degrees=False)
                
            visualizer = SceneVisualizer()
            visualizer.save(path=save_path + "_layoutGT", type='bb', boxes=box_points)
            visualizer.save(path=save_path + "_layout", type='bb', boxes=box_points_pred)
            
            # temporarily visualize on the window
            # visualizer.visualize(type='bb', boxes=box_points) # GT
            # visualizer.visualize(type='bb', boxes=box_points_pred) # pred
        else:
            logger.info("There's no sufficient number of objects to visualize in {}. ".format(scan) +
                        "Object labels overlapped with ShapeNet are lesser than 4.")
    
    return val_losses

#============================================================
# Main
#============================================================

if __name__ == '__main__':
    # Arguments for training
    args, arg_msg, device_msg = SGVAETrainOptions().parse()

    if args.debug:
        args.data_dir = '/root/hdd1/G2S/SceneGraphData'
        args.name = 'G2S_SGVAE_230609_all_graph_64_1_1_01'
        args.gpu_ids = '0' # only 0 is available while debugging
        args.exps_dir = '/root/hdd1/G2S/practice'
        args.verbose = True
        
        args.num_epochs = 200
        args.train_batch_size = 128
        args.num_treads = 8
        args.lr = 1e-5
        args.save_freq = 50

        args.use_wandb = False
        args.wandb_entity = 'ray_park'
        args.wandb_project_name = 'G2S'
        args.visualize = True
        args.vis_freq = 50
        
        args.gconv_dim = 64 # TODO: 32, 64, 128 비교
        args.residual = True
        args.box_weight = 1
        args.angle_weight = 1
        args.kl_weight = 0.1

    # get logger and checkpoint manager
    exp_dir = os.path.join(args.exps_dir, args.name, "ckpts")
    vis_dir = os.path.join(args.exps_dir, args.name, "validation")
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
    
    # # prepare pre-trained autoencoder used to shape generation
    # logger.info("Loading pre-trained autoencoder... ")
    # ckpt = torch.load(args.path2ae, map_location=args.device)
    # model = DiffusionAE(ckpt['args']).to(args.device)
    # model.load_state_dict(ckpt['state_dict'])

    # Datasets and loaders
    logger.info("Loading datasets...")
    dataset = SceneGraphDataset(
        args=args,
        data_dir=args.data_dir,
        categories=args.categories,
        split='train'
    )
    
    # randomly split the training dataset and validation dataset
    train_size = int(0.97 * len(dataset)) # 1141
    val_size = len(dataset) - train_size # 36
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.train_batch_size,
                            collate_fn=collate_fn_sgvae,
                            shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_batch_size,
                            collate_fn=collate_fn_sgvae,
                            shuffle=False)

    # Model
    logger.info('Building model...')
    if args.continue_train:
        logger.info("Continue training from checkpoint...")
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
        
        model = LayoutVAE(vocab=dataset.vocab, embedding_dim=args.gconv_dim,
                          residual=args.residual, gconv_pooling=args.pooling).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
    else:
        model = LayoutVAE(vocab=dataset.vocab, embedding_dim=args.gconv_dim,
                          residual=args.residual, gconv_pooling=args.pooling).to(args.device)

    if args.use_wandb:
        wandb.watch(model, log="all")
    logger.info(print_model(model, verbose=args.verbose))


    # Optimizer and scheduler
    params = filter(lambda x: x.requires_grad, list(model.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Main loop
    logger.info("Start training...")
    try:
        for epoch in range(1, args.num_epochs + 1):
            epoch_start_time = time.time()
            train_loss = train_one_epoch(args, model, train_loader, optimizer)
            logger.info('[Train] Epoch {}/{} | Loss {:.6f} | Time {:.4f} sec'.format(epoch,
                args.num_epochs, train_loss['total_loss'], time.time() - epoch_start_time))
            
            val_loss = val_one_epoch(args, model, val_loader, res_dir=vis_dir,
                                     epoch=epoch, logger=logger, dataset=dataset)
            logger.info('[Val] Loss {:.6f}'.format(val_loss['total_loss']))
            
            if epoch % args.save_freq == 0:
                opt_states = {
                    'optimizer': optimizer.state_dict()
                }
                ckpt_mgr.save(model, args, score=val_loss['total_loss'],
                            others=opt_states, step=epoch)

        logger.info("Training completed!")
        logger.flush()
        if args.use_wandb:
            wandb.finish()

    except KeyboardInterrupt:
        logger.info("Terminating...")
        logger.flush()
        if args.use_wandb:
            wandb.finish()