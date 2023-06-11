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

from data.dpmpc_dataset import ShapeNetDataset
from data.sg_dataset import SceneGraphDataset, collate_fn_sgvae
from models.diffusionae import DiffusionAE
from models.scenegraphvae import LayoutVAE
from options.base_options import TestOptions
from options.diffusionae_options import DiffusionAETestOptions
from options.scenegraphvae_options import SGVAETestOptions
from utils.util import seed_all, CheckpointManager, print_model
from utils.viz_util import params_to_8points, fit_shapes_to_box, batch_torch_denormalize_box_params
from jjuke.logger import CustomLogger
from utils.visualizer import SceneVisualizer
from utils.graph_visualizer import vis_graph

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))


#============================================================
# Scene generation
#============================================================

@torch.no_grad()
def generate_shape(model, test_loader, args, ckpt_args):
    model.eval()
    
    all_recons = []
    all_labels = []
    for i, batch in enumerate(tqdm(test_loader)):
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        label = batch['cate']
        
        code = model.encode(ref) # (B, z_dim)
        recons = model.decode(code, flexibility=ckpt_args.flexibility) # (B, num_points, 3)
        recons = recons * scale + shift
        
        all_recons.append(recons.detach().cpu())
        all_labels.append(label)
    
    all_recons = torch.cat(all_recons, dim=0) # (num_all_objects, num_points, 3)
    all_labels = np.concatenate(all_labels, axis=0) # (num_all_objects)

    # swap y-axis values and z-axis values
    recons = torch.zeros_like(all_recons)
    recons[:, :, 0], recons[:, :, 1], recons[:, :, 2] = \
        all_recons[:, :, 0], all_recons[:, :, 2], all_recons[:, :, 1]
    
    points = {} # {label: points}
    for i in range(len(all_labels)):
        if str(all_labels[i]) not in points.keys():
            points.update({str(all_labels[i]): [all_recons[i]]})
        else:
            max_val, _ = recons[i].max(dim=-2)
            min_val, _ = recons[i].min(dim=-2)
            recons[i] -= (max_val.unsqueeze(0) + min_val.unsqueeze(0)) / 2.0
            points[str(all_labels[i])].append(recons[i])
    return points


@torch.no_grad()
def generate_scene(args, model, test_loader, res_dir, logger, shapes, num_points, dataset=None):
    model.eval()
    test_losses = {'recon_loss': [], 'angle_loss': [], 'KL_Gauss_loss': [], 'total_loss': []}
    for i, data in enumerate(tqdm(test_loader, desc='Evaluate')):
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
        
        boxes_pred, angles_pred = model.sample_box(mean_est, cov_est_, objs, triples)
        
        boxes_denorm = batch_torch_denormalize_box_params(boxes)
        boxes_pred_denorm = batch_torch_denormalize_box_params(boxes_pred)
        
        # loss calculation
        total_loss = 0
        recon_loss = F.l1_loss(boxes_pred, boxes)
        angle_loss = F.nll_loss(angles_pred, angles)

        test_losses['recon_loss'].append(recon_loss.item())
        test_losses['angle_loss'].append(angle_loss.item())
        
        total_loss = recon_loss + angle_loss
        
        test_losses['total_loss'].append(total_loss.item())
        
        logger.info("<Scene id: {}>".format(scan))
        logger.info("Reconstruction loss: {:.6f}".format(recon_loss.item()))
        logger.info("Angle loss: {:.6f}".format(angle_loss.item()))
        
        if args.visualize:
            angles_pred = torch.argmax(angles_pred, dim=1, keepdim=True) * 15.0 # 24 * 15 = 360

            save_path = os.path.join(res_dir, scan + "_" + split)
            classes = {}
            for k, v in dataset.classes_to_vis.items(): # {label: idx}
                classes[v] = k # {idx: label}

            objs_in_scene = []
            for global_id in objs.tolist():
                if global_id in classes.keys():
                    objs_in_scene.append(classes[global_id])

            if len(objs_in_scene) >= 4:
                logger.info("Number of objects to visualize: {}".format(len(objs_in_scene)))

                # visualize the scene graph
                vis_graph(use_sampled_graphs=False, scan_id=scan, split=str(split), data_dir=args.data_dir,
                        outfolder=res_dir, train_or_test='test')
                
                with open(save_path + "_info.txt", "w") as write_file:
                    write_file.write("objects in the scene: \n{}".format(objs_in_scene))
                
                # visualize the scene for only trained objects     
                points = np.zeros((len(objs_in_scene), 2048, 3))
                points_pred = np.zeros((len(objs_in_scene), 2048, 3))

                for i in range(len(objs_in_scene)):
                    idx = random.randint(0, len(shapes[objs_in_scene[i]]) - 1)
                    points[i] = shapes[objs_in_scene[i]][idx]
                    points_pred[i] = shapes[objs_in_scene[i]][idx]
                
                angles = angles.unsqueeze(1) # (9) -> (9, 1)
                box_points = np.zeros((len(objs_in_scene), 8, 3))
                box_points_pred = np.zeros((len(objs_in_scene), 8, 3))
                for i in range(len(objs_in_scene)):
                    box_and_angle = torch.cat([boxes_denorm[i].float(), angles[i].float()])
                    box_points[i] = params_to_8points(box_and_angle, degrees=False)
                    points[i] = fit_shapes_to_box(box_and_angle, points[i])
                    box_and_angle_pred = torch.cat([boxes_pred_denorm[i].float(), angles_pred[i].float()])
                    box_points_pred[i] = params_to_8points(box_and_angle_pred, degrees=False)
                    points_pred[i] = fit_shapes_to_box(box_and_angle_pred, points_pred[i])
                
                visualizer = SceneVisualizer()
                visualizer.save(path=(save_path + "_layoutGT"), type='all', shape_type="pc",
                                boxes=box_points, points=points)
                visualizer.save(path=(save_path + "_layout"), type='all', shape_type="pc",
                                boxes=box_points_pred, points=points_pred)
                
                # # temporarily visualize on the window
                # visualizer.visualize(type="all", shape_type="pc", boxes=box_points, points=points) # GT
                # visualizer.visualize(type="all", shape_type="pc", boxes=box_points_pred, points=points_pred) # pred
            else:
                logger.info("There's no sufficient number of objects to visualize in {}. ".format(scan) +
                            "Object labels overlapped with ShapeNet are lesser than 4.")

    for key in test_losses.keys():
        test_losses[key] = np.asarray(test_losses[key])
        test_losses[key] = np.mean(test_losses[key])
    
    return test_losses



#============================================================
# Scene Composition
#============================================================

if __name__ == "__main__":
    # Arguments
    args_dpmpc, arg_msg_dpmpc, device_msg_dpmpc = DiffusionAETestOptions().parse()
    args_sgvae, arg_msg_sgvae, device_msg_sgvae = SGVAETestOptions().parse()
    args, args_msg, device_msg = TestOptions().parse()
    
    if args_sgvae.debug:
        # for shape generation
        args_dpmpc.data_dir = "/root/hdd1/DPMPC"
        args_dpmpc.data_name = "shapenet.hdf5"
        args_dpmpc.name = 'G2S_DPMPC_practice_230529'
        args_dpmpc.gpu_ids = '0' # only 0 is available while debugging
        args_dpmpc.exps_dir = '/root/hdd1/G2S/practice'
        args_dpmpc.ckpt_name = 'ckpt_200000.pt'
        args_dpmpc.test_batch_size = 128
        args_dpmpc.use_randomseed = True
        args_dpmpc.visualize = True
        
        # for layout generation
        args_sgvae.data_dir = '/root/hdd1/G2S/SceneGraphData'
        args_sgvae.name = 'G2S_SGVAE_230609_all_graph_1e-5_32'
        args_sgvae.gpu_ids = '0' # only 0 is available while debugging
        args_sgvae.exps_dir = '/root/hdd1/G2S/practice'
        args_sgvae.ckpt_name = "ckpt_100.pt"
        args_sgvae.verbose = False
        
        args_sgvae.test_batch_size = 1
        args_sgvae.visualize = True
        args_sgvae.use_randomseed = True
        
        # for scene composition
        args.exps_dir = '/root/hdd1/G2S/exps'
        args.name = 'G2S_all_graph_1e-5_32'
        args.gpu_ids = '0' # only 0 is available while debugging
        args.use_randomseed = True
        
    # get logger
    dpmpc_exp_dir = os.path.join(args_dpmpc.exps_dir, args_dpmpc.name, "ckpts")
    dpmpc_ckpt_path = os.path.join(dpmpc_exp_dir, args_dpmpc.ckpt_name)
    
    sgvae_exp_dir = os.path.join(args_sgvae.exps_dir, args_sgvae.name, "ckpts")
    sgvae_ckpt_path = os.path.join(sgvae_exp_dir, args_sgvae.ckpt_name)
    
    res_dir = os.path.join(args.exps_dir, args.name, "results")
    
    logger = CustomLogger(res_dir, isTrain=args.isTrain)
    
    logger.info("Shape generation model info:")
    logger.info(arg_msg_dpmpc)
    logger.info(device_msg_dpmpc)
    logger.info("Layout generation model info:")
    logger.info(arg_msg_sgvae)
    logger.info(device_msg_sgvae)
    
    if args.use_randomseed:
        args.seed = random.randint(1, 10000)
        args_sgvae.seed = args.seed
        args_dpmpc.seed = args.seed
    seed_all(args.seed)
    logger.info("Seed: {}".format(args.seed))
    
    # Shape generation
    start_time = time.time()
    logger.info("Loading DPMPC datasets...")
    dataset_path = os.path.join(args_dpmpc.data_dir, args_dpmpc.data_name)
    test_dataset = ShapeNetDataset(
        path=dataset_path,
        cates=args_dpmpc.categories,
        split='test',
        scale_mode=args_dpmpc.scale_mode
    )
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args_dpmpc.test_batch_size,
                             num_workers=0)
    
    logger.info("Loading DPMPC model...")
    ckpt = torch.load(dpmpc_ckpt_path, map_location=args.device)
    model = DiffusionAE(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    
    logger.info("Start shape generation...")
    try:
        points = generate_shape(model, test_loader, args_dpmpc, ckpt_args=ckpt['args'])
        logger.info("Shape generation completed!")
        
    except KeyboardInterrupt:
        logger.info("Terminating...")
        logger.flush()
    
    # Layout generation
    logger.info("Loading SGVAE datasets...")
    test_dataset = SceneGraphDataset(
        args=args_sgvae,
        data_dir=args_sgvae.data_dir,
        categories=args_sgvae.categories,
        split='test'
    )
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args_sgvae.test_batch_size,
                             collate_fn=collate_fn_sgvae,
                             shuffle=False)
    
    logger.info("Loading SGVAE model...")
    ckpt = torch.load(sgvae_ckpt_path, map_location=args_sgvae.device)
    model = LayoutVAE(vocab=test_dataset.vocab, embedding_dim=ckpt['args'].gconv_dim,
                        residual=ckpt['args'].residual, gconv_pooling=ckpt['args'].pooling).to(args_sgvae.device)
    model.load_state_dict(ckpt['state_dict'])

    logger.info(print_model(model, verbose=args_sgvae.verbose))
    
    logger.info("Start layout generation and scene composition")
    try:
        generate_scene(args_sgvae, model, test_loader, res_dir, shapes=points,
                       num_points=args_dpmpc.num_points, logger=logger, dataset=test_dataset)
        
        logger.info("Time {:.4f} sec".format(time.time() - start_time))
        logger.info("Scene generation completed!")
        logger.flush()
    except KeyboardInterrupt:
        logger.info("Terminating...")
        logger.flush()