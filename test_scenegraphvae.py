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
from models.scenegraphvae import LayoutVAE
from options.scenegraphvae_options import SGVAETestOptions
from utils.util import seed_all, print_model
from utils.viz_util import params_to_8points, batch_torch_denormalize_box_params
from jjuke.logger import CustomLogger
from utils.visualizer import SceneVisualizer
from utils.graph_visualizer import vis_graph

os.environ["OMP_NUM_THREADS"] = str(min(16, mp.cpu_count()))

#============================================================
# Evaluation
#============================================================

@torch.no_grad()
def evaluate(args, model, test_loader, res_dir, logger, dataset):
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
        
        boxes_pred, angles_pred = model.sample_box(mean_est, cov_est_, objs, triples) # cov_est? cov_est_?
        
        boxes_denorm = batch_torch_denormalize_box_params(boxes)
        boxes_pred_denorm = batch_torch_denormalize_box_params(boxes_pred)
        
        # loss calculation
        total_loss = 0
        recon_loss = F.l1_loss(boxes_pred, boxes)
        angle_loss = F.nll_loss(angles_pred, angles)

        test_losses['recon_loss'].append(recon_loss.item())
        test_losses['angle_loss'].append(angle_loss.item())
        
        total_loss = args.box_weight * recon_loss + args.angle_weight * angle_loss
        
        test_losses['total_loss'].append(total_loss.item())
        
        logger.info("<Scene id: {}>".format(scan))
        logger.info("Reconstruction loss: {:.6f}".format(recon_loss.item()))
        logger.info("Angle loss: {:.6f}".format(angle_loss.item()))
        
        if args.visualize:
            angles_pred = torch.argmax(angles_pred, dim=1, keepdim=True) * 15.0 # 24 * 15 = 360
            
            # visualize the scene for only trained objects
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

    for key in test_losses.keys():
        test_losses[key] = np.asarray(test_losses[key])
        test_losses[key] = np.mean(test_losses[key])
    
    return test_losses

#============================================================
# Main
#============================================================

if __name__ == '__main__':
    # Arguments for evaluation
    args, arg_msg, device_msg = SGVAETestOptions().parse()

    if args.debug:
        args.data_dir = '/root/hdd1/G2S/SceneGraphData'
        args.name = 'G2S_SGVAE_230609_all_graph_1e-5_16'
        args.gpu_ids = '0' # only 0 is available while debugging
        args.exps_dir = '/root/hdd1/G2S/practice'
        args.ckpt_name = "ckpt_250.pt"
        args.verbose = True
        
        args.test_batch_size = 1
        args.visualize = True

    # get logger
    exp_dir = os.path.join(args.exps_dir, args.name, "ckpts")
    ckpt_path = os.path.join(exp_dir, args.ckpt_name)
    res_dir = os.path.join(args.exps_dir, args.name, "results")
    
    logger = CustomLogger(res_dir, isTrain=args.isTrain)
    
    logger.info(arg_msg)
    logger.info(device_msg)
    
    # set seed
    if args.use_randomseed:
        args.seed = random.randint(1, 10000)
    logger.info("Seed: {}".format(args.seed))
    seed_all(args.seed)

    # Datasets and loaders
    logger.info("Loading datasets...")
    test_dataset = SceneGraphDataset(
        args=args,
        data_dir=args.data_dir,
        categories=args.categories,
        split='test'
    )
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.test_batch_size,
                             collate_fn=collate_fn_sgvae,
                             shuffle=False)
    
    logger.info("Number of test data: {}".format(len(test_loader)))

    # Model
    logger.info("Loading model...")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = LayoutVAE(vocab=test_dataset.vocab, embedding_dim=ckpt['args'].gconv_dim,
                        residual=ckpt['args'].residual, gconv_pooling=ckpt['args'].pooling).to(args.device)
    model.load_state_dict(ckpt['state_dict'])

    logger.info(print_model(model, verbose=args.verbose))

    # Main loop
    logger.info("Start evaluation...")
    try:
        eval_start_time = time.time()
        test_loss = evaluate(args, model, test_loader, res_dir,
                             logger=logger, dataset=test_dataset)
        logger.info("[Test] Loss {:.6f} | Time {:.4f} sec".format(
            test_loss['total_loss'], time.time() - eval_start_time))

        logger.info("Evaluation completed!")
        logger.flush()

    except KeyboardInterrupt:
        logger.info("Terminating...")
        logger.flush()