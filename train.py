"""Template of general-purpose training script.

This script works for various models (with option '--model') and different datasets (with option '--dataset_mode').
You need to specify the dataset ('--data_dir'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a [ex_model] model:
        python train.py --data_dir [./dataset/dir] --name [exp_name] --model [ex_model]

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from collections import OrderedDict

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from jjuke.logger import CustomLogger
# from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # set logger
    log_dir = os.path.join(opt.exps_dir, opt.name)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "train_log.log")
    logger = CustomLogger(os.path.join(opt.exps_dir, opt.name))
    logger.info("experiment directory: {}".format(opt.exps_dir))
    logger.info(opt)

    train_dataset = create_dataset(opt, split='train')  # create a training dataset
    val_dataset = create_dataset(opt, split='val')  # create a validation dataset
    logger.info("The number of training dataset: {}".format(len(train_dataset)))
    logger.info("The number of validation dataset: {}".format(len(val_dataset)))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    logger.info(model.print_networks(opt.verbose))

    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    logger.info("Start training...")
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        epoch_losses = OrderedDict()
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if opt.save_by_iter and total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print("saving the latest model (epoch {}, total_iters {})".format(epoch, total_iters))
                save_suffix = "iter_{}".format(total_iters)
                model.save_networks(save_suffix)

            losses = model.get_current_losses()
            for k, v in losses.items():
                epoch_losses[k] += v / opt.batch_size
            
            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print("saving the model at the end of epoch {}, iters {}".format(epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # print training losses at the end of epoch
        log = "End of epoch {} / {} \t Time Taken: {} sec".format(epoch, opt.n_epochs + opt.n_epochs_decay,
                                                                  time.time() - epoch_start_time)
        for k, v in epoch_losses.items():
            log += "{}: {:4f} \t".format(k, v)
        logger.info(log)
    
    logger.info("Training completed!!")