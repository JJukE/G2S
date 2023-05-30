import os
import random

import torch
import numpy as np

#============================================================
# print models
#============================================================

def print_model(model, verbose=False, print_flag=False):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    message = 'Model info\n'
    message += '---------- Networks initialized -------------\n'
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    if verbose:
        message += repr(model)
    message += '[Network {}] Total number of parameters : {:.3f} M\n'.format(type(model).__name__, num_params / 1e6)
    message += '-----------------------------------------------'

    if print_flag:
        print(message)
    return message

#============================================================
# Checkpoint Manager
#============================================================

class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):
    """Checkpoint manager
    Checkpoint should be saved as "ckpt_[iteration or epoch].pt(h)"
    """

    def __init__(self, save_dir, isTrain=False, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        if not isTrain:
            for f in os.listdir(self.save_dir):
                if f[:4] != 'ckpt':
                    continue
                _, score, it = f.split('_')
                it = it.split('.')[0]
                self.ckpts.append({
                    'score': float(score),
                    'file': f,
                    'iteration': int(it),
                })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None):

        if step is None:
            fname = 'ckpt_{}_.pt'.format(float(score))
        else:
            fname = 'ckpt_{}.pt'.format(int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt

#============================================================
# for setting seeds (torch, numpy, python)
#============================================================

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#============================================================
# data iterator for training
#============================================================

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()