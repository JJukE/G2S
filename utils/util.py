import torch
import numpy as np
import os

#============================================================
# print models
#============================================================

def print_model(model, verbose=False, flag=False):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    message = ''
    message += '---------- Networks initialized -------------\n'
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    if verbose:
        message += repr(model)
    message += '[Network {}] Total number of parameters : {:.3f} M\n'.format(type(model).__name__, num_params / 1e6)
    message += '-----------------------------------------------'

    if flag:
        print(message)
    return message

#============================================================
# common
#============================================================