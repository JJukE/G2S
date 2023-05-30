import torch
import numpy as np

def get_rotation(z, degree=True):
    """ Get rotation matrix given rotation angle along the z axis.
    :param z: angle of z axos rotation
    :param degree: boolean, if true angle is given in degrees, else in radians
    :return: rotation matrix as np array of shape[3,3]
    """
    if degree:
        z = np.deg2rad(z)
    rot = np.array([[np.cos(z), -np.sin(z),  0],
                    [np.sin(z),  np.cos(z),  0],
                    [        0,          0,  1]])
    return rot


def normalize_box_params(box_params, scale=3):
    """ Normalize the box parameters for more stable learning utilizing the accumulated dataset statistics

    :param box_params: float array of shape [7] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :return: normalized box parameters array of shape [7]
    """
    mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
    std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])

    return scale * ((box_params - mean) / std)


def denormalize_box_params(box_params, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float array of shape [params] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :param params: number of bounding box parameters. Expects values of either 6 or 7. 6 omits the angle
    :return: denormalized box parameters array of shape [params]
    """
    if params == 6:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753])
    elif params == 7:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])
    else:
        raise NotImplementedError
    return (box_params * std) / scale + mean