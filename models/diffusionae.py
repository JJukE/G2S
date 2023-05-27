import torch
import torch.nn as nn

from .dpmpc import *


class DiffusionAE(nn.Module):
    """Implementation of Diffusion based Autoencoder for learning 3D shape

    """

    def __init__(self, args, bottleneck_size=1024, nb_primitives=1):
        super().__init__()

        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        feat, _ = self.encoder(x)
        return feat

    def decode(self, feat, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(self.args.num_points, feat, flexibility=flexibility, ret_traj=ret_traj)
    
    def forward(self, x):
        feat = self.encode(x)
        loss = self.diffusion.get_loss(x, feat)
        return loss