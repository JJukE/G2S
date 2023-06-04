import numpy as np
import os

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutVAE(nn.Module):

    def __init__(self, vocab=None, embedding_dim=64, residual=False, gconv_pooling='avg', num_box_params=6):
        super().__init__()

        self.vocab = vocab
        self.vae_box = SceneGraphVAE(vocab, embedding_dim=embedding_dim, decoder_cat=True, mlp_normalization="layer",
                            input_dim=num_box_params, residual=residual, gconv_pooling=gconv_pooling, gconv_num_layers=5)

    def forward(self, objs, triples, boxes, angles=None, attributes=None):

        (mu_boxes, logvar_boxes) = self.vae_box.encoder(objs, triples, boxes, attributes=attributes, angles_gt=angles)
        std_box = torch.exp(0.5 * logvar_boxes) # reparameterization
        eps_box = torch.randn_like(std_box) # standard sampling
        z_boxes = eps_box.mul(std_box).add_(mu_boxes)
        boxes_pred, angles_pred = self.vae_box.decoder(z_boxes, objs, triples, attributes) # output: d3_pred, angles_pred
        return mu_boxes, logvar_boxes, boxes_pred, angles_pred

    def load_networks(self, exp, epoch, strict=True):
        self.vae_box.load_state_dict(
            torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
            strict=strict
        )

    def compute_statistics(self, exp, epoch, stats_dataloader, force=False):
        box_stats_f = os.path.join(exp, 'checkpoint', 'model_stats_box_{}.pkl'.format(epoch))

        if os.path.exists(box_stats_f) and not force:
            stats = pickle.load(open(box_stats_f, 'rb'))
            self.mean_est_box, self.cov_est_box = stats[0], stats[1]
        else:
            self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
            pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))

    @torch.no_grad()
    def sample_box(self, mean_est, cov_est, objs, triplets, attributes=None):
        z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float().cuda()
        return self.vae_box.decoder(z, objs, triplets, attributes)


class SceneGraphVAE(nn.Module):
    """
    VAE-based network for layout generation of the scene from a scene graph. (No manipulation)
    It has an embedding of bounding box latents.
    """
    def __init__(self, vocab, embedding_dim=128, batch_size=32, decoder_cat=True, input_dim=6,
                 gconv_pooling='avg', gconv_num_layers=5, mlp_normalization='none', vec_noise_dim=0,
                 residual=False):
        super().__init__()

        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        box_embedding_dim = int(embedding_dim)

        angle_embedding_dim = int(embedding_dim / 4)
        box_embedding_dim = int(embedding_dim * 3 / 4)
        Nangle = 24
        obj_embedding_dim = embedding_dim

        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.decoder_cat = decoder_cat
        self.vocab = vocab
        self.vec_noise_dim = vec_noise_dim

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])

        # build encoder and decoder nets
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_ec = nn.Embedding(num_preds, embedding_dim * 2)
        self.obj_embeddings_dc = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim)
        if self.decoder_cat:
            self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim * 2)
        self.d3_embeddings = nn.Linear(input_dim, box_embedding_dim)
        self.angle_embeddings = nn.Embedding(Nangle, angle_embedding_dim)

        # weight sharing of mean and var
        self.mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.mean = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.var = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.angle_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                        batch_norm=mlp_normalization)
        self.angle_mean = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.angle_var = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)        # graph conv net
        self.gconv_net_ec = None
        self.gconv_net_dc = None

        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2,
            'input_dim_pred': gconv_dim * 2,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_dc = {
            'input_dim_obj': gconv_dim,
            'input_dim_pred': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        if self.decoder_cat:
            gconv_kwargs_dc['input_dim_obj'] = gconv_dim * 2
            gconv_kwargs_dc['input_dim_pred'] = gconv_dim * 2

        self.gconv_net_ec = GraphTripleConvNet(**gconv_kwargs_ec)
        self.gconv_net_dc = GraphTripleConvNet(**gconv_kwargs_dc)

        net_layers = [gconv_dim * 2, gconv_hidden_dim, input_dim]
        self.d3_net = make_mlp(net_layers, batch_norm=mlp_normalization, norelu=True)

        # angle prediction net
        angle_net_layers = [gconv_dim * 2, gconv_hidden_dim, Nangle]
        self.angle_net = make_mlp(angle_net_layers, batch_norm=mlp_normalization, norelu=True)

        # initialization
        self.d3_embeddings.apply(_init_weights)
        self.mean_var.apply(_init_weights)
        self.mean.apply(_init_weights)
        self.var.apply(_init_weights)
        self.d3_net.apply(_init_weights)
        self.angle_mean_var.apply(_init_weights)
        self.angle_mean.apply(_init_weights)
        self.angle_var.apply(_init_weights)

    def encoder(self, objs, triples, boxes_gt, attributes, angles_gt=None):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_ec(objs)
        pred_vecs = self.pred_embeddings_ec(p)
        d3_vecs = self.d3_embeddings(boxes_gt)

        angle_vecs = self.angle_embeddings(angles_gt)
        obj_vecs = torch.cat([obj_vecs, d3_vecs, angle_vecs], dim=1)

        if self.gconv_net_ec is not None:
            obj_vecs, pred_vecs = self.gconv_net_ec(obj_vecs, pred_vecs, edges)

        obj_vecs_3d = self.mean_var(obj_vecs)
        mu = self.mean(obj_vecs_3d)
        logvar = self.var(obj_vecs_3d)

        obj_vecs_angle = self.angle_mean_var(obj_vecs)
        mu_angle = self.angle_mean(obj_vecs_angle)
        logvar_angle = self.angle_var(obj_vecs_angle)
        
        mu = torch.cat([mu, mu_angle], dim=1)
        logvar = torch.cat([logvar, logvar_angle], dim=1)

        return mu, logvar

    def decoder(self, z, objs, triples, attributes):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        pred_vecs = self.pred_embeddings_dc(p)

        if self.decoder_cat: # concatenate noise first
            obj_vecs = torch.cat([obj_vecs, z], dim=1)
            obj_vecs, pred_vecs = self.gconv_net_dc(obj_vecs, pred_vecs, edges)
        else: # concatenate noise after gconv
            obj_vecs, pred_vecs = self.gconv_net_dc(obj_vecs, pred_vecs, edges)
            obj_vecs = torch.cat([obj_vecs, z], dim=1)

        d3_pred = self.d3_net(obj_vecs)
        angles_pred = F.log_softmax(self.angle_net(obj_vecs), dim=1)
        return d3_pred, angles_pred

    def forward(self, objs, triples, enc, attributes):
        mu, logvar = self.encoder(objs, triples, enc, attributes)

        std = torch.exp(0.5 * logvar) # reparameterization
        eps = torch.randn_like(std) # standard sampling
        z = eps.mul(std).add_(mu)
        pred = self.decoder(z, objs, triples, attributes)
        return mu, logvar, pred

    @torch.no_grad()
    def sampleBoxes(self, mean_est, cov_est, objs, triplets, attributes=None):
        z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float().cuda()
        return self.decoder(z, objs, triplets, attributes)

    # TODO: 정확히 어떻게 쓰이는지 training code 짠 후 확인
    def collect_train_statistics(self, train_loader):
        # model = model.eval()
        mean_cat = None

        for idx, data in enumerate(train_loader):
            if data == -1:
                continue
            try:
                objs, triples, tight_boxes, objs_to_scene, triples_to_scene = data['decoder']['objs'], \
                                                                              data['decoder']['tripltes'], \
                                                                              data['decoder']['boxes'], \
                                                                              data['decoder']['obj_to_scene'], \
                                                                              data['decoder']['tiple_to_scene']

                if 'feats' in data['decoder']:
                    encoded_points = data['decoder']['feats']
                    encoded_points = encoded_points.cuda()

            except Exception as e:
                print('Exception', str(e))
                continue

            objs, triples, tight_boxes = objs.cuda(), triples.cuda(), tight_boxes.cuda()
            boxes = tight_boxes[:, :6]
            angles = tight_boxes[:, 6].long() - 1
            angles = torch.where(angles > 0, angles, torch.zeros_like(angles))
            attributes = None

            mean = mean.data.cpu().clone()

            if mean_cat is None:
                mean_cat = mean
            else:
                mean_cat = torch.cat([mean_cat, mean], dim=0)

        mean_est = torch.mean(mean_cat, dim=0, keepdim=True)  # size 1*embed_dim
        mean_cat = mean_cat - mean_est
        cov_est_ = np.cov(mean_cat.numpy().T)
        n = mean_cat.size(0)
        d = mean_cat.size(1)
        cov_est = np.zeros((d, d))
        for i in range(n):
            x = mean_cat[i].numpy()
            cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
        mean_est = mean_est[0]

        return mean_est, cov_est_


#============================================================
# PyTorch modules for dealing with scene graphs
#============================================================
#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2021 Helisa Dhamo, Fabian Manhardt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def make_mlp(dim_list, activation='relu', batch_norm='none', dropout=0, norelu=False):
  return build_mlp(dim_list, activation, batch_norm, dropout, final_nonlinearity=(not norelu))


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class WeightNetGCN(nn.Module):
    """ predict a weight array for the subject and the objects """
    def __init__(self, feat_dim_in1=256, feat_dim_in2=256, feat_dim=128, separate_s_o=True):
        super(WeightNetGCN, self).__init__()

        self.separate = separate_s_o

        if self.separate:
            self.Net_s = nn.Sequential(
                nn.Linear(3*feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

            self.Net_o = nn.Sequential(
                    nn.Linear(3*feat_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                    )
        else:
            self.Net = nn.Sequential(
                    nn.Linear(3*feat_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                    )

        self.down_sample_obj = nn.Linear(feat_dim_in1, feat_dim)
        self.down_sample_pred = nn.Linear(feat_dim_in2, feat_dim)

    def forward(self, s, p, o):

        s = self.down_sample_obj(s)
        p = self.down_sample_pred(p)
        o = self.down_sample_obj(o)

        if self.separate:
            feat1 = torch.cat([s, o, p], 1)
            w_s = self.Net_s(feat1)

            feat2 = torch.cat([s, o, p], 1)
            w_o = self.Net_o(feat2)
        else:
            feat = torch.cat([s, o, p], 1)
            w_o = self.Net(feat)
            w_s = w_o

        return w_s, w_o


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """
    def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, hidden_dim=512,
                             pooling='avg', mlp_normalization='none', residual=True):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim_obj
        self.input_dim_obj = input_dim_obj
        self.input_dim_pred = input_dim_pred
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.residual = residual

        assert pooling in ['sum', 'avg', 'wAvg'], 'Invalid pooling "%s"' % pooling

        self.pooling = pooling
        net1_layers = [2 * input_dim_obj + input_dim_pred, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

        if self.residual:
            self.linear_projection = nn.Linear(input_dim_obj, output_dim)
            self.linear_projection_pred = nn.Linear(input_dim_pred, output_dim)

        if self.pooling == 'wAvg':
            self.weightNet = WeightNetGCN(hidden_dim, output_dim, 128)

    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (num_objs, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (num_triples, D) giving vectors for all predicates
        - edges: LongTensor of shape (num_triples, 2) where edges[k] = [i, j] indicates the
            presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (num_objs, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (num_triples, D) giving new vectors for predicates
        """

        dtype, device = obj_vecs.dtype, obj_vecs.device
        num_objs, num_triples = obj_vecs.size(0), pred_vecs.size(0)
        Din_obj, Din_pred, H, Dout = self.input_dim_obj, self.input_dim_pred, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (num_triples,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (num_triples, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (num_triples, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (num_triples, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (num_triples, H) and
        # p vecs have shape (num_triples, Dout)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H+Dout)]
        new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]
 
        # Allocate space for pooled object vectors of shape (num_objs, H)
        pooled_obj_vecs = torch.zeros(num_objs, H, dtype=dtype, device=device)

        if self.pooling == 'wAvg':

            s_weights, o_weights = self.weightNet(new_s_vecs.detach(),
                                                  new_p_vecs.detach(),
                                                  new_o_vecs.detach())

            new_s_vecs = s_weights * new_s_vecs
            new_o_vecs = o_weights * new_o_vecs

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (num_triples, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'wAvg':
            pooled_weight_sums = torch.zeros(num_objs, 1, dtype=dtype, device=device)
            pooled_weight_sums = pooled_weight_sums.scatter_add(0, o_idx.view(-1, 1), o_weights)
            pooled_weight_sums = pooled_weight_sums.scatter_add(0, s_idx.view(-1, 1), s_weights)

            pooled_obj_vecs = pooled_obj_vecs / (pooled_weight_sums + 0.0001)

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(num_objs, dtype=dtype, device=device)
            ones = torch.ones(num_triples, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (num_objs, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

        if self.residual:
            projected_obj_vecs = self.linear_projection(obj_vecs)
            new_obj_vecs = new_obj_vecs + projected_obj_vecs
            # new
            new_p_vecs = new_p_vecs + self.linear_projection_pred(pred_vecs)

        return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers    """
    def __init__(self, input_dim_obj, input_dim_pred, num_layers=2, hidden_dim=512,
                             residual=False, pooling='avg',
                             mlp_normalization='none', output_dim=None):
        super().__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim_obj': input_dim_obj,
            'input_dim_pred': input_dim_pred,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'residual': residual,
            'mlp_normalization': mlp_normalization,
        }
        gconv_kwargs_out = {
            'input_dim_obj': input_dim_obj,
            'input_dim_pred': input_dim_pred,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'residual': residual,
            'mlp_normalization': mlp_normalization,
            'output_dim': output_dim
        }
        for i in range(self.num_layers):
            if output_dim is not None and i >=  self.num_layers - 1:
                self.gconvs.append(GraphTripleConv(**gconv_kwargs_out))
            else:
                self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
  layers = []
  for i in range(len(dim_list) - 1):
    dim_in, dim_out = dim_list[i], dim_list[i + 1]
    layers.append(nn.Linear(dim_in, dim_out))
    final_layer = (i == len(dim_list) - 2)
    if not final_layer or final_nonlinearity:
      if batch_norm == 'batch':
        layers.append(nn.BatchNorm1d(dim_out))
      elif batch_norm == 'layer':
        layers.append(nn.LayerNorm(dim_out))
      if activation == 'relu':
        layers.append(nn.ReLU())
      elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))

  return nn.Sequential(*layers)