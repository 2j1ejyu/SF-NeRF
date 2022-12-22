""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

class TransientNet(nn.Module):
    def __init__(self,
                 N_vocab: int = 1500,
                 deg_pixel: int = 10,
                 beta_activation: str = 'softplus',
                 alpha_activation: str = 'softplus',
                 rgb_activation: str = 'sigmoid',
                 net_activation: str = 'relu',
                 final_feat_dim: int = 128,  # if 0, don't use
                 beta_min: float = 0.1,
                 t_dim: int = 16,
                 PE_dropout_p: float = 0.,
                 concrete_tmp: float = 0.1,
                 concrete_u_min: float = 0.0,
                 concrete_u_max: float = 1.0,
                 concrete_t_min: float = 0.5,
                 concrete_gamma:  float = 3e-5,
                 concrete_tmp_anneal_step: int = 4000,
                 mlp_net_depth_A: int = 3,
                 mlp_net_width_A: int = 128,
                 mlp_net_depth_B: int = 3,
                 mlp_net_width_B: int = 128,
                 mlp_net_depth_feat: int = 3,
                 mlp_net_width_feat: int = 128,):

        super(TransientNet, self).__init__()
        assert final_feat_dim >= 0
        
        self.deg_pixel = deg_pixel
        self.final_feat_dim = final_feat_dim
        self.beta_min = beta_min
        self.t_dim = t_dim

        self.embedding_t = nn.Embedding(N_vocab, t_dim)
        self.pxl_pos_enc = PosEmbedding(deg_pixel,False)

        self.xy_encoder = MLP(net_depth=mlp_net_depth_A, 
                              net_width=mlp_net_width_A, 
                              skip_index=100,  # no skip 
                              input_dim=self.pxl_pos_enc.get_dim() + t_dim, 
                              output_dim=mlp_net_width_A,
                              activation=net_activation,
                              last_activation=False)
        
        self.final_encoder = MLP(net_depth=mlp_net_depth_B, 
                                 net_width=mlp_net_width_B, 
                                 skip_index=100,  # no skip 
                                 input_dim=mlp_net_width_A+final_feat_dim, 
                                 output_dim=mlp_net_width_B,
                                 activation=net_activation)

        if final_feat_dim > 0:
            self.feat_mapper = MLP(net_depth=mlp_net_depth_feat, 
                                   net_width=mlp_net_width_feat, 
                                   skip_index=100,  # no skip 
                                   input_dim=384,   # DINO feat dimension
                                   output_dim=final_feat_dim,
                                   activation=net_activation,
                                   last_activation=False)

        self.alpha_layer = nn.Sequential(nn.Linear(mlp_net_width_B, 1),
                                         actv_dict[alpha_activation])
        self.beta_layer = nn.Sequential(nn.Linear(mlp_net_width_B, 1),
                                         actv_dict[beta_activation])
        self.rgb_layer = nn.Sequential(nn.Linear(mlp_net_width_B, 3),
                                         actv_dict[rgb_activation])
        self.concrete = Concrete(tmp=concrete_tmp, u_min=concrete_u_min, u_max=concrete_u_max, t_min=concrete_t_min, gamma=concrete_gamma, concrete_tmp_anneal_step=concrete_tmp_anneal_step)

        self.PE_dropout = nn.Dropout(p=PE_dropout_p)

    def forward(self, p, ts, step, feat=None, randomized=False): # p: (B, 2), ts: (B, 1)
        t_emb = self.embedding_t(ts[...,0])    # (B, t_dim)
        pxl_enc = self.pxl_pos_enc(p)        # (B, deg_pixel*4 + 2)
        pxl_enc = self.PE_dropout(pxl_enc)
        h = self.xy_encoder(torch.cat([pxl_enc,t_emb], -1))  # (B, C_A)

        if self.final_feat_dim > 0:
            assert feat is not None
            feat = self.feat_mapper(feat) # (B, C_f)
            h = torch.cat([h, feat], -1) # (B, C_f+C_A)
        
        h = self.final_encoder(h)        # (B, C_B)
        rgb = self.rgb_layer(h)                         # (B, 1)

        beta = self.beta_layer(h) + self.beta_min       # (B, 1)
        alpha = self.alpha_layer(h)                 # (B, 1)
        alpha, tmp = self.concrete(alpha, step, randomized)
        return rgb, alpha, beta, pxl_enc, tmp

    def get_params(self, lr, final_lr, feat_lr, train_feat=False):
        params = [
            {'params': self.embedding_t.parameters(), 'lr': lr},
            {'params': self.xy_encoder.parameters(), 'lr': lr},
            {'params': self.alpha_layer.parameters(), 'lr': lr if final_lr is None else final_lr},
            {'params': self.beta_layer.parameters(), 'lr': lr if final_lr is None else final_lr},
            {'params': self.rgb_layer.parameters(), 'lr': lr if final_lr is None else final_lr},
        ]
        params.append({'params': self.final_encoder.parameters(), 'lr': lr if final_lr is None else final_lr})
        if self.final_feat_dim > 0 and train_feat:
            params.append({'params': self.feat_mapper.parameters(), 'lr': lr if feat_lr is None else feat_lr})
        return params

    


