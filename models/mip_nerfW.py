import torch
import torch.nn as nn
from einops import repeat
from collections import namedtuple

from models.mip import sample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering, resample_along_rays
from models.utils import *

class MipNerfW(nn.Module):
    """Nerf NN Model with both coarse and e MLPs."""

    def __init__(self, num_samples: int = 128,
                 resample_padding: float = 0.01,
                 stop_resample_grad: bool = True,
                 use_viewdirs: bool = True,
                 disparity: bool = False,
                 ray_shape: str = 'cone',
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_activation: str = 'softplus',
                 density_noise: float = 0.,
                 density_bias: float = 0.,
                 rgb_activation: str = 'sigmoid',
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,
                 append_identity: bool = True,
                 mlp_net_depth: int = 8,
                 mlp_net_width: int = 256,
                 mlp_net_depth_color: int = 1, 
                 mlp_net_width_color: int = 128,  
                 mlp_skip_index: int = 4,
                 mlp_net_activation: str = 'relu',
                 #################################
                 encode_appearance: bool = False,
                 N_vocab: int = 1000,
                 a_dim: int = 48):

        super(MipNerfW, self).__init__()
        self.num_samples = num_samples  # The number of samples per level.
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
        self.ray_shape = ray_shape  # The shape of cast rays ('cone' or 'cylinder').
        self.disable_integration = disable_integration  # If True, use PE instead of IPE.
        self.min_deg_point = min_deg_point  # Min degree of positional encoding for 3D points.
        self.max_deg_point = max_deg_point  # Max degree of positional encoding for 3D points.
        self.use_viewdirs = use_viewdirs  # If True, use view directions as a condition.
        self.deg_view = deg_view  # Degree of positional encoding for viewdirs.
        self.density_noise = density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding  # Dirichlet/alpha "padding" on the histogram.
        self.stop_resample_grad = stop_resample_grad  # If True, don't backprop across levels')
        mlp_xyz_dim = (max_deg_point - min_deg_point) * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim

        if encode_appearance:
            self.embedding_a = nn.Embedding(N_vocab, a_dim)

        self.encode_appearance = encode_appearance
        self.a_dim = a_dim if encode_appearance else 0
        self.xyz_encoder = MLP(net_depth=mlp_net_depth, 
                               net_width=mlp_net_width, 
                               skip_index=mlp_skip_index, 
                               input_dim=mlp_xyz_dim, 
                               output_dim=mlp_net_width,
                               activation=mlp_net_activation)

        self.xyz_encoder_final = nn.Linear(mlp_net_width, mlp_net_width)

        self.sigma_layer = nn.Sequential(nn.Linear(mlp_net_width, 1),
                                            Shift(self.density_bias), 
                                            actv_dict[density_activation])

        if self.use_viewdirs:
            self.dir_encoder = MLP(net_depth=mlp_net_depth_color, 
                                    net_width=mlp_net_width_color, 
                                    skip_index=mlp_skip_index, 
                                    input_dim=mlp_view_dim+self.a_dim+mlp_net_width, 
                                    output_dim=mlp_net_width_color,
                                    activation=mlp_net_activation)
        else:
            self.dir_encoder = MLP(net_depth=mlp_net_depth_color, 
                                    net_width=mlp_net_width_color, 
                                    skip_index=mlp_skip_index, 
                                    input_dim=self.a_dim+mlp_net_width, 
                                    output_dim=mlp_net_width_color,
                                    activation=mlp_net_activation)

        # _xavier_init(self.sigma_layer[0])
        self.rgb_layer = nn.Sequential(nn.Linear(mlp_net_width_color, 3), 
                                            actv_dict[rgb_activation])
        # _xavier_init(self.rgb_layer[0])

        self.rgb_padding = rgb_padding

    def forward(self, rays: namedtuple,  randomized: bool, white_bkgd: bool, debug: bool=False, a_emb_: torch.Tensor=None):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.   (B, 11)
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """
        if debug:
            debug_ret = []

        ret = []
        t_samples, weights = None, None

        for i_level in range(2):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )
            if self.disable_integration:
                means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
            samples_enc = integrated_pos_enc(
                means_covs,
                self.min_deg_point,
                self.max_deg_point,
            )  # samples_enc: [B, N, 2*3*L1]  L1:(max_deg_point - min_deg_point)

            sample_num = samples_enc.shape[1]
            xyz_encoded = self.xyz_encoder(samples_enc) # (B, N, C1)
            sigma = self.sigma_layer(xyz_encoded)   # (B, N, 1)

            xyz_encoded = self.xyz_encoder_final(xyz_encoded)
            dir_input = [xyz_encoded]
            if self.use_viewdirs:
                viewdirs = rays.directions / torch.linalg.norm(rays.directions, dim=-1, keepdims=True)
                viewdirs_enc = pos_enc(
                    viewdirs,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True,
                )  # viewdirs_enc: [B, 2*3*L2]  L2:(deg_view)
                viewdirs_enc = repeat(viewdirs_enc, 'batch feature -> batch sample feature', sample=sample_num)
                dir_input += [viewdirs_enc]
            if self.encode_appearance:
                if a_emb_ is None:
                    a_emb = self.embedding_a(rays.ts[...,0])    # (B, C_a)
                    a_emb = repeat(a_emb, 'batch feature -> batch sample feature', sample=sample_num)  # (B, N, C_a)
                else:
                    a_emb = repeat(a_emb_, 'feature -> batch sample feature', batch=rays.ts.shape[0], sample=sample_num) # (B, N, C_a) 
                
                dir_input += [a_emb]
            dir_input = torch.cat(dir_input, -1)  # (B, N, C1 + 2*3*L2 + C_a)
            dir_encoded = self.dir_encoder(dir_input)  # (B, N, C2)
            rgb = self.rgb_layer(dir_encoded)      # (B, N, 3)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

            if randomized and (self.density_noise > 0):
                noise = self.density_noise * torch.randn_like(sigma)
                sigma = torch.relu(sigma + noise)

            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                sigma,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )

            if debug:
                debug_ret.append({"rgb":rgb, 
                                  "sigma":sigma,
                                  "t_samples":t_samples,
                                  "color":comp_rgb,
                                  "distance":distance,
                                  "opacity":acc,
                                  "weights":weights})
            # Add noise to regularize the density predictions if needed.

            ret.append((comp_rgb, sigma, distance, acc))

        if debug:
            return debug_ret

        return ret
