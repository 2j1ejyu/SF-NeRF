# This file is modified from official mipnerf
"""Different datasets implementation plus a general port for all the datasets."""
# import json
import os
# from os import path
import cv2
import numpy as np
import torch
import random
# from PIL import Image
import collections
import pickle
from torch.utils.data import Dataset
import glob
import pandas as pd
from prepare_phototourism import process_data

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'radii', 'near', 'far', 'ts'))
Rays_keys = Rays._fields

class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, val_idcs=[], white_bkgd=False, get_seg_feats=False, preload_feat=False, seg_feat_dir=None, use_transient=False, pxl_augment=False, fewshot=-1, get_pts=False, chunk_size=-1):
        """
        img_downscale: how much scale to downsample the training images.
        """
        self.get_pts = get_pts
        self.fewshot = fewshot
        self.val_idcs = val_idcs  # 0 <= idcs <= train_img_num-1
        self.chunk_size = chunk_size
        self.get_seg_feats = get_seg_feats
        self.preload_feat = preload_feat
        self.seg_feat_dir = seg_feat_dir
        self.use_transient = use_transient
        self.pxl_augment = pxl_augment

        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
        # if split == 'val': # image downscale=1 will cause OOM in val mode
        #     self.img_downscale = max(2, self.img_downscale)

        self.read_meta()
        self.white_bkgd = white_bkgd

    def read_meta(self):
        # read all files in the tsv first (split to train)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
            self.img_ids = pickle.load(f)   # list of ids of imgs (ordered)
        with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
            self.image_paths = pickle.load(f)   # dict of paths for each imgs (key=id)

        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='train']


        self.N_images_train = len(self.img_ids_train)


        # with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
        #     self.Ks = pickle.load(f)

        self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))   # follow order of self.img_ids (N, 3, 4)

        if self.get_pts:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))  # points of the object

        with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
            self.nears = pickle.load(f)    # dict of nears for each imgs (key=id)
        with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
            self.fars = pickle.load(f)      # dict of nears for each imgs  (key=id)
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)} # dict of poses (key=id)

        with open(os.path.join(self.root_dir, f'cache/rays{self.img_downscale}.pkl'),'rb') as f:
            self.all_rays = pickle.load(f)  # list of rays : each entry is [rays_o, rays_d, nears, fars, rays_t], (H,W,9)
        with open(os.path.join(self.root_dir, f'cache/rgbs{self.img_downscale}.pkl'), 'rb') as f:
            self.images = pickle.load(f)  # list of images : each entry shape (H,W,3)
        assert len(self.all_rays) == len(self.images)

        if self.fewshot != -1:          # few shot  
            txt_path = os.path.join(self.root_dir, f"few_{self.fewshot}.txt")
            with open(txt_path, 'r') as f:
                few_list = f.read().splitlines()
            idcs = [i for i,id_ in enumerate(self.img_ids_train) if self.image_paths[id_] in few_list]
            self.img_ids_train = [self.img_ids_train[i] for i in idcs]
            self.all_rays = [self.all_rays[i] for i in idcs]
            self.images = [self.images[i] for i in idcs]

            assert len(self.img_ids_train) == len(few_list)
        # else:                           # all
        #     idcs = [i for i in range(self.N_images_train)]

        if len(self.val_idcs) != 0:
            self.img_ids_train = [self.img_ids_train[i] for i in self.val_idcs]
            self.all_rays = [self.all_rays[i] for i in self.val_idcs]
            self.images = [self.images[i] for i in self.val_idcs]
            # idcs = [idcs[i] for i in self.val_idcs]


        if self.get_seg_feats:
            self.feat_names = [self.image_paths[id_].replace('.jpg','.npy') for id_ in self.img_ids_train]
            if self.preload_feat:
                self.seg_feats = []
                for i, name in enumerate(self.feat_names):
                    H, W, _ = self.images[i].shape
                    feat_map = np.load(os.path.join(self.seg_feat_dir,name))                  # [C,H',W']
                    feat_map = torch.from_numpy(cv2.resize(feat_map.transpose(1,2,0),(W,H)))
                    # feat_map = torch.nn.functional.interpolate(torch.from_numpy(feat_map).unsqueeze(0), (H,W), mode='bilinear', align_corners=True).squeeze()  
                    self.seg_feats.append(feat_map)

        self.radii = []   # pixel radius
        self.img_idx = []
        self.pxl_idx = []
        for idx in range(len(self.all_rays)):            
            v = self.all_rays[idx][...,3:6] # (H, W, 3)
            dx = np.sqrt(np.sum((v[:,:-1,:]-v[:,1:,:])**2,-1)) # (H, W-1)
            dx = np.concatenate([dx, dx[:, -2:-1]], 1)      # (H, W)
            self.radii.append(dx[...,None] * 2 / np.sqrt(12))   # N x (H_n, W_n, 1)
            H_, W_, _ = self.images[idx].shape
            n_ = H_*W_ // self.chunk_size if self.split == 'train' else 1 # num of chunk for each img
            if self.split == 'train':
                self.pxl_idx.extend(list(torch.randperm(H_*W_).split(self.chunk_size))[:n_])
            self.img_idx.append(np.ones(n_).astype(np.int64)*idx)

        self.img_idx = np.concatenate(self.img_idx)

    def reset_pxl_idx(self):
        if self.split != 'train':
            return
        self.pxl_idx = []
        for img in self.images:            
            H_, W_, _ = img.shape
            n_ = H_*W_ // self.chunk_size 
            self.pxl_idx.extend(list(torch.randperm(H_*W_).split(self.chunk_size))[:n_])

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        if self.split == 'train': 
            i = self.img_idx[idx]
            img = self.images[i]
            H, W, _ = img.shape
            indices = self.pxl_idx[idx]
            # indices = torch.randperm(H*W)[:self.chunk_size]
            h = torch.div(indices, W, rounding_mode='floor')
            w = torch.remainder(indices, W)
            rays_ = self.all_rays[i][h,w]   # (9,)
            radii = self.radii[i][h,w]
            ray_o, ray_d, near, far, ray_t = rays_[:,:3], rays_[:,3:6], rays_[:,6:7], rays_[:,7:8], rays_[:,8:]
            rays = Rays(
                        origins=ray_o,
                        directions=ray_d,
                        radii=radii,
                        near=near,
                        far=far,
                        ts=ray_t.astype(np.int64)
                    )

            items = [rays, img[h,w]]

            if self.use_transient:
                if self.pxl_augment:
                    beta_dist = torch.distributions.beta.Beta(torch.Tensor([4]),torch.Tensor([4]))
                    noise = beta_dist.sample((h.shape[0],2)).squeeze(-1)  # (chunk_size, 2)
                    pxl_coord = torch.stack([(h.float()+noise[:,0])/H, (w.float()+noise[:,1])/W],dim=-1) # (chunk_size,2)
                else:
                    pxl_coord = torch.stack([h.float()/H, w.float()/W],dim=-1) # (chunk_size,2)
                items.append(pxl_coord)
            if self.get_seg_feats:
                if self.preload_feat:
                    feat_map = self.seg_feats[i] # [C,H',W']
                else:
                    feat_map = np.load(os.path.join(self.seg_feat_dir,self.feat_names[i]))         # [C,H',W']
                    feat_map = torch.from_numpy(cv2.resize(feat_map.transpose(1,2,0),(W,H)))  # [H,W,C]
                # feat_map = torch.nn.functional.interpolate(torch.from_numpy(feat_map).unsqueeze(0), (H,W), mode='bilinear', align_corners=True).squeeze()  
                feat_map = feat_map[h,w,:]     # [chunk_size,C]
                items.append(feat_map)
            
            return items

        else:  # val
            img = self.images[idx]
            H, W, _ = img.shape
            rays_ = self.all_rays[idx]   # (H, W, 9)
            ray_o, ray_d, near, far, ray_t = rays_[...,:3], rays_[...,3:6], rays_[...,6:7], rays_[...,7:8], rays_[...,8:]
            radii = self.radii[idx]    # (H, W, 1)
            rays = Rays(
                        origins=ray_o,
                        directions=ray_d,
                        radii=radii,
                        near=near,
                        far=far,
                        ts=ray_t.astype(np.int64)
                    )
            items = [rays, img]
            if self.use_transient:
                pxl_coord = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).float()  # [H,W,2]
                if self.pxl_augment:
                    pxl_coord = pxl_coord + 0.5
                pxl_coord[...,0] = pxl_coord[...,0]/H
                pxl_coord[...,1] = pxl_coord[...,1]/W
                items.append(pxl_coord)

            if self.get_seg_feats:
                if self.preload_feat:
                    feat_map = self.seg_feats[idx]          # [C,H',W']
                else:
                    feat_map = np.load(os.path.join(self.seg_feat_dir,self.feat_names[idx]))                  # [C,H',W']
                    feat_map = torch.from_numpy(cv2.resize(feat_map.transpose(1,2,0),(W,H)))  # [H,W,C]
                # feat_map = torch.nn.functional.interpolate(torch.from_numpy(feat_map).unsqueeze(0), (H,W), mode='bilinear', align_corners=True).squeeze()  
                items.append(feat_map)

            return items

#####################################################################

class PhototourismDataset_test(Dataset):
    def __init__(self, root_dir, img_downscale=1, val_idcs=[], white_bkgd=False):
        """
        img_downscale: how much scale to downsample the images.
        """
        self.val_idcs = val_idcs
        self.root_dir = root_dir
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale

        self.read_meta()
        self.white_bkgd = white_bkgd

    def read_meta(self):
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]

        self.img_ids_test, self.img_names, self.nears, self.fars, self.all_rays, self.images = \
        process_data(self.root_dir, self.img_downscale, test=True)

        self.N_images_test = len(self.img_ids_test)

        # with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
        #     self.Ks = pickle.load(f)

        assert len(self.all_rays) == len(self.images)

        if len(self.val_idcs) != 0:
            self.all_rays = [self.all_rays[i] for i in self.val_idcs]
            self.images = [self.images[i] for i in self.val_idcs]

        self.radii = []

        for idx in range(len(self.all_rays)):
            W, H, _ = self.images[idx].shape
            i, j = np.meshgrid(np.arange(W),np.arange(H),indexing='ij')
            i, j = i.reshape(-1), j.reshape(-1)
            v = self.all_rays[idx][...,3:6] # (H, W, 3)
            dx = np.sqrt(np.sum((v[:,:-1,:]-v[:,1:,:])**2,-1)) # (H, W-1)
            dx = np.concatenate([dx, dx[:, -2:-1]], 1)      # (H, W)
            self.radii.append(dx[...,None] * 2 / np.sqrt(12))   # N x (H_n, W_n, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rays_ = torch.from_numpy(self.all_rays[idx])   # (H, W, 9)
        H, W, _ = rays_.shape
        b_idx = int(W/2)
        ray_o, ray_d, near, far, ray_t = rays_[...,:3], rays_[...,3:6], rays_[...,6:7], rays_[...,7:8], rays_[...,8:].type(torch.int64)
        radii = torch.from_numpy(self.radii[idx])    # (H, W, 1)
        rays_train = Rays(
                    origins=ray_o[:,:b_idx,:],
                    directions=ray_d[:,:b_idx,:],
                    radii=radii[:,:b_idx,:],
                    near=near[:,:b_idx,:],
                    far=far[:,:b_idx,:],
                    ts=ray_t[:,:b_idx,:]
                )

        rays_test = Rays(
                    origins=ray_o[:,b_idx:,:],
                    directions=ray_d[:,b_idx:,:],
                    radii=radii[:,b_idx:,:],
                    near=near[:,b_idx:,:],
                    far=far[:,b_idx:,:],
                    ts=ray_t[:,b_idx:,:]
                )
# width index to separate left,right part of image 
        return rays_train, torch.from_numpy(self.images[idx][:,:b_idx,:]), rays_test, torch.from_numpy(self.images[idx][:,b_idx:,:])



class Tensor2Dataset(Dataset):
    def __init__(self, rays, img):  # [data1, data2...]  datas shape (N, ...)
        self.rays = rays
        self.img = img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        rays = Rays(*[r[idx] for r in self.rays])
        img = self.img[idx]

        return rays, img



