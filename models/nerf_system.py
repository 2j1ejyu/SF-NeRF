import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import os
from tqdm import tqdm
from glob import glob
from collections import namedtuple
from PIL import Image
import numpy as np
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from models.mip_nerfW import MipNerfW
from models.mip import rearrange_render_image
from models.transient_net import TransientNet
from utils.metrics import calc_psnr, calc_lpips, calc_ssim
from datasets import PhototourismDataset, Tensor2Dataset
from datasets import Rays as Ray_Collection

from torch.utils.data import DataLoader
from utils.vis import stack_rgb, visualize_depth

def cos_aneal(x):
    return 1-(np.cos(x)+1)/2

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.train_randomized = hparams['train.randomized']
        self.val_randomized = hparams['val.randomized']
        self.white_bkgd = hparams['train.white_bkgd']
        self.val_chunk_size = hparams['val.chunk_size']
        self.batch_size = self.hparams['train.batch_size']
        self.mip_nerf = MipNerfW(
            num_samples=hparams['nerf.num_samples'],
            resample_padding=hparams['nerf.resample_padding'],
            stop_resample_grad=hparams['nerf.stop_resample_grad'],
            use_viewdirs=hparams['nerf.use_viewdirs'],
            disparity=hparams['nerf.disparity'],
            ray_shape=hparams['nerf.ray_shape'],
            min_deg_point=hparams['nerf.min_deg_point'],
            max_deg_point=hparams['nerf.max_deg_point'],
            deg_view=hparams['nerf.deg_view'],
            density_activation=hparams['nerf.density_activation'],
            density_noise=hparams['nerf.density_noise'],
            density_bias=hparams['nerf.density_bias'],
            rgb_activation=hparams['nerf.rgb_activation'],
            rgb_padding=hparams['nerf.rgb_padding'],
            disable_integration=hparams['nerf.disable_integration'],
            append_identity=hparams['nerf.append_identity'],
            mlp_net_depth=hparams['nerf.mlp.net_depth'],
            mlp_net_width=hparams['nerf.mlp.net_width'],
            mlp_net_depth_color=hparams['nerf.mlp.net_depth_color'],
            mlp_net_width_color=hparams['nerf.mlp.net_width_color'],
            mlp_skip_index=hparams['nerf.mlp.skip_index'],
            mlp_net_activation=hparams['nerf.mlp.net_activation'],
            encode_appearance=hparams['nerf.encode_appearance'],
            N_vocab=hparams['N_vocab'],
            a_dim=hparams['nerf.a_dim'],
        )
        if hparams['t_net.use_transient']:
            self.transient_net = TransientNet(
                N_vocab=hparams['N_vocab'],
                deg_pixel=hparams['t_net.deg_pixel'],
                beta_activation=hparams['t_net.beta_activation'],
                alpha_activation=hparams['t_net.alpha_activation'],
                rgb_activation=hparams['t_net.rgb_activation'],
                final_feat_dim=hparams['t_net.final_feat_dim'] if hparams['t_net.use_seg_feat'] else 0,
                beta_min=hparams['t_net.beta_min'],
                t_dim=hparams['t_net.t_dim'],
                alpha_noise=hparams['t_net.alpha_noise'],
                PE_dropout_p=hparams['t_net.PE_dropout_p'],
                concrete_tmp=hparams['t_net.concrete_tmp'],
                concrete_t_min=hparams['t_net.concrete_t_min'],
                concrete_u_min=hparams['t_net.concrete_u_min'],
                concrete_u_max=hparams['t_net.concrete_u_max'],
                concrete_gamma=hparams['t_net.concrete_gamma'],
                concrete_tmp_anneal_step=hparams['t_net.concrete_tmp_anneal_step'],
                net_activation=hparams['t_net.mlp.net_activation'],
                mlp_net_depth_A=hparams['t_net.mlp.net_depth_A'],
                mlp_net_width_A=hparams['t_net.mlp.net_width_A'],
                mlp_net_depth_B=hparams['t_net.mlp.net_depth_B'],
                mlp_net_width_B=hparams['t_net.mlp.net_width_B'],
                mlp_net_depth_feat=hparams['t_net.mlp.net_depth_feat'],
                mlp_net_width_feat=hparams['t_net.mlp.net_width_feat'])
        self.criterion = NeRFLoss(
                lambda_sparsity=self.hparams['loss.sparsity_mult'],
                )

        if self.hparams['t_net.pretrained_path']:
            state_dict = self.transient_net.state_dict()
            state_dict_ = torch.load(self.hparams['t_net.pretrained_path'])['state_dict']
            update_dict = {}
            # for k,v in state_dict_.items():
            #     if k.startswith('transient_net.feat_mapper'):
            #         m = k.split('.',1)[-1]
            #         update_dict[m] = v
            #     if k.startswith('transient_net.final_encoder'):
            #         m = k.split('.',1)[-1]
            #         update_dict[m] = v
            for k,v in state_dict_.items():
                if k.startswith('transient_net.feat_mapper'):
                    m = k.split('.',1)[-1]
                    update_dict[m] = v
                if k.startswith('transient_net.final_encoder'):
                    m = k.split('.',1)[-1]
                    update_dict[m] = v
                if k.startswith('transient_net.alpha_layer'):
                    m = k.split('.',1)[-1]
                    update_dict[m] = v
                if k.startswith('transient_net.beta_layer'):
                    m = k.split('.',1)[-1]
                    update_dict[m] = v
                if k.startswith('transient_net.rgb_layer'):
                    m = k.split('.',1)[-1]
                    update_dict[m] = v
                if k.startswith('transient_net.xy_encoder'):
                    m = k.split('.',1)[-1]
                    update_dict[m] = v
            state_dict.update(update_dict)
            self.transient_net.load_state_dict(state_dict)
            if not self.hparams['t_net.finetune']:
                for param in self.transient_net.feat_mapper.parameters():
                    param.requires_grad = False

    def forward(self, batch_rays: namedtuple, randomized: bool, white_bkgd: bool, debug: bool=False, a_emb: torch.Tensor=None):
        res = self.mip_nerf(batch_rays, randomized, white_bkgd, debug, a_emb) 

        return res

    def setup(self, stage):
        self.train_dataset = PhototourismDataset(
            root_dir=self.hparams['data_path'],
            split='train',
            img_downscale=self.hparams['train.img_downscale'],
            white_bkgd=self.hparams['train.white_bkgd'],
            get_seg_feats=self.hparams['t_net.use_seg_feat'],
            preload_feat=self.hparams['t_net.preload_feat'],
            seg_feat_dir=self.hparams['t_net.seg_feat_dir'],
            use_transient=self.hparams['t_net.use_transient'],
            pxl_augment=self.hparams['t_net.pxl_augment'],
            chunk_size=self.hparams['train.chunk_size'],
            fewshot=self.hparams['fewshot'],
            )
        self.val_dataset = PhototourismDataset(
            root_dir=self.hparams['data_path'],
            split='val',
            img_downscale=self.hparams['val.img_downscale'],
            get_seg_feats=self.hparams['t_net.use_seg_feat'],
            preload_feat=self.hparams['t_net.preload_feat'],
            seg_feat_dir=self.hparams['t_net.seg_feat_dir'],
            use_transient=self.hparams['t_net.use_transient'],
            val_idcs=self.hparams['val.img_idx'],
            white_bkgd=self.hparams['val.white_bkgd'],
            fewshot=self.hparams['fewshot'],
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams['train.num_work'],
                          batch_size=self.hparams['train.batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        # must give 1 worker
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=True,
                          persistent_workers=False)

    def training_epoch_end(self, training_step_outputs):
        self.train_dataset.reset_pxl_idx()
    
    def configure_optimizers(self):
        params_to_train = [{'params':self.mip_nerf.parameters()}]
        if self.hparams['t_net.use_transient']:
            params_to_train.extend(self.transient_net.get_params(\
                    self.hparams['optimizer.transient_lr_init'],
                    self.hparams['optimizer.fin_mlp_lr_init'],
                    self.hparams['optimizer.f_mlp_lr_init'],
                    self.hparams['t_net.finetune'] or (self.hparams['t_net.pretrained_path'])==''))
        optimizer = torch.optim.Adam(params_to_train, lr=self.hparams['optimizer.lr_init'])
        # scheduler = MipLRDecay(optimizer, self.hparams['optimizer.lr_init'], self.hparams['optimizer.lr_final'],
        #                        self.hparams['optimizer.max_steps'], self.hparams['optimizer.lr_delay_steps'],
        #                        self.hparams['optimizer.lr_delay_mult'])
        # scheduler = StepLR(
        #     optimizer, 
        #     step_size=self.hparams['optimizer.lr_decay_steps'],
        #     gamma=self.hparams['optimizer.lr_decay_mult'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams['optimizer.max_steps'])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_nb):
        rays, rgb_gt = batch[0], batch[1] # [batch_size, chunk_size, ~]

        if self.mip_nerf.encode_appearance:
            a_emb = self.mip_nerf.embedding_a(rays.ts[:,0,0]) # [batch_size, C_a]
        else:
            a_emb = None

        BN, CN, _ = rgb_gt.shape    # BN: batch size, CN: Chunk size
        rgb_gt = rgb_gt.reshape(BN*CN, -1)
        rays = Ray_Collection(*[r.reshape(BN*CN,-1) for r in rays])

        if self.hparams['t_net.use_transient']:
            pxl_coords = batch[2].reshape(BN*CN,-1)
            if self.hparams['loss.alpha_smooth_mult'] > 0:
                pxl_coords.requires_grad = True
            if self.hparams['t_net.use_seg_feat']:
                seg_feats = batch[3].reshape(BN*CN,-1)
            else:
                seg_feats = None

        ret = self(rays, self.train_randomized, self.white_bkgd)

        beta, t_rgb, t_alpha = None, None, None
        if self.hparams['t_net.use_transient']:
            t_rgb, t_alpha, beta, pxl_enc, tmp = self.transient_net(pxl_coords, rays.ts, self.global_step, seg_feats, self.train_randomized)
        
        # calculate loss for coarse and fine
        reclosses = []
        for i, (rgb, sigmas, _, _) in enumerate(ret):
            reclosses.append(self.criterion(
                s_rgb=rgb,
                gt=rgb_gt[...,:3], 
                sigmas=sigmas,
                # a_emb=a_emb,
                t_rgb=t_rgb, 
                t_alpha=t_alpha,
                beta=beta, 
                fine=(i==1)))
        # The loss is a sum of coarse and fine MSEs
        mse_coarse, mse_fine = reclosses
        loss = self.hparams['loss.coarse_loss_mult']*mse_coarse + \
               mse_fine
        loss += self.hparams['loss.appearance_reg_mult'] * \
                torch.mean(torch.sum(a_emb**2, dim=-1)-1) + \
                self.hparams['loss.alpha_reg_mult'] * \
                torch.mean(t_alpha)

        if self.hparams['t_net.use_transient'] and self.hparams['loss.alpha_smooth_mult'] > 0:
            if self.transient_net.pxl_pos_enc.identity:
                raise Exception('not implemented') 
            scale = self.transient_net.pxl_pos_enc.freqs.unsqueeze(-1).repeat(1,4).view(1,-1).to(self.device)
            grad_pxl = torch.autograd.grad(t_alpha,[pxl_enc],grad_outputs=torch.ones_like(t_alpha), create_graph=True)[0]
            loss += self.hparams['loss.alpha_smooth_mult'] * torch.norm(scale*grad_pxl,p=1).mean()

        with torch.no_grad():
            if (t_alpha is not None) and (t_rgb is not None): 
                pred_rgb = ret[1][0]*(1-t_alpha) + t_rgb*t_alpha
            else:
                pred_rgb = ret[1][0]
            psnr_fine = calc_psnr(pred_rgb, rgb_gt[..., :3])

        # self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_fine, prog_bar=True)

        lr = self.optimizers().optimizer.param_groups[0]['lr']
        
        start_step = 12000
        recover_step = 10000
        step_size = math.pi/recover_step
        for pg in self.optimizers().optimizer.param_groups:
            if pg['lr'] != lr:
                if self.global_step < start_step:
                    pg['lr'] = pg['initial_lr']
                elif self.global_step < start_step+recover_step:
                    # p = (1/(recover_step-self.global_step))**1.2
                    # pg['lr'] = (1-p)*pg['lr'] +p*lr
                    u = cos_aneal((self.global_step-start_step)*step_size)
                    n_u = cos_aneal((self.global_step-start_step+1)*step_size)
                    pg['lr'] += (lr-pg['lr'])*(n_u-u)/(1-u)
                else:
                    pg['lr'] = lr

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgb_gt = batch[:2]
        rgb_gt = rgb_gt[..., :3]
        img_idx = self.hparams['val.img_idx'][batch_nb]
        coarse_rgb, fine_rgb, distances = self.render_image(rays, rgb_gt)
        
        val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)

        # log = {'val/loss': val_loss, 'val/psnr': val_psnr_fine}
        
        log = {'val/psnr': val_psnr_fine}
        
        # stack = stack_rgb(rgb_gt, coarse_rgb, fine_rgb)  # (3, 3, H, W)
        # self.logger.experiment.add_images('val/GT_coarse_fine',
        #                                   stack, self.global_step)
        rgb_gt = rgb_gt.squeeze(0).permute(2, 0, 1).cpu()
        coarse_rgb = coarse_rgb.squeeze(0).permute(2, 0, 1).cpu()
        fine_rgb = fine_rgb.squeeze(0).permute(2, 0, 1).cpu()

        if self.hparams['t_net.use_transient']:
            pxl_coords = batch[2][0]
            if self.hparams['t_net.use_seg_feat']:
                seg_feats = batch[3][0]
            else:
                seg_feats = None
            with torch.no_grad():
                t_rgb, t_alpha, beta, _, _ = self.transient_net(pxl_coords, rays.ts[0], self.global_step, seg_feats, self.val_randomized)
            t_rgb = t_rgb.permute(2,0,1).cpu()        # [3,H,W]
            t_alpha = t_alpha.permute(2,0,1).cpu()    # [1,H,W]
            beta = beta.permute(2,0,1).cpu()          # [1,H,W]

            self.logger.log_image(f'val_{img_idx}/transient_all', [t_rgb])
            self.logger.log_image(f'val_{img_idx}/transient', [t_rgb*t_alpha+(1-t_alpha)])
            self.logger.log_image(f'val_{img_idx}/alpha', [t_alpha])
            self.logger.log_image(f'val_{img_idx}/beta', [beta])

        self.logger.log_image(f'val_{img_idx}/distance', [distances])
        self.logger.log_image(f'val_{img_idx}/GT', [rgb_gt])
        self.logger.log_image(f'val_{img_idx}/coarse', [coarse_rgb])
        self.logger.log_image(f'val_{img_idx}/fine', [fine_rgb])
        self.log(f'val_{img_idx}/psnr', val_psnr_fine)
        return log

    def validation_epoch_end(self, outputs):
        # mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        # self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)

    def render_image(self, rays, rgb_gt):
        _, height, width, _ = rgb_gt.shape  # N H W C
        single_image_rays = rearrange_render_image(
            rays, self.val_chunk_size)
        coarse_rgb, fine_rgb = [], []
        distances = []
        with torch.no_grad():
            for batch_rays in single_image_rays:
                (c_rgb, _, _, _), (f_rgb, _, distance, _) = self(
                    batch_rays, self.val_randomized, self.white_bkgd)
                coarse_rgb.append(c_rgb)
                fine_rgb.append(f_rgb)
                distances.append(distance)

        coarse_rgb = torch.cat(coarse_rgb, dim=0)
        fine_rgb = torch.cat(fine_rgb, dim=0)
        distances = torch.cat(distances, dim=0)
        distances = distances.reshape(1, height, width)  # [1,H,W]
        distances = visualize_depth(distances)
        # self.logger.experiment.add_image('distance', distances, self.global_step)

        coarse_rgb = coarse_rgb.reshape(
            1, height, width, coarse_rgb.shape[-1])  # [N,H,W,C]
        fine_rgb = fine_rgb.reshape(
            1, height, width, fine_rgb.shape[-1])  # [N,H,W,C]
        return coarse_rgb, fine_rgb, distances

    def eval_model(self, img_name, train_rays, test_rays, train_img, test_img, lr, epochs, chunk, save_root, test_only=False):
        """
            for eval.py
            optimize a_emb with left side of the image and test with right side
            
            img_name: str
            train_rays, torch.tensor namedtuple, each [H*W, ...]
            test_rays: torch.tensor namedtuple, each [H, W, ...]
            train_img, torch.tensor [H*W, 3]
            test_img: torch.tensor [H, W, 3]
            lr: float
            epochs: int
            chunk: int
            save_root: str
            test_only: bool
        """
        save_dir = os.path.join(save_root, img_name)
        os.makedirs(save_dir, exist_ok=True)
        # self.setup(None)
        # train_data_num = len(self.train_dataset)

        H_test, W_test, _ = test_img.shape

        # emb_list = [sorted(glob(os.path.join(save_dir,'epoch=*.npy')))]
        emb_list = []
        
        if not test_only:
            if len(emb_list) != 0:
                start_epoch = int(emb_list[-1].split('/')[-1].rstrip('.npy').lstrip('epoch='))
                print("load {}".format(emb_list[-1]))
                with open(emb_list[-1],'rb') as f:
                    a_emb = torch.from_numpy(np.load(f)).to(self.device)
            else:
                a_emb = self.mip_nerf.embedding_a(torch.tensor([0]).long().to(self.device)).clone().detach()
                # nn.init.xavier_uniform_(a_emb)
                nn.init.normal_(a_emb)
                a_emb = a_emb[0]
                start_epoch = 0
            a_emb.requires_grad = True
            optim = torch.optim.SGD(lr=lr, params=[a_emb])
            scheduler = StepLR(
                optim, 
                step_size=20,
                gamma=0.1)

            train_loader = DataLoader(Tensor2Dataset(train_rays, train_img), batch_size=chunk, shuffle=True, num_workers=4, pin_memory=True)

            print("optimizing {}".format(img_name))
            with tqdm(total=(epochs-start_epoch)*len(train_loader)) as pbar:
                for epoch in range(start_epoch,epochs):
                    for step, (batch_rays, rgbs) in enumerate(train_loader):
                        batch_rays = Ray_Collection(*[r.to(self.device) for r in batch_rays])
                        rgbs = rgbs.to(self.device)
                        ret = self(batch_rays, True, self.white_bkgd, False, a_emb=a_emb)
                        loss = 0.
                        rgb = ret[1][0]
                        loss = 0.5*((rgb - rgbs[..., :3]) ** 2).mean()

                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        pbar.update(1)
                    if (epoch+1) % 15 == 0:
                        emb_list.append(os.path.join(save_dir, "epoch={}.npy".format(str(epoch+1).zfill(3))))
                        with open(os.path.join(save_dir, "epoch={}.npy".format(str(epoch+1).zfill(3))), 'wb') as f:
                            np.save(f, a_emb.detach().cpu().numpy())
                    scheduler.step()

        test_rays = Ray_Collection(*[r.to(self.device) for r in test_rays])
        single_image_rays = rearrange_render_image(test_rays, chunk)

        for emb_path in emb_list:
            with open(emb_path,'rb') as f:
                emb = torch.from_numpy(np.load(f)).to(self.device)
            fine_rgb = []
            # distances = []
            with torch.no_grad():
                for batch_rays in single_image_rays:
                    ret = self(batch_rays, False, self.white_bkgd, False, a_emb=emb)
                    f_rgb = ret[1][0]
                    fine_rgb.append(f_rgb.detach().cpu())
                    # distances.append(distance)

            fine_rgb = torch.cat(fine_rgb, dim=0)

            # distances = torch.cat(distances, dim=0)
            # distances = distances.reshape(1, H_test, W_test)  # H W
            # distances = visualize_depth(distances)

            fine_rgb = fine_rgb.reshape(H_test, W_test, fine_rgb.shape[-1])  # H W C
            
            im = Image.fromarray((fine_rgb.numpy()*255).astype(np.uint8))
            im.save(emb_path.replace('.npy','.png'))

            test_img = test_img.float()
            psnr = calc_psnr(fine_rgb, test_img)
            ssim = calc_ssim(fine_rgb.permute(2,0,1)[None,...],
                             test_img.permute(2,0,1)[None,...])
            lpips = calc_lpips(fine_rgb.permute(2,0,1)[None,...],
                               test_img.permute(2,0,1)[None,...])

            print(emb_path.split('/')[-1])
            print("psnr : {}".format(float(psnr)))
            print("ssim : {}".format(float(ssim)))
            print("lpips : {}".format(float(lpips)))

        # fine_rgb = []
        # # distances = []
        # with torch.no_grad():
        #     for batch_rays in single_image_rays:
        #         _, (f_rgb, distance, _) = self(batch_rays, False, self.white_bkgd, False, a_emb=a_emb)
        #         fine_rgb.append(f_rgb.detach().cpu())
        #         # distances.append(distance)

        # fine_rgb = torch.cat(fine_rgb, dim=0)

        # # distances = torch.cat(distances, dim=0)
        # # distances = distances.reshape(1, H_test, W_test)  # H W
        # # distances = visualize_depth(distances)

        # fine_rgb = fine_rgb.reshape(H_test, W_test, fine_rgb.shape[-1])  # H W C
        
        # im = Image.fromarray((fine_rgb.numpy()*255).astype(np.uint8))
        # im.save(os.path.join(save_dir,"final.png"))
        # im = Image.fromarray((test_img.numpy()*255).astype(np.uint8))
        # im.save(os.path.join(save_dir,"gt.png"))

        # psnr = calc_psnr(fine_rgb, test_img)

        # print("final psnr : {}".format(str(float(psnr))))

        return {'psnr':psnr, 'ssim':ssim, 'lpips':lpips}

class NeRFLoss():
    def __init__(self, lambda_sparsity=0.):
        super().__init__()
        self.lambda_sparsity = lambda_sparsity

    def __call__(self, s_rgb, gt, sigmas=None, t_rgb=None, t_alpha=None, beta=None, fine=True):
        if fine:
            if (t_rgb is not None) and (t_alpha is not None):
                pred = s_rgb*(1-t_alpha) + t_rgb*t_alpha
            else:
                pred = s_rgb

            if beta is None:
                loss = torch.mean((pred-gt)**2)
            else: 
                loss1 = torch.mean((pred-gt)**2/(2*(beta**2)))
                loss2 = 3 + torch.log(beta.squeeze()).mean()
                loss = loss1 + loss2


        else:  
            if t_alpha is None:
                loss = torch.mean((s_rgb-gt)**2)
            else:
                loss = torch.mean((1-t_alpha.detach())*((s_rgb-gt)**2))
            # reg_s = torch.mean(t_alpha.detach().unsqueeze(1)*torch.log(1 + 2*(sigmas**2)))  # sparsity loss
            reg_s = torch.mean(torch.log(1 + 2*(sigmas**2)))  # sparsity loss
            loss += self.lambda_sparsity*reg_s
        return loss