import argparse
import os
import torch
from tqdm import tqdm
from datasets import PhototourismDataset_test
from datasets import Rays as Ray_Collection
from datasets import Rays_keys, Rays
from utils.metrics import eval_errors, summarize_results
from utils.vis import save_images
from models.mip import rearrange_render_image
from models.nerf_system import NeRFSystem
from configs.config import default, get_from_path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="Path to ckpt.")
parser.add_argument("--data_path", help="Path to data.", default='/hub_data/jaewon/phototourism/brandenburg_gate')
parser.add_argument("--out_dir", help="Output directory.", type=str, default="./eval")
parser.add_argument("--chunk_size", help="Chunck size for render.", type=int, default=3072)
parser.add_argument("--white_bkgd", help="Train set image background color.", type=bool, default=False)
parser.add_argument('--save_image', help='whether save predicted image', action='store_true')
parser.add_argument('--lr', help='learning rate for optimization of appearance embedding', type=float, default=0.01)
parser.add_argument('--img_downscale', help='eval downscale', type=int, required=True)
parser.add_argument('--epochs', help='num epochs to optimize', type=int, default=30)
parser.add_argument('--img_idx', help='img nums to evaluate', type=str, default='')
parser.add_argument('--num_samples', type=int, default=256)
parser.add_argument('--test_only', action='store_true')


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    add_hparam = {}
    hparams = torch.load(args.ckpt)['hyper_parameters']
    default_hparams = get_from_path(hparams['config'])
    for k in default_hparams:
        if k not in hparams:
            add_hparam[k] = default_hparams[k]
    del hparams 

    model = NeRFSystem.load_from_checkpoint(args.ckpt, **add_hparam).to(device)
    model.mip_nerf.num_samples = args.num_samples
    for child in model.mip_nerf.children():
        for param in child.parameters():
            param.requires_grad = False
    model.mip_nerf.eval()
    hparams = model.hparams
    exp_name = hparams['exp_name']
    save_root = os.path.join(args.out_dir, exp_name,"downscale={}".format(args.img_downscale))
    test_dataset = PhototourismDataset_test(
                        root_dir=hparams['data_path'],
                        img_downscale=args.img_downscale,
                        val_idcs=[int(k) for k in args.img_idx.split(',')],
                        white_bkgd=args.white_bkgd,
                        )
    img_names = [test_dataset.img_names[id].replace('.jpg','') for id in test_dataset.img_ids_test]

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for i,data in enumerate(test_dataset):
        train_rays, train_img, test_rays, test_img = data
        train_rays = Ray_Collection(*[r.reshape(-1,r.shape[-1]) for r in train_rays])
        train_img = train_img.reshape(-1, train_img.shape[-1])
        img_name = img_names[i]
        results = model.eval_model(img_name, train_rays, test_rays, train_img, test_img, args.lr, args.epochs, args.chunk_size, save_root, args.test_only)
        psnr_values.append(results['psnr'])
        ssim_values.append(results['ssim'])
        lpips_values.append(results['lpips'])
    
    print("Average PSNR : {}".format(str(sum(psnr_values)/len(psnr_values))))
    print("Average SSIM : {}".format(str(sum(ssim_values)/len(ssim_values))))
    print("Average LPIPS : {}".format(str(sum(lpips_values)/len(lpips_values))))


    
    


if __name__ == '__main__':
    args = parser.parse_args()
    scenes = main(args)
    # I remove the LPIPS metric, if you want to eval it, you should modify eval code simply.
    # print('PSNR | SSIM | Average')
    # if args.scale == 1:
    #     print(summarize_results(args.out_dir, scenes, 1))
    # else:
    #     print(summarize_results(args.out_dir, scenes, args.scale))
