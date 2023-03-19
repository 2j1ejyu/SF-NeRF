# SF-NeRF: Semantic-aware Occlusion Filtering Neural Radiance Fields in the Wild
This repository is the implementation of [SF-NeRF](https://arxiv.org/abs/2303.03966), written by Jaewon Lee for master's thesis, using pytorch-lightning.
![pipeline](https://user-images.githubusercontent.com/59645285/209291784-cbb3831b-126d-4b83-bc43-1d8e431f6665.png)


## Installation
```bash
conda create -n sfnerf python=3.9.12
conda activate sfnerf
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
## Data Preparation
Download phototourism dataset [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)

Download full split files [here](https://nerf-w.github.io/)

Before training, you must run the code below
```bash
python prepare_phototourism.py --root_dir [root directory] --img_downscale [downscale factor (original 1)]
```
where the root directory should follow the structure:
```                                                                                    
├── root directory
│   ├── dense                                                                                                  
│   │   ├── images                                                                                                                             
│   │   │   └── [image1].jpg                                                                      
│   │   │   └── [image2].jpg
│   │   │   └── ...
│   │   ├── sparse                                                                                                                             
│   │   │   └── cameras.bin
│   │   │   └── images.bin   
│   │   │   └── points3D.bin
│   │   ...
|   └── [split].tsv
```

## How To Run
### Training
You can train SF-NeRF by running:
```bash
python train.py \
--dataset_name phototourism \
--config ./configs/phototourism.yaml
```

You can change arguments in bash scripts as:
```bash
python train.py \
--dataset_name phototourism \
--config ./configs/phototourism.yaml \
train.img_downscale 2 \
val.img_downscale 2 \
loss.alpha_reg_mult 0.2 \
loss.sparsity_mult 1e-10 \
loss.alpha_smooth_mult 5e-7 \
data_path [phototourism dir]/brandenburg_gate \
t_net.seg_feat_dir [feature_root] \
t_net.beta_min 0.03 \
t_net.concrete_tmp 1.0 \
train.batch_size 4 \
train.chunk_size 1024 \
optimizer.max_steps 600000 \
optimizer.transient_lr_init 1e-4 \
optimizer.fin_mlp_lr_init 1e-5 \
optimizer.f_mlp_lr_init 1e-5 \
optimizer.lr_init 1e-3 \
val.img_idx [0,1,2,3] \
val.check_interval 1.0 \
exp_name brandengurb_toy
```
---
### Few-shot Training
For few-shot training, you should first make a split text file with the names of the images you want to train:
```                                                                                    
├── root directory
│   ├── dense                                                                                                  
│   │   ├── images                                                                                                                             
│   │   │   └── [image1].jpg                                                                      
│   │   │   └── [image2].jpg
│   │   │   └── ...
│   │   ├── sparse                                                                                                                             
│   │   │   └── cameras.bin
│   │   │   └── images.bin   
│   │   │   └── points3D.bin
│   │   ...
|   └── [split].tsv
|   └── few_30.txt
```
You should add arguments "fewshot" and "t_net.pretrained_path" in the script where "fewshot" indicates the number of images to train and "t_net.pretrained_path" the path of pretrained weight:
```bash
python train.py \
--dataset_name phototourism \
--config ./configs/phototourism.yaml \
train.img_downscale 2 \
val.img_downscale 2 \
loss.alpha_reg_mult 0.2 \
loss.sparsity_mult 1e-10 \
loss.alpha_smooth_mult 5e-7 \
data_path [phototourism dir]/brandenburg_gate \
t_net.seg_feat_dir [feature_root] \
t_net.beta_min 0.03 \
t_net.concrete_tmp 1.0 \
train.batch_size 4 \
train.chunk_size 1024 \
optimizer.max_steps 200000 \
optimizer.transient_lr_init 1e-4 \
optimizer.fin_mlp_lr_init 1e-5 \
optimizer.f_mlp_lr_init 1e-5 \
optimizer.lr_init 1e-3 \
val.img_idx [0,1,2,3] \
val.check_interval 1.0 \
t_net.pretrained_path ./logs/ckpt/[pretrain exp_name]/last.ckpt \
fewshot 30 \
exp_name fewshot_brandenburg_toy
```
---
### Evaluation
After training, you can evaluate your model by running:
```bash
python eval.py \
--ckpt ./logs/ckpt/brandengurb_toy/last.ckpt \
--save_image \
--lr 10 \
--img_downscale 2 \
--num_samples 256 \
--epochs 30
```
## Acknowledege
Our code is based on the pytorch lightning implementation of [NeRF-W](https://github.com/kwea123/nerf_pl/tree/nerfw/) and [mip-NeRF](https://github.com/hjxwhy/mipnerf_pl)
