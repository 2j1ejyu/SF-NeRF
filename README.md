## Installation
```bash
conda create -n sfnerf python=3.9.12
conda activate sfnerf
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
## Data Preparation
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
optimizer.max_steps 200000 \
optimizer.transient_lr_init 1e-4 \
optimizer.fin_mlp_lr_init 1e-5 \
optimizer.f_mlp_lr_init 1e-5 \
optimizer.lr_init 1e-3 \
val.img_idx [0,1,2,3] \
val.check_interval 1.0 \
exp_name brandengurb_toy
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
