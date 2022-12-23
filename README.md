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
|   ├── [split].tsv
```
