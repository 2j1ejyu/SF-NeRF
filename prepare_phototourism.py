import argparse
import numpy as np
import os
import glob
import pickle
import pandas as pd
from PIL import Image
from utils.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import torch
from kornia import create_meshgrid

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H, W, 3), the origin of the rays in world coordinate
        rays_d: (H, W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    return rays_o, rays_d

def get_ray_directions(H, W, K):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def main(args):
    os.makedirs(os.path.join(args.root_dir, 'cache'), exist_ok=True)
    print(f'Preparing cache for scale {args.img_downscale}...')

    assert args.img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'

    img_ids, image_paths, xyz_world, poses, nears, fars, Ks, all_rays, all_rgbs = \
    process_data(args.root_dir, args.img_downscale)

    # save img ids
    path = os.path.join(args.root_dir, f'cache/img_ids.pkl')
    if not os.path.isfile(path):
        with open(os.path.join(args.root_dir, f'cache/img_ids.pkl'), 'wb') as f:
            pickle.dump(img_ids, f, pickle.HIGHEST_PROTOCOL)

    # save img paths
    path = os.path.join(args.root_dir, f'cache/image_paths.pkl')
    if not os.path.isfile(path):
        with open(path, 'wb') as f:
            pickle.dump(image_paths, f, pickle.HIGHEST_PROTOCOL)

    # save scene points
    path = os.path.join(args.root_dir, 'cache/xyz_world.npy')
    if not os.path.isfile(path):
        np.save(path, xyz_world)

    # save poses
    path = os.path.join(args.root_dir, 'cache/poses.npy')
    if not os.path.isfile(path):
        np.save(path, poses)

    # save near and far bounds
    path = os.path.join(args.root_dir, f'cache/nears.pkl')
    if not os.path.isfile(path):
        with open(path, 'wb') as f:
            pickle.dump(nears, f, pickle.HIGHEST_PROTOCOL)

    path = os.path.join(args.root_dir, f'cache/fars.pkl')
    if not os.path.isfile(path):
        with open(path, 'wb') as f:
            pickle.dump(fars, f, pickle.HIGHEST_PROTOCOL)

    # save Ks
    with open(os.path.join(args.root_dir, f'cache/Ks{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(Ks, f, pickle.HIGHEST_PROTOCOL)
    # save rays and rgbs
    with open(os.path.join(args.root_dir, f'cache/rays{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(all_rays, f)
    with open(os.path.join(args.root_dir, f'cache/rgbs{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(all_rgbs, f)

    print(f"Data cache saved to {os.path.join(args.root_dir, 'cache')} !")

def process_data(root_dir, img_downscale, test=False):
    tsv = glob.glob(os.path.join(root_dir, '*.tsv'))[0]
    files = pd.read_csv(tsv, sep='\t')
    files = files[~files['id'].isnull()] # remove data without id
    files.reset_index(inplace=True, drop=True)

    imdata = read_images_binary(os.path.join(root_dir, 'dense/sparse/images.bin'))
    img_path_to_id = {}
    for v in imdata.values():
        img_path_to_id[v.name] = v.id
    img_ids = []
    image_paths = {} # {id: filename}
    for filename in list(files['filename']):
        id_ = img_path_to_id[filename]
        image_paths[id_] = filename
        img_ids += [id_]

    Ks = {} # {id: K}
    camdata = read_cameras_binary(os.path.join(root_dir, 'dense/sparse/cameras.bin'))
    for id_ in img_ids:
        K = np.zeros((3, 3), dtype=np.float32)
        cam = camdata[id_]
        img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
        img_w_, img_h_ = img_w//img_downscale, img_h//img_downscale
        K[0, 0] = cam.params[0]*img_w_/img_w # fx
        K[1, 1] = cam.params[1]*img_h_/img_h # fy
        K[0, 2] = cam.params[2]*img_w_/img_w # cx
        K[1, 2] = cam.params[3]*img_h_/img_h # cy
        K[2, 2] = 1
        Ks[id_] = K

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
    for id_ in img_ids:
        im = imdata[id_]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
    poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
    # Original poses has rotation in form "right down front", change to "right up back"
    poses[..., 1:3] *= -1

    pts3d = read_points3d_binary(os.path.join(root_dir, 'dense/sparse/points3D.bin'))
    xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
    xyz_world_h = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], -1)
    # Compute near and far bounds for each image individually
    nears, fars = {}, {} # {id_: distance}
    for i, id_ in enumerate(img_ids):
        xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
        xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
        nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
        fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

    max_far = np.fromiter(fars.values(), np.float32).max()
    scale_factor = max_far/5 # so that the max far is scaled to 5
    poses[..., 3] /= scale_factor
    for k in nears:
        nears[k] /= scale_factor
    for k in fars:
        fars[k] /= scale_factor
    xyz_world /= scale_factor
    poses_dict = {id_: poses[i] for i, id_ in enumerate(img_ids)}

    img_ids_train = [id_ for i, id_ in enumerate(img_ids) 
                                if files.loc[i, 'split']=='train']
    img_ids_test = [id_ for i, id_ in enumerate(img_ids)
                                if files.loc[i, 'split']=='test']
    
    if test:
        target_ids = img_ids_test
    else:
        target_ids = img_ids_train

    all_rays = []
    all_rgbs = []
    for id_ in target_ids:
        c2w = torch.FloatTensor(poses_dict[id_])

        img = Image.open(os.path.join(root_dir, 'dense/images',
                                        image_paths[id_])).convert('RGB')
        img_w, img_h = img.size
        if img_downscale > 1:
            img_w = img_w//img_downscale
            img_h = img_h//img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
        img = np.array(img) / 255. # (H, W, 3)

        all_rgbs.append(img)
        
        directions = get_ray_directions(img_h, img_w, Ks[id_])
        rays_o, rays_d = get_rays(directions, c2w)   # (H, W, 3), (H, W, 3)
        # viewdirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)  # (H, W, 3)

        rays_t = id_ * torch.ones_like(rays_o[:,:,:1])  # (H, W, 1)

        all_ray = torch.cat([rays_o, rays_d,
                                    nears[id_]*torch.ones_like(rays_o[..., :1]),
                                    fars[id_]*torch.ones_like(rays_o[..., :1]),
                                    rays_t],
                                    -1).numpy() # (H, W, 9)
        all_rays.append(all_ray)

    if test:
        return img_ids_test, image_paths, nears, fars, all_rays, all_rgbs
    else:
        return img_ids, image_paths, xyz_world, poses, nears, fars, Ks, all_rays, all_rgbs
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of data')
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism data')
    args = parser.parse_args()
    
    main(args)