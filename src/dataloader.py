import os
import re
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from utils.preprocess import im_to_torch, to_torch
from utils.preprocess import crop, color_normalize
from utils.preprocess import transform
from utils.preprocess import transform_hm_to_org
from utils.drawgaussian import draw_labelmap


class ObjectKeypointDataset(torch.utils.data.Dataset):
    """
    This class loads a custom dataset.
    Inherits from pytorch data.Dataset class.
    """
    def __init__(self, txt_path, num_feats, inp_res, out_res, is_train=True, visualize=False):
        self.txt_path = txt_path
        self.num_features = num_feats
        self.inp_res = inp_res
        self.out_res = out_res
        self.is_train = is_train
        self.visualize = visualize

        self.data_dir = os.path.dirname(self.txt_path)

        with open(txt_path, 'r') as file:
            lines = [line.rstrip("\n") for line in file.readlines()]
        self.image_list = lines

        #compute mean and std of all train images
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = os.path.join(self.data_dir, 'mean.pth.tar')
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for img_path in self.image_list:
                img = im_to_torch(plt.imread(img_path))
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.image_list)
            std /= len(self.image_list)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
        return meanstd['mean'], meanstd['std']

    def _visualize_data(self, images, heatmaps):
        if (images.dim()==3) and (heatmaps.dim()==3):
            images = images.unsqueeze(0)
            heatmaps = heatmaps.unsqueeze(0)
        for img, maps in zip(images, heatmaps):
            img.add_(self.mean.unsqueeze(1).unsqueeze(1))
            img = (255.0*img).permute(1,2,0).byte().numpy()[:,:,[2,1,0]]
            img = np.ascontiguousarray(img)
            for heat_map in maps:
                peak_loc = np.asarray(np.unravel_index(heat_map.view(-1).max(0)[1].data, heat_map.size()))
                peak_loc = transform_hm_to_org(torch.from_numpy(peak_loc).flip(0),
                                               torch.tensor([self.inp_res/2]),
                                               self.inp_res/200.0).numpy()
                cv2.circle(img, tuple(map(int, peak_loc)), 5, (0,255,0), -1)
            cv2.imshow("win", img)
            cv2.waitKey(0)
        return

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        #get image file name
        image_file = self.image_list[index]
        title, _ = os.path.splitext(os.path.basename(image_file))
        suffix = re.split("_",title)[1]

        #read rgb image
        rgb_image = im_to_torch(plt.imread(image_file).copy())

        #read keypoints, center and scale
        keypts_filename = os.path.join(self.data_dir, "label", "label_" + suffix + ".txt")
        center_filename = os.path.join(self.data_dir, "center", "center_" + suffix + ".txt")
        scale_filename  = os.path.join(self.data_dir, "scale", "scales_" + suffix + ".txt")
        keypts = torch.from_numpy(np.loadtxt(keypts_filename)).float()
        center = torch.from_numpy(np.loadtxt(center_filename)).float()
        scale  = torch.from_numpy(np.loadtxt(scale_filename)).float()

        #if training, apply image augmentation
        if self.is_train:
            center = center*((1+0.5*(np.random.rand()-0.5)))
            scale  = scale*((1+0.5*(np.random.rand()-0.5)))
            #rot_val = torch.randn(1).mul_(15).clamp(-2*15, 2*15)[0] if random.random() <= 0.6 else 0
            rot_val = 0

            rgb_image[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            rgb_image[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            rgb_image[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        else:
            rot_val = 0

        #this is the input image to the network
        inp = crop(rgb_image, center, scale, [self.inp_res, self.inp_res], rot=rot_val)
        inp = color_normalize(inp, self.mean, self.std)

        #this is the target image
        tar = torch.zeros(self.num_features, self.out_res, self.out_res)
        for idx, key_pt in enumerate(keypts):
            pt_tf = to_torch(transform(key_pt, center, scale, [self.out_res, self.out_res], rot=rot_val))
            tar[idx], _ = draw_labelmap(tar[idx].numpy(), pt_tf, 1.0)

        if self.visualize:
            self._visualize_data(inp, tar)

        #meta data
        meta_info = {
            'rgb' : rgb_image,
            'points' : keypts,
            'centers' : center,
            'scales' : scale
        }
        return inp, tar, meta_info
