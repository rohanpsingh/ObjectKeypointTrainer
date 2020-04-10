import torch
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import re
import itertools
import random
from torch.utils.data import Dataset

from utils.preprocess import *
from utils.drawgaussian import *


class ObjectKeypointDataset(Dataset):
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
        for img, maps in zip(images, heatmaps):
            img.add_(self.mean.unsqueeze(1).unsqueeze(1))
            img= (255.0*img).permute(1,2,0).byte().numpy()[:,:,[2,1,0]]
            img = np.ascontiguousarray(img)

            for hm in maps:
                pt = np.asarray(np.unravel_index(hm.view(-1).max(0)[1].data, hm.size()))
                pt = transform_hm_to_org(torch.from_numpy(pt).flip(0), torch.tensor([self.inp_res/2]), self.inp_res/200.0).numpy()
                cv2.circle(img, tuple(map(int, pt)), 5, (0,255,0), -1)
            cv2.imshow("win", img)
            cv2.waitKey(100)
        #cv2.destroyAllWindows()
        return

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        #get image file name
        image_file = self.image_list[index]
        title, ext = os.path.splitext(os.path.basename(image_file))
        suffix = re.split("_",title)[1]

        #read rgb image
        rgb_image = im_to_torch(plt.imread(image_file))

        #read keypoints, center and scale
        keypts_filename = os.path.join(self.data_dir, "label", "label_" + suffix + ".txt")
        center_filename = os.path.join(self.data_dir, "center", "center_" + suffix + ".txt")
        scale_filename  = os.path.join(self.data_dir, "scale", "scales_" + suffix + ".txt")
        keypts = torch.from_numpy(np.loadtxt(keypts_filename))
        center = torch.from_numpy(np.loadtxt(center_filename))
        scale  = torch.from_numpy(np.loadtxt(scale_filename))

        #if training, apply image augmentation
        if self.is_train:
            center = center*((1+0.25*(np.random.rand()-0.5)))
            scale  = scale*((1+0.25*(np.random.rand()-0.5)))
            rot_val = torch.randn(1).mul_(15).clamp(-2*15, 2*15)[0] if random.random() <= 0.6 else 0

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
        for idx, pt in enumerate(keypts):
            keypts[idx] = to_torch(transform(pt, center, scale, [self.out_res, self.out_res], rot=rot_val))
            tar[idx], _ = draw_labelmap(tar[idx].numpy(), keypts[idx], 1.0)

        #meta data
        meta_info = {
            'rgb' : rgb_image,
            'points' : keypts,
            'centers' : center,
            'scales' : scale
        }
        return inp, tar, meta_info

    
    def manual_load(self, batch_size):
        """
        This function is deprecated.
        Superseded by torch.utils.data.Dataloader
        """

        grouped_batches = list(zip(*[iter(self.image_list)] * batch_size))
        random.shuffle(grouped_batches)

        load_data = []
        for indx, batch in enumerate(grouped_batches):
            rgb_batch = torch.zeros(batch_size, 3, 480, 640).float()
            cen_batch = torch.zeros(batch_size, 2).float()
            sca_batch = torch.zeros(batch_size, 1).float()
            tar_batch = torch.zeros(batch_size, self.num_features, self.out_res, self.out_res).float()
            pts_batch = torch.zeros(batch_size, self.num_features, 2).float()

            for i in range(len(batch)):

                title,ext = os.path.splitext(os.path.basename(batch[i]))
                suffix = re.split("_",title)[1]

                filename = os.path.join(batch[i])
                rgb_batch[i] = im_to_torch(plt.imread(filename))

                filename = os.path.join(self.data_dir, "label", "label_" + suffix + ".txt")
                imgpts = np.loadtxt(filename)
                pts_batch[i] = torch.from_numpy(imgpts).float()

                filename = os.path.join(self.data_dir, "center", "center_" + suffix + ".txt")
                cen_batch[i] = torch.from_numpy(np.loadtxt(filename))

                filename = os.path.join(self.data_dir, "scale", "scales_" + suffix + ".txt")
                sca_batch[i] = torch.from_numpy(np.loadtxt(filename))

                random_f = 1+0.25*(np.random.rand()-0.5)
                sca_batch[i] = sca_batch[i]*(random_f)
                cen_batch[i] = cen_batch[i]*(random_f)

            inp_batch = torch.zeros(batch_size, 3, self.inp_res, self.inp_res)
            for idx, (img, center, scale) in enumerate(zip(rgb_batch, cen_batch, sca_batch)):
                r = 0
                if self.is_train:
                    r = torch.randn(1).mul_(15).clamp(-2*15, 2*15)[0] if random.random() <= 0.6 else 0
            
                    img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                    img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                    img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

                inp = crop(img, center, scale, [self.inp_res, self.inp_res], rot=r)
                inp_batch[idx] = color_normalize(inp, self.mean, self.std)
                for i in range(self.num_features):
                    pt = transform(pts_batch[idx, i], center, scale, [self.inp_res, self.inp_res], rot=r)
                    pt = torch.from_numpy(pt).float()
                    pts_batch[idx, i] = pt
                    pt = transform_org_to_hm(pt, center, scale)
                    tar_batch[idx, i] = torch.from_numpy(DrawGaussian(tar_batch[idx, i].numpy(), pt, 1.0))

            if self.visualize:
                self._visualize_data(inp_batch, pts_batch)

            meta_data = {
                'rgb' : rgb_batch,
                'points' : pts_batch,
                'centers' : cen_batch,
                'scales' : sca_batch
            }
            load_data.append([inp_batch, tar_batch, meta_data])

        return load_data
