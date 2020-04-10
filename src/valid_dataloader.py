import torch
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import re
import itertools
import random

from utils.preprocess import *
from utils.drawgaussian import *

def valid_loader(valid_data_path, valid_batch_size, net_inp_size, net_out_size, num_feats, visualize_valid_data=False):
    """
    This function is not used anymore.
    """


    valid_data_dir = os.path.dirname(valid_data_path)
    grouped_batches = []
    valid_txt_path = os.path.join(valid_data_dir, "valid.txt")
    f = open(valid_txt_path)
    lines = [line.rstrip("\n") for line in f.readlines()]
    grouped_batches.append(list(zip(*[iter(lines)] * valid_batch_size)))
    random.shuffle(grouped_batches)
    grouped_batches = list(itertools.chain.from_iterable(grouped_batches))

    random.shuffle(grouped_batches)
    valid_data = []

    for indx, batch in enumerate(grouped_batches):
        valid_rgb_batch = torch.zeros(valid_batch_size, 3, 480, 640).float()
        valid_cen_batch = torch.zeros(valid_batch_size, 2).float()
        valid_sca_batch = torch.zeros(valid_batch_size, 1).float()
        valid_tar_batch = torch.zeros(valid_batch_size, num_feats, net_out_size, net_out_size).float()
        valid_pts_batch = torch.zeros(valid_batch_size, num_feats, 2).float()

        for i in range(len(batch)):
            title,ext = os.path.splitext(os.path.basename(batch[i]))
            suffix = re.split("_",title)[1]

            filename = os.path.join(batch[i])
            valid_rgb_batch[i] = im_to_torch(plt.imread(filename))

            filename = os.path.join(valid_data_dir, "label", "label_" + suffix + ".txt")
            imgpts = np.loadtxt(filename)
            valid_pts_batch[i] = torch.from_numpy(imgpts).float()[:num_feats]

            filename = os.path.join(valid_data_dir, "center", "center_" + suffix + ".txt")
            valid_cen_batch[i] = torch.from_numpy(np.loadtxt(filename))

            filename = os.path.join(valid_data_dir, "scale", "scales_" + suffix + ".txt")
            valid_sca_batch[i] = torch.from_numpy(np.loadtxt(filename))

            random_f = 1+0.5*(np.random.rand()-0.5)
            valid_sca_batch[i] = valid_sca_batch[i]*(random_f)
            valid_cen_batch[i] = valid_cen_batch[i]*(random_f)

            for j in range(num_feats):
                point = torch.Tensor((imgpts[j,0], imgpts[j,1]))
                point = transform_org_to_hm(point, valid_cen_batch[i], valid_sca_batch[i])
                valid_tar_batch[i][j] = torch.from_numpy(DrawGaussian(valid_tar_batch[i][j].numpy(), point, 1.0)).float()

            if visualize_valid_data:
                img = (255.0*valid_rgb_batch[i]).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
                center = (int(valid_cen_batch[i][0]), int(valid_cen_batch[i][1]))
                cv2.circle(img, center, 5, (0,0,255), -1)
                cv2.putText(img, str(i+1), (40,80), cv2.FONT_HERSHEY_PLAIN,5,(0,0,0),3,cv2.LINE_AA)
                for j in range(num_feats):
                    pt = (int(imgpts[j,0]), int(imgpts[j,1]))
                    cv2.circle(img, pt, 5, (255,0,0), -1)
                s = valid_sca_batch[i]*100
                tl = (center[0]-s, center[1]-s)
                br = (center[0]+s, center[1]+s)
                if br[0]>640 or br[1]>480:
                    print filename
                cv2.rectangle(img,tl,br,(0,255,0),3)
                cv2.imshow("win", img)
                cv2.waitKey(200)

        valid_inp_batch = preprocess(valid_rgb_batch, valid_cen_batch, valid_sca_batch, net_inp_size)
        '''
        valid_meta_data = {'rgb' : valid_rgb_batch, 
                           'points' : valid_pts_batch,
                           'centers' : valid_cen_batch, 
                           'scales' : valid_sca_batch}
        '''
        valid_meta_data = []
        valid_data.append([valid_inp_batch, valid_tar_batch, valid_meta_data])

    cv2.destroyAllWindows()
    return valid_data
