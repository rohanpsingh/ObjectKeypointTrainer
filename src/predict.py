from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import re
import random
import argparse
import matplotlib.pyplot as plt

from utils.drawgaussian import *
from utils.evaluatepreds import EvaluatePreds
from utils.visualizepreds import VisualizePreds
from models.StackedHourGlass import *

np.set_printoptions(suppress=True)
net_inp_res = 256
net_out_size = 64
eval_batch_size = 1

def initialize_net(trained_weights, num_feats):
    net = StackedHourGlass(256, 2, 2, 4, num_feats).cuda()
    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    net.eval()
    net.load_state_dict(torch.load(trained_weights))
    manualSeed = 0
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    cudnn.deterministic=True
    cudnn.benchmark = False
    print("====Loaded weights====")
    return net

def get_predictions(net, opt, num_feats, model_points, visualize=False, verbose=False):

    #load camera intrinsics (needed for 3D errors and visualization)
    camera_mat = np.load(os.path.join(opt.dataset, 'camera_matrix.npy'))

    #set up evaluator
    evaluator = EvaluatePreds(opt, model_points, camera_mat)

    if visualize:
        vis = VisualizePreds(opt.obj_off, camera_mat)

    #load pre-computed mean and std of dataset
    meanstd_file = os.path.join(opt.dataset, 'mean.pth.tar')
    if os.path.isfile(meanstd_file):
        meanstd = torch.load(meanstd_file)
        mean, std = meanstd['mean'], meanstd['std']
    else:
        mean, std = torch.zeros(3), torch.zeros(3)

    #path to txt file containing paths to test images
    images_txt_path = os.path.join(opt.dataset, 'valid.txt')

    with open(images_txt_path) as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    random.shuffle(lines)
    grouped_batches = list(zip(*[iter(lines)] * eval_batch_size))
    random.shuffle(grouped_batches)
    print(len(grouped_batches), "batches of size ", eval_batch_size)

    for indx, batch in enumerate(grouped_batches):
        input_rgb_batch  = torch.zeros(eval_batch_size, 3, 480, 640).float()
        input_cen_batch  = torch.zeros(eval_batch_size, 2).float()
        input_sca_batch  = torch.zeros(eval_batch_size, 1).float()
        input_pts_batch  = torch.zeros(eval_batch_size, num_feats, 2).float()
        input_tar_batch  = torch.zeros(eval_batch_size, num_feats, net_out_size, net_out_size).float()

        for i in range(eval_batch_size):
            title,ext = os.path.splitext(os.path.basename(batch[i]))
            suffix = re.split("_",title)[1]

            filename = os.path.join(batch[i])
            input_rgb_batch[i] = im_to_torch(plt.imread(filename))

            filename = os.path.join(os.path.dirname(filename), "..", "label", "label_" + suffix + ".txt")
            imgpts = np.loadtxt(filename)
            input_pts_batch[i] = torch.from_numpy(imgpts).float()[:num_feats]

            filename = os.path.join(os.path.dirname(filename), "..", "center", "center_" + suffix + ".txt")
            input_cen_batch[i] = torch.from_numpy(np.loadtxt(filename))

            filename = os.path.join(os.path.dirname(filename), "..", "scale", "scales_" + suffix + ".txt")
            input_sca_batch[i] = torch.from_numpy(np.loadtxt(filename))

            for j in range(num_feats):
                point = torch.Tensor((imgpts[j,0], imgpts[j,1]))
                point = transform_org_to_hm(point, input_cen_batch[i], input_sca_batch[i])
                input_tar_batch[i][j] = torch.from_numpy(DrawGaussian(input_tar_batch[i][j].numpy(), point, 1.0)).float()

        #input_sca_batch = input_sca_batch*(1+0.5*np.random.rand())
        #input_cen_batch = input_cen_batch*(1+0.5*np.random.rand())

        inp = torch.zeros(eval_batch_size, 3, net_inp_res, net_inp_res)
        for idx, (img, center, scale) in enumerate(zip(input_rgb_batch, input_cen_batch, input_sca_batch)):
            img = crop(img, center, scale, [net_inp_res, net_inp_res], rot=0)
            inp[idx] = color_normalize(img, mean, std)

        inp = torch.autograd.Variable(inp.cuda())
        out = net(inp)
        out_poses = evaluator.getTransformationsUsingPnP(out[1].cpu(), input_cen_batch, input_sca_batch)
        tru_poses = evaluator.getTransformationsUsingPnP(input_tar_batch, input_cen_batch, input_sca_batch)

        print("Batch: ", indx)
        evaluator.calc_2d_errors(out[1].cpu(), input_pts_batch, input_cen_batch, input_sca_batch)
        evaluator.calc_3d_errors(out_poses, tru_poses)

        if visualize:
            vis.draw_keypoints(out[1], input_rgb_batch, input_cen_batch, input_sca_batch, input_pts_batch)
            vis.draw_model(input_rgb_batch, out_poses)
            vis.cv_display(100)

    #plot errors
    evaluator.plot()

    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--obj_off', required=True)
    ap.add_argument('--obj_inf', required=True)
    ap.add_argument('--visualize', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    opt = ap.parse_args()

    print("Evaluating: ", opt.weights)
    print("Dataset path: ", opt.dataset)
    print("Mesh path: ", opt.obj_off)

    #load 3D model points
    with open(opt.obj_inf) as file:
        lines = [[float(i.rsplit('=')[1].rsplit('"')[1])
                  for i in line.split()[1:4]] for line in file.readlines()[8:-1]]
    model_points = np.asarray(lines).astype(np.float64, order='C')
    num_feats = model_points.shape[0]

    #load weights and set seeed
    net = initialize_net(opt.weights, num_feats)

    get_predictions(net, opt, num_feats, model_points, opt.visualize, opt.verbose)

if __name__=='__main__':
    main()
