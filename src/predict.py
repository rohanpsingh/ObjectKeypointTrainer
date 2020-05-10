from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import re
import random
import argparse
import matplotlib.pyplot as plt

from utils.drawgaussian import *
from utils.evaluatepreds import EvaluatePreds
from utils.visualizepreds import VisualizePreds
from dataloader import ObjectKeypointDataset
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
    torch.set_default_tensor_type(torch.FloatTensor)
    cudnn.deterministic=True
    cudnn.benchmark = False
    print("====Loaded weights====")
    return net

def get_predictions(net, dataset_dir, off_file, num_feats, model_points, visualize=False, verbose=False):

    #load camera intrinsics (needed for 3D errors and visualization)
    camera_mat = np.load(os.path.join(dataset_dir, 'camera_matrix.npy'))

    #set up evaluator and visualizer
    evaluator = EvaluatePreds(model_points, camera_mat, verbose)
    vis = VisualizePreds(off_file, camera_mat)

    #load pre-computed mean and std of dataset
    meanstd_file = os.path.join(dataset_dir, 'mean.pth.tar')
    if os.path.isfile(meanstd_file):
        meanstd = torch.load(meanstd_file)
        mean, std = meanstd['mean'], meanstd['std']
    else:
        mean, std = torch.zeros(3), torch.zeros(3)

    #load dataset using dataloader
    eval_set = ObjectKeypointDataset(os.path.join(dataset_dir, "valid.txt"), num_feats, 256, 64, is_train=False)
    eval_data = DataLoader(eval_set, batch_size=eval_batch_size, shuffle=True, num_workers=6)
    print("valid data size is: {} batches of batch size: {}".format(len(eval_data), eval_batch_size))

    with torch.no_grad():
        for b, (inputs, targets, meta) in enumerate(eval_data):
            #forward pass through network
            inp = torch.autograd.Variable(inputs.cuda())
            out = net(inp)

            #get evaluations
            if verbose:
                print("Batch: ", b)
            out_poses = evaluator.getTransformationsUsingPnP(out[1].cpu(), meta['centers'], meta['scales'])
            tru_poses = evaluator.getTransformationsUsingPnP(targets, meta['centers'], meta['scales'])
            evaluator.calc_2d_errors(out[1].cpu(), meta['points'], meta['centers'], meta['scales'])
            evaluator.calc_3d_errors(out_poses, tru_poses)

            #visualize iff necessary
            if visualize:
                vis.draw_keypoints(out[1], meta['rgb'], meta['centers'], meta['scales'], meta['points'])
                vis.draw_model(meta['rgb'], out_poses)
                vis.cv_display(0)

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

    #get predictions on dataset and evaluate wrt ground truth
    get_predictions(net, opt.dataset, opt.obj_off, num_feats, model_points, opt.visualize, opt.verbose)

if __name__=='__main__':
    main()
