import random
import argparse
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.evaluate_preds import EvaluatePreds
from utils.visualize_preds import VisualizePreds
from dataloader import ObjectKeypointDataset
from models.StackedHourGlass import StackedHourGlass

np.set_printoptions(suppress=True)
NET_INP_RES = 256
NET_OUT_RES = 64
EVAL_BATCH_SIZE = 1

def initialize_net(trained_weights, num_feats):
    net = StackedHourGlass(NET_INP_RES, 2, 2, 4, num_feats).cuda()
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    net.load_state_dict(torch.load(trained_weights))
    manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    cudnn.deterministic=True
    cudnn.benchmark = False
    print("====Loaded weights====")
    return net

def get_predictions(args):
    #load 3D model points
    with open(args.obj_inf) as file:
        lines = [[float(i.rsplit('=')[1].rsplit('"')[1])
                  for i in line.split()[1:4]] for line in file.readlines()[8:-1]]
    model_points = np.asarray(lines).astype(np.float64, order='C')
    num_feats = model_points.shape[0]

    #load weights and set seeed
    net = initialize_net(args.weights, num_feats)
    #load camera intrinsics (needed for 3D errors and visualization)
    camera_mat = np.load(os.path.join(args.dataset, 'camera_matrix.npy'))

    #set up evaluator and visualizer
    evaluator = EvaluatePreds(model_points, camera_mat, args.verbose)
    if args.visualize:
        vis = VisualizePreds(args.obj_off, camera_mat)

    #load dataset using dataloader
    eval_set = ObjectKeypointDataset(os.path.join(args.dataset, "valid.txt"), num_feats, NET_INP_RES, NET_OUT_RES, is_train=False)
    eval_data = DataLoader(eval_set, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=0)
    print("valid data size is: {} batches of batch size: {}".format(len(eval_data), EVAL_BATCH_SIZE))

    with torch.no_grad():
        for batch_id, (inputs, targets, meta) in enumerate(eval_data):
            #forward pass through network
            inp = torch.autograd.Variable(inputs.cuda())
            out = net(inp)

            #get evaluations
            if args.verbose:
                print("Batch: ", batch_id)

            est_keypoints = evaluator.get_keypoints_from_heatmaps(out[1].cpu().numpy(), meta['centers'], meta['scales'])
            #tru_keypoints = evaluator.get_keypoints_from_heatmaps(targets.numpy(), meta['centers'], meta['scales'])
            tru_keypoints = [{idx:tuple(i.tolist()) for idx,i in enumerate(batch)} for batch in meta['points']]
            #tru_with_noise = [{idx:tuple((i*(1 + (np.random.rand()-0.5)*0.02)).tolist()) for idx,i in enumerate(batch)} for batch in meta['points']]

            out_poses = evaluator.get_pose_using_pnp(est_keypoints)
            tru_poses = evaluator.get_pose_using_pnp(tru_keypoints)

            evaluator.calc_2d_errors(est_keypoints, tru_keypoints)
            evaluator.calc_3d_errors(out_poses, tru_poses)

            #visualize iff necessary
            if args.visualize:
                vis.set_canvas(meta['rgb'])
                vis.draw_keypoints(est_keypoints, tru_keypoints)
                vis.draw_model(out_poses)
                vis.draw_model(tru_poses, color=(0, 255, 0))
                vis.cv_display(500, "batch: " + repr(batch_id))
                #vis.visualize_3d(out_poses[0], tru_poses[0])
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

    #get predictions on dataset and evaluate wrt ground truth
    get_predictions(opt)
    return

if __name__=='__main__':
    main()
