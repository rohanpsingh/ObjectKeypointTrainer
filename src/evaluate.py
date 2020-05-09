from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import cv2
import matplotlib.pyplot as plt
import re
import random
import itertools
import transforms3d as tf3
import math
import argparse
import statistics
from scipy.linalg import logm, expm

from utils.preprocess import *
from utils.drawgaussian import *
from models.StackedHourGlass import *

counter = 0

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.array(verts), np.array(faces)

def getTransformationsUsingPnP(x, cen, sca):                                    #returns pose of camera in object frame
    T = torch.zeros(x.size(0), 4, 4)
    for i in range(x.size(0)):
        pts_2d = []
        pts_3d = []
        for j in range(x.size(1)):
            pk = np.asarray(np.unravel_index(x[i][j].view(-1).max(0)[1].data, x[i][j].size()))
            pt = transform_hm_to_org(torch.from_numpy(pk).flip(0).float(), cen[i], sca[i])
            if x[i,j].max() != 0.0:                                              #do not consider points outside the frame
                pts_2d.append(pt.numpy())
                pts_3d.append(obj_points[j])
        a = np.ascontiguousarray(np.asarray(pts_2d)).reshape((len(pts_2d),1,2))
        b = np.ascontiguousarray(np.asarray(pts_3d)).reshape((len(pts_3d),1,3))
        _, rvec, tvec, inl = cv2.solvePnPRansac(b, a, camera_rgb_intrinsics, None, None, None, False, 1000, 1, 0.95, None, cv2.SOLVEPNP_EPNP)
        #_, rvec, tvec = cv2.solvePnP(b, a, camera_rgb_intrinsics, None, None, None, False, cv2.SOLVEPNP_EPNP)
        tf = np.eye(4)
        tf[:3,:3] = cv2.Rodrigues(rvec)[0]
        tf[:3, 3] = tvec[:,0]
        T[i] = torch.from_numpy(tf)                                             #To return pose of camera instead of object
    return T

def viz_keypoints(x, rgb_batch, cen_batch, sca_batch, pts_batch):
    batch_size = x.size(0)
    num_feats  = x.size(1)
    for i in range(batch_size):
        img = (255.0*rgb_batch[i]).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
        for j in range(num_feats):
            trupt = (int(pts_batch[i,j,0]), int(pts_batch[i,j,1]))
            points = np.unravel_index(x[i][j].cpu().view(-1).max(0)[1].data, x[i][j].size())
            points = transform_hm_to_org(torch.from_numpy(np.asarray(points)).flip(0).float(), cen_batch[i], sca_batch[i]).numpy()
            cv2.circle(img, tuple(points), 2, (0,255,0), -1)
            cv2.circle(img, tuple(trupt), 2, (255,0,0), -1)

        imgfn = os.path.dirname(model_file) + "/valids/kp_image_" + repr(i) + "_" + repr(img_fn_counter) + ".jpg"
        cv2.imshow("window", img)
        cv2.waitKey(200)
        cv2.imwrite(imgfn,img)
    cv2.destroyAllWindows()
    return

def viz_model(rgb_batch, cen_batch, sca_batch, pos_batch):
    batch_size = rgb_batch.size(0)
    for i in range(batch_size):
        img = (255.0*rgb_batch[i]).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
        tf = pos_batch[i].numpy()
        rvec,_ = cv2.Rodrigues(tf[:3,:3])
        tvec = tf[:3,3]
        imgpoints,_ = cv2.projectPoints(verts, rvec, tvec, camera_rgb_intrinsics, None)

        face_pts = [[tuple((int(imgpoints[idx,0,0]), int(imgpoints[idx,0,1]))) for idx in face] for face in faces]
        for face in random.sample(face_pts, 10000):
            cv2.fillPoly(img, [np.asarray(face)], (0,255,255))

        for pt in random.sample(imgpoints, 10000):
            cv2.circle(img, tuple((int(pt[0,0]), int(pt[0,1]))), 1, (255,0,0), -1)

        imgfn = os.path.dirname(model_file) + "/valids/mesh_image_" + repr(i) + "_" + repr(img_fn_counter) + ".jpg"
        cv2.imshow("window", img)
        cv2.waitKey(10)
        cv2.imwrite(imgfn, img)
    cv2.destroyAllWindows()
    return

def plot_errors(est_pos_batch, tru_pos_batch):
    batch_size = est_pos_batch.size(0)
    global counter
    for i in range(batch_size):
        est_tf = est_pos_batch[i].numpy()
        tru_tf = tru_pos_batch[i].numpy()
        pos_error = est_tf[:3,3] - tru_tf[:3,3]
        rot_error = np.subtract(tf3.euler.mat2euler(est_tf[:3,:3]), tf3.euler.mat2euler(tru_tf[:3,:3]))
        geo_error = np.linalg.norm(logm(np.dot(np.linalg.inv(tru_tf[:3,:3]), est_tf[:3,:3]), disp=False)[0])/np.sqrt(2)
        if (np.linalg.norm(pos_error)>0.1) or  (np.sum(rot_error*180/math.pi) > 30):
            counter += 1
        average_pos_error.append(pos_error)
        average_rot_error.append(rot_error*180/math.pi)
        average_geo_error.append(geo_error*180/math.pi)
        average_est_rot.append(np.asarray(tf3.euler.mat2euler(est_tf[:3,:3]))*180/math.pi)
        average_tru_rot.append(np.asarray(tf3.euler.mat2euler(tru_tf[:3,:3]))*180/math.pi)
    return

def get_object_definition(pp_file):
    with open(pp_file, 'r') as file:
        lines = [[float(i.rsplit('=')[1].rsplit('"')[1]) for i in line.split()[1:4]] for line in file.readlines()[8:-1]]
    return np.asarray(lines)
        
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--obj_off', required=True)
    ap.add_argument('--obj_inf', required=True)
    ap.add_argument('--visualize', required=False, default=False)
    ap.add_argument('--batch', required=False, default=8, type=int)
    opt = ap.parse_args()

    model_file = opt.weights
    dataset_path = opt.dataset
    mesh_filename = opt.obj_off
    visualize = (opt.visualize.lower()=='true')
    model_info_path = opt.obj_inf
    camera_mat_path = os.path.join(os.path.dirname(dataset_path), "camera_matrix.npy")
    eval_batch_size = opt.batch
    print("Evaluating: ", opt.weights)
    print("Dataset path: ", opt.dataset)
    print("Mesh path: ", opt.obj_off)

    manualSeed = 0
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    cudnn.deterministic=True
    cudnn.benchmark = False

    dataset_dir = os.path.dirname(dataset_path)
    with open(dataset_path) as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    random.shuffle(lines)
    grouped_batches = list(zip(*[iter(lines)] * eval_batch_size))
    random.shuffle(grouped_batches)
    print(len(grouped_batches), "batches of size ", eval_batch_size)

    camera_rgb_intrinsics = np.load(camera_mat_path)
    picked_points = get_object_definition(model_info_path)
    obj_points = (picked_points[:,:3]).astype(np.float64, order='C')

    if visualize:
        with open(mesh_filename) as f:
            verts, faces = read_off(f)

    inp_res = 256
    net_out_size = 64
    num_feats = picked_points.shape[0]
    net = StackedHourGlass(256, 2, 2, 4, num_feats).cuda()
    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    net.eval()
    net.load_state_dict(torch.load(model_file))

    average_kpt_error = 0
    average_pos_error = []
    average_rot_error = []
    average_geo_error = []
    average_est_rot = []
    average_tru_rot = []
    img_fn_counter = 0
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
                #input_tar_batch[i][j] = torch.from_numpy(draw_labelmap(input_tar_batch[i][j].numpy(), point, 1.0)).float()

        #input_sca_batch = input_sca_batch*(1+0.5*np.random.rand())
        #input_cen_batch = input_cen_batch*(1+0.5*np.random.rand())

        meanstd_file = os.path.join(dataset_dir, 'mean.pth.tar')
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        mean, std = meanstd['mean'], meanstd['std']

        inp = torch.zeros(eval_batch_size, 3, inp_res, inp_res)
        for idx, (img, center, scale) in enumerate(zip(input_rgb_batch, input_cen_batch, input_sca_batch)):
            img = crop(img, center, scale, [inp_res, inp_res], rot=0)
            inp[idx] = color_normalize(img, mean, std)

        inp = torch.autograd.Variable(inp.cuda())
        out = net(inp)
        out_poses = getTransformationsUsingPnP(out[1].cpu(), input_cen_batch, input_sca_batch)
        tru_poses = getTransformationsUsingPnP(input_tar_batch, input_cen_batch, input_sca_batch)

        error = []
        for i in range(eval_batch_size):
            for j in range(num_feats):
                out_points = np.asarray(np.unravel_index(out[1][i][j].cpu().view(-1).max(0)[1].data, out[1][i][j].size()))
                out_points = transform_hm_to_org(torch.from_numpy(np.asarray(out_points)).flip(0).float(), input_cen_batch[i], input_sca_batch[i]).numpy()
                tru_points = input_pts_batch[i][j].numpy()
                dist = torch.from_numpy(out_points - tru_points).float().norm(2).data
                error.append(dist)

        plot_errors(out_poses, tru_poses)
        print("Batch: ", indx)
        print("\tKeypoint error: ", sum(error)/len(error))
        print("\tPosition error: ", average_pos_error[-1])
        average_kpt_error += float(sum(error)/len(error))

        if visualize:
            viz_keypoints(out[1], input_rgb_batch, input_cen_batch, input_sca_batch, input_pts_batch)
            viz_model(input_rgb_batch, input_cen_batch, input_sca_batch, out_poses)
            img_fn_counter += 1


    print("Average error: ", average_kpt_error/len(grouped_batches))
    print("Average pos error: ", sum(average_pos_error)/len(average_pos_error))
    print("Average rot error: ", sum(average_rot_error)/len(average_rot_error))
    print("num of outliers: ", counter)
    print("==================")

    t_errors = [float(np.linalg.norm(e)) for e in average_pos_error]
    print(average_geo_error[0], len(average_geo_error))
    print("Mean pos error: ", statistics.mean(t_errors))
    print("Median pos error: ", statistics.median(t_errors))
    print("Mean geo error: ", statistics.mean(average_geo_error))
    print("Median geo error: ", statistics.median(average_geo_error))

    fig, axs = plt.subplots(3, 3)
    fig.suptitle("Avg errors in position(m), avg errors in rotation(deg), and absolute true+estimated rotation(deg)")

    axs[0,0].plot(np.asarray(average_pos_error)[:, 0])
    axs[0,0].set(ylabel='pos_error_x')
    axs[0,0].grid(True)

    axs[1,0].plot(np.asarray(average_pos_error)[:, 1])
    axs[1,0].set(ylabel='pos_error_y')
    axs[1,0].grid(True)

    axs[2,0].plot(np.asarray(average_pos_error)[:, 2])
    axs[2,0].set(ylabel='pos_error_z')
    axs[2,0].grid(True)

    axs[0,1].plot(np.asarray(average_rot_error)[:, 0])
    axs[0,1].set(ylabel='rot_error_x')
    axs[0,1].grid(True)

    axs[1,1].plot(np.asarray(average_rot_error)[:, 1])
    axs[1,1].set(ylabel='rot_error_y')
    axs[1,1].grid(True)

    axs[2,1].plot(np.asarray(average_rot_error)[:, 2])
    axs[2,1].set(ylabel='rot_error_z')
    axs[2,1].grid(True)

    axs[0,2].plot(zip(np.asarray(average_tru_rot)[:, 0], np.asarray(average_est_rot)[:, 0]))
    axs[0,2].set(ylabel='rot_track_x')
    axs[0,2].grid(True)

    axs[1,2].plot(zip(np.asarray(average_tru_rot)[:, 1], np.asarray(average_est_rot)[:, 1]))
    axs[1,2].set(ylabel='rot_track_y')
    axs[1,2].grid(True)

    axs[2,2].plot(zip(np.asarray(average_tru_rot)[:, 2], np.asarray(average_est_rot)[:, 2]))
    axs[2,2].set(ylabel='rot_track_z')
    axs[2,2].grid(True)

    plt.show()
