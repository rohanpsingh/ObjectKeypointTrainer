from __future__ import print_function
import os
import numpy as np
import torch
import statistics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import logm
import transforms3d as tf3
import math
import cv2
from utils.preprocess import *

class EvaluatePreds(object):
    def __init__(self, model_points, camera_mat, verbose):

        self.camera_mat = camera_mat
        self.model_points = model_points
        self.verbose = verbose

        self.peak_thresh = 0.7
        self.outlier_count = 0
        self.list_kpt_error = []
        self.list_pos_error = []
        self.list_rot_error = []
        self.list_geo_error = []
        self.list_est_rot = []
        self.list_tru_rot = []


    def getTransformationsUsingPnP(self, heatmaps, centers, scales):
        '''
        Estimates pose of camera in object frame
        given the heatmap and bounding box predictions.
        Input: Heatmaps, bbox centers, bbox scales
        Output: 4x4 homogenous matrix
        '''
        batch_size = heatmaps.size(0)
        num_feats = heatmaps.size(1)
        T = torch.zeros(batch_size, 4, 4)
        for i in range(batch_size):
            pts_2d = []
            pts_3d = []
            for j in range(num_feats):
                pk = np.asarray(np.unravel_index(heatmaps[i][j].view(-1).max(0)[1].data, heatmaps[i][j].size()))
                pt = transform_hm_to_org(torch.from_numpy(pk).flip(0).float(), centers[i], scales[i])
                if heatmaps[i,j].max() > self.peak_thresh:
                    pts_2d.append(pt.numpy())
                    pts_3d.append(self.model_points[j])
            a = np.ascontiguousarray(np.asarray(pts_2d)).reshape((len(pts_2d),1,2))
            b = np.ascontiguousarray(np.asarray(pts_3d)).reshape((len(pts_3d),1,3))
            tf = np.eye(4)
            try:
                #_, rvec, tvec, inl = cv2.solvePnPRansac(b, a, self.camera_mat, None, None, None, False, 1000, 1, 0.95, None, cv2.SOLVEPNP_EPNP)
                _, rvec, tvec = cv2.solvePnP(b, a, self.camera_mat, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                tf[:3,:3] = cv2.Rodrigues(rvec)[0]
                tf[:3, 3] = tvec[:,0]
            except Exception as e:
                print(e)
            T[i] = torch.from_numpy(tf)                                             #To return pose of camera instead of object
        return T

    def calc_3d_errors(self, est_pos_batch, tru_pos_batch):
        batch_size = est_pos_batch.size(0)
        for i in range(batch_size):
            est_tf = est_pos_batch[i].numpy()
            tru_tf = tru_pos_batch[i].numpy()
            #position errors
            pos_error = np.abs(est_tf[:3,3] - tru_tf[:3,3])
            #rotation errors
            est_euler = np.asarray(tf3.euler.mat2euler(est_tf[:3,:3]))*180/math.pi
            tru_euler = np.asarray(tf3.euler.mat2euler(tru_tf[:3,:3]))*180/math.pi
            rot_error = np.abs(np.asarray([(e-360) if e>180 else ((e+360) if e<-180 else e)  for e in (est_euler-tru_euler)]))
            #rotation errors (geodesic distance)
            geo_error = np.linalg.norm(logm(np.dot(np.linalg.inv(tru_tf[:3,:3]), est_tf[:3,:3]), disp=False)[0])/np.sqrt(2)
            if (np.linalg.norm(pos_error)>0.1) or  (np.sum(rot_error) > 30):
                self.outlier_count += 1
            if not (est_tf==np.eye(4)).all():
                self.list_pos_error.append(pos_error.round(4))
                self.list_rot_error.append((rot_error).round(2))
                self.list_geo_error.append(geo_error*180/math.pi)
                self.list_est_rot.append(est_euler)
                self.list_tru_rot.append(tru_euler)

        if self.verbose:
            print("\tPosition error(meters): {}".format(self.list_pos_error[-1]))
            print("\tRotation error(degree): {}".format(self.list_rot_error[-1]))
        return


    def calc_2d_errors(self, heatmaps, tru_pts_batch, centers, scales):
        batch_kpt_error = []
        batch_size = heatmaps.size(0)
        num_feats = heatmaps.size(1)
        for i in range(batch_size):
            for j in range(num_feats):
                out_points = np.asarray(np.unravel_index(heatmaps[i][j].cpu().view(-1).max(0)[1].data, heatmaps[i][j].size()))
                out_points = transform_hm_to_org(torch.from_numpy(np.asarray(out_points)).flip(0).float(), centers[i], scales[i]).numpy()
                tru_points = tru_pts_batch[i][j].numpy()
                dist = torch.from_numpy(out_points - tru_points).float().norm(2).data
                batch_kpt_error.append(round(dist.item(),2))

        self.list_kpt_error.append(float(sum(batch_kpt_error)/len(batch_kpt_error)))
        if self.verbose:
            print("\tKeypoint error(pixels): {} \t(avg: {})".format(batch_kpt_error, self.list_kpt_error[-1]))
        return

    def plot(self):
        print("Average pix error: ", sum(self.list_kpt_error)/len(self.list_kpt_error))
        print("Average pos error: ", sum(self.list_pos_error)/len(self.list_pos_error))
        print("Average rot error: ", sum(self.list_rot_error)/len(self.list_rot_error))
        print("num of outliers: ", self.outlier_count)
        print("==================")

        t_errors = [float(np.linalg.norm(e)) for e in self.list_pos_error]
        print("Mean pos error: ", statistics.mean(t_errors))
        print("Median pos error: ", statistics.median(t_errors))
        print("Mean geo error: ", statistics.mean(self.list_geo_error))
        print("Median geo error: ", statistics.median(self.list_geo_error))

        grid = GridSpec(4, 3)
        fig = plt.figure()
        fig.clf()

        ax0 = fig.add_subplot(grid[0,0])
        ax1 = fig.add_subplot(grid[1,0])
        ax2 = fig.add_subplot(grid[2,0])
        ax3 = fig.add_subplot(grid[0,1])
        ax4 = fig.add_subplot(grid[1,1])
        ax5 = fig.add_subplot(grid[2,1])
        ax6 = fig.add_subplot(grid[0,2])
        ax7 = fig.add_subplot(grid[1,2])
        ax8 = fig.add_subplot(grid[2,2])
        ax9 = fig.add_subplot(grid[3,:])
        fig.suptitle("Avg errors in position(m), avg errors in rotation(deg), and absolute true+estimated rotation(deg)")

        ax0.plot(np.asarray(self.list_pos_error)[:, 0])
        ax0.set_ylabel('pos_error_x')
        ax0.grid(True)

        ax1.plot(np.asarray(self.list_pos_error)[:, 1])
        ax1.set_ylabel('pos_error_y')
        ax1.grid(True)

        ax2.plot(np.asarray(self.list_pos_error)[:, 2])
        ax2.set_ylabel('pos_error_z')
        ax2.grid(True)

        ax3.plot(np.asarray(self.list_rot_error)[:, 0])
        ax3.set_ylabel('rot_error_x')
        ax3.grid(True)

        ax4.plot(np.asarray(self.list_rot_error)[:, 1])
        ax4.set_ylabel('rot_error_y')
        ax4.grid(True)

        ax5.plot(np.asarray(self.list_rot_error)[:, 2])
        ax5.set_ylabel('rot_error_z')
        ax5.grid(True)

        ax6.plot(zip(np.asarray(self.list_tru_rot)[:, 0], np.asarray(self.list_est_rot)[:, 0]))
        ax6.set_ylabel('rot_track_x')
        ax6.grid(True)

        ax7.plot(zip(np.asarray(self.list_tru_rot)[:, 1], np.asarray(self.list_est_rot)[:, 1]))
        ax7.set_ylabel('rot_track_y')
        ax7.grid(True)

        ax8.plot(zip(np.asarray(self.list_tru_rot)[:, 2], np.asarray(self.list_est_rot)[:, 2]))
        ax8.set_ylabel('rot_track_z')
        ax8.grid(True)

        ax9.plot(zip(np.asarray(self.list_kpt_error)))
        ax9.set_ylabel('kpt_pix_error')
        ax9.grid(True)

        plt.show()
        return
