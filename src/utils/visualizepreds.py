import cv2
import random
from utils.preprocess import *

class VisualizePreds(object):
    def __init__(self, mesh_filename, camera_mat):

        self.camera_mat = camera_mat
        v,f  = self.read_off(mesh_filename)
        self.verts = v
        self.faces = f

        self.keypoints_img = None
        self.model_img = None
        
    def read_off(self, filename):
        with open(filename) as f:
            if 'OFF' != f.readline().strip():
                raise('Not a valid OFF header')
            n_verts, n_faces, n_dontknow = tuple([int(s) for s in f.readline().strip().split(' ')])
            verts = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
            faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return np.array(verts), np.array(faces)

    def draw_keypoints(self, heatmaps, rgb_batch, cen_batch, sca_batch, pts_batch):
        batch_size = heatmaps.size(0)
        num_feats  = heatmaps.size(1)
        for i in range(batch_size):
            img = (255.0*rgb_batch[i]).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
            for j in range(num_feats):
                trupt = (int(pts_batch[i,j,0]), int(pts_batch[i,j,1]))
                points = np.unravel_index(heatmaps[i][j].cpu().view(-1).max(0)[1].data, heatmaps[i][j].size())
                points = transform_hm_to_org(torch.from_numpy(np.asarray(points)).flip(0).float(), cen_batch[i], sca_batch[i]).numpy()
                cv2.circle(img, tuple(points), 2, (0,255,0), -1)
                cv2.circle(img, tuple(trupt), 2, (255,0,0), -1)
        self.keypoints_img = img
        return

    def draw_model(self, rgb_batch, pos_batch):
        batch_size = rgb_batch.size(0)
        for i in range(batch_size):
            img = (255.0*rgb_batch[i]).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
            tf = pos_batch[i].numpy()
            rvec,_ = cv2.Rodrigues(tf[:3,:3])
            tvec = tf[:3,3]
            imgpoints,_ = cv2.projectPoints(self.verts, rvec, tvec, self.camera_mat, None)

            face_pts = [[tuple((int(imgpoints[idx,0,0]), int(imgpoints[idx,0,1]))) for idx in face] for face in self.faces]
            for face in random.sample(face_pts, 10000):
                cv2.fillPoly(img, [np.asarray(face)], (0,255,255))

            for pt in random.sample(imgpoints, 10000):
                cv2.circle(img, tuple((int(pt[0,0]), int(pt[0,1]))), 1, (255,0,0), -1)

        self.model_img = img
        return

    def cv_display(self, delay=0, cvwindow='preds_window'):
        if (self.keypoints_img is None) or (self.model_img is None):
            return
        np_horizontal = np.concatenate((self.keypoints_img, self.model_img), axis=1)
        cv2.imshow(cvwindow, np_horizontal)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
        return
