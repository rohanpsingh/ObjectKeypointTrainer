import cv2
import random
from utils.preprocess import *

class VisualizePreds(object):
    def __init__(self, mesh_filename, camera_mat):
        #camera intrinsics
        self.camera_mat = camera_mat
        #read object .off file to get vertices and faces
        v,f  = self.read_off(mesh_filename)
        self.verts = v
        self.faces = f
        #list of images to display
        self.disp_images = []
        
    def read_off(self, filename):
        """
        Parses a '.off' file and return lists of vertices and faces as numpy arrays.
        Input: path to .off file
        Returns: vertices, faces
        """
        with open(filename) as f:
            if 'OFF' != f.readline().strip():
                raise('Not a valid OFF header')
            n_verts, n_faces, n_dontknow = tuple([int(s) for s in f.readline().strip().split(' ')])
            verts = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
            faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return np.array(verts), np.array(faces)

    def draw_keypoints(self, heatmaps, rgb_batch, cen_batch, sca_batch, pts_batch):
        """
        Extracts peaks from heatmaps and draws points at corresponding positions
        on BGR image using OpenCV draw functions.
        Appends canvas to list.
        Returns: None
        """
        batch_size = heatmaps.size(0)
        num_feats  = heatmaps.size(1)
        for i in range(batch_size):
            img = (255.0*rgb_batch[i]).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
            for j in range(num_feats):
                trupt = (int(pts_batch[i,j,0]), int(pts_batch[i,j,1]))
                points = np.unravel_index(heatmaps[i][j].cpu().view(-1).max(0)[1].data, heatmaps[i][j].size())
                points = transform_hm_to_org(torch.from_numpy(np.asarray(points)).flip(0).float(), cen_batch[i], sca_batch[i]).numpy()
                cv2.circle(img, tuple(points), 3, (0,0,255), -1)  #predicted points in red
                cv2.circle(img, tuple(trupt), 3, (0,255,0), -1)   #ground-truth points in green
        self.disp_images.append(img)
        return

    def draw_model(self, rgb_batch, pos_batch):
        """
        Performs 3D->2D projection of vertices and faces and draws on BGR image.
        Appends canvas to list.
        Returns: None
        """
        batch_size = rgb_batch.size(0)
        for i in range(batch_size):
            img = (255.0*rgb_batch[i]).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
            tf = pos_batch[i].numpy()
            rvec,_ = cv2.Rodrigues(tf[:3,:3])
            tvec = tf[:3,3]
            imgpoints,_ = cv2.projectPoints(self.verts, rvec, tvec, self.camera_mat, None)
            face_pts = [[tuple((int(imgpoints[idx,0,0]), int(imgpoints[idx,0,1]))) for idx in face] for face in self.faces]
            #add some faces
            for face in random.sample(face_pts, 10000):
                cv2.fillPoly(img, [np.asarray(face)], (0,255,255))
            #add some vertices
            for pt in random.sample(imgpoints, 10000):
                cv2.circle(img, tuple((int(pt[0,0]), int(pt[0,1]))), 1, (255,0,0), -1)
        self.disp_images.append(img)
        return

    def cv_display(self, delay=0, cvwindow='preds_window'):
        """
        Diplays all images in list in a horizontal stack.
        Returns: None
        """
        if not self.disp_images:
            return
        np_horizontal = np.concatenate(self.disp_images, axis=1)
        cv2.imshow(cvwindow, np_horizontal)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
        self.disp_images = []
        return
