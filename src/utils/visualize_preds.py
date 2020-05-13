import cv2
import random
import numpy as np

class VisualizePreds(object):
    def __init__(self, mesh_filename, camera_mat):
        #camera intrinsics
        self.camera_mat = camera_mat
        #read object .off file to get vertices and faces
        v,f  = self.read_off(mesh_filename)
        self.verts = v
        self.faces = f
        #list of images to display
        self.org_canvas = []
        self.out_images = []
        
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

    def set_canvas(self, input_tensor):
        """
        Converts pytorch tensor (CxHxW) image to
        numpy array of (WxHxC) shape (OpenCV BGR)
        format and appends to list of canvas.
        """
        for image in input_tensor:
            image = (255.0*image).permute(1,2,0).byte()[:,:,[2,1,0]].numpy()
            self.org_canvas.append(image)
        return

    def draw_keypoints(self, est_pts_batch, tru_pts_batch, color=(0, 0, 255)):
        """
        Draws circles on each canvas at given input positions
        using OpenCV draw functions.
        Appends canvas to list.
        Returns: None
        """
        assert(len(est_pts_batch)==len(self.org_canvas))
        assert(len(tru_pts_batch)==len(self.org_canvas))
        for feats1, feats2, img in zip(est_pts_batch, tru_pts_batch, self.org_canvas):
            canvas = img.copy()
            if isinstance(feats1, dict):
                feats1 = feats1.values()
            if isinstance(feats2, dict):
                feats2 = feats2.values()
            for pt1, pt2 in zip(feats1, feats2):
                cv2.circle(canvas, tuple(map(int, pt1)), 3, color, -1)
                cv2.circle(canvas, tuple(map(int, pt2)), 3, (0, 255, 0), -1)
            self.out_images.append(canvas)
        return

    def draw_model(self, tf_batch, color=(0, 0, 255)):
        """
        Performs 3D->2D projection of vertices and faces and draws on BGR image.
        Appends canvas to list.
        Returns: None
        """
        for tf, img in zip(tf_batch, self.org_canvas):
            canvas = img.copy()
            rvec,_ = cv2.Rodrigues(tf[:3,:3])
            tvec = tf[:3,3]
            imgpoints,_ = cv2.projectPoints(self.verts, rvec, tvec, self.camera_mat, None)
            face_pts = [[tuple((int(imgpoints[idx,0,0]), int(imgpoints[idx,0,1]))) for idx in face] for face in self.faces]
            #add some faces
            for face in random.sample(face_pts, int(0.1*(len(face_pts)))):
                cv2.fillPoly(canvas, [np.asarray(face)], (0,255,255))
            #add some vertices
            for pt in random.sample(imgpoints, int(0.1*(len(imgpoints)))):
                cv2.circle(canvas, tuple((int(pt[0,0]), int(pt[0,1]))), 1, color, -1)
            self.out_images.append(canvas)
        return

    def cv_display(self, delay=0, cvwindow='preds_window'):
        """
        Diplays all images in list in a horizontal stack.
        Returns: None
        """
        if not self.out_images:
            return
        #create fixed size opencv window
        cv2.namedWindow(cvwindow, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cvwindow, 1800, 400)
        #concatenate all images in list to display simultaneously
        np_horizontal = np.concatenate(self.out_images, axis=1)
        cv2.imshow(cvwindow, np_horizontal)
        #clear canvases
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
        self.org_canvas = []
        self.out_images = []
        return
