

# Stacked-Hourglass networks for Object Keypoints

This is a PyTorch code for the training and evaluation of a stacked-hourglass network for detecting pre-defined semantic keypoints of a rigid, non-articulated 3D object in RGB images. Keypoint predictions from the stacked-hourglass network can be fed to a P*n*P module to obtain the full 6-DoF pose of the object in camera frame. The code currently only supports single-instance case, although it should be easy to modify the scripts for multiple objects.
We provide scripts for training the network, evaluating its predictions with respect to a given ground-truth and visualizing the predictions in several ways. Visualizing the predictions in 2D and 3D are frequently needed for gaining a better understanding of the network performance and for debugging potential bugs in the dataset or training procedure.

The code in this repository forms Part-2 of the full software:
![pose-estimation-github](https://user-images.githubusercontent.com/16384313/84745705-ec04bf00-afef-11ea-9966-c88f24c9a3ba.png)

Links to other parts:
- Part-1: [RapidPoseLabels](https://github.com/rohanpsingh/rapidposelabels) 
- Part-3: Not-yet-available

## Dependencies

All or several parts of the given Python 3.7.4 code are dependent on the following:
- PyTorch==1.5.0
- torchvision==0.6.0
- OpenCV
- [open3d](http://www.open3d.org/docs/release/getting_started.html)
- [transforms3d](https://matthew-brett.github.io/transforms3d)
- [statistics](https://pypi.org/project/statistics)

We recommend satisfying above dependencies to be able to use all scripts, though it should be possible to bypass some requirements depending to the use case. We recommend working in a [conda](https://docs.conda.io/en/latest/) environment.

## Usage
### Preparing the dataset
Each raw RGB image containing a single object should be labeled with a square bounding-box around the object (center pixel-coordinates and scale) and pixel coordinates of the keypoint locations.
Primarily, the labeled training dataset is supposed to be generated using a semi-automated technique proposed in **Rapid Pose Label Generation through Sparse Representation of Unknown Objects**. The code is expected to be made available soon here: [RapidPoseLabels](https://github.com/rohanpsingh/rapidposelabels). RapidPoseLabels is able to generate object keypoint labels, bounding-box labels and a sparse keypoint-based 3D object model, without requiring a previously built object CAD.
Nevertheless, if the user can prepare a **custom dataset** by labelling raw RGB images either manually or through some other technique (like manually aligning object CAD model in 3D space and then reprojecting pre-defined keypoints to image plane), it needs to be in the expected format. Hints to prepare the dataset:

1. Place raw frames in ```frames``` dir as ```frame_00000.jpg, frame_00001.jpg ... frame_0xxxx.jpg```. Then, for each raw frame ```frame_0xxxx.jpg```:
2. Place relative coordinates of the bounding box centers in ```center``` directory in *.txt files. Format: ```center/center_0xxxx.txt```
	```
	<x_center_rel>
	<y_center_rel>
	```
	where ``` <x_center_rel> = <center_x_coordinate>/<image_width> ``` and  ``` <y_center_rel> = <center_y_coordinate>/<image_width> ```.

3. Place scales of the bounding box centers in ```scale``` directory in *.txt files. Format: ```scale/scales_0xxxx.txt```
	```
	<scale>
	```
	where scale is defined as ```<scale> = max(<bounding_box_width>, <bounding_box_height>)/200```

4. Place absolute pixel-coordinates of the ```k```pre-defined object keypoints in the image in ```label``` directory in *.txt files. Format: ```label/label_0xxxx.txt```
	```
	<keypoint_0_x> <keypoint_0_y>
	<keypoint_1_x> <keypoint_1_y>
		|				|
	<keypoint_k_x> <keypoint_k_y>
	```
5. Store the Numpy array for the camera intrinsic parameter matrix in ```camera_matrix.npy``` (only required for full-pose estimation from keypoints)
6. Finally, run split.py to train/valid split the dataset. For example: ```$ python split.py wrench 10``` will look into all jpg images under the ```frames``` directory, and do a random 90-10 split to generate ```wrench/train.txt``` and ```wrench/valid.txt```.

Overview of the ```data``` directory tree is here:
```
root_dir/
├── data/
│   ├── split.py
│   ├── wrench/
│	│	├── center/
│	│	├── frames/
│	│	├── label/
│	│	├── scale/
│	│	├── train.txt
│	│	├── train.txt
│	│	└── camera_matrix.npy
│   ├── another_dataset_1/
│   └── another_dataset_2/
├── src/...
└── exps/...
```
### Training
To run the training script with default options, use:
```
$ cd /path-to-base-dir/src/
$ python train.py --dataset ../data/wrench --num_keypts 7
```
Setting ```--num_keypts 7``` will make the code to expect ```u,v``` coordinates of 7 keypoints in each of ```label_*.txt```. Hence, please ensure the consistency in the number of keypoints. 

Optional command lines to training script can be obtained:
```
$ python train.py --help
```
Providing output directory ```--outdir``` option will save training log and weights to the output directory (weights are not saved without this option!!).
### Plotting the loss curves
For convenience, we provide a script to parse the training log file and plot the train and valid loss curves.
```
$ python plot_log.py ../exps/0/log
```
During experimentation, it may be required to compare how to two logs compare to each other. We can plot the valid loss curves of two different experiments as follows:
```
$ python plot_multiple.py --log1 ../exps/0/log --log2 ../exps/1/log 
```
### Evaluation
We provide a script to use the trained model to make predictions on the validation set (any labeled set) and plot the errors with respect to the ground-truth. The predictions can be visualized with the ```--visualize``` option, which will display, in an OpenCV window, the keypoint predictions and 3D model mesh projected on the 2D RGB image according to the 6-DoF pose estimated using P*n*P. The mesh is expected as *.off model file and the selected object keypoint definitions in the object frame are provided as a [MeshLab]([http://www.meshlab.net/](http://www.meshlab.net/)) *.pp file.
```
$ python predict.py --weights <path_to_weights_file> --dataset <path_to_dataset>
 --obj_off <path_to_off_file> --obj_inf <path_to_pp_file> --verbose --visualize
```
This will, by default, display a cv window which shows the predicted keypoints in red and ground-truth keypoints in green (on the right), the projection of pose estimated from predicted keypoints in middle and that from ground-truth keypoints on the right.
![example visualization for facom electronic torque wrench](https://user-images.githubusercontent.com/16384313/82292563-190c8480-99e6-11ea-96ad-3f2b0ec8eb57.png)

Sometimes, instead of a 2D projection, user may require visualizing the estimated 6-DoF pose and the pose estimated from ground-truth keypoint annotations in a 3D environment for better debugging. For this, we use [open3d](http://www.open3d.org/docs/release/getting_started.html) to provide a rather basic visualization feature. To enable this, un-comment the following line in predict.py: ```#vis.visualize_3d(out_poses[0], tru_poses[0])```

The open3d visualization will look somewhat as follows (TODO: fix camera viewpoint). Here the green object is the true pose and the red is the predicted one.
<p align="center">
<img src="https://user-images.githubusercontent.com/16384313/82295952-735c1400-99eb-11ea-8141-7c0dadf65196.png" alt="open3d_window" width="80%">
<p>

### Example .pp file
Picking points in MeshLab using the PickPoints tool produces a *.pp file like below, which is then parsed to obtain the 3D object points. Again, please ensure the sequence and number of keypoints is consistent.
```
<!DOCTYPE PickedPoints>
<PickedPoints>
 <DocumentData>
  <DateTime time="13:57:53" date="2020-05-01"/>
  <User name="rohan"/>
  <DataFileName name="facom_tool.off"/>
  <templateName name="new Template"/>
 </DocumentData>
 <point x="-0.13846" y="-0.02119" z="0.0218537" active="1" name="0"/>
 <point x="-0.0970746" y="-0.0211021" z="0.0218537" active="1" name="1"/>
 <point x="-0.0983204" y="-0.0596115" z="0.0218537" active="1" name="2"/>
 <point x="0.0416884" y="-0.0156214" z="0.0117291" active="1" name="3"/>
 <point x="0.0728224" y="-0.0261951" z="0.011768" active="1" name="4"/>
 <point x="0.104821" y="-0.020917" z="0.0136446" active="1" name="5"/>
 <point x="0.208068" y="-0.0219506" z="0.0258971" active="1" name="6"/>
</PickedPoints>
```

## Potential Issues
- predict.py script currently works only for batch_size=1 (a larger batch size may be required for quicker evaluation of large datasets).
- Ideally, predict.py shouldn't require .off model file if visualization is not enabled.
- Tested only with all raw images in 640x480 resolution.
- Image cropping in preprocessing step is inefficient due to large zero-padded area.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

The general idea of object pose estimation using semantic keypoints is mainly based on [6-DoF Object Pose from Semantic Keypoints](https://www.seas.upenn.edu/~pavlakos/projects/object3d/). Some parts of this code are inspired and adapted from the following works on human pose estimation. If you use this code consider citing ours and the respective works.
- [https://github.com/princeton-vl/pose-hg-demo](https://github.com/princeton-vl/pose-hg-demo)  
- [https://github.com/bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose)
- [https://github.com/princeton-vl/pytorch_stacked_hourglass](https://github.com/princeton-vl/pytorch_stacked_hourglass)
