# Gripping Demo
The main repository for 3D Perception and Gripping Prototype using 3D Cameras.

## Overview
This repository includes
- gripping demostration several kinds of depth camera, pose detectors and xARM5 robot arm

- python wrapper for Kinect for Azure and our own structure light depth camera

- interface to our own structure light depth camera

- pose detectors
    1. passive IR pose detector that using 2 IR markers to determine a pose of object
    2. interface to pose detector based on deep neural network

- a rule-based robot arm motion planner

- a program for Camera-RobotArm calibration

- a user-end program to record tasks with 3D perception capability for the robot arm

## Caution
- During use, people should stay away from the robot arm to avoid accidental injury or damage to other items by the robot arm.
- Please make sure the arm don't encounter obstacles.
- Protect the arm before unlocking the motor.
- When deploy to a new environment, set the proper moving range, set calibration points and set home position first 

## Installation
Tested on ubuntu18.04 and python3.6

### Optional Kinect for Azure
- Install AzureKinect SDK

  https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md

  There is no need to compile, just installing

- Install pyk4a

  ```
  git clone https://github.com/etiennedub/pyk4a
  cd pyk4a
  pip install -e .
  ```
  
  Make sure viewer_transformation.py in example of pyk4a works. You may need to run "pip install typing-extensions" if it throws an error.

  If on windows, extra link path and includes may be needed:
  ```
  pip install pyk4a --no-use-pep517 --global-option=build_ext --global-option="-LC:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\lib;C:\Users\lenovo\Desktop\workspace\vcpkg\installed\x64-windows\lib" --global-option="-IC:\Users\lenovo\Desktop\workspace\vcpkg\installed\x64-windows\include\python3.8;C:\Program Files\Azure Kinect SDK v1.4.1\sdk\include"
  ```
  Change the path accrodingly. 

### xArm-Python-SDK
- Install xArm-Python-SDK

  ```
  git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
  cd xArm-Python-SDK
  pip install -e .
  ```
### Huarry Camera SDK
- Install huarry camera SDK if our own structure light depth camera is used

  http://download.huaraytech.com/pub/sdk/Ver2.2.3/
  http://download.huaraytech.com/pub/sdk/Ver2.2.5/

### Others

  ```
  pip install opencv-python pynput pyserial cmake apriltag hidapi open3d

  ```

## PVN3D
If pose-detected is based on PVN3D, this section is needed.

### PyTorch1.6.0
  pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

### Cuda10.1 and cudnn7.6.2, Nvidia driver 450.80.02
  Remove old drivers, CUDA, and cuDNN
  Download driver from Nvidia and install. Disable secure boot in bios, otherwise nvidia driver may fail.
  For CUDA and CUDNN refer to https://blog.csdn.net/lemonxiaoxiao/article/details/105693389

### Pointcloud library and python-pcl
  Make sure libpcl-dev version is 1.8 (the default version to install on Ubuntu18.04).
  ```
  sudo apt install libpcl-dev
  ```
  The python bindings python-pcl with ubuntu18.04 fix:
  ```
  git clone https://github.com/Tuebel/python-pcl
  cd python-pcl
  pip install -e .
  ```
  Do not use "pip install python-pcl" or install the original one from [https://github.com/strawlab/python-pcl]. This will bring dependency problem (mainly libVTK) on ubuntu whose version is higher than 16.04.

### Checkout and build PVN3D
  Then follow the instruction of the repo, put it to the same path as this repo, as described in PVN3DPoseDetector of pose_dectector.py.
  Remove the requiremts for `torch-1.0.1.post2` and `torchvision` when pip install requiremts.txt

  https://github.com/Jialn/PVN3D
  PVN3D is working with CUDA-9, cuDNN and Pytorch 1.0.1.post2, not compatiable with our main env

### A workaround to make PVN3D working with the new env: pytorch 1.6 and Cuda10.1
  Checkout PVN3D, https://github.com/Jialn/PVN3D, put it to the same path as this repo, as described in PVN3DPoseDetector of pose_dectector.py.
  Follow the instruction of the repo, install the requirements
  Download the pre-trained model and YCB-dataset, put to the folder described in PVN3D.
  cd ../PVN3D
  git checkout pytorch-1.5
  rm -r build
  run build ext: `python3 setup.py build_ext`, The compile should be successful. If so, switch back to master, It should work with new dependencies now.
  git checkout master

## For d-cv-multitask 2D segmentation

### d-cv-multitask and xdwx-mxnet1.5.0
  Do not use official pre-built mxnet like `pip install mxnet-cu101==1.5.1`. Need xdwx-mxnet.

  If the the nevironment are exactly the same as above (default version on ubuntu like, opencv3.2, python3.6; cuda 10.1, etc) the copied version should work. otherwise need to rebuild. need set "export TMPDIR=$(pwd)" before build.
  Copy the folder `d-cv-multitask` to the same path as this repo. The folder should with pre-built mxnet in subfold 3rdpatry.

  Install the dependency:
  `sudo apt install libopencv-dev libopenblas-dev`
  `pip3 install Polygon3 pyclipper shapely easydict imageio`

  Run `python scripts/multitask/test_multitask_vis.py` in path d-cv-multitask, to test d-cv-multitask and xdwx-mxnet are working.
  May also need to install packages "chardet-4.0.0 graphviz-0.8.4 idna-2.10 requests-2.25.1 urllib3-1.26.2"
  
  If rebuild is needed, ask algorthim-team for help. Need to copy the libmxnet.so by hand after compling: `cp lib/libmxnet.so python/mxnet/`

## Test
  
- Test robot arm, be careful. 测试时周边不要有障碍物！
  ```
  python arm_wrapper/arm_wrapper.py
  ```

- Test 3D camera
  ```
  cd ../x3d_camera
  python x3d_camera.py
  ```

- Test pose_detector
  ```
  python pose_detector.py
  ```

## Usage
  
- Run the gripping demostration to excute task.json:
   
  ```
  python gripping_demo.py
  ```
   Should have generate calibration file first, and set destination by recording task properly.

- To generate new task by put the robot into soft teaching mode:
   
  ```
  python gripping_demo.py rec
  ```
   This will enter task recording mode. Set the destination and waypoints by moving the robots. The configuration will be saved to ./task.json

## Progress of Calibration

一、标定结构光相机：参考结构光相机的repo

二、相机-机械臂标定：
1. ``python calibra_cam2world.py  ``
放置标定的Tag中心位置大概在机械抓末端中心位置，默认参数进行标定流程，检查calibra_cam2world.py中以下是否为默认参数:
``use_existing_cali_data = False; run_cali_test = False``

2. 以下选项开启则测试流程，修改后重新运行``python calibra_cam2world.py  ``可查看测试误差:
``use_existing_cali_data = True; run_cali_test = True``

三、运行抓取演示程序

1. ``python gripping_demo.py``

2. 启动后，按空格键执行一次3D成像和抓取, 按ESC可退出

## Subfolders

./arm_wrapper

contains wrappers for different kinds of Robot arms (xARM5, xARM6, DexARM)

../x3d_camera  

python interface for structure light depth camera. Includes driver and python interface for huarry industrial camera and PDC-03 projector.

./tools

tools for code-style checking

## Contributing Workflow
1. Install code style tools
```bash
pip install pre-commit cpplint pydocstyle
sudo apt install clang-format
```

2. Make local changes
```bash
git co -b PR_change_name origin/master
```

  Make change to your code

3. Run pre-commit before commit

```bash
pre-commit run --files ./*
```
  This will format the code and do checking. Then commit if passed.

4. Make pull request:
```bash
git push origin PR_change_name
```
