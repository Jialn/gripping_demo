# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-18
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
Gripping demo using Kinect for Azure and simple pose detector.

To calibrate:
Run the calibra_k4_passiveIR.py first to generate ./calibration.json file.

To Generate task:
Run "python gripping_demo.py rec" to enter task recording mode. The configuration
    will be saved to ./task.json

To run the demo:
Run "python gripping_demo.py"
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import importlib
from config import Config
import transform_helper as trans_helper
# from k4a_wrapper import K4aWrapper
from pose_detector import PassiveMarkerPoseDetector, PVN3DPoseDetector, DCVMultiTask2DDetector
lib_path = os.path.abspath(os.path.join('../x3d_camera'))
sys.path.append(lib_path)
import x3d_camera
from x3d_camera import X3DCamera
from calibra_cam2world import Cam2World
from arm_wrapper.arm_wrapper import xARMWrapper
from planner import Planner
from pynput.keyboard import Key, KeyCode
from pynput.keyboard import Listener as kListener

keyflag = None

def on_press(key):
    print('{0} pressed'.format(key))


def on_release(key):
    global keyflag
    print('{0} release'.format(key))
    if isinstance(key, KeyCode):
        if key.char == 'q':
            keyflag = 'q'
        else:
            keyflag = key.char
    # if key is a special key
    if isinstance(key, Key):
        if key == Key.esc:
            keyflag = 'q'
        elif key == Key.space:
            keyflag = " "

listener_k = kListener(on_press=on_press, on_release=on_release)
listener_k.start()

if len(sys.argv) > 1:
    if sys.argv[1] == 'rec':
        # task recording mode
        from task_recorder import task_recording
        task_recording()
        exit()

visulize_pose_res_for_first_time = True
# pose detector type and camera type
POSE_DETECT_TYPE = Config.pose_detector_type  # 'mask2d'  # 'ir_marker' or 'pvn3d' or 'mask2d'
DEPTH_CAM_CLS = X3DCamera  # X3DCamera or K4aWrapper
# run the task group for limited times(1-n) or repeatedly(-1). if repeat_times == -1, press ESC to exit
repeat_times = 1

# init camera, detector, arm and planner
depth_camera = DEPTH_CAM_CLS(logging=False)
calibra = Cam2World()
depth_camera.re_capture_image_using_env_exposure = True

need_undistort = True
need_convert_world_coord = True
with open("./task_general.json") as cfg_file:  # the task description
    task_group = json.load(cfg_file)
if POSE_DETECT_TYPE == 'ir_marker':
    need_undistort = False
    detector = PassiveMarkerPoseDetector(logging=False, calibra_src=depth_camera)
    with open("./task_ir_marker.json") as cfg_file:  # the task description
        task_group = json.load(cfg_file)
elif POSE_DETECT_TYPE == 'pvn3d':
    detector = PVN3DPoseDetector(logging=False, calibra_src=depth_camera)
elif POSE_DETECT_TYPE == 'mask2d':
    detector = DCVMultiTask2DDetector(logging=visulize_pose_res_for_first_time, calibra_src=depth_camera, calibra_world=calibra)
    need_convert_world_coord = False  # DCVMultiTask2DDetector output world coord
else:
    print("unsupported pose detector!")
    depth_camera.close()
    exit()

arm = xARMWrapper(init_to_home=False)
planner = Planner(arm=arm, logging=True)

# go to initial position
planner.set_waiting_pos(task_group['waiting_pos'])
planner.go_waiting_pos()


def detect_objects():
    if Config.use_phsft:  # re-import gpu struli, because mxnet-cuda from pose dector will destory pycuda contex
        importlib.reload(x3d_camera.struli)
    rgb_img, gray_img, depth_img = depth_camera.get_images(undistort=need_undistort)
    detect_res = detector.get_obj_pose(rgb=rgb_img, gray=gray_img, depth=depth_img)
    detector._logging = False
    return detect_res

while True:
    if keyflag == ' ':
        detect_res = {}
        skip_det_next_time = False
        for task in task_group['tasks']:
            obj_type = task['object_type']
            planner.set_task(task)
            if task['source_pose'] is not None:  # using a fixed src pose
                obj_pose_matrix = trans_helper.xyzrpy2mat(task['source_pose'][:6])
                planner.plan(object_pose=obj_pose_matrix, grip_pose_transfrom=False)
            else:
                while True:
                    if not skip_det_next_time:
                        detect_res = detect_objects()
                    skip_det_next_time = False
                    if obj_type in detect_res.keys():  # object is detected
                        pose_list = detect_res[obj_type]
                        obj_pose_trans_matrix_cam = pose_list[0]
                        if need_convert_world_coord:
                            # translate to world coordinate, i.e., robot arm base coordinate
                            obj_pose_matrix = np.dot(calibra.get_transform_matrix(), obj_pose_trans_matrix_cam)
                            planner.plan(object_pose=obj_pose_matrix)
                        else:
                            planner.plan(object_pose=obj_pose_trans_matrix_cam)
                        # pose_list.pop(0)
                        # if len(pose_list) == 0: # no such object after excution, pop and exits
                        #     detect_res.pop(obj_type)
                        #     break
                    else:
                        skip_det_next_time = True
                        break  # there is no such object, obj_type not in detect_res.keys()

            # check early end condition
            if len(detect_res.keys()) == 0: break
            
        planner.set_waiting_pos(task_group['waiting_pos'])
        planner.go_waiting_pos()
        detect_res = {}
        depth_camera.close_projector()
        keyflag = None
    
    elif keyflag == 'q':
        print("q pressed, quit")
        break 
    else:
        pass

# release camera and robot arm
arm.close()
depth_camera.close()