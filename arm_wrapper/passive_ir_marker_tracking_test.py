# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-09
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
A real-time position traking demo using robot arms.
This operation is in dervo mode j motion mode, which is lack of protection and dangerous.
Move slowly and be careful.
"""

import os
import time
import cv2
import numpy as np
from ..k4a_wrapper import K4aWrapper
from ..k4a_wrapper import K4aMarkerDetector
from arm_wrapper import xARMWrapper
from ..calibra_cam2world import Cam2World


def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
offset = np.array([0, 300, -50])
max_step = 60   
max_step_per_update = 20  # max step per 1/30s, in mili-meter

depth_camera = K4aWrapper(logging=True)
detector = K4aMarkerDetector(logging=False, calibra_src=depth_camera)
transformer = Cam2World()
arm = xARMWrapper()
arm._arm.motion_enable(enable=True)
arm._arm.set_mode(1)  # servo_j_mode
arm._arm.set_state(state=0)

while True:
    rgb_img, gray_img, depth_img = depth_camera.get_images(undistort=False)
    marker_pos = detector.get_marker_from_img(gray_img, depth_img)
    if len(marker_pos) == 1:
        cam_pos = marker_pos[0]
        pos = transformer.cam_2_world(np.array(cam_pos[:3]))[:3]
        pos = pos - offset
        arm_pos = arm.get_pos_cache()[:3]
        dis = distance(pos, arm_pos)
        if dis < 2.0:
            pass
        elif dis <= max_step_per_update:
            # set arm pos
            arm.set_pos_mode_j(x = pos[0], y=pos[1], z=pos[2])
        elif dis < max_step:
            # set arm pos by step for safty
            ratio = max_step_per_update / dis
            target_pos = np.array(arm_pos) * (1.0 - ratio) + pos * ratio
            arm.set_pos_mode_j(x = target_pos[0], y=target_pos[1], z=target_pos[2])
        else:
            # too far
            print("step too far, no update! dis:" + str(dis))

    cv2.imshow("rgb_img", rgb_img)
    key = cv2.waitKey(1)
    if key == 27:  # Esc key to stop
        break

arm.close()
depth_camera.close()
