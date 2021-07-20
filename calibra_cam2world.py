# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-12
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
Using passive infrared marker or april-tag to get the 3D coordinate transform from 
camera to robot arm.

Usage:
Change the Parameters in this file properly (USE_PASSIV_IR_MARKER or ariltag, gripper_init_angle etc) 
Run: python calibra_cam2world.py

The calibra points is set in calibration.json. If the file not existing, a recording program will triggered.
Remove "calibration.json" when deploy to a new environment.
The marker is pasted on calibrating board. It should be on the center of the 
gripper. Do as the program's instruction.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from arm_wrapper.arm_wrapper import xARMWrapper
import apriltag
lib_path = os.path.abspath(os.path.join('../x3d_camera'))
sys.path.append(lib_path)
from x3d_camera import X3DCamera

# Parameters
# use ir marker + kinect or apriltag + x3d camera
USE_PASSIV_IR_MARKER = False
# only do the math with old data, do not use real robot
use_existing_cali_data = False
# should interactively put the caliboard to gripper
interactive_install_cali_board = True
# run the test or not, for most time it's not necessary
run_cali_test = False
# cali-board direction, should towards the camera
gripper_init_angle = 100  # 130
# cali-board width for gripping
# caliboard in SHANGHAI, 70cm; Beijing 72cm
CALI_BOARD_GRIP_CLOSE = 720

if USE_PASSIV_IR_MARKER:
    from k4a_wrapper import K4aWrapper
    from k4a_wrapper import K4aMarkerDetector
    DepthCameraCls = K4aWrapper
else:
    apriltag_detector = apriltag.Detector()
    DepthCameraCls = X3DCamera

# how many frame used for one robot pose when using USE_PASSIV_IR_MARKER
ir_marker_repeat_num = 15


class Cam2World():
    """ Helper class for transforming camera coordinate to world coordinate
    """

    def __init__(self):
        """Init.
        """
        with open("./calibration.json") as cfg_file:
            cali_cfgs = json.load(cfg_file)
        self._transform_cam2arm = np.array(cali_cfgs["transform_cam2arm"])

    def get_transform_matrix(self):
        """ Get transform matrix
        """
        return self._transform_cam2arm

    def cam_2_world(self, cam_pos):
        """Transform camera to world coordinate
        Input: 
            cam_pos: a signle or a list of cam_pos
        Return:
            numpy.array, the position in world coordinate
        """
        cam_pos = cam_pos.reshape(1, 3)
        ones = np.ones(len(cam_pos))
        cam_pos_c = np.insert(cam_pos, 3, values=ones, axis=1)
        cam_pos_t = cam_pos_c.T
        world_pos = np.dot(self._transform_cam2arm, cam_pos_t).T[0][:3]
        return world_pos

    def cam_2_world_sigle_points(self, cam_pos, overide_matrix=None):
        """Transform camera to world coordinate, faster than cam_2_world for sigle convertion
        Input: 
            cam_pos: list, a signle cam_pos
        Return:
            numpy.array, the position in world coordinate
        """
        cam_pos = cam_pos[:]
        cam_pos.append(1)
        cam_pos = np.array(cam_pos)
        if overide_matrix is not None:
            world_pos = np.dot(overide_matrix, cam_pos.T).T[:3]
        else:
            world_pos = np.dot(self._transform_cam2arm, cam_pos.T).T[:3]
        return world_pos


def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def homogeneous_coord_t(pos_3d):
    """
    Convert to homogeneous coordinate and then transpose the matrix
    """
    ones = np.ones(len(pos_3d))
    pos_homogeneous = np.insert(pos_3d, 3, values=ones, axis=1)
    return pos_homogeneous.T


def remove_newline_around_digits(json_str):
    """ Remove \n around digits to make the config file more readable
    """
    jason_str_list = list(json_str)
    i = len(jason_str_list) - 1
    last_right_idx = i
    only_digits = False
    while (i > 2):
        if jason_str_list[i] == ']':
            last_right_idx = i
            only_digits = True
        elif jason_str_list[i] == '[' and only_digits:
            for j in range(last_right_idx, i, -1):
                if jason_str_list[j] == ' ' or jason_str_list[j] == '\n':
                    jason_str_list.pop(j)
            only_digits = False
        elif jason_str_list[i].isdigit() or jason_str_list[i] in [' ', '\n', '-', ',', 'e', '.']:
            pass
        else:
            only_digits = False
        i -= 1
    res = ''.join(jason_str_list)
    return res


def get_apriltag_3dposition(gray_img, depth_img, kp):
    """ Get 3d points of apriltag from images and camera kp
    """
    result = apriltag_detector.detect(gray_img)
    if len(result) > 0:
        cxf, cyf = result[0].center[0], result[0].center[1]
        corners = result[0].corners
        cv2.drawMarker(gray_img, (round(cxf), round(cyf)), color=255)
        cv2.drawMarker(gray_img, (round(corners[0][0]), round(corners[0][1])), color=255)
        half_size = round(0.747 * distance(np.array([cxf, cyf]), np.array([corners[0][0], corners[0][1]])))
        if half_size > 40: half_size = 40
        if half_size < 5: half_size = 5
        cz_list = []
        cx, cy = round(cxf), round(cyf)
        for i in range(half_size):
            for j in range(half_size):
                z_ij = depth_img[cy + i, cx + j]
                if z_ij > 30.0: cz_list.append(z_ij)
        cz_list.sort()
        cz_len = len(cz_list)
        print("valid points for apriltag z:" + str(cz_len))

        czf = np.mean(cz_list[cz_len // 3:cz_len // 3 + cz_len // 3])
        point_raw = np.array([cxf * czf, cyf * czf, czf])
        kp_inv = np.linalg.inv(kp)
        point = np.dot(kp_inv, point_raw.T).T
        print("pos:" + str(point))
        return point


# get an 3d position measurments of marker or tag
def get_marker_pos(depth_camera):
    if USE_PASSIV_IR_MARKER:
        detector = K4aMarkerDetector(logging=False, calibra_src=depth_camera)
        marker_pos = None
        for _ in range(ir_marker_repeat_num):
            _, gray_img, depth_img = depth_camera.get_images(undistort=False)
            if marker_pos is None:
                marker_pos = np.array(detector.get_marker_from_img(gray_img, depth_img)[0])
            else:
                marker_pos += np.array(detector.get_marker_from_img(gray_img, depth_img)[0])
            cv2.imshow("gray_img", gray_img)
            cv2.waitKey(50)
        return marker_pos[:3] / ir_marker_repeat_num
    else:
        _, gray_img, depth_img = depth_camera.get_images()
        kp = depth_camera.get_camera_kp()
        marker_pos = get_apriltag_3dposition(gray_img, depth_img, kp)
        cv2.imshow("img", gray_img)
        cv2.waitKey(50)
        return marker_pos


def generate_cali_points_list():
    arm = xARMWrapper(init_to_home=False, teaching_mode=True)
    depth_camera = DepthCameraCls(logging=True)
    print("calibration.json does not exits, now entering soft teaching mode to generate calibra-points:")
    print("move the robot to a proper position and then \n \
        enter 'c' to set one cali-point  \n \
        enter 't' to set one test point  \n \
        enter 'h' to set as recommanded home position/waitting position for applications")
    print("at least 6 cali-points are needed, the more the better")
    print("at least 4 test points are needed, the more the better")

    def get_cali_point():
        """ Get the calibra point
        """
        code, pose = arm._arm.get_position(is_radian=False)
        if code != 0:
            print("geting pos ERROR")
        return pose[:3]

    def get_rec_point():
        """ Get the angles and pose from the robot arm 
        """
        angles = arm.get_servo_angle(is_radian=False)
        code, pose = arm._arm.get_position(is_radian=False)
        if code != 0:
            print("geting pos ERROR")
        return angles, pose

    cfg_dict = {}
    cali_points = []
    test_points = []
    while True:
        if USE_PASSIV_IR_MARKER:
            rgb_img, gray_img, depth_img = depth_camera.get_images(undistort=not USE_PASSIV_IR_MARKER)
            cv2.imshow("gray_img", gray_img)
            cv2.imshow("rgb_img", rgb_img)
        else:
            gray_img = depth_camera.get_one_frame(camera_id=1)
            cv2.imshow("gray_img", gray_img)
        c = cv2.waitKey(20)
        if c != -1:
            c = chr(c)
            if c == 'C' or c == 'c':
                cali_points.append(get_cali_point())
                print("cali_points" + str(cali_points))
            if c == 'T' or c == 't':
                test_points.append(get_cali_point())
                print("test_points" + str(test_points))
            if c == 'H' or c == 'h':
                angles, pose = get_rec_point()
                cfg_dict['home_position'] = {'angles': angles, 'pose': pose}
                print("home_position:" + str(cfg_dict['home_position']))
            if len(cali_points) >= 6 and len(test_points) >= 3:
                print("press 'E' and Enter, to end the procedure.")
                if c == 'E' or c == 'e':
                    cfg_dict['calibra_points'] = cali_points
                    cfg_dict['test_points'] = test_points
                    break

    print(cfg_dict)
    with open("./calibration.json", "w") as cfg_file:
        json.dump(cfg_dict, cfg_file)
    print("cali-points saved to ./calibration.json. Now you can re-run for calibration. Exiting")
    arm.close()
    depth_camera.close()


def get_points_in_camera_coord():
    depth_camera = DepthCameraCls(logging=True)
    arm = xARMWrapper(init_to_home=False)
    arm.set_slow_mode()

    with open("./calibration.json") as cfg_file:
        cfg_dict = json.load(cfg_file)
    arm_pos_list = cfg_dict['calibra_points']
    camera_pos_list = []

    pos = arm_pos_list[0]
    arm.set_pos(x=pos[0], y=pos[1], z=pos[2], yaw=gripper_init_angle, wait=True)
    if interactive_install_cali_board:
        arm.set_gripper(CALI_BOARD_GRIP_CLOSE+50, normalized=False)
        print("Put the calibrating board to the gripper, press Enter to continue")
        input()
        arm.set_gripper(CALI_BOARD_GRIP_CLOSE, normalized=False)
        print("If gripped, press Enter, otherwise press N")
        c = input()
        print(c)
        if len(c) != 0:
            arm.close()
            depth_camera.close()
            exit()

    actual_arm_pos_list = []
    for pos in arm_pos_list:
        print("set robot pos:" + str(pos))
        arm.set_pos(x=pos[0], y=pos[1], z=pos[2], yaw=gripper_init_angle, wait=True)
        time.sleep(2)
        marker_pos = get_marker_pos(depth_camera)
        if marker_pos is not None:
            _, pose = arm._arm.get_position(is_radian=False)
            print("get cam pos:" + str(marker_pos))
            if len(marker_pos) > 0 and not np.isnan(marker_pos[0]):
                camera_pos_list.append(list(marker_pos[:3]))
                actual_arm_pos_list.append(pose[:3])

    print("arm_pos_list:" + str(arm_pos_list))
    print("actual_arm_pos_list:" + str(actual_arm_pos_list))
    print("camera_pos_list:" + str(camera_pos_list))

    # save captured data
    cfg_dict["arm_pos_list"] = actual_arm_pos_list
    cfg_dict["camera_pos_list"] = camera_pos_list
    with open("./calibration.json", "w") as cfg_file:
        json.dump(cfg_dict, cfg_file)
    print("data capture done")
    arm.close()
    depth_camera.close()


def test_cali_result():
    depth_camera = DepthCameraCls(logging=True)
    detector = K4aMarkerDetector(logging=False, calibra_src=depth_camera)
    arm = xARMWrapper(init_to_home=False)
    arm.set_slow_mode()
    print("*" * 50)
    print("tesing:")
    with open("./calibration.json") as cfg_file:
        cfg_dict = json.load(cfg_file)
    rd_transform_cam2arm = np.array(cfg_dict["transform_cam2arm"])
    test_points_list = cfg_dict['test_points']
    error = 0
    valid_tests = 0
    for pos in test_points_list:
        print("setting robot pos:" + str(pos))
        arm.set_pos(x=pos[0], y=pos[1], z=pos[2], yaw=gripper_init_angle, wait=True)
        time.sleep(2)
        marker_pos = get_marker_pos(depth_camera, detector)
        if marker_pos is not None and len(marker_pos) > 0:
            cam_pos = marker_pos[:3]
            cam_pos_t = homogeneous_coord_t(cam_pos.reshape(1, 3))
            caculate_arm_pos = np.dot(rd_transform_cam2arm, cam_pos_t).T[0][:3]
            print("arm_pos:" + str(pos))
            print("caculate_arm_pos:" + str(caculate_arm_pos))
            dis = distance(pos, caculate_arm_pos)
            error += dis
            valid_tests += 1
        else:
            print("can not find marker!")
    avg_error = error / valid_tests
    print("average error in mili-meter:" + str(avg_error))
    print("testing done!")
    print("after holding the cali board, press Enter to continue")
    input()
    arm.set_gripper(800, normalized=False)
    print("press Enter to exit")
    input()
    arm.go_home()
    arm.close()
    depth_camera.close()
    print("exiting suc!")


def get_cali_matrix():
    with open("./calibration.json") as cfg_file:
        cali_cfg = json.load(cfg_file)
    arm_pos_list = np.array(cali_cfg["arm_pos_list"][:])
    camera_pos_list = np.array(cali_cfg["camera_pos_list"][:])

    camera_pos_transpose = homogeneous_coord_t(camera_pos_list)
    camera_pos_inv = np.linalg.pinv(camera_pos_transpose)
    arm_pos_transpose = homogeneous_coord_t(arm_pos_list)
    transform_cam2arm = np.dot(arm_pos_transpose, camera_pos_inv)
    print("transform: cam to arm:" + str(transform_cam2arm))

    # write the tranform matrix, you can get base coord from cam coord as:
    #     [arm_base_coord, 1].T = np.dot(transform_cam2arm, [cam_base_coord, 1].T)
    cali_cfg["transform_cam2arm"] = transform_cam2arm.tolist()
    with open("./calibration.json", "w") as cfg_file:
        cfg_str = json.dumps(cali_cfg, indent=2)
        cfg_str = remove_newline_around_digits(cfg_str)
        cfg_file.write(cfg_str)
        cfg_file.close()
    print("calibration done!, saved to './calibration.json'")

    for i in range(len(camera_pos_list)):
        cam_pos, arm_pos = camera_pos_list[i], arm_pos_list[i]
        cam_pos_t = homogeneous_coord_t(cam_pos.reshape(1, 3))
        caculate_arm_pos = np.dot(transform_cam2arm, cam_pos_t).T[0][:3]
        print("arm_pos:" + str(arm_pos))
        print("caculate_arm_pos:" + str(caculate_arm_pos))
        dis = distance(arm_pos, caculate_arm_pos)
        print("distance:" + str(dis))


if __name__ == "__main__":
    if not os.path.exists("./calibration.json"):
        # if calibration.json not exits, entering soft teaching mode to generate calibra-points list
        generate_cali_points_list()
        exit()
    if not use_existing_cali_data:
        get_points_in_camera_coord()
    get_cali_matrix()
    if run_cali_test:
        test_cali_result()
