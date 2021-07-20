# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-18
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
Do the planning for gripping task given the pose of the object.
"""

import math
import numpy as np
import transform_helper as trans_helper
import time

global_object_pos_offset = [5, -5, -5] # in mili-meter

# Define the gripping property of each objects
#    'grip_pose': the desired griping pose in object coordinate, can be None if grip_axis is provided
#    'grip_axis': the orthogonal axis where the gripper should be. e.g., [0, 0, 1] means grip the z-axis of the bottle
#    'extra_distance': extra distance should be kept before gripping
#    'gripper_open': distance when open gripper
#    'gripper_close': distance when open gripper
gripping_properties = {
    'coffe_bottle': {
        'grip_pose': None,
        'grip_axis': [0, 0, 1],
        'pos_offset': [0, 0, 5],
        'extra_distance': 100,
        'gripper_open': 850,
        'gripper_close': 525
    },
    'large_marker': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, 5],
        'extra_distance': 100,
        'gripper_open': 650,
        'gripper_close': 135
    },
    'banana': {
        'grip_pose': None,
        'grip_axis': [-0.1, 0.95, 0],
        'pos_offset': [0, 0, 3],
        'extra_distance': 120,
        'gripper_open': 850,
        'gripper_close': 300
    },
    'cup': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [25, 25, 20],
        'extra_distance': 120,
        'gripper_open': 351,
        'gripper_close': 30
    },
    'bottle': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, 5],
        'extra_distance': 120,
        'gripper_open': 551,
        'gripper_close': 420
    },
    'mouse': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, 2],
        'extra_distance': 120,
        'gripper_open': 851,
        'gripper_close': 420
    },
    'orange': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, 2],
        'extra_distance': 120,
        'gripper_open': 851,
        'gripper_close': 470
    },
    'carrot': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, 2],
        'extra_distance': 120,
        'gripper_open': 700,
        'gripper_close': 280
    },
    'apple': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, -10],
        'extra_distance': 120,
        'gripper_open': 851,
        'gripper_close': 720
    }
}
"""
    # not grabing well
    'scissors': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, 2],
        'extra_distance': 120,
        'gripper_open': 851,
        'gripper_close': 320
    },
    'cell_phone': {
        'grip_pose': None,
        'grip_axis': [0, 1, 0],
        'pos_offset': [0, 0, 10],
        'extra_distance': 120,
        'gripper_open': 851,
        'gripper_close': 851
    }
"""


class Planner():
    """
    Class for planning
    Usage:  1. init the planner
            2. set the destination by planner.set_destination()
            3. then do the planning by planner.plan()
    """

    def __init__(self, arm=None, logging=False):
        """
        Init
        """
        self._arm = arm
        self._logging = logging
        self._task_cfg = None
        self._waiting_position_in_angle = None
        self._target_position = None
        self._waypoints = None
        self._current_obj_type = None

    def set_task(self, task_cfg):
        """Set the task description
        """
        self._task_cfg = task_cfg
        self._target_position = task_cfg['destination']['pose'][:3]
        self._waypoints = task_cfg['waypoints']

    def set_waiting_pos(self, waiting_position):
        # use key "angles", to avoid XARM inverse kinematic error
        self._waiting_position_in_angle = waiting_position['angles']

    def go_waiting_pos(self, wait=True):
        self._arm.set_servo(angle=self._waiting_position_in_angle, wait=wait)

    def go_obj_pos(self, pos, yaw, wait=True):
        self._arm.set_pos(x=pos[0], y=pos[1], z=pos[2], yaw=yaw, radius=-1.0, wait=True)

    def go_above_obj_pos(self, pos, yaw, wait=True):
        extra_distance = gripping_properties[self._current_obj_type]['extra_distance']
        self._arm.set_pos(x=pos[0], y=pos[1], z=pos[2] + extra_distance, yaw=yaw, radius=30, wait=False)

    def go_waypoints(self, wait=True):
        if len(self._waypoints) == 0:
            return
        for wp in self._waypoints[:-1]:
            self._arm.set_servo(angle=wp['angles'], wait=False)
        wp = self._waypoints[-1]
        self._arm.set_servo(angle=wp['angles'], wait=wait)

    def go_target_pos(self, wait=True):
        if self._target_position is not None:
            pos = self._target_position
            self._arm.set_pos(x=pos[0], y=pos[1], z=pos[2], yaw=0, radius=45.0, wait=wait)

    def go_above_target_pos(self, wait=True):
        if self._target_position is not None:
            hover_pos = self._target_position[:]
            hover_pos[2] += gripping_properties[self._current_obj_type]['extra_distance']
            self._arm.set_pos(x=hover_pos[0], y=hover_pos[1], z=hover_pos[2], yaw=0, radius=45.0, wait=wait)

    def open_gripper(self):
        self._arm.set_gripper(gripping_properties[self._current_obj_type]['gripper_open'])

    def close_gripper(self):
        self._arm.set_gripper(gripping_properties[self._current_obj_type]['gripper_close'])

    def plan(self, object_pose=None, grip_pose_transfrom=True):
        """
        Do the planning and excute motion command

        Args:
            object_pose(numpy.array): 6DoF pose of object, 4x4 rotation matrix with homogeneous coordinate
            grip_pose_transfrom(bool): do the transfromation to get grip_pose or not.
                if True, grip_pose will be caculated from object_pose and the gripping parameters of object_type
                if False, object_pose will be used as grip_pose directly, used in case like the pose is recorded by robot hand rather than detected from camera.
        """

        def get_grip_angle(x, y):
            angle = np.arctan(y / x)
            # translate from radin to degree
            angle = angle * 360 / 2 / np.pi
            return angle

        self._current_obj_type = self._task_cfg['object_type']

        if grip_pose_transfrom:
            position = np.dot(object_pose, np.array([0, 0, 0, 1]).T).T[:3]
            # get the z axis of object in world coordinate
            grip_axis = gripping_properties[self._current_obj_type]['grip_axis'][:]
            grip_axis.append(1)
            grip_axis = np.array(grip_axis)
            grip_axis_rot_vec = np.dot(object_pose, np.array(grip_axis.T)).T[:3] - position
            pos_offset = gripping_properties[self._current_obj_type]['pos_offset']
            position += np.array(pos_offset)
            print("object position:" + str(position) + ", object rotation vector:" + str(grip_axis_rot_vec))
            yaw = get_grip_angle(grip_axis_rot_vec[0], grip_axis_rot_vec[1])
        else:
            position = np.dot(object_pose, np.array([0, 0, 0, 1]).T).T[:3]
            rpy = trans_helper.mat2euler(object_pose)
            rpy = trans_helper.to_deg(rpy)
            yaw = rpy[2]

        position = position + np.array(global_object_pos_offset)
        self.go_waiting_pos()

        self.open_gripper()
        self.go_above_obj_pos(pos=position, yaw=yaw, wait=False)
        self.go_obj_pos(pos=position, yaw=yaw, wait=True)
        self.close_gripper()
        self.go_above_obj_pos(pos=position, yaw=yaw, wait=False)

        self.go_waypoints(wait=True)
        self.go_above_target_pos(wait=True)
        self.go_target_pos(wait=True)
        self.open_gripper()
        self.go_above_target_pos(wait=False)
        self.go_waiting_pos(wait=True)

        print("The task has been Done!")


# Test for Planner
if __name__ == "__main__":
    pass
