# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-11-03
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 

A helper to recording task for gripping demo.

To Generate tasks:
Run "python gripping_demo.py rec" to enter task recording mode.

This will generate a task group contains multipe sub-tasks:

task_group = [sub_task1, sub_task2, ...]

A sub_task is described as a dict:

sub_task = {
    "object_type": "coffe_bottle",          # type of object to be operated with
    "source_pose": None or [x,y,z,r,p,y]    # source pose of object, if None, the pose is captured by RGBD-sensor
    "destination": {                        # destination where the object will be released, a dict with key angles and pose 
        "angles": [angle1, angle2, ...]     # angle of the servos
        "pose": [x,y,z,r,p,y]               # the cartisian pose, in mili-meter or degree, [x,y,z,r,p,y]
        "wait": 0.01                        # waitting time for this key frame, float, in seconds
    }
    'waypoints'  [wp1, wp2, ...]            # waypoints, a list of waypoint, which is a dict with key angles, pose and wait_time
                                            # the structure of waypoint is the same as to destination
}

The configuration file will be saved to ./task.json, can be edited manually.
"""

import json
from arm_wrapper.arm_wrapper import xARMWrapper


def task_recording():
    print("Entering recording mode...")
    print("=" * 50)
    print("A task configuration includes object_type, destination and waypoints \n \
    object_type is fixed now, only need to specify destination and waypoints \n \
    move the robot to your target position, then \n \
    press 'S' and Enter, to set as source pose of the object. If not set, the object pose will be captured by RGBD-sensor \n \
    press 'W' and Enter, to set as waypoint, waypoint is optional, multiple waypoint will be excuted in the same sequence as recording \n \
    press 'D' and Enter, to set as destination where the object will be released \n \
    setting destination by 'D' will complete one task and start a new one.\n \n \
    press 'E' and Enter, to end the recording mode.\n \n \
    e.g. generate one task group with 3 sub-tasks: WWD WWD SWWD E \
    ")

    def get_rec_point():
        """ Get the angles and pose from the robot arm at current recording point
        """
        angles = arm.get_servo_angle(is_radian=False)
        code, pose = arm._arm.get_position(is_radian=False)
        if code != 0:
            print("geting pos ERROR")
        return angles, pose

    def remove_newline_for_keys(json_str, keys):
        """ Remove \n for the specified keys to make the config file more readable, as we may 
        manually edit the config file
        """
        jason_str_list = list(json_str)
        keyword_indexs = []
        for key in keys:
            key = "\"" + key + "\": ["
            i = 2
            while (i < len(jason_str_list) - 2):
                index = json_str.find(key, i)
                if index != -1:
                    keyword_indexs.append(index)
                    i = index + 1
                else:
                    break
        keyword_indexs.sort(reverse=True)
        for index in keyword_indexs:
            i = index
            while (jason_str_list[i] != ']'):
                if jason_str_list[i] == '\n' or jason_str_list[i] == ' ':
                    jason_str_list.pop(i)
                else:
                    i += 1
        res = ''.join(jason_str_list)
        return res

    arm = xARMWrapper(init_to_home=False, teaching_mode=True)

    print("=" * 50)
    print("Enter 'S'/'W'/'D' to record a new task or enter 'E' to end the recording.")
    print("If needed, enter 'waitpos' to override the home position from calibrating process.")

    description = "The configuration of task group, editable. For detailed description please refer to task_recorder.py"

    # Get the default waiting pos from calibra
    with open("./calibration.json") as cali_cfg_file:
        cali_dict = json.load(cali_cfg_file)
    default_wait_pos = cali_dict['home_position']
    task_group = {'description': description, 'waiting_pos': default_wait_pos, 'tasks': []}
    task_cfg = None
    while True:
        c = input()
        if c == 'E' or c == 'e':
            break
        if c == 'waitpos':
            angles, pose = get_rec_point()
            task_group['waiting_pos'] = {'angles': angles, 'pose': pose}
            print("wait_pos:" + str(task_group['waiting_pos']))

        if task_cfg is None:
            task_cfg = {}
            task_cfg['object_type'] = 'coffe_bottle'  # reserved for now
            task_cfg['source_pose'] = None
            task_cfg['destination'] = {}  # a dict with key angles and pose [x,y,z,r,p,y]
            task_cfg['waypoints'] = []  # a list of dict with key angles and pose

        if c == 'S' or c == 's':
            angles, pose = get_rec_point()
            task_cfg['source_pose'] = pose
            print("source_pose:" + str(task_cfg['source_pose']))
        if c == 'W' or c == 'w':
            angles, pose = get_rec_point()
            wp_temp = {}
            wp_temp['angles'] = angles
            wp_temp['pose'] = pose
            wp_temp['wait'] = 0.0
            task_cfg['waypoints'].append(wp_temp)
            print("waypoints:" + str(task_cfg['waypoints']))
        if c == 'D' or c == 'd':
            angles, pose = get_rec_point()
            task_cfg['destination']['angles'] = angles  # angles of the servo
            task_cfg['destination']['pose'] = pose  # the cartisian pose, in mili-meter or degree
            task_cfg['destination']['wait'] = 0.0  # waitting time for this key frame, float, in s
            print("destination:" + str(task_cfg['destination']))
            task_group['tasks'].append(task_cfg)
            print("=" * 50)
            print("A task has been recorded, enter 'S'/'W'/'D' to record a new task or enter 'E' to end the recording.")
            task_cfg = None

    tg_str = json.dumps(task_group, indent=2)
    tg_str = remove_newline_for_keys(tg_str, ["angles", "pose", "source_pose"])
    with open("./task.json", "w") as cfg_file:
        print("writting task configuration:")
        print(tg_str)
        cfg_file.write(tg_str)
        cfg_file.close()
    print("task configuration saved to ./task.json, done!")
