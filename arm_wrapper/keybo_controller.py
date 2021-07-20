# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-09
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
A keyborad and mouse control program to test robot arms.

"W,A,S,D" for left and right; "E,F" for up and down;
Use middle botton to switch controlling mode;
Scroll wheel on mouse to open or close the gripper.
"""

import os
import time
import numpy as np
from pynput.mouse import Button
from pynput.mouse import Listener as mListener
from pynput.keyboard import Key, KeyCode
from pynput.keyboard import Listener as kListener
from arm_wrapper import xARMWrapper

exiting_flag = 0
# serov_j mode or normal position mode
servo_j_mode_for_mouse_control = True
arm = None
arm_init_pos = None
# 0 disable; 1 use mouse; 2 for teaching mode (move robot by hand and get the position)
pos_control_flag = 0
# 0-10, init with 6, wide open a little bit
mouse_scroll_value = 6
mouse_init_xy_val = [0, 0]


def on_move(x, y):
    global mouse_init_xy_val, pos_control_flag, arm_init_pos
    if pos_control_flag == 1:
        pos = arm.get_pos_cache()
        posy = x - mouse_init_xy_val[0]
        posx = y - mouse_init_xy_val[1]
        # divid 10: map 1000 pixel to 100mm
        posx, posy = posx / 10.0 + 300, posy / 10.0
        if not servo_j_mode_for_mouse_control:
            arm.set_pos(x=posx, y=posy, z=pos[2], wait=False)
        else:
            arm.set_pos_mode_j(x=posx, y=posy, z=pos[2])


def on_click(x, y, button, pressed):
    global pos_control_flag, mouse_init_xy_val
    print('{0} {1} at {2}'.format(button, 'Pressed' if pressed else 'Released', (x, y)))
    if pressed and button == Button.middle:
        pos_control_flag += 1
        if pos_control_flag == 3: pos_control_flag = 0
        arm._arm.motion_enable(enable=True)
        if pos_control_flag == 1 and servo_j_mode_for_mouse_control:
            arm._arm.set_mode(1)
        elif pos_control_flag == 2:
            arm._arm.set_mode(2)
        else:
            arm._arm.set_mode(0)
        arm._arm.set_state(state=0)
        print('pos_control_flag: {0}'.format(pos_control_flag))
        mouse_init_xy_val = [x, y]
    if pressed and button == Button.right:
        pass


def on_scroll(x, y, dx, dy):
    global arm, mouse_scroll_value
    mouse_scroll_value += dy * 2
    if mouse_scroll_value < 0: mouse_scroll_value = 0
    if mouse_scroll_value > 10: mouse_scroll_value = 10
    if arm:
        arm.set_gripper(mouse_scroll_value / 10.0, normalized=True)
    print('Scrolled {0}, scroll value {1}'.format((dx, dy), mouse_scroll_value))


def on_press(key):
    print('{0} pressed'.format(key))


def on_release(key):
    global exiting_flag
    move_step_keybo = 20
    print('{0} release'.format(key))
    if isinstance(key, KeyCode):
        if key.char == 'q':
            exiting_flag = 2
        else:
            pos = arm.get_pos()
            if pos[0] is not None:
                if key.char == 'e':
                    arm.set_pos(z=pos[2] + move_step_keybo)
                if key.char == 'f':
                    arm.set_pos(z=pos[2] - move_step_keybo)
                if key.char == 'a':
                    arm.set_pos(x=pos[0] - move_step_keybo)
                if key.char == 'd':
                    arm.set_pos(x=pos[0] + move_step_keybo)
                if key.char == 'w':
                    arm.set_pos(y=pos[1] - move_step_keybo)
                if key.char == 's':
                    arm.set_pos(y=pos[1] + move_step_keybo)
                angles = arm.get_servo_angle(is_radian=False)
                print("servo_angle:" + str(angles))
    # if key is a special key
    if isinstance(key, Key):
        if key == Key.esc:
            exiting_flag = 1


listener_m = mListener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
listener_m.start()
listener_k = kListener(on_press=on_press, on_release=on_release)
listener_k.start()

arm = xARMWrapper()  # DexArmWrapper() xArm6Wrapper()

while True:
    time.sleep(0.1)
    if pos_control_flag == 2:
        angles = arm.get_servo_angle()
        pos = arm.get_pos_and_rotation()
        print("servo_angle:" + str(angles) + ", position:" + str(pos))
    if exiting_flag == 1 or exiting_flag == 2:  # Esc or Q key to stop
        break

arm.close()
listener_m.stop()
listener_k.stop()
