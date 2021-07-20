# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-09-28
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
A Robot Arm Wrapper, suit for different kinds of Robot arms
"""
from xarm.wrapper import XArmAPI
import os
import time
import numpy as np

class BaseArmWrapper():
    """ Base Class for Arm
    """

    def __init__(self):
        pass

    def setup(self):
        pass

    def go_home(self):
        pass 

    def set_pos(self, x, y, z):
        pass

    def get_pos(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass

class DexArmWrapper(BaseArmWrapper):
    """ Base Class for Arm
    """

    def __init__(self,
                 logging=False):
        """
        Args:
            logging (bool): log or not.
        """
        self.init_offset = [0, 300 ,0]
        from .pydexarm import Dexarm
        self._arm = Dexarm("/dev/ttyACM0") # COM* in Windows, /dev/tty* in Ubuntu
        self._logging = logging
        self.setup()

    def setup(self):
        self._arm.go_home()

    def go_home(self):
        self._arm.go_home()

    def set_pos(self, x=None, y=None, z=None, wait=False):
        if x is not None: x = - x
        if y is not None: y = y + 300
        self._arm.move_to(x,y,z, wait=False, feedrate=2000)

    def get_pos(self):
        pos = list(self._arm.get_current_position())
        if pos[0] is not None:
            pos[0] = - pos[0]
            pos[1] = pos[1] - 300
        return pos[:3]

    def set_gripper(self, open_pos):
        if 0.3 < open_pos <= 0.5:
            self._arm.soft_gripper_pick()
        elif 0.7 <= open_pos < 0.9:
            self._arm.soft_gripper_place()
        else:
            self._arm.soft_gripper_nature()

    def reset(self, wait=True):
        self._arm.go_home()

    def close(self):
        self._arm.close()


class xARMWrapper(BaseArmWrapper):
    """ Class for xArm
    """
    ## Parameters
    # xARM5
    # _ip, _is_xARM6 = '192.168.1.212', False
    # xARM6
    _ip, _is_xARM6 = '192.168.1.235', True

    minimal_z = 170.0 if _is_xARM6 else 165.0
    # the condition of exit waitting for set position function, in mili-meter
    wait_precision = 3.0
    # wait timeout, in second, for set position function
    wait_timeout = 16.0
    # initial pos
    home_pose_in_angle = [0, -13.4, -16.3, 0.0, 29.7, 0.0, 0.0]

    # Motion Parameters
    # the extreme limits: {'speed': 1000, 'acc': 20000, 'angle_speed': 180, 'angle_acc': 3600}
    # original params in xARM example: {'speed': 100, 'acc': 2000, 'angle_speed': 20, 'angle_acc': 500}
    slow_params = {'speed': 120, 'acc': 600, 'angle_speed': 20, 'angle_acc': 200}
    normal_params = {'speed': 300, 'acc': 800, 'angle_speed': 60, 'angle_acc': 300}
    # fast_params sometimes could trigger current protection
    fast_params = {'speed': 400, 'acc': 2000, 'angle_speed': 90, 'angle_acc': 500}
    gripper_speed = 2000  # rpm, max 6000-7200

    def __init__(self,
                 init_to_home=True,
                 teaching_mode=False,
                 logging=True):
        """
        Args:
            logging (bool): log or not.
        """
        self._logging = logging
        self._position = None
        self._arm = self.setup(teaching_mode=teaching_mode)
        self._params = self.normal_params

        if init_to_home:
            self.set_servo(angle=self.home_pose_in_angle, is_radian=False, wait=True)
            self._arm.set_gripper_position(800, wait=True)

    def setup(self, teaching_mode):
        """
        Setup xARM, this will connect the robot arm, reset the mode, and move
        """
        self._params = self.slow_params
        print('=' * 50)
        print("_is_xARM6 flag:" + str(self._is_xARM6))
        print("conneting xARM, please double check the settings, and make sure no human or obstacles in working area")
        arm = XArmAPI(self._ip, do_not_open=False, is_radian=False)
        time.sleep(0.2)
        arm.motion_enable(enable=True)
        arm.set_mode(0) # 0 for position control mode. 1 for servo motion mode, 2 for joint teaching mode
        arm.set_state(state=0) # 0: enable motion, 3: pause state, 4: stop state

        if self._logging:
            print('xarm version:', arm.get_version())
            # get_state: 1 in motion, 2 ready but no motion(hibernate), 3 pause(could have instruction cache), 4 stop 
            print('state:', arm.get_state())
            print('err_warn_code:', arm.get_err_warn_code())
            print('position(°):', arm.get_position(is_radian=False))
            print('angles(°):', arm.get_servo_angle(is_radian=False))

        arm.set_gripper_mode(0) # location mode, xarm gripper only support location mode for now
        arm.set_gripper_enable(True)
        arm.set_gripper_speed(self.gripper_speed)  # unit:r/min
        arm.set_pause_time(0.0)        
        # clear warn code
        if arm.warn_code != 0:
            arm.clean_warn()
        if arm.error_code != 0:
            arm.clean_error()

        if teaching_mode:
            print("Entering teaching mode")
            arm.motion_enable(enable=True)
            arm.set_mode(2) # 0 for position control mode. 1 for servo motion mode, 2 for joint teaching mode
            arm.set_state(state=0) # 0: enable motion, 3: pause state, 4: stop state
        return arm

    def set_slow_mode(self):
        # the slow motion mode, using for scenraio like calibration
        self._params = self.slow_params

    def set_fast_mode(self):
        # the fast motion mode, using with caution
        self._params = self.fast_params

    def go_home(self): 
        self.set_servo(angle=self.home_pose_in_angle)

    def set_servo(self, angle, is_radian=False, radius=-1.0, wait=True, ignore_joints_conversion=False):
        """ set the servo angle, angle is a list, e.g., [0, -13.4, -16.3, 29.7, 0.0, 0.0, 0.0]
        For xARM5, the last 2 is always zero, to be compatiable with xARM7
        For xARM6, the last 1 is always zero, to be compatiable with xARM7
        The angle should be writen as xARM6, so move one out
        For xARM5, ignored the index 3 joint, and append an zero to be length of 7
        """
        if (not ignore_joints_conversion) and (not self._is_xARM6):
            # the difference of xARM5 and xARM6 is xARM5 lack of joint3
            angle = angle[:]
            _ = angle.pop(3)
            angle.append(0.0)
        self._arm.set_servo_angle(angle=angle, is_radian=is_radian, radius=radius, speed=self._params['angle_speed'],
            mvacc=self._params['angle_acc'], wait=wait)
        self._position = None # force local buffer to be updated

    def set_pos(self, x=None, y=None, z=None, yaw=0, radius=-1.0, use_built_in_ik=True, wait=True):
        """
        Set pos. could insert mid point to avoid stright line can not reach problem
        Args:
            x, y, z, yaw: pos and rotation
            use_built_in_ik: move line or use costomized inverse kinematics to move angles
                             use_built_in_ik=False is for xARM5 only, xARM6's costomized IK has problem
            radius: if use_built_in_ik, the move radius
                if radius is None or radius less than 0, will MoveLine, else MoveArcLine, more smoother
        """
        if self._position is None:
            self._position = self.get_pos()
        if x is None: x = self._position[0]
        if y is None: y = self._position[1]
        if z is None: z = self._position[2]
        if z < self.minimal_z: z = self.minimal_z

        start = np.array(self._position)
        target = np.array([x,y,z])
        start_complex = complex(start[0], start[1])
        target_complex = complex(target[0], target[1])
        start_angle = np.angle(start_complex)  # in radian, (-pi, pi]
        target_angle = np.angle(target_complex)
        absangle = abs(target_angle - start_angle)
        mid_angle = (start_angle + target_angle) / 2.0
        if self._logging: 
            print("start:" + str(start_complex))
            print("target:" + str(target_complex))
            print("start_angle:" + str(start_angle*180.0/np.pi))
            print("target_angle:" + str(target_angle*180.0/np.pi))
            print("absangle:" + str(absangle*180.0/np.pi))
            print("mid_angle:" + str(mid_angle*180.0/np.pi))
            
        if absangle > np.pi / 2:  # > 90 deg
            # insert a mid point
            mid_r = 350
            mid_x = mid_r * np.cos(mid_angle)
            mid_y = mid_r * np.sin(mid_angle)
            mid_z = (start[2] + target[2]) / 2.0
            print("add mid_point:" + str((mid_x, mid_y, mid_z)))
            self.set_pos_step(x=mid_x, y=mid_y, z=mid_z, yaw=yaw, radius=200.0, use_built_in_ik=use_built_in_ik, wait=False)
            self.set_pos_step(x=x, y=y, z=z, yaw=yaw, radius=radius, use_built_in_ik=use_built_in_ik, wait=wait)
        else:
            self.set_pos_step(x=x, y=y, z=z, yaw=yaw, radius=radius, use_built_in_ik=use_built_in_ik, wait=wait)
        self._position = [x,y,z]

    def set_pos_step(self, x, y, z, yaw=0, radius=-1.0, use_built_in_ik=True, wait=True):
        """ Set pos with a single step
        Args:
            x, y, z, yaw: pos and rotation
            use_built_in_ik: move line use built in IK or use costomized inverse kinematics to move angles
            radius: if use built in IK, the move radius
                if radius is None or radius less than 0, will MoveLine, else MoveArcLine, more smoother
        """
        if self._logging: print("set pos one step:"+str((x, y, z)))
        if use_built_in_ik:  # use built in inverse kinematics
            self._arm.set_position(x=x, y=y, z=z, roll=-180, pitch=0, yaw=yaw, radius=radius, speed=self._params['speed'],
                mvacc=self._params['acc'], wait=wait)
        else:
            pose = [x, y, z, -180, 0, yaw]  # [x, y, z, r, p, y]
            code, angles = self._arm._arm.get_inverse_kinematics(pose, input_is_radian=False, return_is_radian=False)
            if code == 0:
                print("IK angles:" + str(angles))
                self.set_servo(angle=angles, radius=radius, wait=False, ignore_joints_conversion=True)
            else:
                print("inverse_kinematics error!")
                return
            if wait:
                pos1 = np.array([x,y,z])
                for _ in range((int)(self.wait_timeout/0.025)):
                    pos2 = np.array(self.get_pos())
                    if np.linalg.norm(pos1 - pos2) < self.wait_precision:
                        break
                    if _ == 200:
                        print("waiting for arm to achieve the position")
                        print(np.linalg.norm(pos1 - pos2))
                    time.sleep(0.025)
                time.sleep(0.2)

    def set_pos_mode_j(self, x=None, y=None, z=None, yaw=0):
        if self._position is None:
            self._position = self.get_pos()
        if x is None: x = self._position[0]
        if y is None: y = self._position[1]
        if z is None: z = self._position[2]
        if z < self.minimal_z: z = self.minimal_z
        if self._logging: print("set pos in mode j:"+str((x, y, z)))
        mvpose = [x, y, z, -180, 0, 0]
        self._arm.set_servo_cartesian(mvpose, speed=50, mvacc=1000)
        self._position = [x,y,z]

    def get_pos(self):
        """Get position
        Return:
            list, [x,y,z]
        """
        _, pos = self._arm.get_position(is_radian=False)
        return pos[:3]

    def get_pos_cache(self):
        """Get position cache, may not update accroding to control mode

        Return:
            list, [x,y,z]
        """
        if self._position is None:
            self._position = self.get_pos()
        return self._position

    def get_pos_and_rotation(self):
        """Get both position and rotation

        Return:
            list, [x,y,z,r,p,y]. for xARM5, pitch and roll should be -180 and 0
        """
        _, pos = self._arm.get_position(is_radian=False)
        return pos[:6]
    
    def get_servo_angle(self, is_radian=False):
        _, angle = self._arm.get_servo_angle(is_radian=is_radian)
        if not self._is_xARM6:
            # the difference of xARM5 and xARM6 is xARM5 lack of joint3
            angle[5]=angle[4]
            angle[4]=angle[3]
            angle[3]=0.0
        return angle

    def set_gripper(self, open_pos, normalized=False, wait=True):
        # normalized: if normalized, open_pos is from 0 to 1.0; otherwise the raw input
        if self._arm.connected:  # and self._arm.get_cmdnum()[1] <= 2:
            if normalized: pos = open_pos * 850.0  # -10 to 850
            else: pos = open_pos
            code = self._arm.set_gripper_position(min(pos, 850), wait=False)
            if self._logging: print('set gripper pos, code={}'.format(code))
            if wait:
                # get_gripper_position is not real feedback from sensor, use this estimation
                time.sleep(1600.0/self.gripper_speed)

    def reset(self, wait=True):
        print('=' * 50)
        print("reseting xARM, please wait..")
        self._arm.reset(wait)

    def close(self):
        # self._arm.reset()
        self._arm.disconnect()


if __name__ == "__main__":
    arm = xARMWrapper(init_to_home=False)
    use_built_in_ik = True  # use_built_in_ik=False for XARM5 only, high level costomized IK is not woking on XRAM6
    wait = False
    radius = 100.0  # -1.0
    yaw = 0
    pos_list = [
        [250, 0, 300],
        [500, 0, 400],
        [400, 300, 400],
        [-200, 350, 300],
        [-200, -350, 700],
        [400, -300, 300],
        [400, 0, 300],
    ]
    for pos in pos_list:
        print("setting pos:" + str(pos))
        arm.set_pos(x=pos[0], y=pos[1], z=pos[2], yaw=yaw, radius=radius, use_built_in_ik=use_built_in_ik, wait=wait)
        
    pos = [250, 0, 300]
    arm.set_pos(x=pos[0], y=pos[1], z=pos[2], yaw=yaw, radius=radius, use_built_in_ik=use_built_in_ik, wait=True)