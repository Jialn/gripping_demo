# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-11-11
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
Helper for 3D transformation

About "axes_para" parameter in this file:
    axes_para is an optional axis-specification. The format is (firstaxis, parity, repetition, frame).
    A fast look up table for axes_para:
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0), 'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0), 'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1), 'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1), 'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)
    First char 's/r' stands for static(extrinsic) or rotating(intrinsic) coordinate axes
        extrinsic: rotation is about the static fixed axis
        intrinsic: rotation is about the rotated new axis
    The following 3 characters 'x/y/z' give the order in which the rotations will be performed.
    The defalut is sxyz, i.e., static RollPitchYaw. 

"""
import numpy as np
import math


def to_rad(deg):
    """Return radian form degree.
    """
    return np.array(deg) * np.pi / 180.0


def to_deg(rad):
    """Return degree from radian.
    """
    return np.array(rad) * 180.0 / np.pi


def xyzrpy2mat(pose_xyzrpy):
    """Return homogenous rotation matrix from the robot arm [x,y,z,r,p,y] representation.
    """
    pos = pose_xyzrpy[:3]
    rpy = to_rad(pose_xyzrpy[3:6])
    R = euler2mat(rpy[0], rpy[1], rpy[2])
    R_with_pos = np.insert(R, 3, values=pos, axis=1)
    # rotation matrix with homogenous coordinates
    trans_matrix = np.insert(R_with_pos, 3, values=np.array([0, 0, 0, 1]), axis=0)
    # print(R)
    # print(trans_matrix)
    return trans_matrix


def euler2mat(ai, aj, ak, axes_para=(0, 0, 0, 0)):
    """
    Return rotation matrix from Euler angles and axis sequence.

    Args:
        ai : First rotation angle (according to `axes_para`).
        aj : Second rotation angle (according to `axes_para`).
        ak : Third rotation angle (according to `axes_para`).
        axes_para : optional Axis specification. The format is (firstaxis, parity, repetition, frame).
            The defalut is sxyz, i.e., static RollPitchYaw, for details refer to description of this file

    Return:
        numpy.array in shape (3,3), the rotation matrix, 3D non-homogenous coordinates
    """
    _NEXT_AXIS = [1, 2, 0, 1]
    i, parity, repetition, frame = axes_para
    j, k = _NEXT_AXIS[i + parity], _NEXT_AXIS[i - parity + 1]
    if frame: ai, ak = ak, ai
    if parity: ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def mat2euler(mat, axes_para=(0, 0, 0, 0)):
    """
    Give Euler angles from rotation matrix. Note that many Euler angle triplets can describe one matrix.

    Args:
        mat : array-like shape (3, 3). homogenous (4, 4) will be trunked to (3, 3)
        axes_para : optional Axis specification. The format is (firstaxis, parity, repetition, frame).

    Return:
        a list, rotation angle for specified axis sequence.
    """
    _EPS4 = np.finfo(float).eps * 4.0  # For testing whether a number is close to zero
    _NEXT_AXIS = [1, 2, 0, 1]
    i, parity, repetition, frame = axes_para
    j, k = _NEXT_AXIS[i + parity], _NEXT_AXIS[i - parity + 1]
    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS4:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS4:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return [ax, ay, az]
