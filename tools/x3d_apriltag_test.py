# Copyright (c) 2020. All Rights Reserved.

import os
import time
import cv2
import numpy as np
import pickle
lib_path = os.path.abspath(os.path.join('../x3d_camera'))
sys.path.append(lib_path)
from x3d_camera import X3DCamera
from calibra_cam2world import get_apriltag_3dposition, Cam2World
import apriltag

# python x3d_apriltag_test.py
detector = apriltag.Detector()
"""
options = apriltag.DetectorOptions(families='tag36h11',
                                 border=1,
                                 nthreads=4,
                                 quad_decimate=1.0,
                                 quad_blur=0.0,
                                 refine_edges=True,
                                 refine_decode=False,
                                 refine_pose=False,
                                 debug=False,
                                 quad_contours=True)
"""

def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def find_apriltag(gray_img):
    result = detector.detect(gray_img)
    print(result)
    if len(result) > 0:
        cx, cy = result[0].center[0], result[0].center[1]
        corners = result[0].corners
        cv2.drawMarker(gray_img, (round(cx), round(cy)), color=255)
        cv2.drawMarker(gray_img, (round(corners[0][0]), round(corners[0][1])), color=255)
        return [cx, cy]


# 3D test
depth_camera = X3DCamera(camera_ids=[1, 2], hw_trigger=True, scale=0.5, logging=True)
_, gray_img, depth_img = depth_camera.get_images()
kp = depth_camera.get_camera_kp()
tag_pos_cam = get_apriltag_3dposition(gray_img, depth_img, kp)
cv2.imshow("gray_img_pts", gray_img)

calibra = Cam2World()
tag_pos = calibra.cam_2_world(tag_pos_cam)
print("camera coord position:" + str(tag_pos_cam) + "\tworld coord position:" + str(tag_pos))
key = cv2.waitKey(1000)
depth_camera.close()
cv2.destroyAllWindows()

exit()

# 2D test
camera_id = 1
depth_camera = X3DCamera(camera_ids=[camera_id], hw_trigger=False, scale=0.5, logging=True)
while True:
    gray_img = depth_camera.get_one_frame(camera_id=camera_id)
    find_apriltag(gray_img)
    cv2.imshow("gray_img_pts", gray_img)
    key = cv2.waitKey(50)
    if key == 27: break

depth_camera.close()
cv2.destroyAllWindows()

exit()
