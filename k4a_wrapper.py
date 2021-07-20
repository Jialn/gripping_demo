# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-12
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
A Wrapper of Kinect for Azure using pyk4a.
This file also provid a class to track a passive retro-reflective IR Marker's position
    - The marker should be made of retro-reflecting material, around 1*1 cm^2 size
    - IR power of kinect should be dimmed, e.g., put a semi-transparent adhesive tape on the IR
        projector. Otherwise the marker zone will be overflow and depth info will be lost.
"""

import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A

# the image size of depth/ir assuming depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED, change it otherwise
FRAME_WIDTH = 640
FRAME_HEIGHT = 576

# distortion coefficients [k1, k2, p1, p2, k3, k4, k5, k6]
rgb_dist = np.array([0.8208, -2.88168, 0.000468124, 0.000267001, 1.59698, 0.702313, -2.72892, 1.5369])
depth_dist = np.array([4.43991, 3.21018, -0.0000901, 0.0000315402, 0.175744, 4.76839, 4.68595, 0.902381])
rgb_mtx = np.array([[599.372, 0, 635.526], [0, 598.893, 366.672], [0, 0, 1]])
depth_mtx = np.array([[504.168, 0, 316.97], [0, 504.316, 325.306], [0, 0, 1]])

# parameters for contours filter
marker_zone_threshold = 24
# in num of pixel, 25 at 1.2m, 400 about 25cm
contours_area_range = [6, 1200]
# z range, a bit larger than 20cm - 1.5 m
z_range = [150, 1800]
# z * sqrt(area), in mm*pixel_num depends on size of marker
volume_factor_range = [1000, 12000]


def undistort(img, mtx, dist):
    """ 
    Used to undistort a image. Camera matrix and distortion coefficients is needed.
    For detail refer to https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#using-cv2-undistort

    Args:
        img: the image
        mtx: camera matrix
        dist: distortion coefficients
    Returns:
        new camera matrix and un-distorted image
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # un-distort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return newcameramtx, dst


class K4aWrapper():
    """ Class for get image of Kinect
    """

    def __init__(self, logging=False):
        """
        Init the kinect.

        Args:
            logging (bool): log or not.
        """
        self._logging = logging
        self._k4a = self.init_k4a()
        self._depth_point_cloud = None
        self._rectified_rgb_img = None
        self._depth_rectified_to_rgb = None
        self._camera_kp = None

    def init_k4a(self):
        """ Init Kinect for Azure
        """
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            ))
        k4a.start()
        return k4a

    def get_images(self, undistort=True, trunk_bits=4):
        """ 
        Get undistored the images: rgb, gray/IR and depth image

        Args:
            undistort: undistort or not. 
                If undistort, will return undistorted and rectified images
                If not, will return the original images
            trunk_bits: ir image from kinect is 16bit, we will trunk the high 4 bits (0 by most of the time),
        and then select the higher 8 bits

        Return:
            rgb image, ir image and depth_img, 8 bit or 16 bit numy.array
        """
        capture = self._k4a.get_capture()
        # Reshape to image
        depth_img_full = np.frombuffer(capture.depth, dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()
        ir_img_full = np.frombuffer(capture.ir, dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()
        rgb_img = capture.color[:, :, :3]
        # update the cache
        self._depth_point_cloud = capture.depth_point_cloud
        self._rectified_rgb_img = capture.transformed_color[:, :, :3]
        self._depth_rectified_to_rgb = np.frombuffer(
            capture.transformed_depth, dtype=np.uint16).reshape(720, 1280).copy()
        # Type convert and bit trunk
        depth_img = depth_img_full.astype(np.float32)  # in milimeter
        ir_img = np.clip(ir_img_full, 0, 2**(8 + trunk_bits) - 1)
        ir_img = np.array(np.right_shift(ir_img, trunk_bits), np.uint8)
        if undistort:
            _, undistort_rgb_img = self.undistort_depth(self._rectified_rgb_img)
            _, undistort_ir = self.undistort_depth(ir_img)
            newcameramtx, undistort_depth_img = self.undistort_depth(depth_img)
            self._camera_kp = newcameramtx
            return undistort_rgb_img, undistort_ir, undistort_depth_img
        else:
            self._camera_kp = depth_mtx
            return rgb_img, ir_img, depth_img

    def get_pointcloud(self):
        """ Get point cloud in [[x1,y1,z1], [x2,y2,z2], ...] format.
        Need to run after "get_images()" to update the cache.
        """
        points = self._depth_point_cloud.reshape((-1, 3))
        return points.astype(np.float32) / 1000.0

    def get_rectified_rgb_image(self, transformed=True):
        """ Get the RGB image rectified to depth. Call "get_images()" to update the cache.
        """
        return self._rectified_rgb_img

    def get_rectified_depth_image(self, transformed=True):
        """ Get depth image rectified RGB. Call "get_images()" to update the cache.
        """
        return self._depth_rectified_to_rgb

    def get_transformed_xyz(self, img_x, img_y):
        """
        Do the transformation

        Args:
            imgx, imgy (int):  (x, y) pixel in the image
        Return:
            x, y, z in camera coordinate in milimeter
        """
        return list(self._depth_point_cloud[img_y, img_x])

    def undistort_rgb(self, img):
        """ get the undistorted 720p rgb image. used for rgb or rectified depth
        """
        newcameramtx, dst = undistort(img, rgb_mtx, rgb_dist)
        return newcameramtx, dst

    def undistort_depth(self, img):
        """ get the undistorted 576p depth image. used for depth or rectified rgb
        """
        newcameramtx, dst = undistort(img, depth_mtx, depth_dist)
        return newcameramtx, dst

    def get_camera_kp(self):
        """ Get Camera Kp. Call "get_images()" to update the cache.
        """
        return self._camera_kp

    def close(self):
        self._k4a.stop()


class K4aMarkerDetector():
    """ Class for tracking a passive IR Marker position using Azure Kinect
    The marker should be made of retro-reflecting material, ~ 1*1 cm^2 size
    IR power of kinect should be dimmed
    """

    def __init__(self, logging=True, calibra_src=None):
        """
        Args:
            logging (bool): log or not.
        """
        self._logging = logging
        self._calibra_src = calibra_src

    def get_marker_from_img(self, gray_img, depth_img):
        """
        Args:
            gray_img, depth_image: the image
        Return:
            (x, y, z, size) x, y, z in camera coordinate in milimeter, and size of the marker
        """
        thres, gray_bin = cv2.threshold(gray_img, marker_zone_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_bin = cv2.morphologyEx(gray_bin, cv2.MORPH_CLOSE, kernel, iterations=1)  # erode then dilate
        marker_results = []
        # find contours
        _, contours, h = cv2.findContours(gray_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # if contours exits, run the filter
        if (len(contours) >= 1):
            marker_contours = []
            for marker_idx in range(len(contours)):
                area_i = cv2.contourArea(contours[marker_idx])
                if contours_area_range[0] < area_i < contours_area_range[1]:
                    cntou = contours[marker_idx]
                    moments = cv2.moments(cntou)  # to extract centroid
                    if moments['m00'] != 0:
                        cxf = int(moments['m10'] / moments['m00'])
                        cyf = int(moments['m01'] / moments['m00'])
                        cx, cy = int(cxf), int(cyf)
                        result_cam_coord = self._calibra_src.get_transformed_xyz(cx, cy)
                        result_cam_coord = list(map(float, result_cam_coord))
                        volume_factor = np.sqrt(area_i) * result_cam_coord[2]
                        if volume_factor_range[0] < volume_factor < volume_factor_range[1]:
                            if self._logging:
                                print("idx:" + str(marker_idx) + " xyz:" + str(result_cam_coord) + " vfactor:" +
                                      str(volume_factor))
                            result_cam_coord.append(volume_factor)
                            marker_results.append(result_cam_coord)
                            marker_contours.append(contours[marker_idx])
            if len(marker_results) > 0:
                marker_results.sort(key=lambda x: x[-1], reverse=True)  # sort by area in reversed order
                if self._logging:
                    cv2.drawContours(gray_bin, marker_contours, 0, 192, -1)
                    cv2.imshow("gray_bin", gray_bin)
        return marker_results


# Test for k4a wrapper and marker detection
if __name__ == "__main__":
    import sys
    from pose_detector import scale_crop_helper

    if len(sys.argv) < 2 or sys.argv[1] not in ['show', 'save', 'marker', 'showpc']:
        print("run with arg: \n \
        show        - show rgb, ir and depth image \n \
        save ./temp - capture and save one frame rgb, ir image and pointcloud to './temp', need pcl \n \
        marker      - test passive ir marker detector \n \
        showpc      - show saved point cloud, need pcl ")
        exit()

    if sys.argv[1] == 'save' or sys.argv[1] == 'showpc':
        import pcl
        import pcl.pcl_visualization

    if sys.argv[1] == 'showpc':
        cloud = pcl.load_XYZRGBA('./temp/cloud.pcd')
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud, b'cloud')
        v = True
        while v:
            v = not (visual.WasStopped())
        exit()

    depth_camera = K4aWrapper(logging=True)
    detector = K4aMarkerDetector(logging=True, calibra_src=depth_camera)

    while (True):
        rgb_img, gray_img, depth_img = depth_camera.get_images(undistort=False)
        rectified_rgb_img = depth_camera.get_rectified_rgb_image()
        points = depth_camera.get_pointcloud()

        cv2.imshow("rgb_img", rgb_img)

        if sys.argv[1] == 'show':
            cv2.imshow("ir_img", gray_img)
            cv2.imshow("depth_img", depth_img)

        if sys.argv[1] == 'save':
            if len(sys.argv) >= 3: save_path = sys.argv[2]
            else: save_path = './temp/'
            cv2.imwrite(save_path + 'rgb_img.png', rgb_img)
            cv2.imwrite(save_path + 'rectified_rgb_img.png', rectified_rgb_img)
            cv2.imwrite(save_path + 'gray_img.png', gray_img)
            depth_img = depth_img * 10.0
            depth_img = depth_img.astype(np.uint16)
            cv2.imwrite(save_path + 'depth_img.png', depth_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # for ycb dataset
            rectify_depth_to_rgb = True
            if not rectify_depth_to_rgb:
                crop_color = rectified_rgb_img[96 // 2:480 + 96 // 2, :]
                crop_depth = depth_img[96 // 2:480 + 96 // 2, :]
            else:
                _, undistort_rgb_img = depth_camera.undistort_rgb(rgb_img)
                _, undistort_rectified_depth = depth_camera.undistort_rgb(depth_camera._depth_rectified_to_rgb)
                scale = 1.0
                h, w = undistort_rgb_img.shape[:2]
                size = (800, 600)
                margin_h_f, margin_w_f = (h * scale - size[1]) / 2, (w * scale - size[0]) / 2
                margin_h, margin_w = round(margin_h_f), round(margin_w_f)
                crop_color = undistort_rgb_img[margin_h:size[1] + margin_h, margin_w:margin_w + size[0]]
                crop_depth = depth_camera._depth_rectified_to_rgb[margin_h:size[1] + margin_h, margin_w:margin_w + size[0]]
                crop_depth = crop_depth * 10.0
                crop_depth = crop_depth.astype(np.uint16)
            cv2.imwrite(save_path + '000001-color.png', crop_color)
            cv2.imwrite(save_path + '000001-depth.png', crop_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # for ycb

            rgb = rectified_rgb_img.reshape((-1, 3)).astype(np.uint32)
            red, green, blue = rgb[:, 2], rgb[:, 1], rgb[:, 0]
            rgb_combine = np.left_shift(red, 16) + np.left_shift(green, 8) + np.left_shift(blue, 0)
            rgb_combine = rgb_combine.reshape(-1, 1).astype(np.float32)
            color_points = np.concatenate((points, rgb_combine), axis=1)
            color_cloud = pcl.PointCloud_PointXYZRGBA(color_points)
            pcl.save_XYZRGBA(cloud=color_cloud, path=save_path + '000001-cloud.pcd', format=None, binary=False)
            pcl.save_XYZRGBA(cloud=color_cloud, path=save_path + '000001-cloud.ply', format=None, binary=False)
            depth_camera.close()
            print("images saved to " + save_path)
            break

        if sys.argv[1] == 'marker':
            marker_pos = detector.get_marker_from_img(gray_img, depth_img)
            if len(marker_pos) > 0:
                print(marker_pos)

        key = cv2.waitKey(1)
        if key == 27:  # Esc key to stop
            depth_camera.close()
            break
