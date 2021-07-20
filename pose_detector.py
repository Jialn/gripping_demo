# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-18
# Autor:
"""
Description: 
Pose detector wrappers. Take input from depth camera and give 6DoF pose of object.
"""

import os
import sys
import time
import cv2
import numpy as np
from abc import ABC, abstractmethod
from tools.pca import pca_with_svd

def scale_crop_helper(image, camera_kp, scale, size=None, interpolation=cv2.INTER_LINEAR):
    """
    Scale the image and crop to a new size, and return the new camera Kp
    
    Args:
        image: the image 
        camera_kp: camera intrinsic matrix
        scale: the scale of the resized new image
        size: cropped size, (w, h), e.g., (640, 480). By default it is the same as original image

    Return:
        new camera matrix and cropped_img
    """
    h, w = image.shape[:2]
    if size is None: size = (w, h)
    margin_h_f, margin_w_f = (h * scale - size[1]) / 2, (w * scale - size[0]) / 2
    margin_h, margin_w = round(margin_h_f), round(margin_w_f)

    new_camera_kp = np.zeros_like(camera_kp)
    new_camera_kp[0][0] = scale * camera_kp[0][0]
    new_camera_kp[1][1] = scale * camera_kp[1][1]
    new_camera_kp[0][2] = scale * camera_kp[0][2] - margin_w_f
    new_camera_kp[1][2] = scale * camera_kp[1][2] - margin_h_f
    new_camera_kp[2][2] = 1.0
    scaled_img = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=interpolation)
    cropped_img = scaled_img[margin_h:margin_h + size[1], margin_w:margin_w + size[0]]

    return new_camera_kp, cropped_img

def homogeneous_coord_t(pos_3d):
    """
    Add ones to be homogeneous coordinate and then transpose the matrix
    """
    ones = np.ones(len(pos_3d))
    pos_homogeneous = np.insert(pos_3d, 3, values=ones, axis=1)
    return pos_homogeneous.T


class BasePoseDetector(ABC):
    """ Aabstract Class to different kind of Pose Detectors
    """

    def __init__(self, logging=False):
        """
        Init
        """
        self._logging = logging

    @abstractmethod
    def get_obj_pose(self, rgb=None, gray=None, depth=None):
        """
        Get objects and their poses from images
        Abstractmethod of getting object pose from images or point clouds. The method have to be implemented
        in sub calsses

        Input:
            rgb(numpy.array, 8bit): rgb image, could be None
            gray(numpy.array, 8bit): gray image, could be None
            depth(numpy.array, 16bit, in mili-meter): depth image, could be None
        
        Output:
            A dict, {object_type_1: [pose_1, pose_2, ...], object_type_2: [pose_1, pose_2, ...], ...}
                object_type(string): the type of object
                pose(numpy.array): 6DoF pose of object. The format is 4x4 rotation matrix with homogeneous coordinate.
                    Can be seen as a transform matrix from object coordinate to camera coordinate
                A Example: {'marker_pen': [pose_of_marker_pen1], 'bowl': [pose_of_bowl1, pose_of_bowl2]}
        """
        return {}


class PassiveMarkerPoseDetector(BasePoseDetector):
    """ A simple Pose Detector using Passive IR marker to find the pose of object
    - The pose of a object is calculated from 2 markers.
    - Markers should be pasted along the longer dimension.
    """

    def __init__(self, logging=False, calibra_src=None):
        """
        Init
        """
        self._logging = logging
        from k4a_wrapper import K4aMarkerDetector
        self._marker_detector = K4aMarkerDetector(logging=logging, calibra_src=calibra_src)

    def get_obj_pose(self, gray, depth, rgb=None, z_offset=25):
        """
        Get the raw pose in camera coordinate, only support for coffe_bottle with 2 markers now

        Input:
            gray(numpy.array): gray image
            depth(numpy.array): depth image
        
        Output:
            A dict, {object_type_1: [pose_1, pose_2, ...], object_type_2: [pose_1, pose_2, ...], ...}
        """
        # get markers
        markers = self._marker_detector.get_marker_from_img(gray, depth)
        # use 2 marker as the indicator. Largest one is on the bottom of the bottle
        if len(markers) == 2:
            # # print(markers)
            markers = np.array(markers)
            center = (0.5 * markers[0][:3] + 0.5 * markers[1][:3])
            center[2] += z_offset
            vector_z = markers[1][:3] - markers[0][:3]

            # vx1, vx2, vx3 = 1, 1, -(vector_z[1][0] + vector_z[1][1])/vector_z[1][2]
            # vector_x = np.array([vx1, vx2, vx3])
            vector_x = np.ones_like(vector_z)
            max_dim = np.argmax(np.absolute(vector_z))  # none singular dimension
            vector_x[max_dim] = -(np.sum(vector_z) - vector_z[max_dim]) / vector_z[max_dim]
            vector_x_norm = vector_x / np.linalg.norm(vector_x)
            vector_y = np.cross(vector_z, vector_x_norm)  # or np.cross(vector_x_norm, vector_z) ?
            vector_y_norm = vector_y / np.linalg.norm(vector_y)
            vector_z_norm = vector_z / np.linalg.norm(vector_z)

            cam_key_points = [center, center + vector_x_norm, center + vector_y_norm, center + vector_z_norm]
            obj_key_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            # get transformation from obj coord to camera coord
            obj_points_inv = np.linalg.inv(homogeneous_coord_t(obj_key_points))
            cam_points_t = homogeneous_coord_t(cam_key_points)
            trans_matrix = np.dot(cam_points_t, obj_points_inv)
            if self._logging:
                print("none_singular_dim:" + str(max_dim))
                print("vector_x_norm:" + str(vector_x_norm))
                print("vector_y_norm:" + str(vector_y_norm))
                print("vector_z_norm:" + str(vector_z_norm))
                print("cam_key_points:" + str(cam_key_points))
                print("trans_matrix:" + str(trans_matrix))
            return {"coffe_bottle": [trans_matrix]}
        else:
            return {}


class PVN3DPoseDetector(BasePoseDetector):
    """ Class to Wrap PVN3D Pose Detector
    Need to put PVN3D folder side by side with this repo, i.e., "../"
    """

    def __init__(self, calibra_src, logging=False):
        """
        Init
        """
        self._logging = logging
        self._depth_camera = calibra_src
        self._ycb_path = "../PVN3D/pvn3d/datasets/ycb/YCB_Video_Dataset/data/0000/"
        self._pose_path = "../PVN3D/pvn3d/train_log/ycb/eval_results/pose_vis/0_pose_dict.npy"
        self._res_vis_path = "../PVN3D/pvn3d/train_log/ycb/eval_results/pose_vis/0.jpg"
        self._pvn3d_run_cmd = "cd ../PVN3D/pvn3d \n \
            rm train_log/ycb/eval_results/pose_vis/0_pose_dict.npy \n \
            python3 -m demo -checkpoint train_log/ycb/checkpoints/pvn3d_best -dataset ycb \n \
            cd - \ "

    def get_obj_pose(self, gray, depth, rgb):
        """
        Get objects and its pose from PVN3D PoseDetector

        Input:
            rgb(numpy.array, 8bit): rgb image
            gray(numpy.array, 8bit): gray image, could be None
            depth(numpy.array, 16bit, in mili-meter): depth image
        
        Output:
            A dict, {object_type_1: [pose_1, pose_2, ...], object_type_2: [pose_1, pose_2, ...], ...}
        """
        camera_kp = self._depth_camera.get_camera_kp()
        h, w = gray.shape[:2]
        scale = max(640 / w, 480 / h)
        scale = scale * 1.8  # scale the image to suit the scene here
        print("imput image scaled by:" + str(scale))
        new_camera_kp, crop_depth = scale_crop_helper(
            depth, camera_kp, scale, size=(640, 480), interpolation=cv2.INTER_NEAREST)
        new_camera_kp, crop_color = scale_crop_helper(rgb, camera_kp, scale, size=(640, 480))
        crop_depth = crop_depth * 10.0
        crop_depth = crop_depth.astype(np.uint16)
        np.savetxt(self._ycb_path + '000001-CameraK.txt', new_camera_kp)
        cv2.imwrite(self._ycb_path + '000001-color.png', crop_color)
        cv2.imwrite(self._ycb_path + '000001-depth.png', crop_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # run pvn3d
        cmd = self._pvn3d_run_cmd
        if self._logging:
            print("saved in: " + self._ycb_path)
            print("running cmd: " + cmd)
        os.system(cmd)  # returns the exit status

        # get results
        try:
            obj_pose_dict = np.load(self._pose_path, allow_pickle=True).item()
        except:
            obj_pose_dict = {}
        for obj_name in obj_pose_dict.keys():
            obj_poses = obj_pose_dict[obj_name]
            for i in range(len(obj_poses)):
                obj_poses[i][:, 3] = obj_poses[i][:, 3] * 1000.0  # convert to mili-meter
                # rotation matrix with homogenous coordinates
                obj_poses[i] = np.insert(obj_poses[i], 3, values=np.array([0, 0, 0, 1]), axis=0)
        # do loging
        if self._logging:
            print(obj_pose_dict)
            res_vis_img = cv2.imread(self._res_vis_path)
            cv2.imshow("pose_vis", res_vis_img)
            cv2.waitKey(100)
        return obj_pose_dict

from planner import gripping_properties, global_object_pos_offset
class DCVMultiTask2DDetector(BasePoseDetector):
    """ Class for using  D-CV-MXNet MultiTask 2D detector
    Need to put d-cv-multitask folder side by side with this repo, i.e., "../"
    """

    def __init__(self, calibra_src, calibra_world, logging=False):
        """
        Init
        """
        self._logging = logging
        self._depth_camera = calibra_src
        self._calibra = calibra_world
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._src_image_path = dir_path + "/../d-cv-multitask/test_images/"
        self._res_save_path = dir_path + "/../d-cv-multitask/pred_mask_res/"
        self._res_vis_path = "../d-cv-multitask/pred_mask_res/0.jpg"
        lib_path = os.path.abspath(os.path.join('../d-cv-multitask/scripts/multitask'))
        sys.path.append(lib_path)
        import mask_predictor as dcv_multitask
        self.dcv_multitask = dcv_multitask

    def get_obj_pose(self, gray, depth, rgb):
        """
        Get objects and its pose

        Input:
            rgb(numpy.array, 8bit): rgb image
            gray(numpy.array, 8bit): gray image, could be None
            depth(numpy.array, 16bit, in mili-meter): depth image
        
        Output:
            A dict, {object_type_1: [pose_1, pose_2, ...], object_type_2: [pose_1, pose_2, ...], ...}
        """
        visualize = self._logging
        visualize_points_list = []
        visualize_colors_list = []
        target_size = (1280, 960)  # (640, 480)
        zoom_scale = 1.0  # zoom the image to suit the scene if needed
        camera_kp = self._depth_camera.get_camera_kp()
        h, w = gray.shape[:2]
        scale = max(target_size[0] / w, target_size[1] / h)
        scale = scale * zoom_scale 
        print("imput image scaled by:" + str(scale))
        new_camera_kp, crop_depth_mm = scale_crop_helper(
            depth, camera_kp, scale, size=(target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)
        new_camera_kp, crop_color = scale_crop_helper(rgb, camera_kp, scale, size=(target_size[0], target_size[1]))
        crop_depth = crop_depth_mm * 10.0
        crop_depth = crop_depth.astype(np.uint16)
        # np.savetxt(self._src_image_path + '0.CameraK.txt', new_camera_kp)
        cv2.imwrite(self._src_image_path + '0.jpg', crop_color)
        # cv2.imwrite(self._src_image_path + '0.png', crop_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        mask_res_dict = self.dcv_multitask.mask_predictor([self._src_image_path + '0.jpg'], self._res_save_path)[0]

        def get_world_pose_from_2dpoints(points_2d, depth, obj_name):
            """ get estimated world pose from a set of points
            """
            # convert to 3d points in world
            points_list = []
            fx, cx, fy, cy = new_camera_kp[0][0], new_camera_kp[0][2], new_camera_kp[1][1], new_camera_kp[1][2]
            for point_2d in points_2d:
                y, x = point_2d[1], point_2d[0]
                if depth[y,x] > 10.0:
                    point_x = depth[y,x] * (x - cx) / fx
                    point_y = depth[y,x] * (y - cy) / fy
                    point_z = depth[y,x]
                    point_cam = [point_x, point_y, point_z]
                    point_wrd = self._calibra.cam_2_world_sigle_points(point_cam)
                    points_list.append(point_wrd)
                    if visualize:
                        visualize_points_list.append(point_wrd)
                        visualize_colors_list.append(np.array([crop_color[y,x][2], crop_color[y,x][1], crop_color[y,x][0]])/255.0)
            points_list_len = len(points_list)
            print("object points count:" + str(points_list_len))
            if points_list_len < 20:
                print("too less points")
                return None
            points = np.array(points_list)
            # np.savetxt('./points.txt', points)
            # get center pos and main direction in x-y plane by PCA SVD
            u, _, _ = pca_with_svd(points[:,:2])
            center = np.mean(points, axis=0, keepdims=False)

            # get keypoints from center and res of PCA
            vector_x = np.array([u[1][0], u[1][1], 0.0])
            # vector_y = np.array([u[0][0], u[0][1], 0.0])
            vector_z = np.array([0.0, 0.0, 1.0])
            vector_x_norm = vector_x / np.linalg.norm(vector_x)
            vector_y = np.cross(vector_x_norm, vector_z)  # or np.cross(vector_x_norm, vector_z) ?
            vector_y_norm = vector_y / np.linalg.norm(vector_y)
            vector_z_norm = vector_z / np.linalg.norm(vector_z)
            key_points = [center, center + vector_x_norm, center + vector_y_norm, center + vector_z_norm]
            obj_key_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            # get transformation of the key points
            obj_points_inv = np.linalg.inv(homogeneous_coord_t(obj_key_points))
            points_t = homogeneous_coord_t(key_points)
            trans_matrix = np.dot(points_t, obj_points_inv)
            if visualize:
                size = [70, 180, 30] # default for banana
                offset = np.array(global_object_pos_offset) + np.array(gripping_properties[obj_name]["pos_offset"])
                if obj_name == "apple": size = [70, 70, 70]
                if obj_name == "mouse": size = [50, 90, 40]
                if obj_name == "orange": size = [50, 50, 50]
                if obj_name == "carrot": size = [50, 130, 50]
                size_x, size_y, size_z = size[0], size[1], size[2]
                flag_points = []
                for i in range(100):
                    flag_points.append([ size_x/2,  size_y/2, size_z*(i-50)/100])
                    flag_points.append([-size_x/2,  size_y/2, size_z*(i-50)/100])
                    flag_points.append([ size_x/2, -size_y/2, size_z*(i-50)/100])
                    flag_points.append([-size_x/2, -size_y/2, size_z*(i-50)/100])
                    flag_points.append([size_x*(i-50)/100,  size_y/2,  size_z/2])
                    flag_points.append([size_x*(i-50)/100,  size_y/2, -size_z/2])
                    flag_points.append([size_x*(i-50)/100, -size_y/2,  size_z/2])
                    flag_points.append([size_x*(i-50)/100, -size_y/2, -size_z/2])
                    flag_points.append([ size_x/2, size_y*(i-50)/100,  size_z/2])
                    flag_points.append([-size_x/2, size_y*(i-50)/100,  size_z/2])
                    flag_points.append([ size_x/2, size_y*(i-50)/100, -size_z/2])
                    flag_points.append([-size_x/2, size_y*(i-50)/100, -size_z/2])
                for flag_point in flag_points:
                    flag_point = self._calibra.cam_2_world_sigle_points(cam_pos=flag_point, overide_matrix=trans_matrix)
                    flag_point = flag_point + offset
                    visualize_points_list.append(flag_point)
                    visualize_colors_list.append(np.array([1.0, 0, 0]))

            return trans_matrix

        start_time = time.time()
        if mask_res_dict is None:
            obj_pose_dict = {}
        else:
            # get obj pose from mask results and depth image/points
            obj_pose_dict = {}
            for i in range(len(mask_res_dict['pred_classes'])):
                obj_name = mask_res_dict['pred_cls_names'][i]
                if self._logging: print(obj_name)
                if obj_name not in gripping_properties.keys(): continue # only allow stable objects for now
                points = mask_res_dict['mask_points'][i]
                pose = get_world_pose_from_2dpoints(points, crop_depth_mm, obj_name=obj_name)
                if pose is None: continue
                if obj_name in obj_pose_dict.keys():
                    obj_pose_dict[obj_name].append(pose)
                else:
                    obj_pose_dict[obj_name] = [pose]
            pass

        for obj_name in obj_pose_dict.keys():
            obj_poses = obj_pose_dict[obj_name]
            for i in range(len(obj_poses)):
                # rotation matrix with homogenous coordinates
                obj_poses[i] = np.insert(obj_poses[i], 3, values=np.array([0, 0, 0, 1]), axis=0)
        print("estimate pose from points time: %.3f s" % (time.time() - start_time))
        
        # do loging
        if self._logging:
            print(obj_pose_dict)
            res_vis_img = cv2.imread(self._res_vis_path)
            cv2.imshow("pose_vis", res_vis_img)
            cv2.waitKey(100)
        if visualize:
            import open3d as o3d
            h, w = crop_depth_mm.shape[:2]
            fx, cx, fy, cy = new_camera_kp[0][0], new_camera_kp[0][2], new_camera_kp[1][1], new_camera_kp[1][2]
            crop_color = cv2.cvtColor(crop_color, cv2.COLOR_BGR2RGB)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(crop_color.astype(np.uint8)),
                o3d.geometry.Image(crop_depth_mm.astype(np.float32)),
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
                depth_trunc=6000.0)
            pcd_env = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy))
            pcd_env.transform(self._calibra.get_transform_matrix())
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(np.array(visualize_points_list))
            pcd_vis.colors = o3d.utility.Vector3dVector(np.array(visualize_colors_list))
            # o3d.io.write_point_cloud('./seg_obj_points.ply', pcd_env+pcd_vis, write_ascii=False, compressed=False)
            # o3d.visualization.draw_geometries([pcd_env, pcd_vis], width=1600, height=900)
            o3d.visualization.draw(geometry=pcd_env+pcd_vis, width=1600, height=900, point_size=1,
                bg_color=(0.5, 0.5, 0.5, 0.5), show_ui=True)
        
        return obj_pose_dict


# Test for PoseDetector
# example: python3 pose_detector.py mask2d
if __name__ == "__main__":
    import sys
    lib_path = os.path.abspath(os.path.join('../x3d_camera'))
    sys.path.append(lib_path)
    from x3d_camera import X3DCamera
    from calibra_cam2world import Cam2World
    from planner import gripping_properties

    if len(sys.argv) < 2 or sys.argv[1] not in ['pvn3d', 'mask2d', 'ir_marker']:
        print("run with args: \n \
        pvn3d, mask2d, ir_marker")
        # pose_detector = 'pvn3d'
        exit()
    else:
        pose_detector = sys.argv[1]

    # run the task group once or repeatedly. if repeatedly, press ESC to exit
    repeatedly = False

    depth_camera = X3DCamera(logging=True)  # K4aWrapper(logging=False)
    depth_camera.re_capture_image_using_env_exposure = True
    calibra = Cam2World()

    need_undistort = True
    need_convert_world_coord = True
    if pose_detector == 'ir_marker':
        need_undistort = False
        detector = PassiveMarkerPoseDetector(logging=True, calibra_src=depth_camera)
    elif pose_detector == 'pvn3d':
        detector = PVN3DPoseDetector(logging=True, calibra_src=depth_camera)
    elif pose_detector == 'mask2d':
        detector = DCVMultiTask2DDetector(logging=True, calibra_src=depth_camera, calibra_world=calibra)
        need_convert_world_coord = False
    else:
        print("unsupported pose detector!")
        depth_camera.close()
        exit()

    while True:
        rgb_img, gray_img, depth_img = depth_camera.get_images(undistort=need_undistort)
        result = detector.get_obj_pose(rgb=rgb_img, gray=gray_img, depth=depth_img)
        print(result)
        for obj_type in result.keys():
            obj_poses = result[obj_type]
            for i in range(len(obj_poses)):
                # print object name and its count
                print(obj_type + ":" + "cnt" + str(i))
                obj_tran_matrix = obj_poses[i]
                # translate to world coordinate, i.e., robot arm base coordinate
                obj_pos_cam = np.dot(obj_tran_matrix, np.array([0, 0, 0, 1]).T).T[:3]
                if obj_type in gripping_properties.keys():
                    grip_axis = gripping_properties[obj_type]['grip_axis'][:]
                    grip_axis.append(1)
                    grip_axis = np.array(grip_axis)
                else:
                    grip_axis = np.array([0, 0, 1, 1])
                # get obj_rot_vec along gripping axis
                obj_rot_vec_cam = np.dot(obj_tran_matrix, grip_axis.T).T[:3] - obj_pos_cam
                # print res
                if need_convert_world_coord:
                    print("\tcamera coord, position:" + str(obj_pos_cam) + ", rotation vector:" + str(obj_rot_vec_cam))
                    obj_position = calibra.cam_2_world(obj_pos_cam)
                    obj_rot_vec = calibra.cam_2_world(obj_rot_vec_cam) - calibra.cam_2_world(np.zeros_like(obj_rot_vec_cam))
                    print("\tworld coord, position:" + str(obj_position) + ", rotation vector:" + str(obj_rot_vec))
                else:
                    print("\tworld coord, position:" + str(obj_pos_cam) + ", rotation vector:" + str(obj_rot_vec_cam))

        key = cv2.waitKey(1000)
        if key == 27 or not repeatedly: break  # Esc key to stop

    depth_camera.close()
