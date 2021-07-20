
import os
import numpy as np
import cv2
import random
lib_path = os.path.abspath(os.path.join('../x3d_camera'))
sys.path.append(lib_path)
from HuaraySDKWrapper import HuarayCamera
from x3d_camera import X3DCamera

# fit and segmentation para
desk_max_height = 170
desk_min_height = 130
objects_min_height = 3 # in mm
objects_max_height = 250 # in mm
hwrb = 150 # half width of robot base, in mm

# test with "python x3d_pcl_seg.py"
if __name__ == "__main__":
    use_saved_images = True

    from calibra_cam2world import Cam2World
    calibra = Cam2World()

    if not use_saved_images:
        depth_camera = X3DCamera(logging=True, scale=None)
        rgb_img, gray_img, depth_img = depth_camera.get_images()  # rgb_img is None
        # pass

        depth_camera.close()
    else:  # test
        # set dir
        file_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file
        dir_path = file_path + "/images/"
        depth_path = dir_path + "depth.png"
        gray_path = dir_path + "gray_unlight.png"
        rgb_path = dir_path + "rgb_image.png"
        kd_yml_path = dir_path + "camera_kd.yml"
        res_path = file_path + "/images/"

        # read camera kp
        ymlfile = cv2.FileStorage(kd_yml_path, cv2.FILE_STORAGE_READ)
        camera_kp = ymlfile.getNode("Kd").mat()
        ymlfile.release()

        # read images
        depth_16bit = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth_16bit * 2000.0 / 60000.0  # convet to mili_meter
        gray = cv2.imread(gray_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        # convert to points
        # x = z * (x-cx) / fx; y is similar; z = z
        print("convert to points")
        fx, cx, fy, cy = camera_kp[0][0], camera_kp[0][2], camera_kp[1][1], camera_kp[1][2]
        h, w = depth.shape[:2]
        points_x, points_y, points_z = depth.copy(), depth.copy(), depth.copy()
        for x in range(w):
            for y in range(h):
                if points_z[y,x] > 10:
                    points_x[y,x] = depth[y,x] * (x - cx) / fx
                    points_y[y,x] = depth[y,x] * (y - cy) / fy

        # convert to world coordinate
        print("convert to world coord")
        for x in range(w):
            for y in range(h):
                if points_z[y,x] > 0.001:
                    point_cam = [points_x[y,x], points_y[y,x], points_z[y,x]]
                    point_wrd = calibra.cam_2_world_sigle_points(point_cam)
                    points_x[y,x] = point_wrd[0]
                    points_y[y,x] = point_wrd[1]
                    points_z[y,x] = point_wrd[2]
        
        # fit desk plane
        print("fit")
        desk_plane_points = []
        for x in range(w):
            for y in range(h):
                if desk_min_height < points_z[y,x] < desk_max_height:
                    point_wrd = np.array([points_x[y,x], points_y[y,x], points_z[y,x]])
                    desk_plane_points.append(point_wrd)
        tmp_A, tmp_b = [], []
        for point in desk_plane_points:
            tmp_A.append([point[0], point[1], 1])  # [x, y, 1].T
            tmp_b.append(point[2])  # z
        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)
        fit = (A.T * A).I * A.T * b
        errors = b - A * fit
        residual = np.linalg.norm(errors)
        print("solution: z = %f x + %f y + %f" % (fit[0], fit[1], fit[2]))
        print("errors:" + str(errors))
        print("residual:" + str(residual))

        # segmentation and show res
        print("segmentation")
        bgr = np.tile(gray[:, :, None], (1, 1, 3))
        points, color = [], []
        for x in range(w):
            for y in range(h):
                if points_z[y,x] > 0.001:
                    desk_height = points_x[y,x] * fit[0] + points_y[y,x] * fit[1] + fit[2]
                    if objects_min_height < points_z[y,x] - desk_height < objects_max_height and \
                        not (-hwrb < points_x[y,x] < hwrb and -hwrb < points_y[y,x] < hwrb):
                        # the valid objects points
                        bgr[y,x,2] = 200
                        points.append([points_x[y,x], points_y[y,x], points_z[y,x]])
                        color.append(rgb[y,x])
                        # print((points_x[y,x], points_y[y,x], points_z[y,x]))

        # write res
        from x3d_camera import gen_rgb_point_clouds
        import pcl
        color_cloud = gen_rgb_point_clouds(points, color)
        pcl.save_XYZRGBA(cloud=color_cloud, path=res_path+'/seg_points.ply', format=None, binary=False)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bgr[:,:,2] = cv2.morphologyEx(bgr[:,:,2], cv2.MORPH_CLOSE, kernel, iterations=1)  # dilate then erode 
        bgr[:,:,2] = cv2.morphologyEx(bgr[:,:,2], cv2.MORPH_OPEN, kernel, iterations=1)  # erode then dilate

        cv2.imwrite(res_path + 'seg_res_bgr.png', bgr)
        print("Done!")

