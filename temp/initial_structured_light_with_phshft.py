# Copyright (c) 2020 XFORWARDAI. All Rights Reserved.
# Created on 2021-04-01
# Autor: Jiangtao <jiangtao.li@xforwardai.com>
"""
Description: xForward 3D Camera structure light algorithm pipeline

These programs implement the xForward structured light 3D camera pipeline.
"""

import os
import cv2
import time
import numpy as np
import numba
from numba import prange
numba.config.NUMBA_DEFAULT_NUM_THREADS=8
import transform_helper
from x3d_camera.stereo_rectify import StereoRectify
from x3d_camera.structured_light import gray_decode, depth_filter, depth_avg_filter

### parameters
image_index_unvalid_thres = 1
image_index_using_positive_pattern_only = True
use_depth_avg_filter = True
use_depth_bi_filter = False

"""
For 4 step phaseshift, phi = np.arctan2(I4-I2, I3-I1), from -pi to pi
"""
@numba.jit ((numba.uint8[:,:,:], numba.uint8[:,:], numba.int64, numba.int64, numba.int64, numba.float32[:,:], numba.int16[:,:], numba.float64),nopython=True, parallel=True)
def phase_shift_decode(src, valid_map, image_num, height, width, img_phase, img_index, unvalid_thres):
    pi = 3.14159265358979
    for h in prange(height):
        for w in range(width):
            if img_index[h,w] == -9999:
                img_phase[h,w] = -9999.0
                continue
            i1, i2, i3, i4 = 1.0*src[0][h,w], 1.0*src[1][h,w], 1.0*src[2][h,w], 1.0*src[3][h,w]
            phi = - np.arctan2(i4-i2, i3-i1)
            # img_phase[h,w] = img_index[h, w] * 2 * phi
            phase = phi+pi
            phase_main_index = img_index[h,w] // 2
            phase_sub_index = img_index[h,w] % 2
            if phase_sub_index == 0 and phase > pi*1.5:
                phase = 0
            if phase_sub_index == 1 and phase < pi*0.5:
                phase = 2 * pi
            img_phase[h,w] = phase
            img_index[h,w] = phase_main_index * 128 + (round)(phase * 128 / (2*pi))

def get_image_index(image_path, appendix, rectifier):
    images_posi = []
    images_nega = []
    unvalid_thres = image_index_unvalid_thres
    for i in range(0, 2):
        fname = image_path + str(i) + appendix
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # rectify_image, accroding to left or right
        if appendix == '_l.bmp': img = rectifier.rectify_image(img.astype(np.uint8))
        else: img = rectifier.rectify_image(img.astype(np.uint8), left=False)
        # posi or negative
        if i % 2 == 0: prj_area_posi = img
        else: prj_area_nega = img
    prj_valid_map = prj_area_posi - prj_area_nega
    if image_index_using_positive_pattern_only:
        positive_pattern_only_avg_thres = (prj_area_posi//2 + prj_area_nega//2)
    thres, prj_valid_map_bin = cv2.threshold(prj_valid_map, unvalid_thres, 255, cv2.THRESH_BINARY)
    
    start_time = time.time()
    for i in range(2, 14):  # (2, 24)
        fname = image_path + str(i) + appendix
        if not os.path.exists(fname): break
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # rectify_image, accroding to left or right
        if appendix == '_l.bmp': img = rectifier.rectify_image(img.astype(np.uint8))
        else: img = rectifier.rectify_image(img.astype(np.uint8), left=False)
        # posi or negative
        if i % 2 == 0: images_posi.append(img)
        else:
            if image_index_using_positive_pattern_only: images_nega.append(positive_pattern_only_avg_thres)
            else: images_nega.append(img)
    print("read and rectify images using %.3f s" % (time.time() - start_time))
    height, width = images_posi[0].shape[:2]
    img_index, src_imgs = np.zeros_like(images_posi[0], dtype=np.int16), np.array(images_posi)
    src_imgs_nega = np.array(images_nega)
    start_time = time.time()
    gray_decode(src_imgs, src_imgs_nega, prj_valid_map_bin, len(images_posi), height,width, img_index, unvalid_thres)
    print("index decoding using %.3f s" % (time.time() - start_time))

    images_phsft = []
    start_time = time.time()
    for i in range(24, 28):  # f = 32 phase shift patern 
        fname = image_path + str(i) + appendix
        if not os.path.exists(fname): break
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # rectify_image, accroding to left or right
        if appendix == '_l.bmp': img = rectifier.rectify_image(img.astype(np.uint8))
        else: img = rectifier.rectify_image(img.astype(np.uint8), left=False)
        # posi or negative
        images_phsft.append(img)
    img_phase = np.zeros_like(images_posi[0], dtype=np.float32)
    images_phsft_src = np.array(images_phsft)
    phase_shift_decode(images_phsft_src, prj_valid_map_bin, len(images_posi), height,width, img_phase, img_index, unvalid_thres)
    print("phase_shift decoding in total using %.3f s" % (time.time() - start_time))

    return img_index, img_phase, rectifier.rectified_camera_kd


@numba.jit ((numba.float64[:,:],numba.float64[:,:], numba.int64,numba.int64, numba.int16[:,:],numba.int16[:,:], numba.float64,numba.float64,numba.float64, numba.float64[:,:],numba.float64[:,:] ), nopython=True, parallel=True, nogil=True)
def get_dmap_from_index_map(dmap,depth_map, height,width, img_index_left,img_index_right, baseline,dmap_base,fx, img_index_left_sub_px,img_index_right_sub_px):
    max_allow_pixel_per_index = 1 + width // 600  # some typical condition: 1 for 640, 3 for 1280, 5 for 2560, 8 for 4200
    right_corres_point_offset_range = width // 10
    for h in prange(height):
        line_r = img_index_right[h,:]
        line_l = img_index_left[h,:]
        possible_points_r = np.zeros(width, dtype=np.int64)
        last_right_corres_point = -1
        for w in range(width):
            if line_l[w] == -9999:   # unvalid
                last_right_corres_point = -1
                continue
            ## find possible right indicator
            cnt = 0
            if last_right_corres_point > 0:
                checking_left_edge = last_right_corres_point - right_corres_point_offset_range
                checking_right_edge = last_right_corres_point + right_corres_point_offset_range
                if checking_left_edge <=0: checking_left_edge=0
                if checking_right_edge >=width: checking_left_edge=width
            else:
                checking_left_edge, checking_right_edge = 0, width
            for i in range(checking_left_edge, checking_right_edge):
                if line_l[w] - 1 <= line_r[i] <= line_l[w] + 1:
                    possible_points_r[cnt] = i
                    cnt += 1
            if cnt == 0:
                cnt_l, cnt_r = 0, 0
                for i in range(width):
                    if line_l[w]-2 >= line_r[i] >= line_l[w]-2:
                        possible_points_r[cnt_r+cnt_l] = i
                        cnt_l += 1
                    elif line_l[w]+2 <= line_r[i] <= line_l[w]+2:
                        possible_points_r[cnt_r+cnt_l] = i
                        cnt_r += 1
                if cnt_l > 0 and cnt_r > 0: cnt = cnt_l + cnt_r
                else: continue
            ## find right indicator w_r in possible_points
            w_r = 0.0
            for i in range(cnt): 
                p = possible_points_r[i]
                if img_index_right_sub_px[h, p] > 0.001: w_r += img_index_right_sub_px[h, p]
                else: w_r += p
            w_r /= cnt
            # check right outliner
            outlier_flag_r = False
            for i in range(cnt): 
                p = possible_points_r[i]
                if abs(p-w_r) >= max_allow_pixel_per_index: outlier_flag_r=True
            if outlier_flag_r: continue
            last_right_corres_point = round(w_r)
            ## refine left index around w
            w_l, w_l_cnt = 0.0, 0
            for i in range(w-max_allow_pixel_per_index, min(w+max_allow_pixel_per_index+1, width)):
                if img_index_left[h,i]==img_index_left[h,w]:
                    w_l_cnt += 1
                    if img_index_left_sub_px[h, i] > 0.001: w_l += img_index_left_sub_px[h, i]
                    else: w_l += i
            # check left outliner
            outlier_flag_l = False
            if w_l_cnt == 1:  # if only one near the checking range has the index, consider it could be an outliner  
                cnt = 0
                for i in range(width):
                    if line_l[i] == line_l[w]: cnt += 1
                    if cnt >= 2:
                        outlier_flag_l = True
                        break
            if outlier_flag_l: continue
            w_l = w_l / w_l_cnt
            ## stereo diff and depth
            stereo_diff = dmap_base + w_l - w_r
            dmap[h,w] =  (w_l - w_r)
            if stereo_diff > 0.000001:
                depth = fx * baseline / stereo_diff
                if 0.1 < depth < 2.0:
                    depth_map[h, w] = depth
                    # if subpix optimization is used
                    if img_index_left_sub_px[h, w] > 0.001 and img_index_left_sub_px[h, w-2] > 0.001:
                        if abs(depth_map[h, w] - depth_map[h, w-2]) < 0.01:
                            dis_r = img_index_left_sub_px[h, w] - (w-1)
                            dis_l = (w-1) - img_index_left_sub_px[h, w-2]
                            if dis_l > 0.0001 and dis_r > 0.0001:
                                diff = depth_map[h, w] - depth_map[h, w-2]
                                inter_depth_w_1 = depth_map[h, w-2] + diff * dis_l / (dis_l + dis_r)
                                # print(depth_map[h, w-1] - inter_depth_w_1)
                                if abs(depth_map[h, w-1] - inter_depth_w_1) < 0.01:
                                    depth_map[h, w-1] = inter_depth_w_1
                                    img_index_left_sub_px[h, w-1] = w-1

def run_stru_li_pipe(pattern_path, res_path, rectifier=None):
    if rectifier is None:
        rectifier = StereoRectify(scale=1.0, cali_file=pattern_path+'calib.yml')

    ### Rectify and Decode to index
    pipe_start_time = start_time = time.time()
    img_index_left, images_phsft_left, camera_kd_l = get_image_index(pattern_path, '_l.bmp', rectifier=rectifier)
    img_index_right, images_phsft_right, camera_kd_r = get_image_index(pattern_path, '_r.bmp', rectifier=rectifier)
    print("index decoding in total using %.3f s" % (time.time() - start_time))

    fx = camera_kd_l[0][0]
    cx, cx_r = camera_kd_l[0][2], camera_kd_r[0][2]
    dmap_base = cx_r - cx
    cam_transform = np.array(rectifier.T)[:,0]
    height, width = img_index_left.shape[:2]
    # TODO
    # print(rectifier.R1)
    baseline = np.linalg.norm(cam_transform) * ( 0.8/(0.8+0.05*0.001) )  # = 0.9999375039060059

    ### Infer DiffMap from DecodeIndex
    print("Generate Depth Map...")
    dmap = np.zeros_like(img_index_left, dtype=np.float)
    depth_map = np.zeros_like(img_index_left, dtype=np.float)  # img_index_left.copy().astype(np.float)
    # subpixel for index
    img_index_left_sub_px = np.zeros_like(img_index_left, dtype=np.float)
    img_index_right_sub_px = np.zeros_like(img_index_left, dtype=np.float)
    # gen depth and diff map
    start_time = time.time()
    get_dmap_from_index_map(dmap, depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px)
    print("depth map generating from index %.3f s" % (time.time() - start_time))

    ### Run Depth Map Filter
    depth_map_raw = depth_map.copy()  # save raw depth map
    start_time = time.time()
    depth_filter(depth_map, depth_map_raw, height, width, camera_kd_l)
    print("flying point filter %.3f s" % (time.time() - start_time))
    if use_depth_avg_filter:
        start_time = time.time()
        depth_avg_filter(depth_map, depth_map_raw, height, width, camera_kd_l)
        print("depth avg filter %.3f s" % (time.time() - start_time))
    if use_depth_bi_filter:
        start_time = time.time()
        depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), d=width//10, sigmaColor=0.0025, sigmaSpace=0.0025, borderType=cv2.BORDER_DEFAULT)
        print("bilateralFilter %.3f s" % (time.time() - start_time))
    print("Total pipeline time: %.3f s" % (time.time() - pipe_start_time))
    cv2.imwrite(res_path + '/diff_map_alg2.png', dmap.astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    depth_map_uint16 = depth_map * 30000
    cv2.imwrite(res_path + '/depth_alg2.png', depth_map_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    depth_map_raw_uint16 = depth_map_raw * 30000
    cv2.imwrite(res_path + '/depth_alg2_raw.png', depth_map_raw_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    
    ### Save Mid Results for debugging
    img_correspondence_l = np.clip(img_index_left//128 * 50 % 255, 0, 255).astype(np.uint8)
    img_correspondence_r = np.clip(img_index_right//128 * 50 % 255, 0, 255).astype(np.uint8)
    images_phsft_left_v = np.clip(images_phsft_left/7*255.0, 0, 255).astype(np.uint8)
    images_phsft_right_v = np.clip(images_phsft_right/7*255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(res_path + "/index_correspondence_l.png", img_correspondence_l)
    cv2.imwrite(res_path + "/index_correspondence_r.png", img_correspondence_r)
    cv2.imwrite(res_path + "/ph_correspondence_l.png", images_phsft_left_v)
    cv2.imwrite(res_path + "/ph_correspondence_r.png", images_phsft_right_v)
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_correspondence_l)
    # plt.subplot(1, 2, 2)
    # plt.imshow(images_phsft_left_v)
    # plt.show()

    ### Prepare results
    depth_map_mm = depth_map * 1000
    gray_img = cv2.imread(pattern_path + "0_l.bmp", cv2.IMREAD_UNCHANGED)
    gray_img = rectifier.rectify_image(gray_img)
    return gray_img, depth_map_mm, camera_kd_l


# test with existing pattern: 
#   python -m x3d_camera.structured_light_with_phshft '/home/ubuntu/workplace/3dperceptionprototype/temp/dataset_render/0000_with_phsft/raw_no_light_bounces/'
if __name__ == "__main__":
    import sys
    import glob
    import shutil
    import open3d as o3d
    import matplotlib.pyplot as plt

    is_render = False
    scale_image = None  # None or 1.0 by default. if calibra file is generated with high res and pattern is downscaled, using this option to upscale back
    
    if len(sys.argv) <= 1:
        print("run with args 'pattern_path'")
    image_path = sys.argv[1]
    if image_path[-1] == "/": image_path = image_path[:-1]

    ### convert to gray and scale the image if needed
    if not os.path.exists(image_path + "_gray"):
        shutil.copytree(image_path + "/",image_path + "_gray/")
        os.system("mkdir " + image_path + "_gray")
        os.system("cp -r " + image_path + "/* " + image_path + "_gray/")
    image_path = image_path + '_gray/'
    images = glob.glob(image_path + '*.bmp')
    for i, fname in enumerate(images):
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(fname, img)
        if scale_image is not None and scale_image != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (round(w*scale_image), round(h*scale_image)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(fname, img)

    ### build up runing parameters and run the pipeline
    res_path = image_path + "../x3d_cam_res"
    if not os.path.exists(res_path): os.system("mkdir " + res_path)
    gray, depth_map_mm, camera_kp = run_stru_li_pipe(image_path, res_path)

    ### save to point cloud
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(gray.astype(np.uint8)),
        o3d.geometry.Image(depth_map_mm.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=6000.0)
    h, w = gray.shape[:2]
    fx, fy, cx, cy = camera_kp[0][0], camera_kp[1][1], camera_kp[0][2], camera_kp[1][2]
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy))
    # flip it if needed
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud(res_path + "/points.ply", pcd, write_ascii=False, compressed=False)
    print("res saved to:" + res_path)
