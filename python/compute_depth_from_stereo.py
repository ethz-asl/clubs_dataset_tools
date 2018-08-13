#!/usr/bin/env python

import argparse
import cv2
import logging as log
import numpy as np

from tqdm import tqdm

from clubs_dataset_tools.stereo_matching import (rectify_images, stereo_match,
                                                 StereoMatchingParams)
from clubs_dataset_tools.filesystem_tools import (
    read_images, find_images_in_folder, find_all_folders,
    find_ir_image_folders, compare_image_names, create_stereo_depth_folder)
from clubs_dataset_tools.common import (CalibrationParams)


def compute_stereo_depth(scene_folder, sensor_folder, stereo_params,
                         calib_params):
    """
    Function that rectifies images and applies SGBM algorithm to compute depth.

    Input:
        scene_folder - Path to the scene folder
        sensor_folder - List containing folder names for left and right ir
        image, as well as the sensor root folder
        stereo_params - Parameters for stereo matching (StereoMatchingParams
        class)
        calib_params - Calibration parameters from the camera
        (CalibrationParams class)
    """

    images_left = find_images_in_folder(scene_folder + sensor_folder[0])
    images_right = find_images_in_folder(scene_folder + sensor_folder[1])

    timestamps = compare_image_names(images_left, images_right)

    if len(timestamps) != 0:
        image_paths_left = [
            scene_folder + sensor_folder[0] + '/' + image_left
            for image_left in images_left
        ]
        image_paths_right = [
            scene_folder + sensor_folder[1] + '/' + image_right
            for image_right in images_right
        ]

        ir_left = read_images(image_paths_left)
        ir_right = read_images(image_paths_right)

        stereo_depth_folder = create_stereo_depth_folder(scene_folder +
                                                         sensor_folder[2])

        stereo_bar = tqdm(total=len(ir_left), desc="Stereo Matching Progress")
        for i in range(len(ir_left)):
            log.debug("Rectifying " + str(i) + ". image pair")
            rectified_l, rectified_r, Q, RL, PL = rectify_images(
                ir_left[i], calib_params.ir1_intrinsics,
                calib_params.ir1_distortion_coeffs, ir_right[i],
                calib_params.ir2_intrinsics,
                calib_params.ir2_distortion_coeffs, calib_params.extrinsics_r,
                calib_params.extrinsics_t)
            log.debug("Stereo matching " + str(i) + '. image pair')
            depth_uint, depth_float, disparity_float = stereo_match(
                rectified_l,
                rectified_r,
                calib_params.extrinsics_t[0],
                calib_params.ir1_intrinsics[0, 0],
                stereo_params,
                scale=10000)

            zero_distortion = np.array([0, 0, 0, 0, 0])
            map_l1, map_l2 = cv2.initUndistortRectifyMap(
                PL[:3, :3], zero_distortion, np.linalg.inv(RL), PL[:3, :3],
                depth_uint.shape[::-1], cv2.CV_16SC2)
            depth_uint = cv2.remap(depth_uint, map_l1, map_l2,
                                   cv2.INTER_LINEAR)

            cv2.imwrite(
                stereo_depth_folder + '/' + timestamps[i] +
                '_stereo_depth.png', depth_uint)
            stereo_bar.update()
        stereo_bar.close()
    else:
        log.error("\nImage names are not consistent for left and right image.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Perform stereo matching for the infrared images and save it as a "
            "depth image. There are two different ways this function can be "
            "called. First one is by passing in the dataset root folder "
            "(flag --dataset_folder) which will create a new folder for each "
            "object/box scene and each sensor (d415 and d435), containing the "
            "depth image obtained through stereo matching. Second way is to "
            "pass object/box scene root folder (flag --scene_folder) which "
            "will do the same for that specific scene."))
    parser.add_argument(
        '--dataset_folder', type=str, help="Path to the dataset root folder.")
    parser.add_argument(
        '--scene_folder', type=str, help="Path to the scene root folder.")
    parser.add_argument(
        '--d415_calib_file',
        type=str,
        default='config/realsense_d415_stereo_depth.yaml',
        help=("Path to RealSense D415 calibration yaml file. By default: "
              "config/realsense_d415_stereo_depth.yaml"))
    parser.add_argument(
        '--d435_calib_file',
        type=str,
        default='config/realsense_d435_stereo_depth.yaml',
        help=("Path to RealSense D435 calibration yaml file. By default: "
              "config/realsense_d435_stereo_depth.yaml"))
    parser.add_argument(
        '--stereo_params_file',
        type=str,
        default='config/default_stereo_params.yaml',
        help=("Path to stereo parameters yaml file. By default: "
              "config/default_stereo_params.yaml"))
    parser.add_argument(
        '--use_only_boxes',
        action='store_true',
        help=("If this flag is set, depth from stereo will only be computed "
              "for the box scenes."))
    parser.add_argument(
        '--log',
        type=str,
        default='CRITICAL',
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    numeric_level = getattr(log, args.log.upper(), None)
    log.basicConfig(level=numeric_level)
    log.debug("Setting log verbosity to " + args.log)

    used_scenes = []

    stereo_params = StereoMatchingParams()
    stereo_params.read_from_yaml(args.stereo_params_file)
    calib_params = CalibrationParams()

    if args.dataset_folder is not None:
        log.debug("Received dataset_folder.")
        object_scenes, box_scenes = find_all_folders(args.dataset_folder)

        if args.use_only_boxes is True:
            log.debug("Processing only box scenes.")
            used_scenes = box_scenes
        else:
            log.debug("Processing both box and object scenes.")
            used_scenes = object_scenes + box_scenes

        progress_bar = tqdm(
            total=len(used_scenes) * 2, desc="Overall Progress")
        for i in range(len(used_scenes)):
            scene = used_scenes[i]
            log.debug("Processing " + str(scene))
            d415_folder, d435_folder = find_ir_image_folders(scene)

            calib_params.read_from_yaml(args.d415_calib_file)
            if d415_folder != []:
                compute_stereo_depth(scene, d415_folder, stereo_params,
                                     calib_params)
            progress_bar.update()
            calib_params.read_from_yaml(args.d435_calib_file)
            if d435_folder != []:
                compute_stereo_depth(scene, d435_folder, stereo_params,
                                     calib_params)
            progress_bar.update()
        progress_bar.close()
    elif args.scene_folder is not None:
        log.debug("Processing single scene " + str(args.scene_folder))

        d415_folder, d435_folder = find_ir_image_folders(args.scene_folder)

        calib_params.read_from_yaml(args.d415_calib_file)
        if d415_folder != []:
            compute_stereo_depth(args.scene_folder, d415_folder, stereo_params,
                                 calib_params)
        calib_params.read_from_yaml(args.d435_calib_file)
        if d435_folder != []:
            compute_stereo_depth(args.scene_folder, d435_folder, stereo_params,
                                 calib_params)
    else:
        parser.print_help()
