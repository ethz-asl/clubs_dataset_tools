#!/usr/bin/env python

import argparse
import sys
import os
import cv2
import numpy as np
import logging as log

from clubs_dataset_tools.stereo_matching import (
    rectify_images, stereo_match, StereoMatchingParams, CalibrationParams)
from clubs_dataset_tools.filesystem_tools import (
    read_images, find_images_in_folder, find_all_folders,
    find_ir_image_folders, compare_image_names, create_stereo_depth_folder)


def compute_stereo_depth(scene_folder, sensor_folder, stereo_params,
                         calib_params):
    """
    Function that rectifies images and applies SGBM algorithm to compute depth.

    Input:
        scene_folder - path to the scene folder
        sensor_folder - list containing folder names for left and right ir
        image, as well as the sensor root folder
        stereo_params - parameters for stereo matching (StereoMatchingParams
        class)
        calib_params - calibration parameters from the camera
        (CalibrationParams class)
    """

    images_left = find_images_in_folder(scene_folder + sensor_folder[0])
    images_right = find_images_in_folder(scene_folder + sensor_folder[1])

    timestamps = compare_image_names(images_left, images_right)

    if len(timestamps) != 0:
        [
            scene_folder + sensor_folder[0] + '/' + image_left
            for image_left in images_left
        ]
        [
            scene_folder + sensor_folder[0] + '/' + image_right
            for image_right in images_right
        ]

        images_left = read_images(
            scene_folder + sensor_folder[0] + '/' + images_left)
        images_right = read_images(
            scene_folder + sensor_folder[1] + '/' + images_right)

        stereo_depth_folder = create_stereo_depth_folder(
            scene_folder + sensor_folder[2])

        for i in range(len(images_left)):
            rectified_l, rectified_r, Q = rectify_images(
                images_left[i], calib_params.camera_matrix_l,
                calib_params.dist_coeffs_l, images_right[i],
                calib_params.camera_matrix_r, calib_params.dist_coeffs_r,
                calib_params.extrinsics_r, calib_params.extrinsics_t)
            depth_uint, depth_float, disparity_float = stereo_match(
                rectified_l,
                rectified_r,
                calib_params.extrinsics_t[0],
                calib_params.camera_matrix_l[0, 0],
                stereo_params,
                scale=10000)
            cv2.imwrite(stereo_depth_folder + '/' + timestamps[i] +
                        '_depth_images.png', depth_uint)
    else:
        log.error("Image names are not consistent for left and right image")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        ("Perform stereo matching for the infrared images and save it as a "
         "depth image. There are three different ways this function can be "
         "called. First one is by passing in the dataset root folder "
         "(flag --dataset_folder) which will create a new folder for each "
         "object/scene and each sensor (d415 and d435), containing the depth "
         "image obtained through stereo matching. Second way is to pass "
         "object/box scene root folder (flag --scene_folder) which will do "
         "the same for that specific scene. Last way is to directly pass in "
         "the right and left image (flags --left_image and --right_image) "
         "which will create a depth map in the current folder."))
    parser.add_argument(
        '--dataset_folder', type=str, help="Path to the dataset root folder.")
    parser.add_argument(
        '--use_only_boxes',
        type=bool,
        default=False,
        help=("If this flag is set to True, depth from stereo will only be "
              "computed for the box scenes."))
    parser.add_argument(
        '--scene_folder', type=str, help="Path to the scene root folder.")
    parser.add_argument(
        '--left_image',
        type=str,
        help="Path to the left image used for stereo matching.")
    parser.add_argument(
        '--right_image',
        type=str,
        help="Path to the right image used for stereo matching.")
    args = parser.parse_args()

    used_scenes = []

    stereo_params = StereoMatchingParams()
    calib_params = CalibrationParams()
    # TODO: add flags for loading from yaml.

    if args.dataset_folder is not None:
        object_scenes, box_scenes = find_all_folders(args.dataset_folder)
        if args.use_only_boxes is True:
            used_scenes.append(box_scenes)
        else:
            used_scenes.append(object_scenes)
            used_scenes.append(box_scenes)

        for scene in used_scenes:
            d415_folder, d435_folder = find_ir_image_folders(scene)

            compute_stereo_depth(scene, d415_folder, stereo_params,
                                 calib_params)
            compute_stereo_depth(scene, d435_folder, stereo_params,
                                 calib_params)

# If two images are passed in then just do the magic

# If two folders are passed in then just call:
# read_images_from_folder before doing magic

# If object is passed in:
#
