#!/usr/bin/env python

import argparse
import cv2
import logging as log

from tqdm import tqdm

from clubs_dataset_tools.filesystem_tools import (
    read_images, find_images_in_folder, find_all_folders,
    find_rgb_d_image_folders, compare_image_names, create_point_cloud_folder,
    create_depth_registered_folder)
from clubs_dataset_tools.common import (CalibrationParams)
from clubs_dataset_tools.point_cloud_generation import (
    convert_depth_uint_to_float, save_register_depth_image,
    save_colored_point_cloud_to_ply)


def generate_point_cloud(scene_folder,
                         sensor_folder,
                         calib_params,
                         save_point_clouds,
                         save_depth_registered,
                         use_stereo_depth=False):
    """
    Function that generates point cloud from RGB and Depth images.

    Input:
        scene_folder - path to the scene folder
        sensor_folder - list containing folder names for rgb and depth
        image, as well as the sensor root folder
        calib_params - calibration parameters from the camera
        (CalibrationParams class)
        use_stereo_depth - if set tu True, stereo depth will be used and
        therefore IR intrinsics instead of depth intrinsics
    """

    images_rgb = find_images_in_folder(scene_folder + sensor_folder[0])
    images_depth = find_images_in_folder(scene_folder + sensor_folder[1])

    timestamps = compare_image_names(images_rgb, images_depth)

    if len(timestamps) != 0:
        image_paths_rgb = [
            scene_folder + sensor_folder[0] + '/' + image_rgb
            for image_rgb in images_rgb
        ]
        image_paths_depth = [
            scene_folder + sensor_folder[1] + '/' + image_depth
            for image_depth in images_depth
        ]

        rgb_images = read_images(
            image_paths_rgb, image_type=cv2.IMREAD_ANYCOLOR)
        depth_images = read_images(image_paths_depth, image_type=cv2.CV_16UC1)

        if save_point_clouds:
            point_cloud_folder = create_point_cloud_folder(
                scene_folder + sensor_folder[2])
        if save_depth_registered:
            depth_registered_folder = create_depth_registered_folder(
                scene_folder + sensor_folder[2])

        stereo_bar = tqdm(
            total=len(rgb_images), desc="PointCloud/DepthRegistered progress")
        for i in range(len(rgb_images)):
            float_depth_image = convert_depth_uint_to_float(
                depth_images[i], calib_params.z_scaling,
                calib_params.depth_scale_mm)
            if save_depth_registered:
                depth_registerd_path = (
                    depth_registered_folder + '/' + timestamps[i] +
                    '_depth_registered_image.png')
                if use_stereo_depth:
                    float_depth_reg, uint_depth_reg = save_register_depth_image(
                        float_depth_image, calib_params.rgb_intrinsics,
                        calib_params.ir1_intrinsics,
                        calib_params.depth_extrinsics,
                        (calib_params.rgb_height, calib_params.rgb_width),
                        depth_registerd_path, calib_params.depth_scale_mm)
                else:
                    float_depth_reg, uint_depth_reg = save_register_depth_image(
                        float_depth_image, calib_params.rgb_intrinsics,
                        calib_params.depth_intrinsics,
                        calib_params.depth_extrinsics,
                        (calib_params.rgb_height, calib_params.rgb_width),
                        depth_registerd_path, calib_params.depth_scale_mm)

            if save_point_clouds:
                point_cloud_path = (point_cloud_folder + '/' + timestamps[i] +
                                    '_point_cloud.ply')
                if use_stereo_depth:
                    save_colored_point_cloud_to_ply(
                        rgb_images[i],
                        float_depth_image,
                        calib_params.rgb_intrinsics,
                        calib_params.rgb_distortion_coeffs,
                        calib_params.ir1_intrinsics,
                        calib_params.depth_extrinsics,
                        point_cloud_path,
                        calib_params.depth_scale_mm,
                        register_depth=False)
                else:
                    save_colored_point_cloud_to_ply(
                        rgb_images[i],
                        float_depth_image,
                        calib_params.rgb_intrinsics,
                        calib_params.rgb_distortion_coeffs,
                        calib_params.depth_intrinsics,
                        calib_params.depth_extrinsics,
                        point_cloud_path,
                        calib_params.depth_scale_mm,
                        register_depth=False)
            stereo_bar.update()
        stereo_bar.close()
    else:
        log.error("\nImage names are not consistent for rgb and depth image.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Generate point clouds from RGB and depth images. There are two "
            "different ways this function can be called. First one is by "
            "passing in the dataset root folder (flag --dataset_folder) which "
            "will create a new folder for each object/scene and each sensor "
            "(d415 and d435), containing the depth image obtained through "
            "stereo matching. Second way is to pass object/box scene root "
            "folder (flag --scene_folder) which will do the same for that "
            "specific scene."))
    parser.add_argument(
        '--dataset_folder', type=str, help="Path to the dataset root folder.")
    parser.add_argument(
        '--scene_folder', type=str, help="Path to the scene root folder.")
    parser.add_argument(
        '--ps_calib_file',
        type=str,
        default='config/primesense.yaml',
        help=("Path to Primesense calibration yaml file. By default: "
              "config/primesense.yaml"))
    parser.add_argument(
        '--d415_calib_file',
        type=str,
        default='config/realsense_hd_d415.yaml',
        help=("Path to RealSense D415 calibration yaml file. By default: "
              "config/realsense_hd_d415.yaml"))
    parser.add_argument(
        '--d435_calib_file',
        type=str,
        default='config/realsense_hd_d435.yaml',
        help=("Path to RealSense D435 calibration yaml file. By default: "
              "config/realsense_hd_d435.yaml"))
    parser.add_argument(
        '--use_only_boxes',
        action='store_true',
        help=("If this flag is set, depth from stereo will only be computed "
              "for the box scenes."))
    parser.add_argument(
        '--use_stereo_depth',
        action='store_true',
        help=("If this flag is set, depth from stereo will be used "
              "for cloud generation"))
    parser.add_argument(
        '--save_depth_registered',
        action='store_true',
        help="If this flag is set, registered depth will be saved.")
    parser.add_argument(
        '--save_point_clouds',
        action='store_true',
        help="If this flag is set, point clouds will be saved.")
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
            total=len(used_scenes) * 3, desc="Overall Progress")
        for i in range(len(used_scenes)):
            scene = used_scenes[i]
            log.debug("Processing " + str(scene))

            ps_folder, d415_folder, d435_folder = find_rgb_d_image_folders(
                scene, args.use_stereo_depth)

            calib_params.read_from_yaml(args.ps_calib_file)
            if ps_folder != []:
                generate_point_cloud(scene, ps_folder, calib_params,
                                     args.save_point_clouds,
                                     args.save_depth_registered)
            progress_bar.update()

            calib_params.read_from_yaml(args.d415_calib_file)
            if d415_folder != []:
                generate_point_cloud(
                    scene, d415_folder, calib_params, args.save_point_clouds,
                    args.save_depth_registered, args.use_stereo_depth)
            progress_bar.update()

            calib_params.read_from_yaml(args.d435_calib_file)
            if d435_folder != []:
                generate_point_cloud(
                    scene, d435_folder, calib_params, args.save_point_clouds,
                    args.save_depth_registered, args.use_stereo_depth)
            progress_bar.update()
        progress_bar.close()
    elif args.scene_folder is not None:
        scene = args.scene_folder
        log.debug("Processing single scene " + str(scene))

        ps_folder, d415_folder, d435_folder = find_rgb_d_image_folders(
            scene, args.use_stereo_depth)

        calib_params.read_from_yaml(args.ps_calib_file)
        if ps_folder != []:
            generate_point_cloud(scene, ps_folder, calib_params,
                                 args.save_point_clouds,
                                 args.save_depth_registered)

        calib_params.read_from_yaml(args.d415_calib_file)
        if d415_folder != []:
            generate_point_cloud(
                scene, d415_folder, calib_params, args.save_point_clouds,
                args.save_depth_registered, args.use_stereo_depth)

        calib_params.read_from_yaml(args.d435_calib_file)
        if d435_folder != []:
            generate_point_cloud(
                scene, d435_folder, calib_params, args.save_point_clouds,
                args.save_depth_registered, args.use_stereo_depth)
    else:
        parser.print_help()
