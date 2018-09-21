import cv2
import logging as log
import numpy as np

from clubs_dataset_tools.point_cloud_generation import (
    convert_depth_float_to_uint)


def register_depth_image(float_depth_image,
                         rgb_intrinsics,
                         depth_intrinsics,
                         extrinsics,
                         rgb_shape,
                         depth_scale_mm=1.0):
    """
    Function that registers a depth image to an rgb image. Registered depth
    image has the same size as the original rgb image. Rgb intrinsics are used
    to convert the registered depth image to 3D points.

    Input:
        float_depth_image[np.array] - Float depth image
        rgb_intrinsics[np.array] - Intrinsic parameters of the rgb camera
        depth_intrinsics[np.array] - Intrinsic parameters of the depth camera
        extrinsics[np.array] - Extrinsic parameters between the rgb and the
        depth cameras
        rgb_shape[tuple(int)] - Image size of the rgb image (rows, columns)
        depth_scale_mm[float] - Conversion factor for the depth (e.g. 1 means
        that value of 1000 in uint16 depth image corresponds to 1.0 in float
        depth image and to 1m in real world)

    Output:
        float_depth_registered[np.array] - Depth image registered to rgb, float
        type
        uint_depth_registered[np.array] - Depth image registered to rgb, uint16
        type
    """

    depth_points_3d = cv2.rgbd.depthTo3d(float_depth_image, depth_intrinsics)
    depth_points_in_rgb_frame = cv2.perspectiveTransform(
        depth_points_3d, extrinsics)

    fx = rgb_intrinsics[0, 0]
    fy = rgb_intrinsics[1, 1]
    cx = rgb_intrinsics[0, 2]
    cy = rgb_intrinsics[1, 2]

    float_depth_registered = np.zeros(rgb_shape, dtype='float')

    log.debug("Computing the registered depth image.")

    for points in depth_points_in_rgb_frame:
        for point in points:
            u = int(fx * point[0] / point[2] + cx)
            v = int(fy * point[1] / point[2] + cy)

            height = rgb_shape[0]
            width = rgb_shape[1]
            if (u >= 0 and u < width and v >= 0 and v < height):
                float_depth_registered[v, u] = point[2]

    uint_depth_registered = convert_depth_float_to_uint(float_depth_registered,
                                                        depth_scale_mm)
    kernel = np.ones((3, 3), np.uint16)
    uint_depth_registered = cv2.morphologyEx(uint_depth_registered,
                                             cv2.MORPH_CLOSE, kernel)

    return float_depth_registered, uint_depth_registered
