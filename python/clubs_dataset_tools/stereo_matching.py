"""Tools for stereo rectification and stereo matching."""

import yaml
import numpy as np
import cv2
import logging as log


class StereoMatchingParams:
    """
    Stereo matching algorithm parameters.
    """

    def __init__(self):
        """
        Constructor for StereoMatchingParams.
        """

        log.debug("Initialized StereoMatchingParams with default values.")

        self.min_disparity = 15
        self.num_disparities = 272
        self.block_size = 25
        self.p1 = 150
        self.p2 = 520
        self.disp_12_max_diff = -1
        self.pre_filter_cap = 10
        self.uniqueness_ratio = 30
        self.speckle_window_size = 2048
        self.speckle_range = 1
        self.mode = cv2.StereoSGBM_MODE_HH4
        self.apply_bilateral_filter = False
        self.bilateral_filter_size = 21
        self.bilateral_filter_sigma = 30
        self.apply_wls_filter = False
        self.wls_filter_sigma_color = 0.8
        self.wls_filter_lambda = 100

    def read_from_yaml(self, yaml_file):
        """
        Function that reads the stereo parameters from the yaml file.

        Input:
            yaml_file - path to the yaml file containing the stereo parameters.
        """

        log.debug("Initialized StereoMatchingParams from yaml file: " +
                  yaml_file)

        with open(yaml_file, 'r') as file_pointer:
            stereo_params = yaml.load(file_pointer)

        self.min_disparity = stereo_params['min_disparity']
        self.num_disparities = stereo_params['num_disparities']
        self.block_size = stereo_params['block_size']
        self.p1 = stereo_params['p1']
        self.p2 = stereo_params['p2']
        self.disp_12_max_diff = stereo_params['disp_12_max_diff']
        self.pre_filter_cap = stereo_params['pre_filter_cap']
        self.uniqueness_ratio = stereo_params['uniqueness_ratio']
        self.speckle_window_size = stereo_params['speckle_window_size']
        self.speckle_range = stereo_params['speckle_range']
        self.mode = stereo_params['mode']
        self.apply_bilateral_filter = stereo_params['apply_bilateral_filter']
        self.bilateral_filter_size = stereo_params['bilateral_filter_size']
        self.bilateral_filter_sigma = stereo_params['bilateral_filter_sigma']
        self.apply_wls_filter = stereo_params['apply_wls_filter']
        self.wls_filter_sigma_color = stereo_params['wls_filter_sigma_color']
        self.wls_filter_lambda = stereo_params['wls_filter_lambda']


def rectify_images(image_l, camera_matrix_l, dist_coeffs_l, image_r,
                   camera_matrix_r, dist_coeffs_r, extrinsics_r, extrinsics_t):
    """
    Function that takes in left and right image, together with their
    intrinsic and extrinsic parameters and returns rectified and
    undistorted images and Q matrix that maps disparity image to 3D.

    Input:
        image_l - left image
        camera_matrix_l - 3x3 calibration matrix of the left image
        dist_coeffs_l - distortion coefficients of the left image
        image_r - right image
        camera_matrix_r - 3x3 calibration matrix of the right image
        dist_coeffs_r - distortion coefficients of the right image
        extrinsics_r - extrinsics 3x3 rotation matrix
        extrinsics_t - extrinsics translation vector

    Output:
        rectified_l - rectified and undistorted left image
        rectified_r - rectified and undistorted right image
        Q - 4x4 perspective transformation matrix:
            [X Y Z W]^t = Q * [u v disp(u,v) 1]^t
        RL - 3x3 rotation matrix of the rectified left image
        PL - 3x4 new calibration matrix for left image
    """

    RL, RR, PL, PR, Q, valid_pix_ROI_l, valid_pix_ROI_r = cv2.stereoRectify(
        camera_matrix_l,
        dist_coeffs_l,
        camera_matrix_r,
        dist_coeffs_r,
        image_l.shape[::-1],
        extrinsics_r,
        extrinsics_t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1)

    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        camera_matrix_l, dist_coeffs_l, RL, PL, image_l.shape[::-1],
        cv2.CV_16SC2)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        camera_matrix_r, dist_coeffs_r, RR, PR, image_r.shape[::-1],
        cv2.CV_16SC2)

    rectified_l = cv2.remap(image_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(image_r, map_r1, map_r2, cv2.INTER_LINEAR)

    log.debug("Rectification results: ")
    log.debug("RL:\n" + str(RL))
    log.debug("RR:\n" + str(RR))
    log.debug("PL:\n" + str(PL))
    log.debug("PR:\n" + str(PR))
    log.debug("Q:\n" + str(Q))
    log.debug("valid_pix_ROI_l:\n" + str(valid_pix_ROI_l))
    log.debug("valid_pix_ROI_r:\n" + str(valid_pix_ROI_r))

    return rectified_l, rectified_r, Q, RL, PL


def stereo_match(undistorted_rectified_l,
                 undistorted_rectified_r,
                 baseline,
                 focal_length,
                 stereo_params,
                 scale=1000):
    """
    Function that performs stereo matching using semi global block
    matcher (SGBM), to obtain disparity map.

    Input:
        undistorted_rectified_l - undistorted and rectified left image
        undistorted_rectified_r - undistorted and rectified right image
        baseline - baseline between the left and right image in m
        focal_length - focal length of the left and right camera (cameras with
        different focal length are not supported)
        stereo_params - StereoMatchingParams class containing parameters for
        the SGBM algorithm
        scale - scaling used to convert to uint depth image (1000 for
        converting m to mm)

    Output:
        depth_uint - depth image represented as a 16-bit uint image
        depth_float - depth image represented as a 32-bit float image
        disparity_float - disparity image represented as a 32-bit float image
    """

    stereo_matcher = cv2.StereoSGBM_create(
        stereo_params.min_disparity, stereo_params.num_disparities,
        stereo_params.block_size, stereo_params.p1, stereo_params.p2,
        stereo_params.disp_12_max_diff, stereo_params.pre_filter_cap,
        stereo_params.uniqueness_ratio, stereo_params.speckle_window_size,
        stereo_params.speckle_range, stereo_params.mode)

    if undistorted_rectified_l.dtype == 'uint8':
        uint8_undistorted_rectified_l = undistorted_rectified_l
    elif undistorted_rectified_l.dtype == 'uint16':
        uint8_undistorted_rectified_l = (
            undistorted_rectified_l / 255).astype('uint8')
    else:
        log.error("\nUnknown image type!")
        return

    if undistorted_rectified_r.dtype == 'uint8':
        uint8_undistorted_rectified_r = undistorted_rectified_r
    elif undistorted_rectified_r.dtype == 'uint16':
        uint8_undistorted_rectified_r = (
            undistorted_rectified_r / 255).astype('uint8')
    else:
        log.error("\nUnknown image type!")
        return

    log.debug("Performing stereo matching.")
    disparity = stereo_matcher.compute(uint8_undistorted_rectified_l,
                                       uint8_undistorted_rectified_r)

    if stereo_params.apply_wls_filter:
        right_macher = cv2.StereoSGBM_create(
            -(stereo_params.min_disparity + stereo_params.num_disparities) + 1,
            stereo_params.num_disparities, stereo_params.block_size,
            stereo_params.p1, stereo_params.p2, stereo_params.disp_12_max_diff,
            stereo_params.pre_filter_cap, stereo_params.uniqueness_ratio,
            stereo_params.speckle_window_size, stereo_params.speckle_range,
            stereo_params.mode)
        disparity_right = right_macher.compute(uint8_undistorted_rectified_r,
                                               uint8_undistorted_rectified_l)

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_matcher)
        wls_filter.setLambda(stereo_params.wls_filter_lambda)
        wls_filter.setSigmaColor(stereo_params.wls_filter_sigma_color)
        filtered_image = wls_filter.filter(
            disparity, uint8_undistorted_rectified_l, None, disparity_right,
            None, uint8_undistorted_rectified_r)
    else:
        filtered_image = disparity

    disparity_float = -filtered_image.astype(np.float32) / 16.0
    if stereo_params.apply_bilateral_filter:
        disparity_float = cv2.bilateralFilter(
            disparity_float, stereo_params.bilateral_filter_size,
            stereo_params.bilateral_filter_sigma,
            stereo_params.bilateral_filter_sigma)

    depth_float = baseline * focal_length / disparity_float

    depth_float[depth_float <= 0] = float('nan')
    depth_float[depth_float == np.amax(depth_float)] = float('nan')

    depth_uint = depth_float * scale
    depth_uint = depth_uint.astype(np.uint16)

    return depth_uint, depth_float, disparity_float
