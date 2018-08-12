"""Tools for stereo rectification and stereo matching."""

import yaml
import numpy as np
import cv2
import logging as log


class CalibrationParams:
    """
    Camera intrinsic and extrinsic parameters.
    """

    def __init__(self):
        """
        Constructor for CalibrationParams.
        """

        log.debug("Initialized CalibrationParams with default values.")

        self.camera_matrix_l = np.array([[1387.426, 0.000,
                                          969.672], [0.000, 1386.698, 559.111],
                                         [0.000, 0.000, 1.000]])
        self.dist_coeffs_l = np.array([
            0.126991973128, -0.351631362871, 0.000823677750, 0.000733769806,
            0.250226895328
        ])
        self.camera_matrix_r = np.array([[1387.668, 0.000,
                                          953.792], [0.000, 1387.664, 553.310],
                                         [0.000, 0.000, 1.000]])
        self.dist_coeffs_r = np.array([
            0.128268524583, -0.368150717004, -0.000893114163, 0.001179459198,
            0.284531882325
        ])
        self.extrinsics_r = np.array(
            [[0.999986130, 0.00125724064,
              -0.00511359975], [-0.00123899938, 0.999992869, 0.00356868523],
             [0.00511805018, -0.00356230478, 0.999980555]])
        self.extrinsics_t = np.array(
            [-0.0550792390, -0.0000461457433, -0.000494267796])

    def read_from_yaml(self, yaml_file):
        """
        Function that reads the calibration parameters from the yaml file.

        Input:
            yaml_file - path to the yaml file containing the calibration
            parameters.
        """

        log.debug("Initialized CalibrationParams from yaml file: " + yaml_file)

        with open(yaml_file, 'r') as file_pointer:
            calibration_params = yaml.load(file_pointer)

        self.camera_matrix_l = np.array(calibration_params['camera_matrix_l'])
        self.dist_coeffs_l = np.array(calibration_params['dist_coeffs_l'])
        self.camera_matrix_r = np.array(calibration_params['camera_matrix_r'])
        self.dist_coeffs_r = np.array(calibration_params['dist_coeffs_r'])
        self.extrinsics_r = np.array(calibration_params['extrinsics_r'])
        self.extrinsics_t = np.array(calibration_params['extrinsics_t'])


class StereoMatchingParams:
    """
    Stereo matching algorithm parameters.
    """

    def __init__(self):
        """
        Constructor for StereoMatchingParams.
        """

        log.debug("Initialized StereoMatchingParams with default values.")

        self.min_disparity = 39
        self.num_disparities = 272
        self.block_size = 17
        self.p1 = 201
        self.p2 = 948
        self.disp_12_max_diff = -1
        self.pre_filter_cap = 9
        self.uniqueness_ratio = 19
        self.speckle_window_size = 2048
        self.speckle_range = 1
        self.mode = cv2.StereoSGBM_MODE_HH4

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

    return rectified_l, rectified_r, Q


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

    disparity_float = disparity.astype(np.float32) / 16.0

    depth_float = baseline * focal_length / disparity_float

    depth_uint = depth_float * scale
    depth_uint = depth_uint.astype(np.uint16)

    return depth_uint, depth_float, disparity_float
