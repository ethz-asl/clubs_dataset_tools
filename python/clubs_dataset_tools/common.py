"""
Common functions and classes.
"""

import yaml
import numpy as np
import logging as log


class CalibrationParams:
    """
    Camera intrinsic and extrinsic parameters.
    """

    def __init__(self):
        """
        Constructor for the CalibrationParams.
        """

        log.debug("Initialized an empty CalibrationParams class.")

        self.rgb_intrinsics = np.array([])
        self.rgb_distortion_coeffs = np.array([])
        self.hand_eye_transform = np.array([])
        self.depth_intrinsics = np.array([])
        self.depth_distortion_coeffs = np.array([])
        self.depth_extrinsics = np.array([])
        self.ir1_intrinsics = np.array([])
        self.ir1_distortion_coeffs = np.array([])
        self.ir2_intrinsics = np.array([])
        self.ir2_distortion_coeffs = np.array([])
        self.ir_extrinsics = np.array([])
        self.extrinsics_r = np.array([])
        self.extrinsics_t = np.array([])
        self.rgb_width = 0.0
        self.rgb_height = 0.0
        self.depth_width = 0.0
        self.depth_height = 0.0
        self.z_scaling = 0.0
        self.depth_scale_mm = 0.0

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

        raw_rgb_intrinsics = calibration_params['rgb_intrinsics']
        self.rgb_intrinsics = np.array(
            np.reshape(
                raw_rgb_intrinsics['data'],
                (raw_rgb_intrinsics['rows'], raw_rgb_intrinsics['cols'])))
        raw_rgb_distortion_coeffs = calibration_params['rgb_distortion_coeffs']
        self.rgb_distortion_coeffs = np.array(
            raw_rgb_distortion_coeffs['data'])
        raw_hand_eye_transform = calibration_params['hand_eye_transform']
        self.hand_eye_transform = np.array(
            np.reshape(raw_hand_eye_transform['data'],
                       (raw_hand_eye_transform['rows'],
                        raw_hand_eye_transform['cols'])))
        raw_depth_intrinsics = calibration_params['depth_intrinsics']
        self.depth_intrinsics = np.array(
            np.reshape(
                raw_depth_intrinsics['data'],
                (raw_depth_intrinsics['rows'], raw_depth_intrinsics['cols'])))
        raw_depth_distortion_coeffs = calibration_params[
            'depth_distortion_coeffs']
        self.depth_distortion_coeffs = np.array(
            raw_depth_distortion_coeffs['data'])
        raw_depth_extrinsics = calibration_params['depth_extrinsics']
        self.depth_extrinsics = np.array(
            np.reshape(
                raw_depth_extrinsics['data'],
                (raw_depth_extrinsics['rows'], raw_depth_extrinsics['cols'])))
        raw_ir1_intrinsics = calibration_params['ir1_intrinsics']
        self.ir1_intrinsics = np.array(
            np.reshape(
                raw_ir1_intrinsics['data'],
                (raw_ir1_intrinsics['rows'], raw_ir1_intrinsics['cols'])))
        raw_ir1_distortion_coeffs = calibration_params['ir1_distortion_coeffs']
        self.ir1_distortion_coeffs = np.array(
            raw_ir1_distortion_coeffs['data'])
        raw_ir2_intrinsics = calibration_params['ir2_intrinsics']
        self.ir2_intrinsics = np.array(
            np.reshape(
                raw_ir2_intrinsics['data'],
                (raw_ir2_intrinsics['rows'], raw_ir2_intrinsics['cols'])))
        raw_ir2_distortion_coeffs = calibration_params['ir2_distortion_coeffs']
        self.ir2_distortion_coeffs = np.array(
            raw_ir2_distortion_coeffs['data'])
        raw_ir_extrinsics = calibration_params['ir_extrinsics']
        self.ir_extrinsics = np.array(
            np.reshape(raw_ir_extrinsics['data'],
                       (raw_ir_extrinsics['rows'], raw_ir_extrinsics['cols'])))

        inv_extrinsics = np.linalg.inv(self.ir_extrinsics)

        self.extrinsics_r = inv_extrinsics[:3, :3]
        self.extrinsics_t = inv_extrinsics[:3, 3]

        self.rgb_width = calibration_params['rgb_width']
        self.rgb_height = calibration_params['rgb_height']
        self.depth_width = calibration_params['depth_width']
        self.depth_height = calibration_params['depth_height']
        self.z_scaling = calibration_params['z_scaling']
        self.depth_scale_mm = calibration_params['depth_scale_mm']
