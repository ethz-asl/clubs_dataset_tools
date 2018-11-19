"""
Common functions and classes.
"""

import yaml
import numpy as np
import logging as log


class CalibrationParams(object):
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
        self.depth_scale = 0.0

    def read_from_yaml(self, yaml_file):
        """
        Function that reads the calibration parameters from the yaml file.

        Input:
            yaml_file[string] - Path to the yaml file containing the
            calibration parameters.
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
        self.rgb_distortion_coeffs = np.array(raw_rgb_distortion_coeffs['data'])
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
        self.ir1_distortion_coeffs = np.array(raw_ir1_distortion_coeffs['data'])
        raw_ir2_intrinsics = calibration_params['ir2_intrinsics']
        self.ir2_intrinsics = np.array(
            np.reshape(
                raw_ir2_intrinsics['data'],
                (raw_ir2_intrinsics['rows'], raw_ir2_intrinsics['cols'])))
        raw_ir2_distortion_coeffs = calibration_params['ir2_distortion_coeffs']
        self.ir2_distortion_coeffs = np.array(raw_ir2_distortion_coeffs['data'])
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
        self.depth_scale = calibration_params['depth_scale']


class SensorTransformations(object):
    """
    Class containing all the sensor poses with respoect to the Realsense D415.
    """

    def __init__(self):
        """
        Constructor for the SensorTransformations.
        """

        log.debug("Initialized an empty SensorTransformations class.")

        self.d415_rgb = np.array([])
        self.d415_depth = np.array([])
        self.d435_rgb = np.array([])
        self.d435_depth = np.array([])
        self.ps_rgb = np.array([])
        self.ps_depth = np.array([])
        self.cham_rgb = np.array([])

    def read_from_yaml(self, yaml_file):
        """
        Function that reads the sensor transformations from the yaml file.

        Input:
            yaml_file[string] - Path to the yaml file containing the
            sensor transformations.
        """

        log.debug("Initialized SensorTransformations from yaml file: " +
                  yaml_file)

        with open(yaml_file, 'r') as file_pointer:
            transformations = yaml.load(file_pointer)

        raw_d415_rgb = transformations['d415_rgb']
        self.d415_rgb = np.array(
            np.reshape(raw_d415_rgb['data'],
                       (raw_d415_rgb['rows'], raw_d415_rgb['cols'])))
        raw_d415_depth = transformations['d415_depth']
        self.d415_depth = np.array(
            np.reshape(raw_d415_depth['data'],
                       (raw_d415_depth['rows'], raw_d415_depth['cols'])))
        raw_d435_rgb = transformations['d435_rgb']
        self.d435_rgb = np.array(
            np.reshape(raw_d435_rgb['data'],
                       (raw_d435_rgb['rows'], raw_d435_rgb['cols'])))
        raw_d435_depth = transformations['d435_depth']
        self.d435_depth = np.array(
            np.reshape(raw_d435_depth['data'],
                       (raw_d435_depth['rows'], raw_d435_depth['cols'])))
        raw_ps_rgb = transformations['ps_rgb']
        self.ps_rgb = np.array(
            np.reshape(raw_ps_rgb['data'],
                       (raw_ps_rgb['rows'], raw_ps_rgb['cols'])))
        raw_ps_depth = transformations['ps_depth']
        self.ps_depth = np.array(
            np.reshape(raw_ps_depth['data'],
                       (raw_ps_depth['rows'], raw_ps_depth['cols'])))
        raw_cham_rgb = transformations['cham_rgb']
        self.cham_rgb = np.array(
            np.reshape(raw_cham_rgb['data'],
                       (raw_cham_rgb['rows'], raw_cham_rgb['cols'])))


def convert_depth_uint_to_float(uint_depth_image,
                                z_scaling=1.0,
                                depth_scale=1.0):
    """
    Function that converts uint16 depth image to float, also considering the
    depth scale and z_scaling.

    Input:
        uint_depth_image[np.array] - Depth image of type uint16
        z_scaling[float] - correction for z values to correspond to true metric
        values
        depth_scale[float] - Conversion factor for depth (e.g. 1 means that
        value of 1000 in uint16 depth image corresponds to 1.0 in float depth
        image and to 1m in real world)

    Output:
        float_depth_image[np.array] - Depth image of type float
    """

    log.debug(("Converting uint depth to float and applying z_scaling and "
               "depth_scaling"))

    return (uint_depth_image / 1000.0 * depth_scale * z_scaling).astype('float')


def convert_depth_float_to_uint(float_depth_image, depth_scale=1.0):
    """
    Function that converts float depth image to uint16, also considering the
    depth scale.

    Input:
        float_depth_image[np.array] - Depth image of type float
        depth_scale[float] - Conversion factor for depth (e.g. 1 means that
        value of 1000 in uint16 depth image corresponds to 1.0 in float depth
        image and to 1m in real world)

    Output:
        uint_depth_image[np.array] - Depth image of type uint16
    """

    log.debug("Converting float depth to uint16 and applying depth_scaling")

    return (float_depth_image * 1000.0 / depth_scale).astype('uint16')
