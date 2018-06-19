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

        log.debug("Initialized CalibrationParams with default values.")

        self.rgb_intrinsics = np.array(
            [[1374.858, 0.000, 948.552],
             [0.000, 1373.845, 541.477],
             [0.000, 0.000, 1.000]])
        self.rgb_distortion_coeffs = np.array([
            0.135775752283, -0.418056082917, 0.000556837458, 0.000123527682,
            0.332401963670])
        self.hand_eye_transform = np.array([[
            -0.000529077278, 0.024926402044, 0.999689148965, 0.047041508000],
            [-0.001915414986, -0.999687480304, 0.024925346720, 0.021963094613],
            [0.999998025629, -0.001901632142, 0.000576656335, -0.034869255532],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])
        self.depth_intrinsics = np.array(
            [[946.041, 0.000, 633.883],
             [0.000, 946.041, 370.771],
             [0.000, 0.000, 1.000]])
        self.depth_distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.depth_extrinsics = np.array([[
            0.999987510895, 0.003531992676, 0.003535969669, 0.014944465045],
            [-0.003538921961, 0.999991826342, 0.001955320976, -0.000049816138],
            [-0.003529034587, -0.001967810077, 0.999991836786, 0.000086832557],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])
        self.ir1_intrinsics = np.array([[1387.426, 0.000, 969.672],
                                        [0.000, 1386.698, 559.111],
                                        [0.000, 0.000, 1.000]])
        self.ir1_distortion_coeffs = np.array([
            0.126991973128, -0.351631362871, 0.000823677750, 0.000733769806,
            0.250226895328])
        self.ir2_intrinsics = np.array([[1387.668, 0.000, 953.792],
                                        [0.000, 1387.664, 553.310],
                                        [0.000, 0.000, 1.000]])
        self.ir2_distortion_coeffs = np.array([
            0.128268524583, -0.368150717004, -0.000893114163, 0.001179459198,
            0.284531882325])
        self.ir_extrinsics = np.array([[
            0.999986135126, -0.001239001799, 0.005118049425, 0.055080948077],
            [0.001257242440, 0.999992864639, -0.003562304364, 0.000113632506],
            [-0.005113599205, 0.003568689602, 0.999980557590, 0.000212769671],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])
        self.extrinsics_r = np.array(
            [[0.999986130, 0.00125724064, -0.00511359975],
             [-0.00123899938, 0.999992869, 0.00356868523],
             [0.00511805018, -0.00356230478, 0.999980555]])
        self.extrinsics_t = np.array(
            [-0.0550792390, -0.0000461457433, -0.000494267796])
        self.rgb_width = 1920
        self.rgb_height = 1080
        self.depth_width = 1280
        self.depth_height = 720
        self.z_scaling = 1.000000
        self.depth_scale_mm = 0.100000

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

        self.rgb_intrinsics = np.array(calibration_params['rgb_intrinsics'])
        self.rgb_distortion_coeffs = np.array(
            calibration_params['rgb_distortion_coeffs'])
        self.hand_eye_transform = np.array(
            calibration_params['hand_eye_transform'])
        self.depth_intrinsics = np.array(
            calibration_params['depth_intrinsics'])
        self.depth_distortion_coeffs = np.array(
            calibration_params['depth_distortion_coeffs'])
        self.depth_extrinsics = np.array(
            calibration_params['depth_extrinsics'])
        self.ir1_intrinsics = np.array(calibration_params['ir1_intrinsics'])
        self.ir1_distortion_coeffs = np.array(
            calibration_params['ir1_distortion_coeffs'])
        self.ir2_intrinsics = np.array(calibration_params['ir2_intrinsics'])
        self.ir2_distortion_coeffs = np.array(
            calibration_params['ir2_distortion_coeffs'])
        self.ir_extrinsics = np.array(calibration_params['ir_extrinsics'])

        inv_extrinsics = np.linalg.inv(self.ir_extrinsics)

        self.extrinsics_r = inv_extrinsics[:3, :3]
        self.extrinsics_t = inv_extrinsics[:3, 3]

        self.rgb_width = calibration_params['rgb_width']
        self.rgb_height = calibration_params['rgb_height']
        self.depth_width = calibration_params['depth_width']
        self.depth_height = calibration_params['depth_height']
        self.z_scaling = calibration_params['z_scaling']
        self.depth_scale_mm = calibration_params['depth_scale_mm']
