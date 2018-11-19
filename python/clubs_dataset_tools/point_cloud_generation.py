import cv2
import logging as log
import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(n_points)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element camera 1
property int viewportx
property int viewporty
end_header
'''

point_ply = '%(x)f %(y)f %(z)f %(r)d %(g)d %(b)d\n'

end_ply = '%(width)d %(height)d\n'

DISTANCE_LOWER_LIMIT = 0.0
DISTANCE_UPPER_LIMIT = 5.0


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

    return (
        uint_depth_image / 1000.0 * depth_scale * z_scaling).astype('float')


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


def save_colored_point_cloud_to_ply(rgb_image,
                                    depth_image,
                                    rgb_intrinsics,
                                    rgb_distortion,
                                    depth_intrinsics,
                                    extrinsics,
                                    cloud_path,
                                    use_registered_depth=False):
    """
    Function that registers depth image to rgb image. Some of the points are
    lost due to discretization errors.

    Input:
        rgb_image[np.array] - Input rgb image
        depth_image[np.array] - Input float depth image in m
        rgb_intrinsics[np.array] - Intrinsic parameters of the rgb camera
        rgb_distortion[np.array] - Distortion parameters of the rgb camera
        depth_intrinsics[np.array] - Intrinsic parameters of the depth camera
        extrinsics[np.array] - Extrinsic parameters between rgb and depth
        cameras
        rgb_shape[tuple(int)] - Image size of rgb image (rows, columns)
        cloud_path[string] - Path where to store the point cloud, including
        file name and extension
        use_registered_depth[bool] - If True, registered depth images will be
        used and therefore the resulting point cloud will be organized in the
        order of the rgb image.
    """

    rgb_image = cv2.undistort(rgb_image, rgb_intrinsics, rgb_distortion)

    if use_registered_depth:
        log.debug(("Using depth_registered image, therefore the resulting "
                   "point cloud is organized in the order of the rgb image."
                   "NOTE: Make sure that the input depth_image is "
                   "registered!"))

        depth_points_3d = cv2.rgbd.depthTo3d(depth_image, rgb_intrinsics)

        n_rows, n_cols, n_coord = np.shape(depth_points_3d)

        with open(cloud_path, 'wb') as ply_file:
            ply_file.write(
                (ply_header % dict(n_points=n_rows * n_cols)).encode('utf-8'))

            for i in range(n_rows):
                for j in range(n_cols):
                    point_x = depth_points_3d[i, j, 0]
                    point_y = depth_points_3d[i, j, 1]
                    point_z = depth_points_3d[i, j, 2]

                    point_b = rgb_image[i, j, 0]
                    point_g = rgb_image[i, j, 1]
                    point_r = rgb_image[i, j, 2]

                    if (point_z > DISTANCE_LOWER_LIMIT and
                            point_z < DISTANCE_UPPER_LIMIT):
                        ply_file.write((point_ply % dict(
                            x=point_x,
                            y=point_y,
                            z=point_z,
                            r=point_r,
                            g=point_g,
                            b=point_b)).encode('utf-8'))
                    else:
                        ply_file.write((point_ply % dict(
                            x=0.0, y=0.0, z=0.0, r=0, g=0,
                            b=0)).encode('utf-8'))

            ply_file.write(
                (end_ply % dict(width=n_cols, height=n_rows)).encode('utf-8'))

    else:
        log.debug(("Using unregistered depth image, therefore the resulting "
                   "point cloud is organized in the order of the "
                   "depth image."))

        depth_points_3d = cv2.rgbd.depthTo3d(depth_image, depth_intrinsics)
        depth_points_in_rgb_frame = cv2.perspectiveTransform(
            depth_points_3d, extrinsics)

        n_rows, n_cols, n_coord = np.shape(depth_points_in_rgb_frame)

        fx = rgb_intrinsics[0, 0]
        fy = rgb_intrinsics[1, 1]
        cx = rgb_intrinsics[0, 2]
        cy = rgb_intrinsics[1, 2]

        with open(cloud_path, 'wb') as ply_file:
            ply_file.write(
                (ply_header % dict(n_points=n_rows * n_cols)).encode('utf-8'))

            for i in range(n_rows):
                for j in range(n_cols):
                    point_x = depth_points_in_rgb_frame[i, j, 0]
                    point_y = depth_points_in_rgb_frame[i, j, 1]
                    point_z = depth_points_in_rgb_frame[i, j, 2]

                    height, width, channels = rgb_image.shape
                    if (point_z > DISTANCE_LOWER_LIMIT and
                            point_z < DISTANCE_UPPER_LIMIT):
                        u = int(fx * point_x / point_z + cx)
                        v = int(fy * point_y / point_z + cy)

                        if (u >= 0 and u < width and v >= 0 and v < height):
                            point_b = rgb_image[v, u, 0]
                            point_g = rgb_image[v, u, 1]
                            point_r = rgb_image[v, u, 2]

                            ply_file.write((point_ply % dict(
                                x=point_x,
                                y=point_y,
                                z=point_z,
                                r=point_r,
                                g=point_g,
                                b=point_b)).encode('utf-8'))
                        else:
                            ply_file.write((point_ply % dict(
                                x=0.0, y=0.0, z=0.0, r=0, g=0,
                                b=0)).encode('utf-8'))
                    else:
                        ply_file.write((point_ply % dict(
                            x=0.0, y=0.0, z=0.0, r=0, g=0,
                            b=0)).encode('utf-8'))

            ply_file.write(
                (end_ply % dict(width=n_cols, height=n_rows)).encode('utf-8'))

    log.debug('Finished writing the file: ' + cloud_path)
