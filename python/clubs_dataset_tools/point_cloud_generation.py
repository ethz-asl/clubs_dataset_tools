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


def convert_depth_uint_to_float(uint_depth_image,
                                z_scaling=1.0,
                                depth_scale_mm=1.0):
    """
    Function that converts uint16 depth image to float, also considering the
    depth scale and z_scaling.

    Input:
        uint_depth_image - depth image of type uint16
        z_scaling - correction for z values to correspond to true metric values
        depth_scale_mm - conversion factor for depth (e.g. 1 means that value
        of 1000 in uint16 depth image corresponds to 1m)

    Output:
        float_depth_image - depth image of type float
    """

    log.debug(("Converting uint depth to float and applying z_scaling and "
               "depth_scaling"))

    return (uint_depth_image / 1000 * depth_scale_mm * z_scaling).astype(float)


def convert_depth_float_to_uint(float_depth_image, depth_scale_mm=1.0):
    """
    Function that converts float depth image to uint16, also considering the
    depth scale.

    Input:
        float_depth_image - depth image of type float
        depth_scale_mm - conversion factor for depth (e.g. 1 means that value
        of 1000 in uint16 depth image corresponds to 1m)

    Output:
        uint_depth_image - depth image of type uint16
    """

    log.debug("Converting float depth to uint16 and applying depth_scaling")

    return (float_depth_image * 1000 / depth_scale_mm).astype('uint16')


def register_depth_image(float_depth_image,
                         rgb_intrinsics,
                         depth_intrinsics,
                         extrinsics,
                         rgb_shape,
                         depth_scale_mm=1.0):
    """

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

    for point in depth_points_in_rgb_frame:
        u = int(fx * point.x / point.z + cx + 0.5)
        v = int(fy * point.y / point.z + cy + 0.5)
        float_depth_registered[u, v] = point.z

    uint_depth_registered = convert_depth_float_to_uint(float_depth_registered)

    return float_depth_registered, uint_depth_registered


def save_colored_point_cloud_to_ply(rgb_image,
                                    depth_image,
                                    rgb_intrinsics,
                                    depth_intrinsics,
                                    extrinsics,
                                    cloud_path,
                                    depth_scale_mm=1.0,
                                    register_depth=False):
    """

    """

    if register_depth:
        log.debug(("Using depth_registered image, therefore the resulting "
                   "point cloud is organized in the order of the rgb image."))

        rgb_rows, rgb_cols, rgb_channels = np.shape(rgb_image)

        float_depth_registered, uint_depth_registered = register_depth_image(
            depth_image, rgb_intrinsics, depth_intrinsics, extrinsics,
            (rgb_rows, rgb_cols), depth_scale_mm)

        depth_points_3d = cv2.rgbd.depthTo3d(float_depth_registered,
                                             rgb_intrinsics)

        n_rows, n_cols, n_coord = np.shape(depth_points_3d)

        with open(cloud_path, 'wb') as ply_file:
            ply_file.write(
                (ply_header % dict(n_points=n_rows * n_cols)).encode('utf-8'))

            for i in range(n_rows):
                for j in range(n_cols):
                    point_x = depth_points_3d[i, j, 0]
                    point_y = depth_points_3d[i, j, 1]
                    point_z = depth_points_3d[i, j, 2]
                    point_r = rgb_image[i, j, 0]
                    point_g = rgb_image[i, j, 1]
                    point_b = rgb_image[i, j, 2]
                    file.write((point_ply % dict(
                        x=point_x,
                        y=point_y,
                        z=point_z,
                        r=point_r,
                        g=point_g,
                        b=point_b)).encode('utf-8'))

            ply_file.write(
                (end_ply % dict(width=n_cols, height=n_rows)).encode('utf-8'))

        return float_depth_registered, uint_depth_registered

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
                    point_x = depth_points_3d[i, j, 0]
                    point_y = depth_points_3d[i, j, 1]
                    point_z = depth_points_3d[i, j, 2]

                    u = int(fx * point_x / point_z + cx + 0.5)
                    v = int(fy * point_y / point_z + cy + 0.5)
                    point_r = rgb_image[u, v, 0]
                    point_g = rgb_image[u, v, 1]
                    point_b = rgb_image[u, v, 2]

                    file.write((point_ply % dict(
                        x=point_x,
                        y=point_y,
                        z=point_z,
                        r=point_r,
                        g=point_g,
                        b=point_b)).encode('utf-8'))

            ply_file.write(
                (end_ply % dict(width=n_cols, height=n_rows)).encode('utf-8'))

    log.debug('Finished writing the file: ' + cloud_path)

    return [], []
