"""Tools for navigating the filesystem to find the proper files."""

import os
import cv2
import libtiff
import logging as log
import glob


def read_images(image_files, image_extension='.png', image_type=cv2.CV_16UC1):
    """
    Function that reads all the images from a list of file paths.

    Input:
        image_files - List containing absolute image full paths
        image_extension - Extension of the images to be used
        image_type - Opencv image type

    Output:
        images - List of numpy arrays containing the images
    """

    images = []

    for file in image_files:
        if file.endswith(image_extension):
            log.debug("Loading " + file)
            if image_extension == '.png' or image_extension == '.jpg':
                image = cv2.imread(file, image_type)
                if image is not None:
                    images.append(image)
            elif image_extension == '.tiff':
                tiff = libtiff.TIFF.open(file, mode='r')
                image = tiff.read_image()
                tiff.close()
                images.append(image)
            else:
                log.error("\nUnknown extension of the image file!")

    if len(images) is 0:
        log.error("\nNo images could be retrieved!\n" +
                  "List containing image full paths:\n" + str(image_files))

    return images


def find_images_in_folder(image_folder, image_extension='.png'):
    """
    Function that returns image filenames.

    Input:
        image_folder - Path to the image folder
        image_extension - Extension of the images to search for

    Output:
        images - List with image filenames
    """

    images = []

    files = os.listdir(image_folder)

    for file in files:
        log.debug("Found " + file)
        if file.endswith(image_extension):
            log.debug(
                "It has the right extension, adding it to the image list")
            images.append(file)

    return sorted(images)


def find_ir_image_folders(input_folder):
    """
    Function that returns left and right infrared folders paths, if they
    exist, for all the sensors. Path is relative to the input folder path.

    Input:
        input_folder - Path to specific object/box folder

    Output:
        d415_image_folders - Realsense d415 root, ir left and right image
        folder path relative to the input_folder and sensor root folder
        d435_image_folders - Realsense d435 root, ir left and right image
        folder path relative to the input_folder and sensor root folder
    """
    expected_number_of_folders = 3

    d415_image_folders = []
    d415_ir_l = '/realsense_d415/ir1_images'
    if os.path.isdir(input_folder + d415_ir_l):
        d415_image_folders.append(d415_ir_l)
    d415_ir_r = '/realsense_d415/ir2_images'
    if os.path.isdir(input_folder + d415_ir_r):
        d415_image_folders.append(d415_ir_r)
    d415_image_folders.append('/realsense_d415')
    if len(d415_image_folders) is not expected_number_of_folders:
        log.error("\nD415 ir folders could not be found!\n" +
                  "Looking for:\n" + str(input_folder + d415_ir_l) + "\n" +
                  str(input_folder + d415_ir_r))
        d415_image_folders = []
    log.debug("Found d415 folders: \n" + str(d415_image_folders))

    d435_image_folders = []
    d435_ir_l = '/realsense_d435/ir1_images'
    if os.path.isdir(input_folder + d435_ir_l):
        d435_image_folders.append(d435_ir_l)
    d435_ir_r = '/realsense_d435/ir2_images'
    if os.path.isdir(input_folder + d435_ir_r):
        d435_image_folders.append(d435_ir_r)
    d435_image_folders.append('/realsense_d435')
    if len(d435_image_folders) is not expected_number_of_folders:
        log.error("\nD435 ir folders could not be found!\n" +
                  "Looking for:\n" + str(input_folder + d435_ir_l) + "\n" +
                  str(input_folder + d435_ir_r))
        d435_image_folders = []
    log.debug("Found d435 folders: \n" + str(d435_image_folders))

    return d415_image_folders, d435_image_folders


def find_rgb_d_image_folders(input_folder, use_stereo_depth=False):
    """
    Function that returns rgb and depth folders paths, if they exist, for all
    the sensors. Path is relative to the input folder path.

    Input:
        input_folder - Path to specific object/box folder
        use_stereo_depth - If True, depth from stereo will be used

    Output:
        ps_image_folders - Primesense root, rgb and depth image folder path
        relative to the input_folder
        d415_image_folders - Realsense d415 root, rgb and depth image folder
        path relative to the input_folder
        d435_image_folders - Realsense d435 root, rgb and depth image folder
        path relative to the input_folder
    """
    expected_number_of_folders = 3

    ps_image_folders = []
    ps_rgb = '/primesense/rgb_images'
    if os.path.isdir(input_folder + ps_rgb):
        ps_image_folders.append(ps_rgb)
    ps_depth = '/primesense/depth_images'
    if os.path.isdir(input_folder + ps_depth):
        ps_image_folders.append(ps_depth)
    ps_image_folders.append('/primesense')
    if len(ps_image_folders) is not expected_number_of_folders:
        log.error("\nPS rgb and depth folders could not be found!\n" +
                  "Looking for:\n" + str(input_folder + ps_rgb) + "\n" +
                  str(input_folder + ps_depth))
        ps_image_folders = []
    log.debug("Found ps folders: \n" + str(ps_image_folders))

    d415_image_folders = []
    d415_rgb = '/realsense_d415/rgb_images'
    if os.path.isdir(input_folder + d415_rgb):
        d415_image_folders.append(d415_rgb)
    if use_stereo_depth:
        d415_depth = '/realsense_d415/stereo_depth_images'
    else:
        d415_depth = '/realsense_d415/depth_images'
    if os.path.isdir(input_folder + d415_depth):
        d415_image_folders.append(d415_depth)
    d415_image_folders.append('/realsense_d415')
    if len(d415_image_folders) is not expected_number_of_folders:
        log.error("\nD415 rgb and depth folders could not be found!\n" +
                  "Looking for:\n" + str(input_folder + d415_rgb) + "\n" +
                  str(input_folder + d415_depth))
        d415_image_folders = []
    log.debug("Found d415 folders: \n" + str(d415_image_folders))

    d435_image_folders = []
    d435_rgb = '/realsense_d435/rgb_images'
    if os.path.isdir(input_folder + d435_rgb):
        d435_image_folders.append(d435_rgb)
    if use_stereo_depth:
        d435_depth = '/realsense_d435/stereo_depth_images'
    else:
        d435_depth = '/realsense_d435/depth_images'
    if os.path.isdir(input_folder + d435_depth):
        d435_image_folders.append(d435_depth)
    d435_image_folders.append('/realsense_d435')
    if len(d435_image_folders) is not expected_number_of_folders:
        log.error("\nD435 rgb and depth folders could not be found!" +
                  "Looking for:\n" + str(input_folder + d435_rgb) + "\n" +
                  str(input_folder + d435_depth))
        d435_image_folders = []
    log.debug("Found d435 folders: \n" + str(d435_image_folders))

    return ps_image_folders, d415_image_folders, d435_image_folders


def find_all_folders(dataset_folder):
    """
    Function that returns folder names for object and box scenes.

    Input:
        dataset_folder - Path to the dataset folder

    Output:
        folders_objects - Paths of folders containing object scenes
        folders_boxes - Paths of folders containing box scenes
    """

    folders_objects = glob.glob(dataset_folder + '/object_scenes/' +
                                '[0-9]' * 3 + '_' + '[0-9]' * 1 + '_' + '*')
    log.debug("Found following object scene folders: \n" +
              str(folders_objects))

    folders_boxes = glob.glob(dataset_folder + '/box_scenes/box' + '_' +
                              '[0-9]' * 3 + '_' + '[0-9]' * 3)
    log.debug("Found following box scene folders: \n" + str(folders_boxes))

    return folders_objects, folders_boxes


def compare_image_names(image_list1, image_list2):
    """
    Function that compares if the two image lists have the same names
    (timestamps).

    Input:
        image_list1 - List of names from the first image set
        image_list2 - List of names from the second image set

    Output:
        timestamps - Timestamps found in both image lists
    """

    timestamps_list1 = [image[:10] for image in image_list1]
    timestamps_list2 = [image[:10] for image in image_list2]

    if len(set(timestamps_list1) & set(timestamps_list2)) == len(
            timestamps_list1):
        log.debug("Timestamp lists are equal.")
        return timestamps_list1

    log.error("\nTwo timestamp lists are not equal: " +
              str(len(timestamps_list1)) + " vs " + str(len(timestamps_list2)))

    return []


def create_stereo_depth_folder(sensor_folder):
    """
    Function that creates the folder for stereo depth images if it does not
    exist.

    Input:
        sensor_folder - Path to the sensor folder

    Output:
        stereo_depth_folder - Path to the created stereo depth folder
    """

    stereo_depth_folder = sensor_folder + '/stereo_depth_images'

    if not os.path.exists(stereo_depth_folder):
        os.makedirs(stereo_depth_folder)

    log.debug("Created a new stereo depth folder: \n" + stereo_depth_folder)

    return stereo_depth_folder


def create_point_cloud_folder(sensor_folder):
    """
    Function that creates the folder for point clouds if it does not exist.

    Input:
        sensor_folder - Path to the sensor folder

    Output:
        point_cloud_folder - Path to the created point_cloud folder
    """

    point_cloud_folder = sensor_folder + '/point_clouds'

    if not os.path.exists(point_cloud_folder):
        os.makedirs(point_cloud_folder)

    log.debug("Created a new point_clouds folder: \n" + point_cloud_folder)

    return point_cloud_folder


def create_depth_registered_folder(sensor_folder):
    """
    Function that creates the folder for registered depth images if it does
    not exist.

    Input:
        sensor_folder - Path to the sensor folder

    Output:
        depth_registered_folder - Path to the created point_cloud folder
    """

    depth_registered_folder = sensor_folder + '/registered_depth_images'

    if not os.path.exists(depth_registered_folder):
        os.makedirs(depth_registered_folder)

    log.debug("Created a new depth_registered_images folder: \n" +
              depth_registered_folder)

    return depth_registered_folder
