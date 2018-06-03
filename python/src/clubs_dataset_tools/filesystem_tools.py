"""Tools for navigating the filesystem to find the proper files."""

import os
import cv2
import libtiff
import logging as log
import glob


def read_images(image_files, image_extension='.png', image_type=cv2.CV_16UC1):
    """
    Function that reads all the images form the folder.

    Input:
        image_files - list containing image paths
        image_extension - extension of the images to be used
        image_type - opencv image type

    Output:
        images - list containing actual images
    """

    images = []

    for file in image_files:
        if file.endswith(image_extension):
            log.debug('Loading ' + file)
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
                log.error("Unknown extension of the image file!")

    if len(images) is 0:
        log.error("No images could be retrieved!")

    return images


def find_images_in_folder(image_folder, image_extension='.png'):
    """
    Function that returns image filenames.

    Input:
        image_folder - path to the image folder
        image_extension - extension of the images to search for

    Output:
        images - list with image filenames
    """

    images = []

    files = os.listdir(image_folder)

    for file in files:
        if file.endswith(image_extension):
            images.append(file)

    return images


def find_ir_image_folders(input_folder):
    """
    Function that returns left and right infrared folders paths, if they
    exist, for all the sensors. Path is relative to the input folder path.

    Input:
        input_folder - path to specific object/box folder

    Output:
        d415_image_folders - realsense d415 ir left and right image folder path
        relative to the input_folder and sensor root folder
        d435_image_folders - realsense d435 ir left and right image folder path
        relative to the input_folder and sensor root folder
    """

    d415_image_folders = []
    d415_ir_l = '/realsense_d415/ir1_images'
    if os.path.isdir(input_folder + d415_ir_l):
        d415_image_folders.append(d415_ir_l)
    d415_ir_r = '/realsense_d415/ir2_images'
    if os.path.isdir(input_folder + d415_ir_r):
        d415_image_folders.append(d415_ir_r)
    d415_image_folders.append('/realsense_d415')
    if len(d415_image_folders) is not 2:
        log.error("D415 ir folders could not be found!")
        d415_image_folders = []

    d435_image_folders = []
    d435_ir_l = '/realsense_d435/ir1_images'
    if os.path.isdir(input_folder + d435_ir_l):
        d435_image_folders.append(d435_ir_l)
    d435_ir_r = '/realsense_d435/ir2_images'
    if os.path.isdir(input_folder + d435_ir_r):
        d435_image_folders.append(d435_ir_r)
    d435_image_folders.append('/realsense_d435')
    if len(d435_image_folders) is not 2:
        log.error("D435 ir folders could not be found!")
        d435_image_folders = []

    return d415_image_folders, d435_image_folders


def find_rgb_d_image_folders(input_folder):
    """
    Function that returns rgb and depth folders paths, if they exist, for all
    the sensors. Path is relative to the input folder path.

    Input:
        input_folder - path to specific object/box folder

    Output:
        ps_image_folders - primesense rgb and depth image folder path relative
        to the input_folder
        d415_image_folders - realsense d415 rgb and depth image folder path
        relative to the input_folder
        d435_image_folders - realsense d435 rgb and depth image folder path
        relative to the input_folder
    """

    ps_image_folders = []
    ps_rgb = '/primesense/rgb_images'
    if os.path.isdir(input_folder + ps_rgb):
        ps_image_folders.append(ps_rgb)
    ps_depth = '/primesense/depth_images'
    if os.path.isdir(input_folder + ps_depth):
        ps_image_folders.append(ps_depth)
    if len(ps_image_folders) is not 2:
        log.error("PS rgb and depth folders could not be found!")
        ps_image_folders = []

    d415_image_folders = []
    d415_rgb = '/realsense_d415/rgb_images'
    if os.path.isdir(input_folder + d415_rgb):
        d415_image_folders.append(d415_rgb)
    d415_depth = '/realsense_d415/depth_images'
    if os.path.isdir(input_folder + d415_depth):
        d415_image_folders.append(d415_depth)
    if len(d415_image_folders) is not 2:
        log.error("D415 rgb and depth folders could not be found!")
        d415_image_folders = []

    d435_image_folders = []
    d435_rgb = '/realsense_d435/rgb_images'
    if os.path.isdir(input_folder + d435_rgb):
        d435_image_folders.append(d435_rgb)
    d435_depth = '/realsense_d435/depth_images'
    if os.path.isdir(input_folder + d435_depth):
        d435_image_folders.append(d435_depth)
    if len(d435_image_folders) is not 2:
        log.error("D435 rgb and depth folders could not be found!")
        d435_image_folders = []

    return ps_image_folders, d415_image_folders, d435_image_folders


def find_all_folders(dataset_folder):
    """
    Function that returns folder names for object and box scenes.

    Input:
        dataset_folder - path to the dataset folder

    Output:
        folders_objects - names of folders containing object scenes
        folders_boxes - names of folders containing box scenes
    """

    folders_objects = glob.glob(dataset_folder + '/object_scenes/[0-9]' * 3 +
                                '_' + '[0-9]' * 1 + '_' + '*')

    folders_boxes = glob.glob(dataset_folder + '/box_scenes/box' + '_' +
                              '[0-9]' * 3 + '_' + '[0-9]' * 3)

    return folders_objects, folders_boxes


def compare_image_names(image_list1, image_list2):
    """
    Function that compares if the two image lists have the same names
    (timestamps).

    Input:
        image_list1 - list of names from the first image set
        image_list2 - list of names from the second image set

    Output:
        timestamps - timestamps found in both image lists
    """

    timestamps_list1 = [image[:10] for image in image_list1]
    timestamps_list2 = [image[:10] for image in image_list2]

    if len(timestamps_list1) != len(timestamps_list2):
        log.error("Two lists do not have equal size!")
        return []

    if len(set(timestamps_list1) & set(timestamps_list2)) == len(
            timestamps_list1):
        return timestamps_list1

    return []


def create_stereo_depth_folder(sensor_folder):
    """
    Function that creates the folder for stereo depth images if it does not
    exist.

    Input:
        sensor_folder - path to the sensor folder

    Output:
        stereo_depth_folder - path to the created stereo depth folder
    """

    stereo_depth_folder = sensor_folder + '/stereo_depth_images'

    if not os.path.exists(stereo_depth_folder):
        os.makedirs(stereo_depth_folder)

    return stereo_depth_folder
