#!/usr/bin/env python

import argparse
import os
import urllib
from zipfile import ZipFile

from tqdm import tqdm

DATASET_URL = 'http://robotics.ethz.ch/~asl-datasets/ijrr_2018_clubs_dataset/'
OBJECTS = 'object_scenes'
BOXES = 'box_scenes'


def get_scene_list(scene_type, dataset_folder):
        """
        Function that fetches the list of object or box scene ids.

        Input:
            scene_type     - type of the scenes to list (OBJECTS or BOXES)
            dataset_folder - path to the dataset root folder

        Output:
            scenes         - list of scene ids
        """
    scene_list_url = DATASET_URL + scene_type + '.txt'
    scene_list_file = os.path.join(dataset_folder, scene_type,
                                   scene_type + '.txt')
    scene_list_dir = os.path.dirname(scene_list_file)
    if not os.path.isfile(scene_list_file):
        if not os.path.exists(scene_list_dir):
            os.makedirs(scene_list_dir)
        urllib.urlretrieve(scene_list_url, filename=scene_list_file)
    f = open(scene_list_file)
    scenes = f.read().splitlines()
    return scenes


class ProgressBar(tqdm):
    """
    Download progress bar based on tqdm.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_scene(scene_id, scene_type, dataset_folder):
    """
    Function that fetches the list of object or box scene ids.

    Input:
        scene_id       - id of the object or box scene to download
        scene_type     - type of the scene to download (OBJECTS or BOXES)
        dataset_folder - path to the dataset root folder
    """
    scene_url = DATASET_URL + scene_type + '/' + scene_id + '.zip'
    scene_dir = os.path.join(dataset_folder, scene_type, scene_id)
    scene_zipfile = os.path.abspath(scene_dir + '.zip')
    if not os.path.isdir(scene_dir):
        with ProgressBar(
                ncols=80,
                unit='B',
                unit_scale=True,
                miniters=1,
                desc=scene_url.split('/')[-1]) as bar:
            urllib.urlretrieve(
                scene_url, filename=scene_zipfile, reporthook=bar.update_to)
        try:
            f = ZipFile(scene_zipfile)
            f.extractall(scene_dir)
            f.close()
        finally:
            os.remove(scene_zipfile)
    else:
        print('WARNING: skipping download of existing scene ' + scene_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        ("Download the CLUBS dataset. Provide a specific object or box scene id, or download ALL_OBJECT scenes, ALL_BOXES scenes or ALL object and box scenes. The list of object and box scene ids can be printed with the --list_scenes flag."
         ))
    parser.add_argument(
        '--dataset_folder',
        type=str,
        default='',
        help="Path to the CLUBS download directory.")
    parser.add_argument(
        '--scene',
        type=str,
        default='ALL',
        help=
        "Specific object or box scene id to download, ALL to download all scenes, ALL_OBJECTS to download all the object scenes or ALL_BOXES to download all the box scenes."
    )
    parser.add_argument(
        '--list_scenes',
        action='store_true',
        help="List all object and box scene ids.")
    args = parser.parse_args()

    object_scenes = get_scene_list(OBJECTS, args.dataset_folder)
    box_scenes = get_scene_list(BOXES, args.dataset_folder)

    if args.list_scenes:
        print('\nOBJECT SCENES: ')
        print('\n'.join(object_scenes))
        print('\nBOX SCENES: ')
        print('\n'.join(box_scenes))
    elif args.scene.lower() == 'ALL'.lower():
        print(
            'Downloading all the object and box scenes.\n' \
            'NOTE: Existing scene folders will be skipped, ' \
            'please delete partially downloaded scenes to re-download.\n'
        )
        for scene_id in object_scenes:
            download_scene(scene_id, OBJECTS, args.dataset_folder)
        for scene_id in box_scenes:
            download_scene(scene_id, BOXES, args.dataset_folder)
    elif args.scene.lower() == 'ALL_OBJECTS'.lower():
        for scene_id in object_scenes:
            download_scene(scene_id, OBJECTS, args.dataset_folder)
    elif args.scene.lower() == 'ALL_BOXES'.lower():
        for scene_id in box_scenes:
            download_scene(scene_id, BOXES, args.dataset_folder)
    elif args.scene:
        scene_id = args.scene.lower()
        if scene_id in object_scenes:
            download_scene(scene_id, OBJECTS, args.dataset_folder)
        elif scene_id in box_scenes:
            download_scene(scene_id, BOXES, args.dataset_folder)
        else:
            print('ERROR: Invalid scene id: ' + scene_id)
    else:
        parser.print_help()
