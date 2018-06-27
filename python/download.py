#!/usr/bin/env python

import argparse
import os
import urllib
from zipfile import ZipFile

BASE_URL = 'http://robotics.ethz.ch/~asl-datasets/ijrr_2018_clubs_dataset/'
OBJECT_SCENES = 'object_scenes.txt'
BOX_SCENES = 'box_scenes.txt'
OBJECTS = 'object_scenes/'
BOXES = 'box_scenes/'


def get_scene_list(scene_type):
    f = urllib.urlopen(BASE_URL + scene_type)
    scenes = f.read().splitlines()
    return scenes


def download_scene(scene_id, scene_type, output_dir):
    scene_url = BASE_URL + scene_type + scene_id + '.zip'
    scene_file = os.path.join(output_dir, scene_type, scene_id)
    scene_zipfile = scene_file + '.zip'
    scene_dir = os.path.dirname(scene_file)
    if not os.path.isfile(scene_file):
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        print(scene_url + ' > ' + scene_zipfile)
        urllib.urlretrieve(scene_url, scene_zipfile)
        f = ZipFile(scene_zipfile)
        f.extractall(scene_dir)
        f.close()
        os.remove(scene_zipfile)
    else:
        print('WARNING: skipping download of existing file ' + scene_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Download CLUBS dataset."))
    parser.add_argument(
        '--scene',
        type=str,
        default='ALL',
        help="Specific object id to download or ALL to download all objects.")
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Path to where the CLUBS dataset is stored.")
    args = parser.parse_args()

    object_scenes = get_scene_list(OBJECT_SCENES)
    box_scenes = get_scene_list(BOX_SCENES)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = '.'

    if args.scene and args.scene != 'ALL':
        scene_id = args.scene
        if scene_id in object_scenes:
            download_scene(scene_id, OBJECTS, output_dir)
        elif scene_id in box_scenes:
            download_scene(scene_id, BOXES, output_dir)
        else:
            print('ERROR: Invalid scene id: ' + scene_id)

    else:
        parser.print_help()
