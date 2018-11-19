"""Contains tools for creating label images."""

import cv2
import random
import numpy as np


def random_color():
    """Generate a random RGB color.

    Returns:
        color (list): Random RGB color.

    """
    color = (random.randint(0, 255), random.randint(0, 255),
             random.randint(0, 255))
    return color


def poly_to_image(img, poly, value):
    """Paint a polygon in an image.

    Note:
        Input image will be modified.

    Args:
        img (np.array): Input image that will be modifed.
        poly (list(float)): List of polygon points.
        value (list(int) or int): Value of the points inside the polygon. Can
            be 3-channel (RGB) or 1-channel (mono) depending on the input
            image.

    """
    mask = np.array([poly], dtype='int32')
    cv2.fillPoly(img, mask, value)


def bbox_to_image(img, bbox, value, line_thickness=2):
    """Paint a bounding box in an image.

    Note:
        Input image will be modified.

    Args:
        img (np.array): Input image that will be modifed.
        bbox (list(float)): Bounding box (x, y, widht, height).
        value (list(int) or int): Value of the points on the bounding box. Can
            be 3-channel (RGB) or 1-channel (mono) depending on the input
            image.
        line_thickness (int, optional): Thickness of the bounding box in
            pixels. Defaults to 2.

    """
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                  (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), value,
                  line_thickness)
