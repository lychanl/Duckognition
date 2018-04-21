import numpy as np
import imageio
import glob
import os
import sys
import logging

__LOGGER__ = logging.getLogger(__name__)

def read_boxes(file_path: str) -> (str, (int, int), (int, int)):
    """
    Parses boundind_box.txt
    Yields tuple (name_of_the_image, (x1, y1), (x2, y2))
    """
    with open(file_path) as file:
        for line in file.readlines():
            raw = line.split(" ")
            name = raw[0].replace("-", "")
            left = tuple(int(p) for p in  raw[1:3])
            right = tuple(int(p) for p in raw[3:5])
            yield (name, left, right)

def iter_images(root_dir_path: str) -> (str, np.array, str):
    """
    Traverses directory tree starting at root_dir_path
    Upon finding an jpg image yields tuple
    (name_of_the_image, loaded_image_as_np_array, path_to_the_image)
    """
    for i, path in enumerate(glob.iglob(f"{root_dir_path}/**/*.jpg")):
        name = path.split("/")[-1][:-4]
        img = imageio.imread(path)
        yield name, img, path
    return i 

def save_image(path: str, image: np.array):
    imageio.imwrite(path, image)

def convert_path(original_path: str, new_path: str) -> str:
    """
    Replaces fist part of the original_path with new_path
    exmpl. 
    original SET_A/2089/img.jpg
    new_path data/images
    
    result data/images/2089/img.jpg 
    """
    path = original_path.split("/")
    path[0] = new_path
    __LOGGER__.debug(f"original: {original_path} new_path: {new_path} result {path}")
    return '/'.join(path)

def crop_image(image, left, right) -> np.array:
    new_image = image[left[1]:right[1], left[0]:right[0]]
    return new_image

def adjust_points(p1, p2) -> ((int, int), (int, int)):
    """
    Creates 2 new points form p1, p2 so that they meet constraint
    p1(x1, y1) < p2(x2, y2)
    """
    p1_0, p2_0 = (p1[0], p2[0]) if p1[0] < p2[0] else (p2[0], p1[0])
    p1_1, p2_1 = (p1[1], p2[1]) if p1[1] < p2[1] else (p2[1], p1[1])
    return (p1_0, p1_1), (p2_0, p2_1) 

"""
Dict of image name and corresponding bounding box
"""
BOUNDING_BOXES = {name: tuple(adjust_points(left, right)) for name, left, right in read_boxes('bounding_boxes.txt')}


def crop_ducks(data_set_path: str, save_path: str):
    """
    Crops images in the dataset according to the bounding_boxes.txt
    then saves in the save_path
    """
    for name, image, path in iter_images(data_set_path):
        __LOGGER__.debug(f"new image {name}, {path}")
        cropped = crop_image(image, *BOUNDING_BOXES[name])
        result_path = convert_path(path, save_path)
        os.makedirs("/".join(result_path.split("/")[:-1]), exist_ok=True)
        try:
            save_image(result_path, cropped)
        except Exception as exc:
            __LOGGER__.debug(f"problem with {name}[{path}]: {BOUNDING_BOXES[name]} -> {exc}\nCROPPED: {cropped}")


if __name__ == "__main__":
    __LOGGER__.debug(f"args: {sys.argv}")
    crop_ducks(sys.argv[1], sys.argv[2])




   

