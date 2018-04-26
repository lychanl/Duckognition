import numpy as np
import glob
import imageio
from resizeimage import resizeimage
from PIL import Image
from collections import defaultdict
import os
from sys import argv
import shutil

IMSIZE = 7


def iter_images(root_dir_path: str) -> (str, Image, str):
    """
    Traverses directory tree starting at root_dir_path
    Upon finding an jpg image yields tuple
    (name_of_the_image, loaded_image_as_np_array, path_to_the_image)
    """
    for i, path in enumerate(glob.iglob(f"{root_dir_path}\\**\\*.jpg")):
        name = path.split("\\")[-1][:-4]
        img = Image.fromarray(imageio.imread(path))
        yield name, img, path
    return i


def join_images(*images: Image) -> np.array:
    images = [*images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('L', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def convert_path(original_path: str, new_path: str) -> str:
    """
    Replaces fist part of the original_path with new_path
    exmpl.
    original SET_A/2089/img.jpg
    new_path data/images

    result data/images/2089/img.jpg
    """
    path = original_path.split("\\")
    path[0] = new_path
    #print(f"original: {original_path} new_path: {new_path} result {path}")
    return '\\'.join(path)


def make_path(path: str):
    os.makedirs("\\".join(path.split("\\")[:-1]), exist_ok=True)


def save_image_with_resize(path: str, im: Image):
    make_path(path)
    im.thumbnail([IMSIZE, IMSIZE], Image.ANTIALIAS)
    im = resizeimage.resize_crop(im, [IMSIZE, IMSIZE])
    #print(f"Scaving scaled image at: {path}")
    im.save(path, im.format)


def save_image(path: str, im: Image):
    make_path(path)
    im.save(path, im.format)


def group_images(root_dir: str):
    groups = defaultdict(list)
    for name, img, path in iter_images(root_dir):
        prefix, suffix = name.split("+")
        groups[prefix].append((name, path))
    return dict(**groups)


def scale_images(root_dir: str, save_path: str):
    for name, img, path in iter_images(root_dir):
        new_path = convert_path(path, save_path)
        save_image_with_resize(new_path, img)


def iter_joined_images(root_dir: str):
    for name, group in group_images(root_dir).items():
        #print(f"{name}:")
        i = 0
        group_path = ""
        for f_name, path in group:
            if i == 0:
                image = Image.fromarray(imageio.imread(path))
                group_path = "\\".join(path.split("\\")[:-1])
            else:
                image = join_images(image, Image.fromarray(imageio.imread(path)))
            i = i + 1
            #print(f"----{f_name}: {i}")
        yield (name, image, group_path)


def join_ducks(root_dir: str, save_path: str):
    scale_images(root_dir, "temp_scaled")
    for name, image, g_path in iter_joined_images("temp_scaled"):
        try:
            image_save_path = convert_path(f"{g_path}\\{name}.jpg", save_path)
            save_image(image_save_path, image)
        except Exception as exc:
            print(f"Problem with image group {name}: {exc}... skipping")
            continue
    print("Removing temp dir")
    shutil.rmtree("temp_scaled")


if __name__ == "__main__":
    join_ducks(argv[1], argv[2])
