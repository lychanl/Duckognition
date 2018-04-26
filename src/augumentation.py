
from PIL import Image
from imagejoiner import (
    iter_images,
    convert_path,
    save_image,
)
from sys import argv


def rotate_image(img: Image, angle: int) -> Image:
    return img.rotate(angle)


def successive_rotates(img: Image) -> tuple:
    return tuple(rotate_image(img, angle) for angle in (90, 180, 270))


def flip_image(img: Image) -> Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def image_transposes(img: Image) -> tuple:
    res = successive_rotates(img)
    res = (*res, flip_image(img),)
    res = (*res, *successive_rotates(res[3]))
    return res


def inject_in_name(original: str, injected: str) -> str:
    name = original.split("+")
    if len(name) == 1:
        return f"{original}{injected}"
    return f"{name[0]}{injected}+{name[1]}"


_IMAGES_NAMES_SUFFIXES = (
    "rot90", "rot180", "rot270",
    "flip",
    "fliprot90", "fliprot180", "fliprot270",
)


def transpose_ducks(root_dir: str, save_dir: str):
    for name, image, path in iter_images(root_dir):
        transposes = image_transposes(image)
        save_image(convert_path(path, save_dir), image)
        for suffix, t_img in zip(_IMAGES_NAMES_SUFFIXES, transposes):
            save_path = convert_path(path, save_dir)
            save_path = "/".join(save_path.split("/")[:-1])
            save_path += f"/{inject_in_name(name, suffix)}.jpg"
            save_image(save_path, t_img)


if __name__ == '__main__':
    transpose_ducks(argv[1], argv[2])
