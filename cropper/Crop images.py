import numpy as np
import imageio
import glob
import os

def read_boxes(file_path: str) -> (str, (int, int), (int, int)):
    with open(file_path) as file:
        for line in file.readlines():
            raw = line.split(" ")
            name = raw[0].replace("-", "")
            left = tuple(int(p) for p in  raw[1:3])
            right = tuple(int(p) for p in raw[3:5])
            yield (name, left, right)

def iter_images(root_dir_path: str) -> (str, np.array, str):
    for i, path in enumerate(glob.iglob(f"{root_dir_path}/**/*.jpg")):
        name = path.split("/")[-1][:-4]
        img = imageio.imread(path)
        yield name, img, path
    return i 

def save_image(path: str, image: np.array):
    imageio.imwrite(path, image)

def convert_path(original_path: str) -> str:
    path = original_path.split("/")
    path[0] = 'converted'
    return '/'.join(path)

def crop_image(image, left, right) -> np.array:
    new_image = image[left[1]:right[1], left[0]:right[0]]
    return new_image

def adjust_points(p1, p2) -> ((int, int), (int, int)):
    p1_0, p2_0 = (p1[0], p2[0]) if p1[0] < p2[0] else (p2[0], p1[0])
    p1_1, p2_1 = (p1[1], p2[1]) if p1[1] < p2[1] else (p2[1], p1[1])
    return (p1_0, p1_1), (p2_0, p2_1) 

BOUNDING_BOXES = {name: tuple(adjust_points(left, right)) for name, left, right in read_boxes('bounding_boxes.txt')}

if __name__ == "__main__":
    for name, image, path in iter_images('SET_B'):
        cropped = crop_image(image, *BOUNDING_BOXES[name])
        save_path = convert_path(path)
        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
        try:
            save_image(save_path, cropped)
        except Exception as exc:
            print(f"problem with {name}[{path}]: {BOUNDING_BOXES[name]}\nCROPPED: {cropped}")




   

