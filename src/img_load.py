import cv2
from image_processing import preprocess_image, extract_obstacle_list_from_img
import pathlib

def img_load(file_name):
    # get the file path
    image_path = pathlib.Path.cwd() / "maps" / file_name
    #read map
    image = cv2.imread(str(image_path))

    # image -> binary img
    binary_img = preprocess_image(image, 127)

    obs_list_x, obs_list_y = extract_obstacle_list_from_img(binary_img)

    return obs_list_x, obs_list_y