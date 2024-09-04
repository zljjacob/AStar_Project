from a_star_planner_without_heapq import AStarPlanner
from image_processing import preprocess_image, extract_obstacle_list_from_img
import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt

import time


show_animation = True

def main():
    # 图像文件名列表
    image_files = ["map1.png", "map2.png", "map3.png", "map4.png", "map5.png"]
    
    for image_file in image_files:
        print(f"Processing {image_file}...")

        # start and goal position
        start_x = 47.0  # [m]
        start_y = 65.0  # [m]
        goal_x = 136.0  # [m]
        goal_y = 11.0  # [m]
        grid_res = 1.0  # [m]
        min_safety_dist = 0.2  # [m]

        # read map
        image = cv2.imread(str(pathlib.Path.cwd()) + "/maps/" + image_file)

        binary_img = preprocess_image(image, 127)
        
        obs_list_x, obs_list_y = extract_obstacle_list_from_img(binary_img)

        # plot map info
        if show_animation:
            plt.figure(figsize=(12, 12))
            plt.axis("equal")
            plt.plot(obs_list_x, obs_list_y, 'sk', markersize=2)
            plt.plot(start_x, start_y, marker='*', color='lime', markersize=8)
            plt.plot(goal_x, goal_y, marker='*', color='r', markersize=8)
        start_time = time.time()

        a_star = AStarPlanner(obs_list_x, obs_list_y, grid_res, min_safety_dist)

        path_x, path_y = a_star.search(start_x, start_y, goal_x, goal_y)

        end_time = time.time()
        print(f"The total running time for {image_file}: {end_time - start_time} seconds")
        
        # plot searched path
        if show_animation:
            plt.plot(path_x, path_y, ".-", color="royalblue")
            plt.show()

if __name__ == '__main__':
    main()
