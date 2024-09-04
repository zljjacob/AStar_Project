# This program uses multiprocess to handle more than one images.


from a_star_planner_numpy import AStarPlanner
from image_processing import preprocess_image, extract_obstacle_list_from_img
import cv2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from img_load import img_load
from concurrent.futures import ThreadPoolExecutor
import time

show_animation = True

def load_images_threaded(file_names):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(img_load, file_name) for file_name in file_names]
        results = [future.result() for future in futures]
    return results

def a_star_search(args):
    result, grid_res, min_safety_dist, start_x, start_y, goal_x, goal_y = args

    if show_animation:
        plt.figure(figsize=(12, 12))
        plt.axis("equal")
        plt.plot(result[0], result[1], 'sk', markersize=2)
        plt.plot(start_x, start_y, marker='*', color='lime', markersize=8)
        plt.plot(goal_x, goal_y, marker='*', color='r', markersize=8)
    
    a_star = AStarPlanner(result[0], result[1], grid_res, min_safety_dist)
    path_x, path_y = a_star.search(start_x, start_y, goal_x, goal_y)  

    if show_animation:
        plt.plot(path_x, path_y, ".-", color="royalblue")
        plt.show()

def main():
    start_time = time.time()

    # start and goal position
    start_x = 47.0  # [m]
    start_y = 65.0  # [m]
    goal_x = 136.0  # [m]
    goal_y = 11.0  # [m]
    grid_res = 1.0  # [m]
    min_safety_dist = 0.2  # [m]

    file_names = ["map1.png", "map2.png", "map3.png", "map4.png", "map5.png"]
    image_results = load_images_threaded(file_names)
    

    # Create arguments for each process
    args_list = [(image_results[i], grid_res, min_safety_dist, start_x, start_y, goal_x, goal_y) for i in range(len(file_names))]

    # Use multiprocessing Pool to parallelize the a_star_search function
    with mp.Pool() as pool:
        pool.map(a_star_search, args_list)

    end_time = time.time()
    print(f"The total running time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
