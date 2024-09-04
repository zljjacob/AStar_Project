import matplotlib.pyplot as plt
import math
import heapq
import numpy as np
import multiprocessing as mp

show_animation = False

class AStarPlanner:

    def __init__(self, obs_list_x, obs_list_y, resolution, min_safety_dist):
        """
        obs_list_x: x list of obstacles
        obs_list_y: y list of obstacles
        resolution: grid map resolution
        min_safety_dist: minimum safety distance to obstacle
        """
        self.resolution = resolution
        self.min_safety_dist = min_safety_dist
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.x_width, self.y_width = 0, 0
        # create 2d grid map
        self.obstacle_map = None
        self.get_obstacle_map(obs_list_x, obs_list_y)
        # create motion model
        self.motion_model = self.get_motion_model()


    class Node:
        def __init__(self, x_idx, y_idx, cost, parent_idx):
            self.x_idx = x_idx  # index of grid map
            self.y_idx = y_idx  # index of grid map
            self.cost = cost # g value
            self.parent_idx = parent_idx
            

        def __str__(self):
            return str(self.x_idx) + "," + str(self.x_idx) + "," + str(
                self.cost) + "," + str(self.parent_idx)

        def __lt__(self, other):
            return self.cost < other.cost

    def search(self, start_x, start_y, goal_x, goal_y): 
        """
        input:
            start_x: start x position
            start_y: start y position
            goal_x: goal x position
            goal_y: goal y position

        output:
            path_x: x list of the final path
            path_y: y list of the final path
        """
        
        # construct start and goal node
        start_node = self.Node(*self.convert_coord_to_idx(start_x, start_y), 0.0, -1)
        goal_node = self.Node(*self.convert_coord_to_idx(goal_x, goal_y), 0.0, -1)

        # create open_set and closed set
        open_set = []
        heapq.heappush(open_set, (self.cal_heuristic_func(start_node, goal_node), start_node))
        open_set_hash = {self.get_vec_index(start_node): start_node}
        closed_set = {}

        while open_set:
            # pop the node with lowest value of the f function from the open_set, and add it to the closedset
            cur_node = heapq.heappop(open_set)[1]
            cur_node_idx = self.get_vec_index(cur_node)

            if cur_node_idx in open_set_hash:
                del open_set_hash[cur_node_idx]

            # plot cur_node
            if show_animation:
                plt.plot(*self.convert_idx_to_coord(cur_node.x_idx, cur_node.y_idx), marker='s', 
                    color='dodgerblue', alpha=0.2)
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.0000001)
            
            # determine whether the current node is the goal, and if so, stop searching
            if (cur_node.x_idx, cur_node.y_idx) == (goal_node.x_idx, goal_node.y_idx):
                print("Find the goal")
                goal_node.cost = cur_node.cost
                goal_node.parent_idx = cur_node.parent_idx
                break

            closed_set[cur_node_idx] = cur_node

            # expand neighbors of the current node
            for i, _ in enumerate(self.motion_model):
                next_node = self.Node(cur_node.x_idx + self.motion_model[i][0],
                                      cur_node.y_idx + self.motion_model[i][1],
                                      cur_node.cost + self.motion_model[i][2],
                                      cur_node_idx)
                next_node_idx = self.get_vec_index(next_node)
                
                if not self.check_node_validity(next_node):
                    continue

                if next_node_idx in closed_set:
                    continue

                if next_node_idx not in open_set_hash:
                    heapq.heappush(open_set, (next_node.cost + self.cal_heuristic_func(goal_node, next_node), next_node))
                    open_set_hash[next_node_idx] = next_node
                else:
                    if open_set_hash[next_node_idx].cost > next_node.cost:
                        open_set_hash[next_node_idx] = next_node
                        for i, (f, node) in enumerate(open_set):
                            if node == next_node:
                                open_set[i] = (next_node.cost + self.cal_heuristic_func(goal_node, next_node), next_node)
                                heapq.heapify(open_set)
                                break

        if not open_set:
            print("open_set is empty, can't find path")
            return [], []

        # backtrack to get the shortest path
        path_x, path_y = self.backtracking(goal_node, closed_set)

        return path_x, path_y

    def backtracking(self, goal_node, closed_set):
        goal_x, goal_y = self.convert_idx_to_coord(goal_node.x_idx, goal_node.y_idx)

        path_x, path_y = [goal_x], [goal_y]

        # backtracking from goal node to start node to extract the whole path
        parent_idx = goal_node.parent_idx
        while parent_idx != -1:
            node = closed_set[parent_idx]
            cur_x, cur_y = self.convert_idx_to_coord(node.x_idx, node.y_idx)
            path_x.append(cur_x)
            path_y.append(cur_y)
            parent_idx = node.parent_idx

        return path_x, path_y

    def cal_heuristic_func(self, node1, node2):
        # implement heuristic function to estimate the cost between node 1 and node 2
        h_value = math.hypot(node1.x_idx - node2.x_idx, node1.y_idx - node2.y_idx)
        return h_value

    def convert_idx_to_coord(self, x_idx, y_idx):
        x_coord = x_idx * self.resolution + self.min_x
        y_coord = y_idx * self.resolution + self.min_y
        return x_coord, y_coord

    def convert_coord_to_idx(self, x_pos, y_pos):
        x_idx = round((x_pos - self.min_x) / self.resolution)
        y_idx = round((y_pos - self.min_y) / self.resolution)
        return x_idx, y_idx

    def get_vec_index(self, node):
        return (node.y_idx - self.min_y) * self.x_width + (node.x_idx - self.min_x)

    def check_node_validity(self, node):
        x_coord, y_coord = self.convert_idx_to_coord(node.x_idx, node.y_idx)
        # check if the current node exceeds the map range
        if x_coord < self.min_x or x_coord > self.max_x \
            or y_coord < self.min_y or y_coord > self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x_idx][node.y_idx]:
            return False

        return True

    # generate 2d grid map from obstacle list
    def get_obstacle_map(self, obs_list_x, obs_list_y):
        print("Grid Map Info: ")
        self.min_x = round(min(obs_list_x))
        self.min_y = round(min(obs_list_y))
        self.max_x = round(max(obs_list_x))
        self.max_y = round(max(obs_list_y))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_length:", self.x_width)
        print("y_length:", self.y_width)

        # Initialize obstacle map with False
        self.obstacle_map = np.zeros((self.x_width, self.y_width), dtype=bool)
        
        obs_array = np.array(list(zip(obs_list_x, obs_list_y)))
        obs_min_idx_x = np.floor((obs_array[:, 0] - self.min_x - self.min_safety_dist) / self.resolution).astype(int)
        obs_max_idx_x = np.ceil((obs_array[:, 0] - self.min_x + self.min_safety_dist) / self.resolution).astype(int)
        obs_min_idx_y = np.floor((obs_array[:, 1] - self.min_y - self.min_safety_dist) / self.resolution).astype(int)
        obs_max_idx_y = np.ceil((obs_array[:, 1] - self.min_y + self.min_safety_dist) / self.resolution).astype(int)

        for (min_idx_x, max_idx_x, min_idx_y, max_idx_y) in zip(obs_min_idx_x, obs_max_idx_x, obs_min_idx_y, obs_max_idx_y):
            self.obstacle_map[min_idx_x:max_idx_x+1, min_idx_y:max_idx_y+1] = True

    def get_motion_model(self):
        # 8-connected motion model
        # dx, dy, cost
        motion_model = [[1, 0, 1],
                        [0, 1, 1],
                        [-1, 0, 1],
                        [0, -1, 1],
                        [-1, -1, math.sqrt(2)],
                        [-1, 1, math.sqrt(2)],
                        [1, -1, math.sqrt(2)],
                        [1, 1, math.sqrt(2)]]

        return motion_model
