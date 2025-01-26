import json
import math
import re
import time
import copy
import random
import logging
import datetime
import numpy as np
import multiprocessing
from rtree import index
import matplotlib.pyplot as plt
import holodeck_utils.prompts as prompts
from langchain import PromptTemplate
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, box, LineString


class FloorObjectGenerator():
    def __init__(self, llm):
        self.json_template = {"assetId": None, "id": None, "kinematic": True,
                              "position": {}, "rotation": {}, "material": None, "roomId": None}
        self.llm = llm
        self.constraint_prompt = PromptTemplate(input_variables=["constraints", "placed_obj"],
                                                template=prompts.object_constraints_prompt)
        self.baseline_prompt = PromptTemplate(input_variables=["room_type", "room_size", "objects"],
                                              template=prompts.floor_baseline_prompt)
        self.grid_density = 30
        self.add_window = False
        self.size_buffer = 10  # add 10 cm buffer to object size

        self.constraint_type = "llm"
        self.use_milp = False
        self.multiprocessing = False
        # self.database_type = database_type
        # if database_type == "objaverse":
        #     with open("/home/yianwang_umass_edu/RoboGen/ljg/Holodeck/data/objaverse_holodeck/09_23_combine_scale/objaverse_holodeck_database.json",
        #         'r') as file:
        #         self.database = json.load(file)
        # elif database_type == "blenderkit":
        #     with open("/project/pi_chuangg_umass_edu/yian/robogen/architect/blenderkit_data/blenderkit_database.json",
        #     'r') as file:
        #         self.database = json.load(file)

    def query_llm(self, query):
        response = self.llm.chat.completions.create(
            model="gpt-4",  # e.g. gpt-35-instant
            messages=[
                {
                    "role": "user",
                    "content": query,
                },
            ],
            max_tokens=2000,
        )
        result = response.choices[0].message.content
        return result

    # def generate_objects(self, scene, use_constraint=True):
    #     room = scene[0]['room_bbox']
    #     room = {
    #         'vertices': [
    #             [room[0][0], room[0][1]],
    #             [room[0][0], room[1][1]],
    #             [room[1][0], room[1][1]],
    #             [room[1][0], room[0][1]],
    #         ],
    #         'x': (room[0][1] - room[0][0]) * 100,
    #         'y': (room[1][1] - room[1][0]) * 100
    #     }
    #     selected_objects = scene[1:]
    #     object_list = [(obj['name'], obj['assetId']) for obj in selected_objects]
    #     size_list = {
    #         obj['name']: {
    #             'x': obj['bbox'][1][0] - obj['bbox'][0][0],
    #             'y': obj['bbox'][1][1] - obj['bbox'][0][1],
    #         } for obj in selected_objects
    #     }
    #
    #     results = self.generate_objects_per_room(room, generate_constraints(selected_objects), object_list, size_list)
    #
    #     # if self.multiprocessing:
    #     #     pool = multiprocessing.Pool(processes=4)
    #     #     all_placements = pool.map(self.generate_objects_per_room, packed_args)
    #     #     pool.close()
    #     #     pool.join()
    #     # else:
    #     #     all_placements = [self.generate_objects_per_room(args) for args in packed_args]
    #
    #     # for placements in all_placements:
    #     #     results += placements
    #
    #     return results

    def generate_objects_per_room(self, room, constraints, object_list, size_list, initial_placement=[]):
        selected_floor_objects = object_list
        object_name2id = {object_name: asset_id for object_name, asset_id in selected_floor_objects}
        room_x, room_z = room['x'], room['y']
        # room_size = f"{room_x} cm x {room_z} cm"
        grid_size = max(room_x // self.grid_density, room_z // self.grid_density)

        object_names = list(object_name2id.keys())
        placed_objects = []
        if len(initial_placement) == 0:
            placed_obj = "No objects are already placed in the scene."
        else:
            placed_obj = ""
            for obj in initial_placement:
                placed_obj += obj["object_name"] + ", "
                placed_objects.append(obj["object_name"])
        constraint_prompt = self.constraint_prompt.format(constraints=constraints, placed_obj=placed_obj)

        constraint_plan = self.query_llm(constraint_prompt)
        logging.debug(f"Final constraint: {constraint_plan}")

        constraints = self.parse_constraints(constraint_plan, object_names, placed_objects)

        # print(f"Final constraint: {constraints}")

        objects_list = [(object_name, (size_list[object_name]['x'] * 100 + self.size_buffer,
                                       size_list[object_name]['y'] * 100 + self.size_buffer)) for object_name
                        in constraints]

        # get initial state
        room_vertices = [(x * 100, y * 100) for (x, y) in room["vertices"]]
        room_poly = Polygon(room_vertices)
        initial_state = self.get_init_placements(initial_placement)

        # solve
        solver = DFS_Solver_Floor(grid_size=grid_size, max_duration=60, constraint_bouns=1)
        solution = solver.get_solution(room_poly, objects_list, constraints, initial_state, use_milp=False)
        # placements = self.solution2placement(solution, object_name2id)

        return solution

    def get_door_window_placements(self, doors, windows, room_vertices, open_walls, add_window=True):
        room_poly = Polygon(room_vertices)
        door_window_placements = {}
        i = 0
        for door in doors:
            door_boxes = door["doorBoxes"]
            for door_box in door_boxes:
                door_vertices = [(x * 100, z * 100) for (x, z) in door_box]
                door_poly = Polygon(door_vertices)
                door_center = door_poly.centroid
                if room_poly.contains(door_center):
                    door_window_placements[f"door-{i}"] = ((door_center.x, door_center.y), 0, door_vertices, 1)
                    i += 1

        if add_window:
            for window in windows:
                window_boxes = window["windowBoxes"]
                for window_box in window_boxes:
                    window_vertices = [(x * 100, z * 100) for (x, z) in window_box]
                    window_poly = Polygon(window_vertices)
                    window_center = window_poly.centroid
                    if room_poly.contains(window_center):
                        door_window_placements[f"window-{i}"] = (
                            (window_center.x, window_center.y), 0, window_vertices, 1)
                        i += 1

        if open_walls != []:
            for open_wall_box in open_walls["openWallBoxes"]:
                open_wall_vertices = [(x * 100, z * 100) for (x, z) in open_wall_box]
                open_wall_poly = Polygon(open_wall_vertices)
                open_wall_center = open_wall_poly.centroid
                if room_poly.contains(open_wall_center):
                    door_window_placements[f"open-{i}"] = (
                        (open_wall_center.x, open_wall_center.y), 0, open_wall_vertices, 1)
                    i += 1

        return door_window_placements

    def get_init_placements(self, placements):
        initial_state = {}

        for object in placements:
            try:
                object_vertices = object["vertices"]
            except:
                continue

            object_poly = Polygon(object_vertices)
            object_center = object_poly.centroid
            initial_state[object["object_name"]] = (
                (object_center.x, object_center.y), object["rotation"]["y"], object_vertices, 1)

        return initial_state

    def get_room_size(self, room):
        floor_polygon = room["floorPolygon"]
        x_values = [point['x'] for point in floor_polygon]
        z_values = [point['z'] for point in floor_polygon]
        return (int(max(x_values) - min(x_values)) * 100, int(max(z_values) - min(z_values)) * 100)

    def solution2placement(self, solutions, object_name2id, size_list):
        placements = []
        for object_name, solution in solutions.items():
            if "door" in object_name or "window" in object_name or "open" in object_name: continue

            placement = self.json_template.copy()
            placement["assetId"] = object_name2id[object_name]
            # placement["id"] = f"{object_name} ({room_id})"
            placement["position"] = {"x": solution[0][0] / 100, "y": size_list[object_name]["z"] / 2, "z": solution[0][1] / 100}
            placement["rotation"] = {"x": 0, "y": solution[1], "z": 0}
            # placement["roomId"] = room_id
            placement["vertices"] = list(solution[2])
            placement["object_name"] = object_name
            placements.append(placement)
        return placements

    def parse_constraints(self, constraint_text, object_names, placed_objects):
        constraint_name2type = {
            "edge": "global",
            "middle": "global",
            "corner": "global",
            "location": "location",
            "in front of": "relative",
            "behind": "relative",
            "left of": "relative",
            "right of": "relative",
            "side of": "relative",
            "around": "relative",
            "face to": "direction",
            "face same as": "direction",
            "aligned": "alignment",
            "center alignment": "alignment",
            "center aligned": "alignment",
            "aligned center": "alignment",
            "edge alignment": "alignment",
            "near": "distance",
            "far": "distance"
        }

        object2constraints = {}
        plans = [plan for plan in constraint_text.split('\n') if "|" in plan]
        for plan in plans:
            # remove index
            pattern = re.compile(r'^(\d+[\.\)]\s*|- )')
            plan = pattern.sub('', plan)
            if plan[-1] == ".": plan = plan[:-1]

            object_name = plan.split("|")[0].replace("*", "").strip()  # remove * in object name

            if object_name not in object_names: continue

            object2constraints[object_name] = []

            constraints = plan.split("|")[1:]
            for constraint in constraints:
                constraint = constraint.strip()
                constraint_name = constraint.split(",")[0].strip()

                if constraint_name == "n/a": continue

                try:
                    constraint_type = constraint_name2type[constraint_name]
                except:
                    logging.warning(f"constraint type {constraint_name} not found")
                    continue

                if constraint_type == "global":
                    object2constraints[object_name].append({"type": constraint_type, "constraint": constraint})
                elif constraint_type in ["relative", "direction", "alignment", "distance"]:
                    try:
                        target = constraint.split(",")[1].strip()
                    except:
                        logging.warning(f"wrong format of constraint: {constraint}")
                        continue

                    if target in object2constraints or target in placed_objects:
                        if constraint_name == "around":
                            object2constraints[object_name].append(
                                {"type": "distance", "constraint": "near", "target": target})
                            # object2constraints[object_name].append(
                            #     {"type": "direction", "constraint": "face to", "target": target})
                        elif constraint_name == "in front of":
                            object2constraints[object_name].append(
                                {"type": "relative", "constraint": "in front of", "target": target})
                            # object2constraints[object_name].append(
                            #     {"type": "alignment", "constraint": "center aligned", "target": target})
                        else:
                            object2constraints[object_name].append(
                                {"type": constraint_type, "constraint": constraint_name, "target": target})
                    else:
                        logging.warning(f"target object {target} not found in the existing constraint plan")
                        logging.warning(object2constraints.keys())
                        logging.warning(placed_objects)
                        continue
                elif constraint_type in ["location"]:
                    try:
                        target = eval(constraint[constraint.find(',') + 1:])
                    except:
                        logging.warning(f"wrong format of constraint: {constraint}")
                        continue

                    object2constraints[object_name].append(
                        {"type": constraint_type, "constraint": constraint_name, "target": target})

                else:
                    logging.warning(f"constraint type {constraint_type} not found")
                    continue

        # clean the constraints
        object2constraints_cleaned = {}
        for object_name, constraints in object2constraints.items():
            constraints_cleaned = []
            constraint_types = []
            for constraint in constraints:
                if constraint["type"] not in constraint_types:
                    constraint_types.append(constraint["type"])
                    constraints_cleaned.append(constraint)
            object2constraints_cleaned[object_name] = constraints_cleaned

        return object2constraints


class SolutionFound(Exception):
    def __init__(self, solution):
        self.solution = solution


class DFS_Solver_Floor():
    def __init__(self, grid_size, random_seed=0, max_duration=10, constraint_bouns=0.2):
        self.grid_size = grid_size
        self.random_seed = random_seed
        self.max_duration = max_duration  # maximum allowed time in seconds
        self.constraint_bouns = constraint_bouns
        self.start_time = None
        self.solutions = []
        self.vistualize = False

        # Define the functions in a dictionary to avoid if-else conditions
        self.func_dict = {
            "global": {
                "edge": self.place_edge,
                "corner": self.place_corner,
                "middle": self.place_middle
            },
            "relative": self.place_relative,
            "direction": self.place_face,
            "alignment": self.place_alignment_center,
            "distance": self.place_distance
        }

        self.constraint_type2weight = {
            "global": 1.0,
            "relative": 0.5,
            "direction": 0.5,
            "alignment": 1.0,
            "distance": 2.5,
            "location": 4.0,
        }

        self.edge_bouns = 0.0  # worth more than one constraint

    def get_solution(self, bounds, objects_list, constraints, initial_state, use_milp=False):
        self.start_time = time.time()
        grid_points = self.create_grids(bounds)
        # grid_points = self.remove_points(grid_points, initial_state)
        try:
            self.dfs(bounds, objects_list, constraints, grid_points, initial_state, 20)
        except SolutionFound as e:
            print(f"Time taken: {time.time() - self.start_time}")

        print(f"Number of solutions found: {len(self.solutions)}")
        max_solution = self.get_max_solution(self.solutions)

        if not use_milp and self.vistualize:
            self.visualize_grid(bounds, grid_points, max_solution)

        return max_solution

    def get_max_solution(self, solutions):
        path_weights = []
        for i, solution in enumerate(solutions):
            path_weights.append((i, sum([obj[-1] for obj in solution.values()]), len(solution)))
        path_weights.sort(key=lambda x: (x[2], x[1]), reverse=True)
        max_index = path_weights[0][0]
        return solutions[max_index]

    def dfs(self, room_poly, objects_list, constraints, grid_points, placed_objects, branch_factor):
        if len(objects_list) == 0:
            self.solutions.append(placed_objects)
            return placed_objects

        if time.time() - self.start_time > self.max_duration:
            logging.warning(f"Time limit reached.")
            raise SolutionFound(self.solutions)

        object_name, object_dim = objects_list[0]
        # print(object_name, object_dim)
        placements = self.get_possible_placements(room_poly, object_dim, constraints[object_name], grid_points,
                                                  placed_objects)
        # print(placements)
        if len(placements) == 0:
            return []
        # if len(placements) == 0 and len(placed_objects) != 0:
            # self.solutions.append(placed_objects)

        paths = []
        # if branch_factor > 1: random.shuffle(placements[4:])  # shuffle the placements of the first object

        for placement in placements[:branch_factor]:
            placed_objects_updated = copy.deepcopy(placed_objects)
            placed_objects_updated[object_name] = placement
            grid_points_updated = self.remove_points(grid_points, placed_objects_updated)

            sub_paths = self.dfs(room_poly, objects_list[1:], constraints, grid_points_updated, placed_objects_updated,
                                 12)  # need change
            paths.extend(sub_paths)

        return paths

    def get_possible_placements(self, room_poly, object_dim, constraints, grid_points, placed_objects):
        # logging.debug("#" * 20)
        solutions = self.get_all_solutions(room_poly, grid_points, object_dim)
        # print(f"solutions after get_all_solutions: {len(solutions)}")
        # logging.debug(f"solutions after get_all_solutions: {len(solutions)}")
        # solutions = self.filter_collision(placed_objects, solutions)
        solutions = self.filter_adjancent(placed_objects, solutions)
        # print(f"solutions after filter_adjancent: {len(solutions)}")
        solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
        # print(f"solutions after filter_facing_wall: {len(solutions)}")
        edge_solutions = self.place_edge(room_poly, copy.deepcopy(solutions), object_dim)
        # print(f"edge_solutions: {len(edge_solutions)}")

        # if len(edge_solutions) == 0: return edge_solutions

        global_constraint = next((constraint for constraint in constraints if constraint["type"] == "global"), None)

        if global_constraint is None: global_constraint = {"type": "global", "constraint": "edge"}

        if global_constraint["constraint"] == "edge":
            candidate_solutions = copy.deepcopy(edge_solutions)  # edge is hard constraint
        elif global_constraint["constraint"] == "corner":
            candidate_solutions = copy.deepcopy(self.place_corner(room_poly, solutions, object_dim))
        elif "middle" in global_constraint["constraint"]:
            horizontal = global_constraint["constraint"].split(',')[1].strip() == "horizontal"
            candidate_solutions = copy.deepcopy(self.place_middle(room_poly, solutions, object_dim, horizontal))
        else:
            if len(constraints) > 1:
                candidate_solutions = copy.deepcopy(solutions)  # edge is soft constraint
            else:
                candidate_solutions = copy.deepcopy(solutions)  # the first object

        candidate_solutions = self.filter_collision(placed_objects,
                                                    candidate_solutions)
        # print("candidate_solutions:", candidate_solutions)

        if candidate_solutions == []:
            print("placed objects:")
            print(placed_objects)
            print("no solution in this round")
            return candidate_solutions
        random.shuffle(candidate_solutions)
        placement2score = {tuple(solution[:3]): solution[-1] for solution in candidate_solutions}

        # add a bias to edge solutions
        for solution in candidate_solutions:
            if solution in edge_solutions and len(constraints) >= 1:
                placement2score[tuple(solution[:3])] += self.edge_bouns

        for constraint in constraints:
            if "target" not in constraint: continue

            if constraint["type"] == "location":
                valid_solutions = candidate_solutions
            else:
                func = self.func_dict.get(constraint["type"])
                valid_solutions = func(constraint["constraint"], placed_objects[constraint["target"]],
                                       candidate_solutions)

            weight = self.constraint_type2weight[constraint["type"]]
            if constraint["type"] == "distance":
                for solution in valid_solutions:
                    bouns = solution[-1]
                    placement2score[tuple(solution[:3])] += bouns * weight
            elif constraint["type"] == "location":
                for solution in valid_solutions:
                    x, y = solution[0]
                    x /= 100
                    y /= 100
                    loc_x, loc_y, loc_z = constraint["target"]
                    distance_to_loc = ((x - loc_x) ** 2 + (y - loc_y) ** 2) ** 0.5
                    if distance_to_loc < 0.5:
                        bouns = 1
                    else:
                        bouns = 1.5 - distance_to_loc
                    placement2score[tuple(solution[:3])] += (bouns + 10) * weight
            else:
                for solution in valid_solutions:
                    placement2score[tuple(solution[:3])] += self.constraint_bouns * weight

        # normalize the scores
        for placement in placement2score: placement2score[placement] /= max(len(constraints), 1)

        sorted_placements = sorted(placement2score, key=placement2score.get, reverse=True)
        sorted_solutions = [list(placement) + [placement2score[placement]] for placement in sorted_placements]
        # if "tv stand-0" in placed_objects and "potted plant-0" in placed_objects:
        #     print("sorted_solutions")
        #     for i in range(20):
        #         print((sorted_solutions[i][2][0][0] + sorted_solutions[i][2][2][0]) / 2 / 100, (sorted_solutions[i][2][0][1] + sorted_solutions[i][2][1][1]) / 2 / 100, sorted_solutions[i][-1])
        return sorted_solutions

    def create_grids(self, room_poly):
        # get the min and max bounds of the room
        min_x, min_z, max_x, max_z = room_poly.bounds

        # create grid points
        grid_points = []
        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_z), int(max_z), self.grid_size):
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))

        return grid_points

    def remove_points(self, grid_points, objects_dict):
        # Create an r-tree index
        idx = index.Index()

        # Populate the index with bounding boxes of the objects
        for i, (_, _, obj, _) in enumerate(objects_dict.values()):
            idx.insert(i, Polygon(obj).bounds)

        # Create Shapely Polygon objects only once
        polygons = [Polygon(obj) for _, _, obj, _ in objects_dict.values()]

        valid_points = []

        for point in grid_points:
            p = Point(point)
            # Get a list of potential candidates
            candidates = [polygons[i] for i in idx.intersection(p.bounds)]
            # Check if point is in any of the candidate polygons
            if not any(candidate.contains(p) for candidate in candidates):
                valid_points.append(point)

        return valid_points

    def get_all_solutions(self, room_poly, grid_points, object_dim):
        obj_length, obj_width = object_dim
        obj_half_length, obj_half_width = obj_length / 2, obj_width / 2

        rotation_adjustments = {
            0: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            90: ((-obj_half_width, -obj_half_length), (obj_half_width, obj_half_length)),
            180: ((-obj_half_length, obj_half_width), (obj_half_length, -obj_half_width)),
            270: ((obj_half_width, -obj_half_length), (-obj_half_width, obj_half_length)),
        }

        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                center_x, center_y = point
                lower_left_adjustment, upper_right_adjustment = rotation_adjustments[rotation]
                lower_left = (center_x + lower_left_adjustment[0], center_y + lower_left_adjustment[1])
                upper_right = (center_x + upper_right_adjustment[0], center_y + upper_right_adjustment[1])
                obj_box = box(*lower_left, *upper_right)

                if room_poly.contains(obj_box):
                    solutions.append([point, rotation, tuple(obj_box.exterior.coords[:]), 1])
        return solutions

    def filter_collision(self, objects_dict, solutions):
        valid_solutions = []
        object_polygons = [Polygon(obj_coords) for _, _, obj_coords, _ in list(objects_dict.values())]
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any(sol_obj.intersects(obj) for obj in object_polygons):
                valid_solutions.append(solution)
        return valid_solutions

    def filter_adjancent(self, objects_dict, solutions):
        valid_solutions = []
        object_polygons = [Polygon(obj_coords) for _, _, obj_coords, _ in list(objects_dict.values())]
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any(sol_obj.distance(obj) < 5 for obj in object_polygons):
                valid_solutions.append(solution)
        return valid_solutions

    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        dist_threshold = 100

        front_center_adjustments = {
            0: (0, obj_half_width + dist_threshold),
            90: (-obj_half_width - dist_threshold, 0),
            180: (0, -obj_half_width - dist_threshold),
            270: (obj_half_width + dist_threshold, 0),
        }

        valid_solutions = []
        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            front_center_adjustment = front_center_adjustments[rotation]
            front_center_x, front_center_y = center_x + front_center_adjustment[0], center_y + front_center_adjustment[
                1]

            front_center_distance = room_poly.boundary.distance(Point(front_center_x, front_center_y))
            front_center_pt = Point(front_center_x, front_center_y)
            
            if room_poly.contains(front_center_pt):
                valid_solutions.append(solution)

        return valid_solutions

    def place_edge(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        back_center_adjustments = {
            0: (0, -obj_half_width),
            90: (obj_half_width, 0),
            180: (0, obj_half_width),
            270: (-obj_half_width, 0),
        }

        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]
            back_center_adjustment = back_center_adjustments[rotation]
            back_center_x, back_center_y = center_x + back_center_adjustment[0], center_y + back_center_adjustment[1]
            back_center_distance = room_poly.boundary.distance(Point(back_center_x, back_center_y))
            center_distance = room_poly.boundary.distance(Point(center_x, center_y))
            if back_center_distance <= self.grid_size and back_center_distance < center_distance:
                solution[-1] += self.constraint_bouns
                # valid_solutions.append(solution) # those are still valid solutions, but we need to move the object to the edge
                # move the object to the edge
                center2back_vector = np.array([back_center_x - center_x, back_center_y - center_y])
                center2back_vector /= np.linalg.norm(center2back_vector)
                offset = center2back_vector * (
                        back_center_distance + 4.5)  # add a small distance to avoid the object cross the wall
                solution[0] = (center_x + offset[0], center_y + offset[1])
                solution[2] = ((solution[2][0][0] + offset[0], solution[2][0][1] + offset[1]), \
                               (solution[2][1][0] + offset[0], solution[2][1][1] + offset[1]), \
                               (solution[2][2][0] + offset[0], solution[2][2][1] + offset[1]), \
                               (solution[2][3][0] + offset[0], solution[2][3][1] + offset[1]))
                valid_solutions.append(solution)

        return valid_solutions

    def place_corner(self, room_poly, solutions, obj_dim):
        obj_length, obj_width = obj_dim
        obj_half_length, _ = obj_length / 2, obj_width / 2

        rotation_center_adjustments = {
            0: ((-obj_half_length, 0), (obj_half_length, 0)),
            90: ((0, -obj_half_length), (0, obj_half_length)),
            180: ((obj_half_length, 0), (-obj_half_length, 0)),
            270: ((0, obj_half_length), (0, -obj_half_length))
        }

        edge_solutions = self.place_edge(room_poly, solutions, obj_dim)

        valid_solutions = []

        for solution in edge_solutions:
            (center_x, center_y), rotation = solution[:2]
            (dx_left, dy_left), (dx_right, dy_right) = rotation_center_adjustments[rotation]

            left_center_x, left_center_y = center_x + dx_left, center_y + dy_left
            right_center_x, right_center_y = center_x + dx_right, center_y + dy_right

            left_center_distance = room_poly.boundary.distance(Point(left_center_x, left_center_y))
            right_center_distance = room_poly.boundary.distance(Point(right_center_x, right_center_y))

            if min(left_center_distance, right_center_distance) < self.grid_size:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_middle(self, room_poly, solutions, obj_dim, horizontal):

        valid_solutions = []

        for solution in solutions:
            xs = [v[0] for v in solution[2]]
            ys = [v[1] for v in solution[2]]
            xs, ys = sorted(xs), sorted(ys)
            x_length, y_length = xs[-1] - xs[0], ys[-1] - ys[0]
            aspect_ratio = x_length / y_length
            solution_horizontal = x_length > y_length
            if solution_horizontal == horizontal or (aspect_ratio < 1.2 and aspect_ratio > 1 / 1.2):
                valid_solutions.append(solution)

        return valid_solutions

    def place_relative(self, place_type, target_object, solutions):
        valid_solutions = []
        _, target_rotation, target_coords, _ = target_object
        target_polygon = Polygon(target_coords)

        min_x, min_y, max_x, max_y = target_polygon.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2

        comparison_dict = {
            'left of': {
                0: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
            },
            'right of': {
                0: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
            },
            'in front of': {
                0: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
                # in front of and centered
                90: lambda sol_center: sol_center[0] > max_x and mean_y - self.grid_size < sol_center[
                    1] < mean_y + self.grid_size,
                180: lambda sol_center: sol_center[1] < min_y and mean_x - self.grid_size < sol_center[
                    0] < mean_x + self.grid_size,
                270: lambda sol_center: sol_center[0] < min_x and mean_y - self.grid_size < sol_center[
                    1] < mean_y + self.grid_size,
            },
            'behind': {
                0: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
                90: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                180: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
                270: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
            },
            "side of": {
                0: lambda sol_center: min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: min_x <= sol_center[0] <= max_x
            }
        }

        compare_func = comparison_dict.get(place_type).get(0)

        for solution in solutions:
            sol_center = solution[0]

            if compare_func(sol_center):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_distance(self, distance_type, target_object, solutions):
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        distances = []
        valid_solutions = []
        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            distance = target_poly.distance(sol_poly)
            distances.append(distance)

            solution[-1] = distance
            valid_solutions.append(solution)

        min_distance = min(distances)
        max_distance = max(distances)

        if distance_type == "near":
            if min_distance < 150:
                points = [(min_distance, 1), (150, 0), (max_distance, 0)]
            else:
                points = [(min_distance, 0), (max_distance, 0)]

        elif distance_type == "far":
            points = [(min_distance, 0), (max_distance, 1)]

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        f = interp1d(x, y, kind='linear', fill_value='extrapolate')

        for solution in valid_solutions:
            distance = solution[-1]
            solution[-1] = float(f(distance))

        return valid_solutions

    def place_face(self, face_type, target_object, solutions):
        if face_type == "face to":
            return self.place_face_to(target_object, solutions)

        elif face_type == "face same as":
            return self.place_face_same(target_object, solutions)

        elif face_type == "face opposite to":
            return self.place_face_opposite(target_object, solutions)

    def place_face_to(self, target_object, solutions):
        # Define unit vectors for each rotation
        unit_vectors = {
            0: np.array([0., 1.]),  # Facing up
            90: np.array([1., 0.]),  # Facing right
            180: np.array([0., -1.]),  # Facing down
            270: np.array([-1., 0.])  # Facing left
        }

        target_coords = target_object[2]
        target_poly = Polygon(target_coords)

        valid_solutions = []

        for solution in solutions:
            sol_center = solution[0]
            sol_rotation = solution[1]

            # Define an arbitrarily large point in the direction of the solution's rotation
            far_point = sol_center + 1e6 * unit_vectors[sol_rotation]

            # Create a half-line from the solution's center to the far point
            half_line = LineString([sol_center, far_point])

            # Check if the half-line intersects with the target polygon
            if half_line.intersects(target_poly):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_same(self, target_object, solutions):
        target_rotation = target_object[1]
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_opposite(self, target_object, solutions):
        target_rotation = (target_object[1] + 180) % 360
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_alignment_center(self, alignment_type, target_object, solutions):
        target_center = target_object[0]
        valid_solutions = []
        eps = 5
        for solution in solutions:
            sol_center = solution[0]
            if abs(sol_center[0] - target_center[0]) < eps or abs(sol_center[1] - target_center[1]) < eps:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)
        return valid_solutions

    def visualize_grid(self, room_poly, grid_points, solutions):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 22

        # create a new figure
        fig, ax = plt.subplots()

        # draw the room
        x, y = room_poly.exterior.xy
        ax.plot(x, y, '-', label='Room', color='black', linewidth=2)

        # draw the grid points
        grid_x = [point[0] for point in grid_points]
        grid_y = [point[1] for point in grid_points]
        ax.plot(grid_x, grid_y, 'o', markersize=2, color="grey")

        # draw the solutions
        for object_name, solution in solutions.items():
            center, rotation, box_coords = solution[:3]
            center_x, center_y = center

            # create a polygon for the solution
            obj_poly = Polygon(box_coords)
            x, y = obj_poly.exterior.xy
            ax.plot(x, y, '-', linewidth=2, color='black')

            # ax.text(center_x, center_y, object_name, fontsize=18, ha='center')

            # set arrow direction based on rotation
            if rotation == 0:
                ax.arrow(center_x, center_y, 0, 25, head_width=10, fc='black')
            elif rotation == 90:
                ax.arrow(center_x, center_y, 25, 0, head_width=10, fc='black')
            elif rotation == 180:
                ax.arrow(center_x, center_y, 0, -25, head_width=10, fc='black')
            elif rotation == 270:
                ax.arrow(center_x, center_y, -25, 0, head_width=10, fc='black')
        # axis off
        ax.axis('off')
        ax.set_aspect('equal', 'box')  # to keep the ratios equal along x and y axis
        create_time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
        plt.savefig(f"{create_time}.pdf", bbox_inches='tight', dpi=300)
        plt.show()


if __name__ == "__main__":
    solver = DFS_Solver_Floor(max_duration=30, grid_size=50)
    solver.test_dfs_placement()
    # solver.test_milp_placement(simple=False, use_milp=True)