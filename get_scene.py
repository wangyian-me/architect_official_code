
import os, sys
from ram_utils.utils_dino15 import mask_and_save, get_bbox, backproject_depth_to_pointcloud, save_point_cloud_as_ply, fit_euler_and_scale_simplified
from gpt_4.prompts.utils import get_tags_and_descriptions, parse_tags_and_descriptions, get_tags_small
from openai import AzureOpenAI, OpenAI
from Retriever import BlenderkitRetriever
import numpy as np
import shutil
import yaml
import math
import cv2
import json
import random
import logging

class ObjectNumUnreached(Exception):
    pass

def along_wall(bbox, room_bbox):
    ret = False
    distance_threshold = 0.4
    if bbox[0][0] < room_bbox[0][0] + distance_threshold:
        ret = True
    if bbox[1][0] > room_bbox[1][0] - distance_threshold:
        ret = True
    if bbox[0][1] < room_bbox[0][1] + distance_threshold:
        ret = True
    if bbox[1][1] > room_bbox[1][1] - distance_threshold:
        ret = True
    if ret == False:
        logging.warning("wall_object deleted!!!")
    return ret

def in_the_corner(bbox, room_bbox):
    ret = False
    distance_threshold = 0.2
    if bbox[0][0] < room_bbox[0][0] + distance_threshold or bbox[1][0] > room_bbox[1][0] - distance_threshold:
        if bbox[0][1] < room_bbox[0][1] + distance_threshold or bbox[1][1] > room_bbox[1][1] - distance_threshold:
            ret = True
    return ret


def on_floor(bbox, room_bbox):
    # this might not be accurate!!! maybe use vlm to do this!!!
    height_threshold = 0.5
    return bbox[0][2] - room_bbox[0][2] < height_threshold # likely to be on_floor, if the object is in DINO detect result but not in VLM answer;

def get_dist(bbox1, bbox2):
    left1, top1, right1, bottom1 = bbox1[0][0], bbox1[0][1], bbox1[1][0], bbox1[1][1]
    left2, top2, right2, bottom2 = bbox2[0][0], bbox2[0][1], bbox2[1][0], bbox2[1][1]

    # Calculate horizontal distance
    if right1 < left2:
        horizontal_distance = left2 - right1
    elif right2 < left1:
        horizontal_distance = left1 - right2
    else:
        horizontal_distance = 0

    # Calculate vertical distance
    if bottom1 < top2:
        vertical_distance = top2 - bottom1
    elif bottom2 < top1:
        vertical_distance = top1 - bottom2
    else:
        vertical_distance = 0

    return (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5


def get_relative(bbox1, bbox2, place_type):
    min_x, min_y, max_x, max_y = bbox1[0][0], bbox1[0][1], bbox1[1][0], bbox1[1][1]
    mean_x = (min_x + max_x) / 2
    mean_y = (min_y + max_y) / 2
    grid_size = 0.1
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
            0: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,  # in front of and centered
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
    center = [(bbox2[0][0] + bbox2[1][0]) / 2, (bbox2[0][1] + bbox2[1][1]) / 2]
    return compare_func(center)


def get_center(bbox):
    return (bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2

def size(bbox):
    return (bbox[1][0] - bbox[0][0]) ** 2 + (bbox[1][1] - bbox[0][1]) ** 2 + (bbox[1][2] - bbox[0][2]) ** 2

def is_in(x, y, z, bbox):
    if x > bbox[0][0] and x < bbox[1][0] and y > bbox[0][1] and y < bbox[1][1] and z > bbox[0][2] and z < bbox[1][2]:
        return True
    return False

def is_on(bbox1, bbox2):
    x, y = get_center(bbox1)
    if is_in(x, y, bbox1[0][2], bbox2):
        return True
    else:
        return False


def get_seg_only(input_img_dir, output_dir, tags):
    masks, pred_phrases, box_list = mask_and_save(input_img_dir, output_dir, tags)

def get_scene(input_dir, output_dir, intrinsic_K, camera_pose, input_mask_dir, input_depth_dir, large_obj=True, scene_name=None, tags=None, image_idx=None, all_edge=False, database_type="objaverse", room_x=5, room_y=6, num_obj=0, ignore_dims=[]):
    input_img_dir = os.path.join(input_dir, 'scene.jpg')
    if image_idx is not None:
        input_img_dir = os.path.join(input_dir, f'scene_{image_idx}.jpg')
    if input_depth_dir is None:
        input_depth_dir = os.path.join(input_dir, 'depth.npy')
    if input_mask_dir is None:
        input_mask_dir = os.path.join(input_dir, 'mask.png')
    previous_scene_dir = os.path.join(input_dir, 'scene.json')
    reference_depth = np.load(input_depth_dir)
    masked_images_dir = os.path.join(output_dir, 'masked_images')

    mask = cv2.imread(input_mask_dir, cv2.IMREAD_COLOR)
    # mask = np.where((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255))
    # logging.debug(mask.shape)
    inpaint_mask = cv2.imread(os.path.join(input_dir, 'inpaint_mask.png'), cv2.IMREAD_COLOR)
    binary_mask = np.all(inpaint_mask == [0, 0, 0], axis=-1).astype(np.uint8)
    object_mask = None
    if not large_obj:
        object_mask = cv2.imread(os.path.join(input_dir, 'mask.png'), cv2.IMREAD_COLOR)
        object_mask = np.all(object_mask == [255, 255, 255], axis=-1).astype(np.uint8)
    # logging.debug(f"length of mask object mask: {binary_mask.shape}")
    mask = np.where((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255))

    object_list = []
    size_list = {}
    name_to_uid = {}

    import json

    if os.path.exists(previous_scene_dir):
        with open(previous_scene_dir, 'r') as file:
            data = json.load(file)
    else:
        data = {'floor_objects': [], 'wall_objects': []}

    previous_floor_placements = data['floor_objects']
    previous_wall_placements = data['wall_objects']

    init_name_dict = {}
    retriever = BlenderkitRetriever("/work/pi_chuangg_umass_edu/yianwang_umass_edu-data/blenderkit_data_annotated")
    for obj in previous_floor_placements:
        name = obj['object_name']
        if name.split('-')[0] in init_name_dict:
            init_name_dict[name.split('-')[0]] = max(init_name_dict[name.split('-')[0]], int(name.split('-')[1]) + 1)
        else:
            init_name_dict[name.split('-')[0]] = int(name.split('-')[1]) + 1
        name_to_uid[name] = obj['assetId']
        uid = obj['assetId']
        x = retriever.database[uid]["bounding_box_extents"][0]
        y = retriever.database[uid]["bounding_box_extents"][2]
        if retriever.database[uid]["front_view_rotation"][1] % 180 == 90:
            x, y = y, x

        # in our dataset, default facing to x-, but in holodeck, default facing to y+
        size_list[name] = {
            'x': y,
            'y': x,
            'z': retriever.database[uid]["bounding_box_extents"][1]
        }

    for obj in previous_wall_placements:
        name = obj['object_name']
        if name.split('-')[0] in init_name_dict:
            init_name_dict[name.split('-')[0]] = max(init_name_dict[name.split('-')[0]], int(name.split('-')[1]) + 1)
        else:
            init_name_dict[name.split('-')[0]] = int(name.split('-')[1]) + 1
        name_to_uid[name] = obj['assetId']

        uid = obj['assetId']
        x = retriever.database[uid]["bounding_box_extents"][0]
        y = retriever.database[uid]["bounding_box_extents"][2]
        if retriever.database[uid]["front_view_rotation"][1] % 180 == 90:
            x, y = y, x
        # in our dataset, default facing to x-, but in holodeck, default facing to y+
        size_list[name] = {
            'x': y,
            'y': x,
            'z': retriever.database[uid]["bounding_box_extents"][1]
        }

    logging.info(f"init name dict: {init_name_dict}")
    logging.info(f"previous floor placements: {previous_floor_placements}")

    if large_obj:
        if tags is not None:
            tags, descriptions, on_floors = parse_tags_and_descriptions(tags)
        else:
            tags, descriptions, on_floors = get_tags_and_descriptions(input_img_dir, scene_name) # get tags by gpt4v and return two list
        with open(os.path.join(output_dir, 'tags_and_description.json'), 'w') as json_file:
            json.dump({tags[i]: {"description": descriptions[i], "on_floor": on_floors[i]} for i in range(len(tags))}, json_file)
        tags_string = ' . '.join(tags)
        logging.info(f"tags: {tags_string}")
        #"sofa, table, rug, painting, vase, bowl, book, cushion"
        logging.info(f"descriptions: {descriptions}")
        descriptions = {tags[i]: descriptions[i] for i in range(len(tags))}
        on_floors = {tags[i]: on_floors[i] for i in range(len(tags))}
        logging.info(f"on_floors: {on_floors}")
    else:
        # tags = get_tags_small(input_img_dir, scene_name)
        # tags_string = ' . '.join(tags)
        tags_string = None
        descriptions = None
        on_floors = None
    result = get_bbox(
        input_img_dir, output_dir,
        tags_string, descriptions, on_floors,
        camera_pose=camera_pose,
        intrinsic_K=intrinsic_K,
        reference_depth=reference_depth,
        reference_mask=mask,
        binary_mask=binary_mask,
        name_dict=init_name_dict,
        filter=(not large_obj),
        object_mask=object_mask,
        mask_path=input_mask_dir,
        depth_path=input_depth_dir
    )
    if len(result) - 1 < num_obj:
        raise ObjectNumUnreached
    with open(os.path.join(output_dir, f'result.json'), 'w') as json_file:
        json.dump(result, json_file, indent=2)

    if not large_obj:
        small_list = []
        for i in range(1, len(result)):
            bbox = result[i]['bbox']
            name = result[i]['name']
            # uid = retriever.retrieve(name, name, np.array(result[i]['bbox'][1]) - np.array(result[i]['bbox'][0]), is_large=False)
            uid = retriever.retrieve_hybrid(
                name, 
                name, 
                np.array(bbox[1]) - np.array(bbox[0]), 
                is_large=False,
                query_image_path=os.path.join(output_dir, "masked_images", f"{name}.png"),
                top_k=5
            )
            object_bbox = np.array(retriever.database[uid]["bounding_box_extents"])
            target_bbox = np.array(result[i]['bbox'][1]) - np.array(result[i]['bbox'][0])
            best_euler, best_scale, min_error = fit_euler_and_scale_simplified(
                object_bbox,
                target_bbox,
                retriever.database[uid]["stable_rotations"],
                ignore_dims=ignore_dims
            )
            pos = (np.array(result[i]['bbox'][1]) + np.array(result[i]['bbox'][0])) / 2
            small_list.append({
                "uid": uid,
                "name": name,
                "pos": pos.tolist(),
                "euler": best_euler,
                "scale": best_scale
            })

        small_list = {"objects": small_list}

        with open(os.path.join(output_dir, 'all_small.json'), 'w') as json_file:
            json.dump(small_list, json_file, indent=2)

        return

    # print(result[0]['room_bbox'])
    room_bbox = result[0]['room_bbox']


    floor_objects = []
    wall_objects = []
    small_objects = []
    sizes_floor = []
    constraint_floor = []
    constraint_wall = []
    sizes = []
    filter_object_names = ['carpet', 'rug', 'fireplace']
    ############ todo: all these codes might be replaced by gpt4v
    for i in range(1, len(result)):
        bbox = result[i]['bbox']
        if result[i]['on_floor'] is None:
            on_floor_result = on_floor(bbox, room_bbox)
        else:
            on_floor_result = result[i]['on_floor']
        filtered = False
        for filter_object_name in filter_object_names:
            if filter_object_name in result[i]['name']:
                filtered = True
                break
        if filtered: continue
        if on_floor_result:
            floor_objects.append(result[i])
            
            if in_the_corner(bbox, room_bbox):
                constraint_floor.append(["corner"])
                # place corner first
                sizes.append(-size(bbox) - 10)
            elif along_wall(bbox, room_bbox) or all_edge:
                constraint_floor.append(["edge"])
                # place edge first
                sizes.append(-size(bbox) - 5)
            else:
                sizes.append(-size(bbox))
                direction = "horizontal" if bbox[1][0] - bbox[0][0] > bbox[1][1] - bbox[0][1] else "vertical"
                constraint_floor.append([f"middle, {direction}"])
            constraint_floor[-1].append(("position", [(bbox[0][j]+bbox[1][j]) / 2 for j in range(3)]))

    logging.info(f"floor_objects: {floor_objects}")
    ###############################################################

    indices = np.argsort(sizes)
    for i in range(len(indices)):
        idxi = indices[i]

        for j in range(i + 1, len(indices)):
            idxj = indices[j]
            bbox1, bbox2 = floor_objects[idxi]['bbox'], floor_objects[idxj]['bbox']
            if get_dist(bbox1, bbox2) < 1.5:
                constraint_floor[idxj].append(("near", idxi))

            cx1, cy1 = get_center(bbox1)
            cx2, cy2 = get_center(bbox2)

            extendx = (bbox1[1][0] - bbox1[0][0] + bbox2[1][0] - bbox2[0][0]) / 2
            extendy = (bbox1[1][1] - bbox1[0][1] + bbox2[1][1] - bbox2[0][1]) / 2

            if abs(cx1 - cx2) < 0.2 * extendx or abs(cy1 - cy2) < 0.2 * extendy:
                constraint_floor[idxj].append(("center aligned", idxi))

            relatives = ['left of', 'right of', 'in front of', 'behind']
            for relative in relatives:
                if get_relative(bbox1, bbox2, relative):
                    constraint_floor[idxj].append((relative, idxi))
                    break

    constraint_floor_string = ""
    for idx in indices:
        location = constraint_floor[idx][1][1]
        location_str = '[' + ', '.join([f'{loc:.3f}' for loc in location]) + ']'
        constraint_floor_string += f"{floor_objects[idx]['name']} | {constraint_floor[idx][0]} | location, {location_str}"
        # constraint_floor_string += floor_objects[idx]['name'] + " | " + constraint_floor[idx][0] + " | " + f"location, {location_str}"
        for j in range(2, len(constraint_floor[idx])):
            constraint_floor_string += " | " + constraint_floor[idx][j][0] + ", " + \
                                       floor_objects[constraint_floor[idx][j][1]]['name']

        for obj in previous_floor_placements:
            vertices = np.array(obj['vertices'])
            x_l = vertices[:, 0].min() / 100
            x_r = vertices[:, 0].max() / 100
            y_l = vertices[:, 1].min() / 100
            y_r = vertices[:, 1].max() / 100
            bbox1 = [[x_l, y_l, 0], [x_r, y_r, obj['position']['y'] * 2]]
            bbox2 = floor_objects[idx]['bbox']
            if get_dist(bbox1, bbox2) < 1.5:
                constraint_floor_string += " | near, " + obj['object_name']
            cx1, cy1 = get_center(bbox1)
            cx2, cy2 = get_center(bbox2)

            extendx = (bbox1[1][0] - bbox1[0][0] + bbox2[1][0] - bbox2[0][0]) / 2
            extendy = (bbox1[1][1] - bbox1[0][1] + bbox2[1][1] - bbox2[0][1]) / 2

            if abs(cx1 - cx2) < 0.2 * extendx or abs(cy1 - cy2) < 0.2 * extendy:
                constraint_floor_string += " | center aligned, " + obj['object_name']

            relatives = ['left of', 'right of', 'in front of', 'behind']
            for relative in relatives:
                if get_relative(bbox1, bbox2, relative):
                    constraint_floor_string += " | " + relative + ", " + obj['object_name']
                    break

        constraint_floor_string += "\n"

    logging.debug(f"constraint_floor_string: {constraint_floor_string}")

    no_retrieved_idx = []

    object_image_dir = os.path.join(output_dir, "masked_images")

    for i in range(len(floor_objects)):
        uid = retriever.retrieve_hybrid(
            floor_objects[i]['name'].split('-')[0], 
            floor_objects[i]['description'], 
            np.array(floor_objects[i]['bbox'][1]) - np.array(floor_objects[i]['bbox'][0]), 
            is_large=True, 
            query_image_path=os.path.join(object_image_dir, f"{floor_objects[i]['name']}.png"),
        )

        if uid is None:
            logging.warning(f"no object retrieved, skip object: {floor_objects[i]['name']}")
            no_retrieved_idx.append(i)
            continue

        object_list.append((floor_objects[i]['name'], uid))
        # note that, the bbox in the database is x, z, y rather than x, y, z

        name_to_uid[floor_objects[i]['name']] = uid
        # bbox = floor_objects[i]['bbox']
        x = retriever.database[uid]["bounding_box_extents"][0]
        y = retriever.database[uid]["bounding_box_extents"][2]
        if retriever.database[uid]["front_view_rotation"][1] % 180 == 90:
            x, y = y, x

        # in our dataset, default facing to x-, but in holodeck, default facing to y+
        size_list[floor_objects[i]['name']] = {
            'x': y,
            'y': x,
            'z': retriever.database[uid]["bounding_box_extents"][1]
        }
        logging.info(f"floor objects: {floor_objects[i]['name']} {uid}")

    for idx in no_retrieved_idx[::-1]:
        floor_objects.pop(idx)

    # print(floor_objects)
    # # This is for object generation plan in floor and wall
    from holodeck_utils.floor_objects import FloorObjectGenerator
    from holodeck_utils.wall_objects import WallObjectGenerator

    llm = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    room = {"vertices": [[0, 0], [0, room_y], [room_x, room_y], [room_x, 0]], "x": room_x*100, "y": room_y*100}
    floor_objects_generator = FloorObjectGenerator(llm)
    # print(room)
    # print(constraint_floor_string)
    # print(object_list)
    # print(size_list)
    # print(previous_floor_placements)
    objects = floor_objects_generator.generate_objects_per_room(room, constraint_floor_string, object_list, size_list, previous_floor_placements)
    placements = floor_objects_generator.solution2placement(objects, name_to_uid, size_list)
    logging.debug(f"floor object plan: {placements}")


    for i in range(1, len(result)):
        bbox = result[i]['bbox']
        if result[i]['on_floor'] is None:
            along_wall_result = along_wall(bbox, room_bbox)
        else:
            along_wall_result = not result[i]['on_floor']
        if along_wall_result:
            wall_objects.append(result[i])
            constraint_wall.append([("position", [(bbox[0][j]+bbox[1][j]) / 2 for j in range(3)])])
            for j in range(len(floor_objects)):
                bbox2 = floor_objects[j]['bbox']
                if get_dist(bbox, bbox2) == 0:
                    constraint_wall[-1].append(("above", j))
                    break


    constraint_wall_string = ""
    logging.debug(f"wall_objects: {wall_objects}")
    logging.debug(f"constraint_wall: {constraint_wall}")
    # wall_objects = wall_objects[:1] #change
    for i in range(len(wall_objects)):
        location = constraint_wall[i][0][1]
        location_str = '[' + ', '.join([f'{loc:.3f}' for loc in location]) + ']'
        constraint_wall_string += wall_objects[i]['name']+ " | " + f"location, {location_str}"
        for j in range(1, len(constraint_wall[i])):
            constraint_wall_string += " | " + constraint_wall[i][j][0] + ", " + \
                                       floor_objects[constraint_wall[i][j][1]]['name']
        constraint_wall_string += '\n'

    logging.debug(f"constraint_wall_string: {constraint_wall_string}")

    with open(os.path.join(output_dir, f'constraint_without_llm.txt'), 'w') as file:
        file.write(f"constraint_floor_string:\n{constraint_floor_string}\n\nconstraint_wall_string:\n{constraint_wall_string}")

    wall_object_list = []
    wall_size_list = {}
    for i in range(len(wall_objects)):
        # todo: also use bbox in select objects
        uid = retriever.retrieve_hybrid(
            wall_objects[i]['name'].split('-')[0], 
            wall_objects[i]['description'], 
            np.array(wall_objects[i]['bbox'][1]) - np.array(wall_objects[i]['bbox'][0]), 
            is_large=True,
            query_image_path=os.path.join(object_image_dir, f"{wall_objects[i]['name']}.png"),
        )

        if uid is None:
            logging.warning(f"no object retrieved, skip object: {wall_objects[i]['name']}")
            continue

        wall_object_list.append((wall_objects[i]['name'], uid))
        # note that, the bbox in the database is x, z, y rather than x, y, z

        if retriever.database[uid]["codimension"]:
            x = retriever.database[uid]["bounding_box_extents"][0]
            y = retriever.database[uid]["bounding_box_extents"][1]
            z = retriever.database[uid]["bounding_box_extents"][2]
            if retriever.database[uid]["front_view_rotation"][0] % 180 == 90:
                y, z = z, y
            if retriever.database[uid]["front_view_rotation"][1] % 180 == 90:
                x, z = z, x
            if retriever.database[uid]["front_view_rotation"][2] % 180 == 90:
                x, y = y, x
            wall_size_list[wall_objects[i]['name']] = {
                'x': y,
                'y': x,
                'z': z
            }

        else:
            # in our dataset, default facing to x-, but in holodeck, default facing to y+
            # bbox = floor_objects[i]['bbox']
            x = retriever.database[uid]["bounding_box_extents"][0]
            y = retriever.database[uid]["bounding_box_extents"][2]
            if retriever.database[uid]["front_view_rotation"][1] % 180 == 90:
                x, y = y, x
            wall_size_list[wall_objects[i]['name']] = {
                'x': y,
                'y': x,
                'z': retriever.database[uid]["bounding_box_extents"][1]
            }

        logging.info(f"wall objects: {wall_objects[i]['name']} {uid}")

    wall_object_generator = WallObjectGenerator(llm)
    wall_objects = wall_object_generator.generate_wall_objects_per_room(room, constraint_wall_string, wall_object_list,
                                                                        wall_size_list, placements + previous_wall_placements)
    # logging.debug(wall_objects)

    data = {'floor_objects': placements, 'wall_objects': wall_objects + previous_wall_placements}

    # Dumping the dictionary into a JSON file
    with open(os.path.join(output_dir, 'scene.json'), 'w') as json_file:
        json.dump(data, json_file, indent=2)

if __name__ == '__main__':

    input_dir = "/project/pi_chuangg_umass_edu/yian/robogen/data/image_to_scene/grocery/beverages_input"
    output_dir = "/project/pi_chuangg_umass_edu/yian/robogen/data/image_to_scene/grocery/beverages_output"

    # predict_fov_and_pos(input_dir, output_dir)
    intrinsic_K = np.load(os.path.join(input_dir, 'intrinsic_K.npy'))
    camera_pose = np.load(os.path.join(input_dir, 'cam_pose.npy'))

    get_scene(input_dir, output_dir, intrinsic_K, camera_pose, large_obj=False, scene_name='beverages', tags='bottles')