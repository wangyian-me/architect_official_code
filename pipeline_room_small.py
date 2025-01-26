from gpt_4.prompts.utils import find_large
from scene_to_all_large import scene_to_all_large
from argparse import ArgumentParser
import numpy as np
import subprocess
import traceback
import logging
import json
import os



parser = ArgumentParser()
parser.add_argument("--mask", action='store_true', default=False)
parser.add_argument("--work_dir", type=str, default='./data/architect_results')
parser.add_argument("--view_id", type=str, default="0")
parser.add_argument("--room_x", type=int, default=5)
parser.add_argument("--room_y", type=int, default=5)
parser.add_argument("--wall_texture_dir", type=str, default="./data/blenderkit_data_annotated/3e8f92f3-2840-47af-afaa-8cd11f9bc641/wall-basecolor.jpg")
parser.add_argument("--floor_texture_dir", type=str, default="./data/wood_floor_diff_4k.jpg")
parser.add_argument("--prompt", type=str, default="living room")
parser.add_argument("--scene_name", type=str, default="living_room_0")
args = parser.parse_args()

if not args.run_depth_only:
    from get_scene import get_scene, ObjectNumUnreached
    from inpaint_sdxl import inpainting

import t2v_metrics.t2v_metrics
clip_flant5_score = t2v_metrics.t2v_metrics.VQAScore(model='clip-flant5-xxl')

def get_command(args):
    command = ['python', 'place_room_small.py']
    command.append('--work_dir')
    command.append(args.work_dir)
    command.append('--output_dir')
    command.append(args.output_dir)
    command.append("--obj_name")
    command.append(args.obj_name)
    command.append("--all_small_path")
    command.append(args.all_small_path)
    command.append('--room_x')
    command.append(str(args.room_x))
    command.append('--room_y')
    command.append(str(args.room_y))
    command.append('--wall_texture_dir')
    command.append(args.wall_texture_dir)
    command.append('--floor_texture_dir')
    command.append(args.floor_texture_dir)
    if args.obj_kind == "table":
        command.append('--table')
    return command

def sub_run(args):
    command = get_command(args)
    logging.info(command)
    result = subprocess.run(command, capture_output=False, text=False)
    # logging.info("STDOUT:", result.stdout)
    # logging.info("STDERR:", result.stderr)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    file_handler = logging.FileHandler(os.path.join(args.work_dir, 'pipeline.log'))
    file_handler.setLevel(logging.CRITICAL)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.disabled = True

basic_dir = args.work_dir
os.makedirs(basic_dir, exist_ok=True)
setup_logging()

base_dir = os.path.join(basic_dir, args.scene_name)
all_large_path = os.path.join(base_dir, 'all_large.json')

if not os.path.exists(all_large_path):
    print("No large objects found! Please place large furniture first!")
    exit(0)

with open(all_large_path, 'r') as file:
    all_large = json.load(file)

# with open(os.path.join(base_dir, "furniture_to_place.json"), 'r') as file:
#     furniture_json = json.load(file)
# furniture_to_place = furniture_json["names"]
# kind = furniture_json["kinds"]
# prompts = furniture_json["prompts"]

furniture_to_place, kind, prompts = find_large(all_large_path, args.prompt)

furniture_json = {"names": furniture_to_place, "kinds": kind, "prompts": prompts}
with open(os.path.join(base_dir, "furniture_to_place.json"), 'w') as file:
    json.dump(furniture_json, file, indent=2)


all_small_json = {}

for k in range(len(furniture_to_place)):
    obj_name = furniture_to_place[k]
    obj_kind = kind[k]
    args.obj_name = obj_name


    inpainting_prompt = prompts[k]
    negative_prompt = "distorted, low quality, cartoon"

    args.work_dir = base_dir
    args.output_dir = os.path.join(base_dir, f"{obj_name}_small")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.all_small_path = "None"
    sub_run(args)

    input_dir = args.output_dir
    output_dir = args.output_dir

    mask_dir = os.path.join(input_dir, f'mask.png')
    depth_dir = os.path.join(input_dir, f'depth.npy')
    img_toinpaint_dir = os.path.join(input_dir, f'prev_scene.png')
    inpaint_img_dir = os.path.join(input_dir, 'scene.jpg')
    inpaint_mask_dir = os.path.join(input_dir, f'inpaint_mask.png')

    max_score = 0
    max_id = 0
    for k in range(10):
        inpaint_img_dir = os.path.join(input_dir, f'scene_{k}.jpg')
        inpainting(img_toinpaint_dir, inpaint_mask_dir, inpaint_img_dir, inpainting_prompt, negative_prompt)
        score = clip_flant5_score(images=[inpaint_img_dir], texts=[inpainting_prompt])
        if score > max_score:
            max_id = k
            max_score = score

    inpaint_img_dir = os.path.join(input_dir, f'scene_{max_id}.jpg')
    print("max_score:", max_score)

    intrinsic_K = np.load(os.path.join(input_dir, 'intrinsic_K.npy'))
    camera_pose = np.load(os.path.join(input_dir, f'cam_pose.npy'))

    ignore_dims = []

    if "shelf" in obj_kind:
        for obj_info in all_large["objects"]:
            if obj_info['name'] == obj_name:
                if 'y' in obj_info["facing"]:
                    ignore_dims = [1]
                else:
                    ignore_dims = [0]
                break

    get_scene(input_dir, output_dir, 
                intrinsic_K, camera_pose, 
                mask_dir, depth_dir, 
                large_obj=False, image_idx=max_id, ignore_dims=ignore_dims)

    args.all_small_path = os.path.join(output_dir, "all_small.json")

    sub_run(args)

    with open(args.all_small_path, 'r') as file:
        all_small = json.load(file)
    
    all_small_json[obj_name] = all_small

with open(os.path.join(base_dir, "all_small.json"), 'w') as file:
    json.dump(all_small_json, file, indent=2)
