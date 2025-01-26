from gpt_4.prompts.utils import get_inpainting_prompt
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
parser.add_argument("--work_dir", type=str, default='/work/pi_chuangg_umass_edu/yianwang_umass_edu-data/architect_results')
parser.add_argument("--view_id", type=str, default="0")
parser.add_argument("--inpaint_only", action='store_true', default=False)
parser.add_argument("--run_depth_only", action='store_true', default=False)
parser.add_argument("--l", type=int, default=0)
parser.add_argument("--r", type=int, default=1)
parser.add_argument("--room_x", type=int, default=5)
parser.add_argument("--room_y", type=int, default=5)
parser.add_argument("--wall_texture_dir", type=str, default="/work/pi_chuangg_umass_edu/yianwang_umass_edu-data/blenderkit_data_annotated/3e8f92f3-2840-47af-afaa-8cd11f9bc641/wall-basecolor.jpg")
parser.add_argument("--floor_texture_dir", type=str, default="/project/pi_chuangg_umass_edu/yian/robogen/data/objaverse_obj/wood_floor_diff_4k.jpg")
parser.add_argument("--database_type", type=str, default="objaverse", choices=["objaverse", "blenderkit"])
parser.add_argument("--prompt", type=str, default="living room")
parser.add_argument("--iter_num", type=int, default=2)
args = parser.parse_args()

if not args.run_depth_only:
    from get_scene import get_scene, ObjectNumUnreached
    from inpaint_sdxl import inpainting

import t2v_metrics.t2v_metrics
clip_flant5_score = t2v_metrics.t2v_metrics.VQAScore(model='clip-flant5-xxl')

def get_command(args):
    command = ['python', 'place_room.py']
    command.append('--work_dir')
    command.append(args.work_dir)
    command.append('--view_id')
    command.append(str(args.view_id))
    command.append('--room_x')
    command.append(str(args.room_x))
    command.append('--room_y')
    command.append(str(args.room_y))
    command.append('--wall_texture_dir')
    command.append(args.wall_texture_dir)
    command.append('--floor_texture_dir')
    command.append(args.floor_texture_dir)
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

scene_list = []
prompt_list = []
for i in range(10):
    scene_list.append(f"living_room_{i + 10}")
    prompt_list.append("living room")

for i in range(10):
    scene_list.append(f"bedroom_{i + 10}")
    prompt_list.append("bedroom")

for i in range(10):
    scene_list.append(f"children_room_{i + 10}")
    prompt_list.append("children room")

for i in range(10):
    scene_list.append(f"dinning_room_{i + 10}")
    prompt_list.append("dinning room")

for i in range(10):
    scene_list.append(f"office_{i + 10}")
    prompt_list.append("office")

for i in range(10):
    scene_list.append(f"study_room_{i + 10}")
    prompt_list.append("study room")

# texture_dir = '/project/pi_chuangg_umass_edu/yian/robogen/data/holodeck_data/data/materials/images'
# textures = os.listdir(texture_dir)

for j in range(args.l, args.r):
    # args.work_dir = f"/project/pi_chuangg_umass_edu/yian/robogen/data/image_to_scene_yian/{scene_list[j]}"

    base_dir = os.path.join(basic_dir, scene_list[j])
    os.makedirs(base_dir, exist_ok=True)
    # args.prompt = "Interior" + prompt_list[j] + "with white plaster walls and a wood floor, in the style of ray tracing, exacting precision, super high detail, photo realistic"
    
    # args.wall_texture_dir = os.path.join(texture_dir, textures[(j + 5) % len(textures)])
    # args.floor_texture_dir = os.path.join(texture_dir, textures[j % len(textures)])

    negative_prompt = "distorted, low quality, cartoon"
    scene_name = prompt_list[j]
    args.prompt = f"{scene_name} with multiple pieces of furnitures, exacting precision, super high detail, photo realistic"

    args.work_dir = os.path.join(base_dir, f"scene_0")
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    args.view_id = str(0)
    if not args.inpaint_only:
        sub_run(args)

    iter_time = 0
    run_again_cnt = 0
    while iter_time < args.iter_num and run_again_cnt < 4:
        i = iter_time
        args.mask = False
        args.work_dir = os.path.join(base_dir, f"scene_{i}")
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)

        # if args.run_depth_only:
        #     args.mask = True
        #     sub_run(args)
        #     break

        # if args.inpaint_only:
        #     args.mask = True
        #     sub_run(args)

        input_dir = args.work_dir
        output_dir = os.path.join(base_dir, f"scene_{i+1}")
        os.makedirs(output_dir, exist_ok=True)
        mask_dir = os.path.join(input_dir, f'mask.png')
        depth_dir = os.path.join(input_dir, f'depth.npy')
        img_toinpaint_dir = os.path.join(input_dir, f'prev_scene.png') #debug
        inpaint_img_dir = os.path.join(input_dir, 'scene.jpg')
        inpaint_mask_dir = os.path.join(input_dir, f'inpaint_mask.png')

        #todo: how to automatically generate positive prompt and negative prompt using gpt
        if args.inpaint_only:
            for k in range(10):
                inpaint_img_dir = os.path.join(input_dir, f'scene_{k}.jpg')
                inpainting(img_toinpaint_dir, inpaint_mask_dir, inpaint_img_dir, args.prompt, negative_prompt)
            break
        else:
            max_score = 0
            max_id = 0
            for k in range(10):
                inpaint_img_dir = os.path.join(input_dir, f'scene_{k}.jpg')
                inpainting(img_toinpaint_dir, inpaint_mask_dir, inpaint_img_dir, args.prompt, negative_prompt)
                score = clip_flant5_score(images=[inpaint_img_dir], texts=[scene_name])
                if score > max_score:
                    max_id = k
                    max_score = score
            inpaint_img_dir = os.path.join(input_dir, f'scene_{max_id}.jpg')
            print("max_score:", max_score)
            if max_score < 0.8:
                break

        intrinsic_K = np.load(os.path.join(input_dir, 'intrinsic_K.npy'))
        camera_pose = np.load(os.path.join(input_dir, f'cam_pose.npy'))

        try:
            get_scene(
                input_dir, output_dir,
                intrinsic_K, camera_pose,
                input_mask_dir=mask_dir,
                input_depth_dir=depth_dir,
                database_type=args.database_type,
                scene_name=prompt_list[j],
                room_x=args.room_x,
                room_y=args.room_y,
                num_obj=3 - iter_time,
                image_idx=max_id
            )
        except ObjectNumUnreached as e:
            logging.error("Number of objects is too little, run again")
            run_again_cnt += 1
            continue
        # except Exception as e:
        #     logging.error(traceback.format_exc())
        #     logging.error("ERROR, run again")
        #     continue
        with open(os.path.join(output_dir, "scene.json"), 'r') as f:
            scene = json.load(f)
        object_names = [floor_object['object_name'].split('-')[0] for floor_object in scene['floor_objects']]
        object_names += [wall_object['object_name'].split('-')[0] for wall_object in scene['wall_objects']]
        negative_objects, positive_objects = get_inpainting_prompt(object_names, scene_name)
        args.prompt = f"{scene_name} with multiple pieces of furnitures, such as {', '.join(positive_objects)}, exacting precision, super high detail, photo realistic"
        # args.prompt = "Interior" + prompt_list[j] + f"with white plaster walls and a wood floor, with {', '.join(positive_objects)}, in the style of ray tracing,  exacting precision, super high detail, photo realistic"
        negative_prompt = f"distorted, low quality, cartoon, {', '.join(negative_objects)}"

        if not args.inpaint_only:
            args.work_dir = os.path.join(base_dir, f"scene_{i+1}")
            args.view_id = str(i + 1)
            sub_run(args)

        iter_time += 1

    scene_path = os.path.join(base_dir, f'scene_{args.iter_num}', 'scene.json')
    if os.path.exists(scene_path):
        result = subprocess.run(['python', 'scene_to_all_large.py', '--scene_path', scene_path, '--output_dir', base_dir], capture_output=False, text=False)

# end of an iteration