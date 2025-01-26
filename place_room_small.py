import os

from genesis_utils.place_small import genesis_shelf, genesis_table
import cv2
import numpy as np

from argparse import ArgumentParser
import random
import time
import json

parser = ArgumentParser()
parser.add_argument("--work_dir", type=str, default='')
parser.add_argument("--output_dir", type=str, default='None')
parser.add_argument("--obj_name", type=str, default='')
parser.add_argument("--all_small_path", type=str, default='None')
parser.add_argument("--room_x", type=int, default=5)
parser.add_argument("--room_y", type=int, default=6)
parser.add_argument("--wall_texture_dir", type=str, default="/project/pi_chuangg_umass_edu/yian/robogen/data/objaverse_obj/plastered_wall_diff_4k.jpg")
parser.add_argument("--floor_texture_dir", type=str, default="/project/pi_chuangg_umass_edu/yian/robogen/data/objaverse_obj/wood_floor_diff_4k.jpg")
parser.add_argument("--table", action='store_true', default=False)
args = parser.parse_args()

args.resolution = (1024, 1024)
args.room = (args.room_x, args.room_y)

if args.output_dir == 'None':
    args.output_dir = args.work_dir

if args.table:
    genesis_table(args)
else:
    genesis_shelf(args)