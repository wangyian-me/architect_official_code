from genesis_utils.place_scene import genesis_room

from argparse import ArgumentParser
import random

parser = ArgumentParser()
parser.add_argument("--mask", action='store_true', default=False)
parser.add_argument("--work_dir", type=str, default=None)
parser.add_argument("--view_id", type=str, default="0")
parser.add_argument("--room_x", type=int, default=5)
parser.add_argument("--room_y", type=int, default=6)
parser.add_argument("--wall_texture_dir", type=str, default="/project/pi_chuangg_umass_edu/yian/robogen/data/objaverse_obj/plastered_wall_diff_4k.jpg")
parser.add_argument("--floor_texture_dir", type=str, default="/project/pi_chuangg_umass_edu/yian/robogen/data/objaverse_obj/wood_floor_diff_4k.jpg")
parser.add_argument("--database_type", type=str, default="blenderkit", choices=["objaverse", "blenderkit"])
args = parser.parse_args()

args.room = (args.room_x, args.room_y)

genesis_room(args)