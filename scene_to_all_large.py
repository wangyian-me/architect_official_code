import json
import os
import trimesh
import genesis as gs
import imageio

dataset_dir = os.environ.get("DATASET_PATH")

def scene_to_all_large(scene_path, output_dir):

    gs.init(seed=0, precision='32', logging_level='debug')
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(gravity=(0, 0, 0), enable_collision=False),
        # renderer=gs.renderers.RayTracer(
        #     # env_euler=(0, 0, 180)
        #     # env_texture=gs.textures.ColorTexture(color=(0.2, 0.2, 0.2)),
        #     env_surface=gs.surfaces.Emission(
        #         emissive_texture=gs.textures.ColorTexture(
        #             # image_path=hdr_path,
        #             color=(0.5, 0.5, 0.5),
        #         )
        #     ),
        #     lights=[
        #         # {'pos': (-4, 2, 4.0), 'color': (255, 220, 170), 'radius': 0.3, 'intensity': 1.0},
        #         # {'pos': (2, 10, 5.0), 'color': (255, 220, 170), 'radius': 0.3, 'intensity': 1.0},
        #         {'pos': (3, 3, 4.3), 'color': (200.0, 200.0, 200.0), 'radius': 0.3, 'intensity': 0.9},
        #         #  {'pos': (70, 4, 2.0), 'color': (255, 255, 200), 'radius': 7, 'intensity': 10.0},
        #         # {'pos': (-4, room[1]/2, 5.0), 'color': (255, 255, 150), 'radius': 0.6, 'intensity': 1.0},
        #     ]
        # ),
    )
    mat_rigid = gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16)

    with open(
            f"{dataset_dir}/blenderkit_database.json",
            'r') as file:
        data = json.load(file)

    with open(scene_path, 'r') as file:
        scene_dict = json.load(file)

    all_large_objects = dict()
    all_large = []
    front_dir = {
            0: 'y+',
            90: 'x-',
            180: 'y-',
            270: 'x+',
        }
    obj_list = []

    for obj_info in scene_dict["floor_objects"]:
        assetId = obj_info["assetId"]
        vertices = obj_info["vertices"]
        position = obj_info["position"]
        rotation = obj_info["rotation"]
        obj_name = obj_info["object_name"]

        bbox = data[assetId]["bounding_box_extents"]
        center = data[assetId]["center"]
        rotation_offset = [90, data[assetId]["front_view_rotation"][1], 0]
        rotation_offset[1] = (rotation_offset[1] + rotation['y'] + 90) % 360

        # height adjust
        half_height = -center[1] + bbox[1] / 2
        angle_mod = rotation_offset[1]

        if angle_mod == 0:
            x_center = center[2]
            y_center = center[0]
        elif angle_mod == 90:
            x_center = center[0]
            y_center = -center[2]
        elif angle_mod == 180:
            x_center = -center[2]
            y_center = -center[0]
        elif angle_mod == 270:
            x_center = -center[0]
            y_center = center[2]
        else:
            # Optional: handle other angles if needed, or raise an error
            raise ValueError(f"Unhandled rotation angle: {angle_mod}")

        obj_path = os.path.join(dataset_dir, f"{assetId}.glb")

        all_large.append({
            "assetId": assetId,
            "name": obj_name,
            "path": obj_path,
            "scale": 1.0,
            "pos": (position['x'] - x_center, position['z'] - y_center, half_height),
            "center": (position['x'], position['z'], bbox[1] / 2),
            "euler": (90, 0, rotation_offset[1]),
            "facing": front_dir[rotation['y']]
        })

        obj = scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                fixed=True,
                file=os.path.join(dataset_dir, f"{assetId}.glb"),
                scale=1.0,
                pos=(position['x'] - x_center, position['z'] - y_center, half_height),
                euler=(90, 0, rotation_offset[1]),
                collision=True,
                convexify=True,
                decompose_nonconvex=False
            ),
            # vis_mode="collision"
        )

        obj_list.append(obj)

    for obj_info in scene_dict["wall_objects"]:
        assetId = obj_info["assetId"]
        vertices = obj_info["vertices"]
        position = obj_info["position"]
        rotation = obj_info["rotation"]
        obj_name = obj_info["object_name"]

        bbox = data[assetId]["bounding_box_extents"]
        center = data[assetId]["center"]

        if data[assetId]["codimension"]:
            # codimension
            rotation_offset = list(data[assetId]["front_view_rotation"])
            rotation_offset[2] = (rotation_offset[2] + rotation['y'] + 90) % 360
            all_large.append({
                "assetId": assetId,
                "name": obj_name,
                "path": os.path.join(dataset_dir, f"{assetId}.glb"),
                "scale": 1.0,
                "pos": (position['x'], position['z'], position['y']),
                "center": (position['x'], position['z'], position['y']),
                "euler": rotation_offset,
                "facing": front_dir[rotation['y']]
            })

            obj = scene.add_entity(
                material=mat_rigid,
                morph=gs.morphs.Mesh(
                    fixed=True,
                    file=os.path.join(dataset_dir, f"{assetId}.glb"),
                    scale=1.0,
                    pos=(position['x'], position['z'], position['y']),
                    euler=rotation_offset,
                    collision=True,
                    convexify=True,
                    decompose_nonconvex=False
                ),
                # vis_mode="collision"
            )

            obj_list.append(obj)

        else:
            rotation_offset = [90, data[assetId]["front_view_rotation"][1], 0]
            rotation_offset[1] = (rotation_offset[1] + rotation['y'] + 90) % 360

            # height adjust
            half_height = -center[1]
            angle_mod = rotation_offset[1]

            if angle_mod == 0:
                x_center = center[2]
                y_center = center[0]
            elif angle_mod == 90:
                x_center = center[0]
                y_center = -center[2]
            elif angle_mod == 180:
                x_center = -center[2]
                y_center = -center[0]
            elif angle_mod == 270:
                x_center = -center[0]
                y_center = center[2]
            else:
                # Optional: handle other angles if needed, or raise an error
                raise ValueError(f"Unhandled rotation angle: {angle_mod}")

            all_large.append({
                "assetId": assetId,
                "name": obj_name,
                "path": os.path.join(dataset_dir, f"{assetId}.glb"),
                "scale": 1.0,
                "pos": (position['x'] - x_center, position['z'] - y_center, position['y'] + half_height),
                "center": (position['x'], position['z'], position['y']),
                "euler": (90, 0, rotation_offset[1]),
                "facing": front_dir[rotation['y']]
            })

            obj = scene.add_entity(
                material=mat_rigid,
                morph=gs.morphs.Mesh(
                    fixed=True,
                    file=os.path.join(dataset_dir, f"{assetId}.glb"),
                    scale=1.0,
                    pos=(position['x'] - x_center, position['z'] - y_center, position['y'] + half_height),
                    euler=(90, 0, rotation_offset[1]),
                    collision=True,
                    convexify=True,
                    decompose_nonconvex=False
                ),
                # vis_mode="collision"
            )

            obj_list.append(obj)

    # x_l = 0
    # y_l = 0
    # y_r = 6
    # x_r = 6
    # cam_1 = scene.add_camera(
    #         pos=(x_l + 0.1, y_r - 0.1, 1.85),
    #         lookat=(x_r - 0.1, y_l + 0.1, 0.3),
    #         res=(1024, 1024),
    #         fov=65,
    #         GUI=False,
    #     )
    
    scene.build()

    for i in range(len(obj_list)):
        obj = obj_list[i]
        aabb = obj.get_AABB().cpu().numpy()
        # print(all_large[i]['name'], aabb)
        extends = aabb[1] - aabb[0]
        all_large[i]['bbox'] = extends.tolist()

    all_large_objects['objects'] = all_large
    scene_path = os.path.join(output_dir, 'all_large.json')
    with open(scene_path, 'w') as file:
        json.dump(all_large_objects, file, indent=2)

    # img = cam_1.render()[0]
    # imageio.imwrite(os.path.join(args.output_dir, "scene.png"), img)

    


if __name__ == '__main__':
    from argparse import ArgumentParser
    import random

    parser = ArgumentParser()
    parser.add_argument("--scene_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    scene_to_all_large(args.scene_path, args.output_dir)


