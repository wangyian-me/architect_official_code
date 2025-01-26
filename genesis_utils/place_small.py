import gzip
import pickle
import trimesh
import numpy as np
from PIL import Image
import os
import random
import genesis as gs

dataset_dir = os.environ.get("DATASET_PATH")
cache_dir = os.environ.get("CACHE_PATH")

def opengl_projection_matrix_to_intrinsics(P: np.ndarray, width: int, height: int):
    """Convert OpenGL projection matrix to camera intrinsics.
    Args:
        P (np.ndarray): OpenGL projection matrix.
        width (int): Image width.
        height (int): Image height
    Returns:
        np.ndarray: Camera intrinsics. [3, 3]
    """

    fx = P[0, 0] * width / 2
    fy = P[1, 1] * height / 2
    cx = (1.0 - P[0, 2]) * width / 2
    cy = (1.0 + P[1, 2]) * height / 2

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K


def save_intrinsic_and_pose(cam, work_dir):
    intrinsic_K = opengl_projection_matrix_to_intrinsics(
        cam._rasterizer._camera_nodes[cam.uid].camera.get_projection_matrix(), width=cam.res[0], height=cam.res[1]
    )

    T_OPENGL_TO_OPENCV = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    cam_pose = cam._rasterizer._camera_nodes[cam.uid].matrix @ T_OPENGL_TO_OPENCV

    np.save(os.path.join(work_dir, "intrinsic_K.npy"), intrinsic_K)
    np.save(os.path.join(work_dir, "cam_pose.npy"), cam_pose)

def generate_mesh_obj_trimesh_with_uv(x_l, x_r, y_l, y_r, a, b, filename="floor.obj", rep=4, remove_region=None, along_axis='z'):
    # Generate grid points for vertices
    gx = np.linspace(x_l, x_r, a)
    gy = np.linspace(y_l, y_r, b)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_z = np.zeros_like(grid_x)

    # Create vertices array
    vertices = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

    # Generate faces indices
    faces = []
    for j in range(b - 1):
        for i in range(a - 1):
            # Indices of vertices in the current quad
            v1 = j * a + i
            v2 = j * a + (i + 1)
            v3 = (j + 1) * a + (i + 1)
            v4 = (j + 1) * a + i
            # Add two triangles for each quad
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # Convert faces to numpy array for easier manipulation
    faces = np.array(faces)

    # Create UV coordinates
    uv_x = np.tile(np.concatenate((np.linspace(0, 1, a // rep + 1)[:-1], np.linspace(1, 0, a // rep + 1)[:-1])), rep // 2)
    uv_y = np.tile(np.concatenate((np.linspace(0, 1, b // rep + 1)[:-1], np.linspace(1, 0, b // rep + 1)[:-1])), rep // 2)
    uv_grid_x, uv_grid_y = np.meshgrid(uv_x, uv_y)
    uvs = np.vstack([uv_grid_x.flatten(), uv_grid_y.flatten()]).T

    if remove_region:
        a1, b1, a2, b2 = remove_region
        # Mask for vertices outside the removal region
        mask_x = (grid_x.flatten() < a1) | (grid_x.flatten() > a2)
        mask_y = (grid_y.flatten() < b1) | (grid_y.flatten() > b2)
        mask = mask_x | mask_y

        # Filter out vertices inside the removal region
        vertices = vertices[mask]
        uvs = uvs[mask]

        # Find the indices of the remaining vertices
        remaining_indices = np.where(mask)[0]
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_indices)}

        # Filter and remap faces
        new_faces = []
        for face in faces:
            if all(idx in index_map for idx in face):
                new_faces.append([index_map[idx] for idx in face])
        faces = np.array(new_faces)

    # Create the mesh with vertices, faces, and uv coordinates
    if along_axis == 'z':
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif along_axis == 'y':
        vertices = vertices[:, [0, 2, 1]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        vertices = vertices[:, [2, 1, 0]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

    # Export to OBJ file
    mesh.export(filename)

def add_skirting_line(scene, x_l, x_r, y_l, y_r, rotation):
    rotation = {
        "x+": (90, 0, 90),
        "x-": (90, 0, 270),
        "y+": (90, 0, 180),
        "y-": (90, 0, 0)
    }[rotation]
    skirting_line_path = f"{dataset_dir}/e795faf4-b642-4f34-abf0-ccea383cd95e.glb"
    mesh = trimesh.load(skirting_line_path)
    bbox = mesh.bounding_box.extents
    real_x_length = bbox[0]
    scale = ((x_r - x_l) / real_x_length, 1.0, 1.0)
    if scale[0] == 0:
        scale = ((y_r - y_l) / real_x_length, 1.0, 1.0)
    scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file=skirting_line_path,
            pos=[(x_l + x_r) / 2, (y_l + y_r) / 2, 0],
            euler=rotation,
            scale=scale,
            fixed=True,
            collision=False
        )
    )

def add_floor(scene, x_l, x_r, y_l, y_r, texture='', texture1=''):

    path = f"{cache_dir}/floor_{x_l}{x_r}{y_l}{y_r}.obj"
    if not os.path.exists(path):
        generate_mesh_obj_trimesh_with_uv(x_l, x_r, y_l, y_r, 64, 64, rep=4, filename=path)
    plane = scene.add_entity(
        material=gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16),
        morph=gs.morphs.Mesh(collision=False,file=path,
                             pos=(0, 0, 0),
                             euler=(0, 0, 0),
                             scale=1.0,
                             fixed=True
                             ),
        surface=gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(
                image_path=texture,
            ),
            roughness=10,
            double_sided=True
        )
    )
    scene.add_entity(
        material=gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16),
        morph=gs.morphs.Mesh(collision=False,file=path,
                             pos=(0, 0, 5.0),
                             euler=(0, 0, 0),
                             scale=1.0,
                             fixed=True
                             ),
        surface=gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(
                image_path=texture1,
            ),
            roughness=10,
            double_sided=True
        )
    )
    return plane

def add_wall(scene, x_l, x_r, y_l, y_r, height=5, remove_region=None, texture=''):
    path = f"{cache_dir}/wall_{x_l}{x_r}{y_l}{y_r}.obj"
    generate = False
    if not os.path.exists(path):
        generate = True
    
    offset = [0, 0, 0]
    if x_l == x_r:
        if generate:
            generate_mesh_obj_trimesh_with_uv(0, height, y_l, y_r, 64, 64, filename=path, rep=4, remove_region=remove_region,
                                              along_axis='x')
        offset = [x_l, 0, 0]
    elif y_l == y_r:
        if generate:
            generate_mesh_obj_trimesh_with_uv(x_l, x_r, 0, height, 64, 64, filename=path, rep=4, remove_region=remove_region,
                                              along_axis='y')
        offset = [0, y_l, 0]
    else:
        print("wall should be 2 dimensions")

    wall = scene.add_entity(
        material=gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16),
        morph=gs.morphs.Mesh(collision=False,file=path,
                             pos=offset,
                             euler=(0, 0, 0),
                             scale=1.0,
                             fixed=True
                             ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path=texture,
            ),
            double_sided=True
        )
    )
    return wall


def genesis_shelf(args):
    import genesis as gs
    import imageio
    import json

    all_large_path = os.path.join(args.work_dir, 'all_large.json')
    if os.path.exists(all_large_path):
        with open(all_large_path, 'r') as file:
            all_large = json.load(file)

    room = args.room
    x_l = 0
    y_l = 0
    x_r = room[0]
    y_r = room[1]
    room_center = [room[0] / 2, room[1] / 2]

    gs.init(seed=0, precision='32', logging_level='debug')


    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(gravity=(0, 0, 0), enable_collision=False),
        renderer=gs.renderers.RayTracer(
            # env_euler=(0, 0, 180)
            # env_texture=gs.textures.ColorTexture(color=(0.2, 0.2, 0.2)),
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(
                    # image_path=hdr_path,
                    color=(0.5, 0.5, 0.5),
                )
            ),
            lights=[
                # {'pos': (-4, 2, 4.0), 'color': (255, 220, 170), 'radius': 0.3, 'intensity': 1.0},
                # {'pos': (2, 10, 5.0), 'color': (255, 220, 170), 'radius': 0.3, 'intensity': 1.0},
                {'pos': (room_center[0], room_center[1], 4.3), 'color': (200.0, 200.0, 200.0), 'radius': 0.3, 'intensity': 0.9},
                #  {'pos': (70, 4, 2.0), 'color': (255, 255, 200), 'radius': 7, 'intensity': 10.0},
                # {'pos': (-4, room[1]/2, 5.0), 'color': (255, 255, 150), 'radius': 0.6, 'intensity': 1.0},
            ]
        ),
    )

    mat_rigid = gs.materials.Rigid()

    walls = []
    floors = []

    plane = add_floor(scene, 0, args.room_x, 0, args.room_y, texture=args.floor_texture_dir, texture=args.wall_texture_dir)

    floors = [plane]

    # wall_path1 = os.path.join(args.work_dir, 'wall1.obj')
    # wall_path2 = os.path.join(args.work_dir, 'wall2.obj')
    #          )
    wall1 = add_wall(scene, 0, 0, 0, args.room_y, texture=args.wall_texture_dir
             ) #1,3 is for height, 2,4 is for width
    wall2 = add_wall(scene, 0, args.room_x, 0, 0, texture=args.wall_texture_dir
             )
    wall3 = add_wall(scene, args.room_x, args.room_x, 0, args.room_y, texture=args.wall_texture_dir
             )
    wall4 = add_wall(scene, 0, args.room_x, args.room_y, args.room_y, texture=args.wall_texture_dir
             )

    add_skirting_line(scene, 0, 0, 0, room[1], "x+")
    add_skirting_line(scene, room[0], room[0], 0, room[1], "x-")
    add_skirting_line(scene, 0, room[0], 0, 0, "y+")
    add_skirting_line(scene, 0, room[0], room[1], room[1], "y-")

    walls = [wall1, wall2, wall3, wall4]

    with open(
            f"{dataset_dir}/blenderkit_database.json",
            'r') as file:
        data = json.load(file)
    objs = {}
    camera_offset = [0, 0, 0]
    object_position = [0, 0, 0]
    object_bbox = []

    for obj_info in all_large["objects"]:

        obj_path = obj_info['path']
        scale = obj_info['scale']
        pos = obj_info['pos']
        name = obj_info['name']
        euler = obj_info['euler']
        facing = obj_info['facing']
        bbox = obj_info['bbox']
        
        objs[name] = scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                fixed=True,
                file=obj_path,
                scale=scale,
                pos=pos,
                euler=euler,
                collision=False
            ),
        )
        
        if name == args.obj_name:
            if facing == 'x+':
                camera_offset = [bbox[0] / 2 + max(bbox[1], bbox[2]), 0, 0]
                object_position = pos
                object_bbox = bbox
            elif facing == 'y+':
                camera_offset = [0, bbox[1] / 2 + max(bbox[0], bbox[2]), 0]
                object_position = pos
                object_bbox = bbox
            elif facing == "x-":
                camera_offset = [-bbox[0] / 2 - max(bbox[1], bbox[2]), 0, 0]
                object_position = pos
                object_bbox = bbox
            elif facing == "y-":
                camera_offset = [0, -bbox[1] / 2 - max(bbox[0], bbox[2]), 0]
                object_position = pos
                object_bbox = bbox

    if os.path.exists(args.all_small_path):
        with open(args.all_small_path, 'r') as file:
            all_small = json.load(file)
        
        for obj_info in all_small["objects"]:

            obj_path = os.path.join(dataset_dir, f"{obj_info['uid']}.glb")
            scale = obj_info['scale']
            pos = obj_info['pos']
            name = obj_info['name']
            euler = obj_info['euler']
            
            objs[name] = scene.add_entity(
                material=mat_rigid,
                morph=gs.morphs.Mesh(
                    fixed=True,
                    file=obj_path,
                    scale=scale,
                    pos=pos,
                    euler=euler,
                    collision=False
                ),
            )
        
    cam_0 = scene.add_camera(
        pos=np.array(object_position) + np.array(camera_offset),
        lookat=object_position,
        res=(1024, 1024),
        fov=55,
        GUI=False,
    )
    
    cube = scene.add_entity(
        material=mat_rigid,
        morph=gs.morphs.Box(
            pos=object_position,
            size=np.array(object_bbox) * 0.8,
            fixed=False,
            collision=False
        )
    )

    scene.build()

    save_intrinsic_and_pose(cam_0, args.output_dir)

    now_qpos = cube.get_qpos()
    cube.set_pos(gs.utils.geom.nowhere())
    scene.visualizer.update()

    rgb_arr, depth, seg_arr, _ = cam_0.render(rgb=True, depth=True, segmentation=True)

    res = (1024, 1024)

    if os.path.exists(args.all_small_path):
        imageio.imwrite(os.path.join(args.output_dir, "placed_scene.png"), rgb_arr)
    else:

        np.save(os.path.join(args.output_dir, "depth.npy"), depth)
        imageio.imwrite(os.path.join(args.output_dir, "prev_scene.png"), rgb_arr)

        cube.set_qpos(now_qpos)
        scene.visualizer.update()

        rgb_arr, depth, seg_arr, _ = cam_0.render(rgb=True, depth=True, segmentation=True)

        inpaint_mask = np.zeros((res[0], res[1], 3), dtype=np.int32)
        inpaint_mask[seg_arr == cube.idx] = np.array([255, 255, 255])
        mask = np.zeros((res[0], res[1], 3), dtype=np.int32)
        mask[seg_arr == objs[args.obj_name].idx] = np.array([255, 255, 255])

        imageio.imwrite(os.path.join(args.output_dir, "mask.png"), mask.astype(np.uint8))
        imageio.imwrite(os.path.join(args.output_dir, "inpaint_mask.png"), inpaint_mask.astype(np.uint8))

        
def genesis_table(args):
    import genesis as gs
    import imageio
    import json

    all_large_path = os.path.join(args.work_dir, 'all_large.json')
    if os.path.exists(all_large_path):
        with open(all_large_path, 'r') as file:
            all_large = json.load(file)

    room = args.room
    x_l = 0
    y_l = 0
    x_r = room[0]
    y_r = room[1]
    room_center = [room[0] / 2, room[1] / 2]

    gs.init(seed=0, precision='32', logging_level='debug')


    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(gravity=(0, 0, 0), enable_collision=False),
        renderer=gs.renderers.RayTracer(
            # env_euler=(0, 0, 180)
            # env_texture=gs.textures.ColorTexture(color=(0.2, 0.2, 0.2)),
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(
                    # image_path=hdr_path,
                    color=(0.5, 0.5, 0.5),
                )
            ),
            lights=[
                # {'pos': (-4, 2, 4.0), 'color': (255, 220, 170), 'radius': 0.3, 'intensity': 1.0},
                # {'pos': (2, 10, 5.0), 'color': (255, 220, 170), 'radius': 0.3, 'intensity': 1.0},
                {'pos': (room_center[0], room_center[1], 4.3), 'color': (200.0, 200.0, 200.0), 'radius': 0.3, 'intensity': 0.9},
                #  {'pos': (70, 4, 2.0), 'color': (255, 255, 200), 'radius': 7, 'intensity': 10.0},
                # {'pos': (-4, room[1]/2, 5.0), 'color': (255, 255, 150), 'radius': 0.6, 'intensity': 1.0},
            ]
        ),
    )

    mat_rigid = gs.materials.Rigid()

    walls = []
    floors = []

    plane = add_floor(scene, 0, args.room_x, 0, args.room_y, texture=args.floor_texture_dir, texture1=args.wall_texture_dir)

    floors = [plane]

    # wall_path1 = os.path.join(args.work_dir, 'wall1.obj')
    # wall_path2 = os.path.join(args.work_dir, 'wall2.obj')
    #          )
    wall1 = add_wall(scene, 0, 0, 0, args.room_y, texture=args.wall_texture_dir
             ) #1,3 is for height, 2,4 is for width
    wall2 = add_wall(scene, 0, args.room_x, 0, 0, texture=args.wall_texture_dir
             )
    wall3 = add_wall(scene, args.room_x, args.room_x, 0, args.room_y, texture=args.wall_texture_dir
             )
    wall4 = add_wall(scene, 0, args.room_x, args.room_y, args.room_y, texture=args.wall_texture_dir
             )

    add_skirting_line(scene, 0, 0, 0, room[1], "x+")
    add_skirting_line(scene, room[0], room[0], 0, room[1], "x-")
    add_skirting_line(scene, 0, room[0], 0, 0, "y+")
    add_skirting_line(scene, 0, room[0], room[1], room[1], "y-")

    walls = [wall1, wall2, wall3, wall4]

    with open(
            f"{dataset_dir}/blenderkit_database.json",
            'r') as file:
        data = json.load(file)
    objs = {}
    camera_offset = [0, 0, 0]
    object_position = [0, 0, 0]
    object_bbox = []
    box_offset = []
    obj_facing = ""

    for obj_info in all_large["objects"]:

        obj_path = obj_info['path']
        scale = obj_info['scale']
        pos = obj_info['pos']
        name = obj_info['name']
        euler = obj_info['euler']
        facing = obj_info['facing']
        bbox = obj_info['bbox']

        if not (name == args.obj_name):
            continue
        
        objs[name] = scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                fixed=True,
                file=obj_path,
                scale=scale,
                pos=pos,
                euler=euler,
                collision=False
            ),
        )
        
        if name == args.obj_name:
            obj_facing = facing
            if facing == 'x+':
                length = bbox[0] / 2 + max(bbox[1], bbox[2]) / 2
                camera_offset = [length / 1.4, 0, bbox[2] / 2 + pos[2] + length / 1.2]
                object_position = pos
                object_bbox = bbox
                box_offset = [bbox[0] * 0.07, 0, 0]
            elif facing == 'y+':
                length = bbox[1] / 2 + max(bbox[0], bbox[2]) / 2
                camera_offset = [0, length / 1.4, bbox[2] / 2 + pos[2] + length / 1.2]
                object_position = pos
                object_bbox = bbox
                box_offset = [0, bbox[1] * 0.07, 0]
            elif facing == "x-":
                length = bbox[0] / 2 + max(bbox[1], bbox[2]) / 2
                camera_offset = [- length / 1.4, 0, bbox[2] / 2 + pos[2] + length / 1.2]
                object_position = pos
                object_bbox = bbox
                box_offset = [-bbox[0] * 0.07, 0, 0]
            elif facing == "y-":
                length = bbox[1] / 2 + max(bbox[0], bbox[2]) / 2
                camera_offset = [0, - length / 1.4, bbox[2] / 2 + pos[2] + length / 1.2]
                object_position = pos
                object_bbox = bbox
                box_offset = [0, -bbox[1] * 0.07, 0]

    if os.path.exists(args.all_small_path):
        with open(args.all_small_path, 'r') as file:
            all_small = json.load(file)
        
        for obj_info in all_small["objects"]:

            obj_path = os.path.join(dataset_dir, f"{obj_info['uid']}.glb")
            scale = obj_info['scale']
            pos = obj_info['pos']
            name = obj_info['name']
            euler = obj_info['euler']
            
            objs[name] = scene.add_entity(
                material=mat_rigid,
                morph=gs.morphs.Mesh(
                    fixed=True,
                    file=obj_path,
                    scale=scale,
                    pos=pos,
                    euler=euler,
                    collision=False
                ),
            )
        
    cam_0 = scene.add_camera(
        pos=np.array(object_position) + np.array(camera_offset),
        lookat=object_position,
        res=(1024, 1024),
        fov=55,
        GUI=False,
    )
    
    cube = scene.add_entity(
        material=mat_rigid,
        morph=gs.morphs.Box(
            pos=[object_position[0] + box_offset[0], object_position[1] + box_offset[1], object_position[2] + object_bbox[2] / 2 + 0.05],
            size=[object_bbox[0] * 0.8, object_bbox[1] * 0.8, 0.1],
            fixed=False,
            collision=False
        )
    )

    cube1 = scene.add_entity(
        material=mat_rigid,
        morph=gs.morphs.Box(
            pos=[object_position[0], object_position[1], object_position[2] + object_bbox[2] / 2 + 0.0005],
            size=[object_bbox[0] * 0.99, object_bbox[1] * 0.99, 0.001],
            fixed=False,
            collision=False
        )
    )

    scene.build()

    save_intrinsic_and_pose(cam_0, args.output_dir)

    now_qpos = cube.get_qpos().cpu().numpy()
    cube.set_pos(gs.utils.geom.nowhere())
    now_qpos1 = cube1.get_qpos().cpu().numpy()
    cube1.set_pos(gs.utils.geom.nowhere())
    scene.visualizer.update()

    rgb_arr, depth, seg_arr1, _ = cam_0.render(rgb=True, depth=True, segmentation=True)

    res = (1024, 1024)

    if os.path.exists(args.all_small_path):
        imageio.imwrite(os.path.join(args.output_dir, "placed_scene.png"), rgb_arr)
    else:

        np.save(os.path.join(args.output_dir, "depth.npy"), depth)
        imageio.imwrite(os.path.join(args.output_dir, "prev_scene.png"), rgb_arr)

        cube.set_qpos(now_qpos)
        cube1.set_qpos(now_qpos1)
        
        scene.visualizer.update()

        rgb_arr, depth, seg_arr, _ = cam_0.render(rgb=True, depth=True, segmentation=True)

        inpaint_mask = np.zeros((res[0], res[1], 3), dtype=np.int32)
        inpaint_mask[seg_arr == cube.idx] = np.array([255, 255, 255])
        mask = np.zeros((res[0], res[1], 3), dtype=np.int32)
        mask[(seg_arr == cube1.idx) & (seg_arr1 == objs[args.obj_name].idx)] = np.array([255, 255, 255])

        imageio.imwrite(os.path.join(args.output_dir, "mask.png"), mask.astype(np.uint8))
        imageio.imwrite(os.path.join(args.output_dir, "inpaint_mask.png"), inpaint_mask.astype(np.uint8))