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

def add_articulation(scene, x_l, x_r, y_l, y_r, z_l, z_r, asset_id):
    path = f"{partnet_dir}/{asset_id}/mobility.urdf"
    pos = [(x_l + x_r) / 2, (y_l + y_r) / 2, 0]
    if x_l == x_r:
        euler = [0, 0, 0]
    else:
        euler = [0, 0, 90]
    with open(f"{partnet_dir}/{asset_id}/bounding_box.json", 'r') as file:
        bbox = json.load(file)
    scale = (z_r - z_l) / (bbox['max'][1] - bbox['min'][1]) * 1.05
    pos[2] = z_l - bbox['min'][1] * scale
    print((bbox['max'][0] - bbox['min'][0]) * scale / 1.05)
    # pos[0] -= bbox['max'][0] * scale
    art = scene.add_entity(
        material=gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16),
        morph=gs.morphs.URDF(
            collision=False,
            file=path,
            scale=scale,
            pos=(pos[0], pos[1], pos[2]),
            euler=(euler[0], euler[1], euler[2])
        ),
    )
    return art

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

def genesis_room(args):
    import genesis as gs
    import imageio
    import json

    scene_path = os.path.join(args.work_dir, 'scene.json')

    if os.path.exists(scene_path):
        with open(scene_path, 'r') as file:
            scene_dict = json.load(file)
    else:
        scene_dict = {"floor_objects": [], "wall_objects": []}

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

    mat_rigid = gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16)

    # path = os.path.join(args.work_dir, 'floor.obj')
    # todo: use existing ones maybe
    # generate_mesh_obj_trimesh_with_uv(x_r, y_r, 800, 960, rep=8, filename=path)

    walls = []
    floors = []

    plane = add_floor(scene, 0, args.room_x, 0, args.room_y, texture=args.floor_texture_dir, texture1=args.wall_texture_dir)

    floors = [plane]

    # wall_path1 = os.path.join(args.work_dir, 'wall1.obj')
    # wall_path2 = os.path.join(args.work_dir, 'wall2.obj')


    # add_articulation(scene, 4, 4, 2.5, 3.5, 0, 2, 9127)
    # wall1 = add_wall(scene, 0, 0, 0, args.room_y, texture=args.wall_texture_dir, id=0,
    #          remove_region=[1.2, 1.2, 3, 2.6]) #1,3 is for height, 2,4 is for width
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
    for obj_info in scene_dict["floor_objects"]:
        assetId = obj_info["assetId"]
        vertices = obj_info["vertices"]
        position = obj_info["position"]
        rotation = obj_info["rotation"]
        obj_name = obj_info["object_name"]

        bbox = data[assetId]["bounding_box_extents"]
        center = data[assetId]["center"]

        rotation_offset = [90, data[assetId]["front_view_rotation"][1], 0]
        # in our dataset, default facing to x-, but in holodeck, default facing to y+, so need to +90
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

        try:
            objs[obj_name] = scene.add_entity(
                material=mat_rigid,
                morph=gs.morphs.Mesh(
                    file=os.path.join(dataset_dir, f"{assetId}.glb"),
                    pos=(position['x'] - x_center, position['z'] - y_center, half_height),
                    euler=[90, 0, rotation_offset[1]],
                    collision=False
                ),
            )
        except:
            print(f"Wrong at {obj_name}, {assetId}")
            raise

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
            try:
                objs[obj_name] = scene.add_entity(
                    material=mat_rigid,
                    morph=gs.morphs.Mesh(
                        file=os.path.join(dataset_dir, f"{assetId}.glb"),
                        pos=(position['x'], position['z'], position['y']),
                        euler=rotation_offset,
                        collision=False
                    ),
                )
            except:
                print(f"Wrong at {obj_name}, {assetId}")
                raise
        
        else:

            rotation_offset = [90, data[assetId]["front_view_rotation"][1], 0]
            # in our dataset, default facing to x-, but in holodeck, default facing to y+, so need to -90
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

            try:
                objs[obj_name] = scene.add_entity(
                    material=mat_rigid,
                    morph=gs.morphs.Mesh(
                        file=os.path.join(dataset_dir, f"{assetId}.glb"),
                        pos=(position['x'] - x_center, position['z'] - y_center, position['y'] + half_height),
                        euler=[90, 0, rotation_offset[1]],
                        collision=False
                    ),
                )
            except:
                print(f"Wrong at {obj_name}, {assetId}")
                raise

    cam_0 = scene.add_camera(
        pos=(room_center[0], room_center[1], 3.8),
        lookat=(room_center[0], room_center[1], 0.0),
        res=(1024,1024),
        fov=84,
        GUI=False,
    )
    pose = []
    lookat = []
    if args.view_id == '0':
        cam_1 = scene.add_camera(
            pos=(x_l + 0.1, y_r - 0.1, 1.85),
            lookat=(x_r - 0.1, y_l + 0.1, 0.3),
            res=(1024, 1024),
            fov=65,
            GUI=False,
        )
        pose = [x_l + 0.1, y_r - 0.1, 1.85]
        lookat = [x_r - 0.1, y_l + 0.1, 0.3]

    elif args.view_id == '1':

        cam_1 = scene.add_camera(
            pos=(x_r / 2, y_l + 0.1, 1.85),
            lookat=(x_r / 2, y_r - 0.1, 0.3),
            res=(1024, 1024),
            fov=65,
            GUI=False,
        )
        pose = [x_r / 2, y_l + 0.1, 1.85]
        lookat = [x_r / 2, y_r - 0.1, 0.3]

        cam_prev = scene.add_camera(
            pos=(x_l + 0.1, y_r - 0.1, 1.85),
            lookat=(x_r - 0.1, y_l + 0.1, 0.3),
            res=(1024, 1024),
            fov=65,
            GUI=False,
        )
    else:
        cam_1 = scene.add_camera(
            pos=(x_l + 0.1, y_l + 0.1, 1.85),
            lookat=(x_r - 0.1, y_r - 0.1, 0.3),
            res=(1024, 1024),
            fov=65,
            GUI=False,
        )
        pose = [x_l + 0.1, y_l + 0.1, 1.85]
        lookat = [x_r - 0.1, y_r - 0.1, 0.3]

        cam_prev = scene.add_camera(
            pos=(x_r / 2, y_l + 0.1, 1.85),
            lookat=(x_r / 2, y_r - 0.1, 0.3),
            res=(1024, 1024),
            fov=65,
            GUI=False,
        )

    scene.build()
    # scene.reset()
    # scene.step()

    blocked_ratio_threshold = 0.6

    while True:
        rgb_arr, depth_arr, seg_arr, _ = cam_1.render(rgb=True, depth=True, segmentation=True)
        mask = np.zeros((1024, 1024, 3), dtype=np.int32)
        inpaint_mask = np.zeros((1024, 1024, 3), dtype=np.int32)
        # floor_mask = np.zeros((1024, 1024, 3), dtype=np.int32)
        # ceil_mask = np.zeros((1024, 1024, 3), dtype=np.int32)
        # wall_mask = np.zeros((1024, 1024, 3), dtype=np.int32)
        mask[:90, :] = mask[-90:, :] = mask[:, :90] = mask[:, -90:] = np.array([255, 255, 255])
        for name in objs:
            mask[seg_arr == objs[name].idx] = np.array([255, 255, 255])
        inpaint_mask = ~mask

        usable_area = np.count_nonzero(inpaint_mask[..., 0] != 0)
        total_area = 1024 ** 2
        ratio = usable_area / total_area
        
        # If ratio is above threshold, we consider the camera to have a decent view
        if ratio > blocked_ratio_threshold:
            break
        else:
            print("limited view detected")
            blocked_ratio_threshold *= 0.9
            pose[0] = pose[0] * 0.9 + room[0] / 2 * 0.1
            pose[1] = pose[1] * 0.9 + room[1] / 2 * 0.1
            cam_1.set_pose(pos=pose, lookat=lookat)
            scene.visualizer.update() 

    # floor_mask[seg_arr == plane.idx] = np.array([255, 255, 255])
    # for obj in walls:
    #     wall_mask[seg_arr == obj.idx] = np.array([255, 255, 255])
    img = cam_1.render()[0]
    imageio.imwrite(os.path.join(args.work_dir, "prev_scene.png"), img)
    if args.view_id != '0':
        img = cam_prev.render()[0]
        imageio.imwrite(os.path.join(args.work_dir, "prev_scene_prev_view.png"), img)

    img = cam_0.render()[0]
    imageio.imwrite(os.path.join(args.work_dir, "top_down.png"), img)

    imageio.imwrite(os.path.join(args.work_dir, f"mask.png"), mask.astype(np.uint8))
    imageio.imwrite(os.path.join(args.work_dir, f"inpaint_mask.png"), inpaint_mask.astype(np.uint8))
    np.save(os.path.join(args.work_dir, f"depth.npy"), depth_arr)

    T_OPENGL_TO_OPENCV = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    cam_pose = cam_1._rasterizer._camera_nodes[cam_1.uid].matrix @ T_OPENGL_TO_OPENCV

    np.save(os.path.join(args.work_dir, f"cam_pose.npy"), cam_pose)
    # imageio.imwrite(os.path.join(args.work_dir, "floor_mask.png"), floor_mask.astype(np.uint8))
    # imageio.imwrite(os.path.join(args.work_dir, "wall_mask.png"), wall_mask.astype(np.uint8))
    # imageio.imwrite(os.path.join(args.work_dir, "ceil_mask.png"), ceil_mask.astype(np.uint8))
    import cv2
    depth = -depth_arr
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized_uint8 = depth_normalized.astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_normalized_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.work_dir, "colored_depth_map.png"), colored_depth)
    save_intrinsic_and_pose(cam_1, args.work_dir)
