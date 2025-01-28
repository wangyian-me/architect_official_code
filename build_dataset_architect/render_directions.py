import genesis as gs
import os
import imageio
import trimesh

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--load_dir", type=str)
parser.add_argument("--image_dir", type=str)
parser.add_argument("--uid", type=str)
parser.add_argument("--front_only", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--rotation", type=float, default=0)
args = parser.parse_args()

# 1. Load the .glb file with trimesh for bounding box info
glb_path = os.path.join(args.load_dir, f'{args.uid}.glb')
mesh = trimesh.load(glb_path, force='mesh')

# bounding box extents
bbox_extents = mesh.bounding_box.extents
scale = max(bbox_extents)

# center offset
offset = mesh.bounding_box.centroid

#
# 2. Create the Genesis scene
#    Z-up => no rotation needed for the object (unless your .glb was authored differently)
#
gs.init(seed=0, precision='32', logging_level='debug')

scene = gs.Scene(
    show_viewer=False,
    rigid_options=gs.options.RigidOptions(enable_collision=False),
    renderer=gs.renderers.RayTracer(
        env_radius=200.0,
        env_surface=gs.surfaces.Emission(emissive=(0.3, 0.3, 0.3)),
    ),
)

obj = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.Mesh(
        file=glb_path,
        euler=(90, 0, 0 + args.rotation),   # No rotation needed for Z-up objects
        collision=False,
    ),
)

# Create one camera that we'll reposition for each view
cam = scene.add_camera(
    res=(1024, 1024),
    pos=(0, 0, 0),     # temporary; will update below
    lookat=offset,     # looking at the mesh center
    fov=45,
    spp=1024
)

scene.build()

#
# 3. Define camera positions for front/back/left/right in Z-up.
#    "front" = +X, "back" = -X, "left" = +Y, "right" = -Y.
#    We also raise the camera in Z by 0.7 * scale for a slight downward angle.
#
positions = [
    # front (+X)
    (offset[0] + 1.5 * scale, offset[2],             offset[1] + 0.7 * scale),
    # back (-X)
    (offset[0] - 1.5 * scale, offset[2],             offset[1] + 0.7 * scale),
    # left (+Y)
    (offset[0],               offset[2] + 1.5 * scale, offset[1] + 0.7 * scale),
    # right (-Y)
    (offset[0],               offset[2] - 1.5 * scale, offset[1] + 0.7 * scale),
]


#
# 4. Render each view
#
for i, cam_pos in enumerate(positions):
    cam.set_pose(pos=cam_pos, lookat=[offset[0], offset[2], offset[1]])
    scene.visualizer.update()  # If you have a real-time viewer, this refreshes it
    
    # Render returns (image_data, something_else). We only need the image_data
    img = cam.render()[0]
    
    # Save
    out_path = os.path.join(args.image_dir, f"{args.uid}_{i}.png")
    if args.test:
        out_path = os.path.join(args.image_dir, f"test_{args.uid}.png")
    imageio.imwrite(out_path, img)
    print(f"Saved view {i} -> {out_path}")

    if args.front_only or args.test:
        break

