import genesis as gs
import os
import imageio
import trimesh
import genesis.utils.geom as gu
import numpy as np
import os
import json
import cv2

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--load_dir", type=str)
parser.add_argument("--uid", type=str)
args = parser.parse_args()

gs.init(seed=0, precision='32', logging_level='error')

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.005),
    show_viewer=False,
    rigid_options=gs.options.RigidOptions(gravity=(0, 0, -9.8), enable_collision=True),
)

obj = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.Mesh(
        file=os.path.join(args.load_dir, f"{args.uid}.glb"),
        euler=(90, 0, 0),   # No rotation needed for Z-up objects
        collision=True,
        convexify=True
    ),
    vis_mode='collision'
)

plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# cam = scene.add_camera(
#     res=(1024, 1024),
#     pos=(0.5, 0.5, 0.5),     # temporary; will update below
#     lookat=(0, 0, 0.0),     # looking at the mesh center
#     fov=45,
# )

for link in obj.links:
    link._inertial_mass = 0.1

# Create one camera that we'll reposition for each view

scene.build()

for dof in range(obj.dof_start, obj.dof_start, 1):
    obj.solver.dofs_info[dof].damping = 0.1

stable_rotations = []

# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1024, 1024))

for x_rotation in [0, 90]:
    for y_rotation in [0, 90]:
        for z_rotation in [0, 90]:
            # Convert Euler angles to quaternion
            euler = [x_rotation, y_rotation, z_rotation]
            quat = gu.xyz_to_quat(np.array(euler))

            # Set the object's rotation
            obj.set_quat(quat)

            # Get the updated AABB after setting the rotation
            aabb = obj.get_AABB()
            bbox_extents = aabb[1] - aabb[0]  # Calculate extents from AABB min and max

            # Reset object's position slightly above the plane
            extend_id = [0, 1, 2]
            if x_rotation == 90:
                extend_id[1], extend_id[2] = extend_id[2], extend_id[1]
            if y_rotation == 90:
                extend_id[0], extend_id[2] = extend_id[2], extend_id[0]
            

            obj.set_pos([0, 0, 0.02 + bbox_extents[extend_id[2]] / 2])


            for _ in range(2000):
                scene.step()

                # if _ % 10 == 0:
                #     img = cam.render()[0]
                #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #     out.write(img)

            # Get the object's final rotation and position
            final_quat = obj.get_quat().cpu().numpy()
            final_pos = obj.get_pos()

            # Check stability by comparing the initial and final quaternions
            # print(final_quat, quat)
            is_stable = np.allclose(final_quat, quat, atol=1e-1)
            # print(is_stable)
            if is_stable:
                stable_rotations.append(euler)


# out.release()

# Save stable rotations to a JSON file
output_path = os.path.join(args.load_dir, f"{args.uid}_stablerot.json")
with open(output_path, 'w') as f:
    json.dump(stable_rotations, f, indent=4)

print(f"Stable rotations saved to {output_path}")


