from argparse import ArgumentParser
from urllib.parse import urlparse
import numpy as np
import requests
import OpenEXR
import trimesh
import Imath
import json
import uuid
import tqdm
import bpy
import cv2
import os

# Note that you need to use another conda environment for running this file!!!!
from mathutils import Vector

def recenter_all_meshes():
    # Make sure we’re in OBJECT mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    # Clear parenting, keep transforms
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # Apply transforms (location, rotation, scale)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Calculate bounding box in world space
    min_coords = Vector((9999999, 9999999, 9999999))
    max_coords = Vector((-9999999, -9999999, -9999999))

    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_coords.x = min(min_coords.x, world_corner.x)
            min_coords.y = min(min_coords.y, world_corner.y)
            min_coords.z = min(min_coords.z, world_corner.z)
            max_coords.x = max(max_coords.x, world_corner.x)
            max_coords.y = max(max_coords.y, world_corner.y)
            max_coords.z = max(max_coords.z, world_corner.z)

    center = (min_coords + max_coords) / 2.0

    # Shift every mesh so that the bounding‐box center is at (0,0,0)
    for obj in mesh_objects:
        obj.location -= center


def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get the header to retrieve image size
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Define the channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    
    # Read the image channels
    rgb_data = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in channels]
    
    # Reshape the channels to an image
    rgb = [channel.reshape((height, width)) for channel in rgb_data]
    
    # Stack the channels to form a color image
    img = np.stack(rgb, axis=-1)
    
    return img

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_filename

def download(asset_base_id, file_type, save_dir, api_key, resolution='1k'):
    selection = { # blend resolution file is too large (up to 300MB)
        'blend': 0,
        '0.5k': 1,
        '1k': 2,
        '2k': 3,
        '4k': 4,
        '8k': 5,
        'thumbnail': 6
    }[resolution]
    search_url = f"https://www.blenderkit.com/api/v1/search/?query=asset_base_id:{asset_base_id}"
    print(search_url)
    result = requests.get(search_url)
    result = json.loads(result.content.decode())
    files = result['results'][0]['files']
    if len(files) - 1 <= selection:
        selection = 0
    file_url = files[selection]['downloadUrl']
    print(file_url)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    scene_uuid = str(uuid.uuid4())
    result = requests.get(f"{file_url}?scene_uuid={scene_uuid}", headers=headers)
    result = json.loads(result.content.decode())
    print(result['filePath'])
    parsed_url = urlparse(result['filePath'])
    suffix = parsed_url.path.split('/')[-1].split('.')[-1]

    file = os.path.join(save_dir, f"{asset_base_id}.{suffix}")
    download_file(result['filePath'], file)

    if file_type == "glb":
        target = os.path.join(save_dir, f"{asset_base_id}.glb")
        try:
            bpy.ops.wm.open_mainfile(filepath=file)
            # Step 2: Pack all external files (textures) into the .blend file
            bpy.ops.file.pack_all()

            # Step 3: Apply all transformations (location, rotation, scale) to ensure they are baked into the mesh
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Step 4: Apply all modifiers on the objects to ensure the final mesh is exported correctly
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    bpy.context.view_layer.objects.active = obj
                    for modifier in obj.modifiers:
                        bpy.ops.object.modifier_apply(modifier=modifier.name)

            # Step 5: Ensure all objects have a UV map (important for textures)
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH' and not obj.data.uv_layers:
                    # print("no uv?")
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.uv.smart_project(angle_limit=66)
                # else:
                #     print("have uv", obj.type)

            recenter_all_meshes()

            # Step 6: Convert materials to use the Principled BSDF shader if not already used
            def convert_to_principled_bsdf(obj):
                for mat in obj.data.materials:
                    if mat and mat.use_nodes:
                        bsdf_node = None
                        for node in mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                bsdf_node = node
                                break
                        if not bsdf_node:
                            bsdf_node = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
                            bsdf_node.location = (0, 0)
                            material_output = mat.node_tree.nodes.get('Material Output')
                            if material_output:
                                mat.node_tree.links.new(bsdf_node.outputs['BSDF'], material_output.inputs['Surface'])

            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    convert_to_principled_bsdf(obj)

            bpy.ops.export_scene.gltf(
                filepath=target,
                export_format='GLB',
                use_visible=True,
                export_apply=True,
                export_materials='EXPORT',
            )
        except:
            print(f"RecursionError encountered while processing file: {file}")
            return

        # mesh = trimesh.load(target)

        # # Ensure the mesh is zero-centered
        # bounding_box_center = mesh.bounds.mean(axis=0)
        # mesh.apply_translation(-bounding_box_center)

        # # Normalize the mesh to fit within the [-1, 1]^3 box
        # max_extent = max(mesh.extents)
        # scale_factor = 2.0 / max_extent
        # mesh.apply_scale(scale_factor)

        # # Save the normalized, zero-centered mesh back to a GLB file
        # output_glb_file_path = target
        # mesh.export(output_glb_file_path)
    elif file_type == "hdr":
        if suffix == "exr":
            exr = read_exr(file)
            assert exr is not None, "failed to open input image!"
            assert exr.shape[2] >= 3, "image should at least contain 3 channels!"
            path, filename = os.path.split(file)
            name, _ = os.path.splitext(filename)
            if exr.shape[2] != 3:
                print("[!] abort some channels, only preserve channel [0:3]")
            cv2.imwrite(os.path.join(path, name + ".hdr"), exr[:,:,0:3])
    elif file_type == "material":
        if suffix == "blend":
            material_dir = os.path.join(save_dir, asset_base_id)
            os.makedirs(material_dir, exist_ok=True)
            bpy.ops.wm.open_mainfile(filepath=file)
            for mat in bpy.data.materials:
                if not mat.use_nodes:
                    continue
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        image = node.image
                        save_path = os.path.join(material_dir, image.name)
                        image.save_render(save_path)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--uid", type=str)
    parser.add_argument("--type", type=str, default="glb", choices=["glb", "material", "hdr"])
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--api_key", type=str, default='')
    args = parser.parse_args()
    args.api_key = os.environ.get("BLENDERKIT_API_KEY")

    download(args.uid, args.type, args.save_dir, args.api_key)
