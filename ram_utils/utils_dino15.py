import os

import json
import torchvision
import sys
from PIL import Image

current_directory = os.getcwd()
# Grounding DINO
sys.path.append(os.path.join(current_directory, 'Grounded-Segment-Anything'))
sys.path.append(os.path.join(current_directory, 'Grounded-Segment-Anything', "GroundingDINO"))
sys.path.append(os.path.join(current_directory, 'Grounded-Segment-Anything', "segment_anything"))
sys.path.append(os.path.join(current_directory, 'Marigold'))
import GroundingDINO.groundingdino.datasets.transforms as T
# from GroundingDINO.groundingdino.models import build_model
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from marigold import MarigoldPipeline
# import supervision as sv

from plyfile import PlyData, PlyElement

# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as TS

import torch
import open3d as o3d


# from perspective2d.utils import draw_perspective_fields, draw_from_r_p_f_cx_cy

def save_point_cloud_as_ply(points, filename):
    """
    Saves a point cloud stored in a NumPy array as a PLY file using the plyfile library.

    Args:
    - points (np.ndarray): A NumPy array of shape (N, 3) containing the point cloud, where N is the number of points.
    - filename (str): The filename of the output PLY file.
    """
    # Create a structured array for the plyfile
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    structured_array = np.array(list(map(tuple, points)), dtype=dtype)

    # Create a PlyElement from the structured array and write to file
    el = PlyElement.describe(structured_array, 'vertex')
    PlyData([el]).write(filename)


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def backproject_depth_to_pointcloud(K: np.ndarray, depth: np.ndarray, pose):
    """Convert depth image to pointcloud given camera intrinsics.
    Args:
        depth (np.ndarray): Depth image.
    Returns:
        np.ndarray: (x, y, z) Point cloud. [n, 4]
        np.ndarray: (r, g, b) RGB colors per point. [n, 3] or None
    """
    _fx = K[0, 0]
    _fy = K[1, 1]
    _cx = K[0, 2]
    _cy = K[1, 2]

    # Mask out invalid depth
    mask = np.where(depth > -1.0)  # all should be valid
    depth = np.maximum(depth, -0.99)
    mask1 = np.where(depth > -1.0)
    x, y = mask1[1], mask1[0]

    # Normalize pixel coordinates
    normalized_x = x.astype(np.float32) - _cx
    normalized_y = y.astype(np.float32) - _cy

    # Convert to world coordinates
    world_x = normalized_x * depth[y, x] / _fx
    world_y = normalized_y * depth[y, x] / _fy
    world_z = depth[y, x]

    pc = np.vstack((world_x, world_y, world_z)).T

    point_cloud_h = np.hstack((pc, np.ones((pc.shape[0], 1))))
    point_cloud_world = (pose @ point_cloud_h.T).T
    point_cloud_world = point_cloud_world[:, :3].reshape(depth.shape[0], depth.shape[1], 3)

    return point_cloud_world, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    scale_factor = 1.0
    image_pil = image_pil.resize((int(scale_factor * image_pil.width), int(scale_factor * image_pil.height)))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


# def load_model(model_config_path, model_checkpoint_path, device):
#     args = SLConfig.fromfile(model_config_path)
#     args.device = device
#     model = build_model(args)
#     checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#     load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     _ = model.eval()
#     return model



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def get_box_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU
    iou = intersection_area / min(box1_area, box2_area)

    return iou


def get_area(box1):
    return (box1[2] - box1[0]) * (box1[3] - box1[1])


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')

    np.save(os.path.join(output_dir, 'mask.npy'), mask_img.numpy())
    json_data = {
        'tags_chinese': tags_chinese,
        'mask': [{
            'value': value,
            'label': 'background'
        }]
    }

    for label, box in zip(label_list, box_list):
        show_box(box.numpy(), plt.gca(), label)
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data['mask'].append({'value': value, 'label': name, 'logit': float(logit), 'box': box.numpy().tolist()})

    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)

current_directory = os.getcwd()
config_file = os.path.join(current_directory, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
ram_checkpoint = os.path.join(current_directory, "Grounded-Segment-Anything/ram_swin_large_14m.pth")
grounded_checkpoint = os.path.join(current_directory, "Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
sam_checkpoint = os.path.join(current_directory, "Grounded-Segment-Anything/sam_vit_h_4b8939.pth")

# model_dino = load_model(config_file, grounded_checkpoint, device="cuda")
predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cuda"))

from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

API_TOKEN = os.environ.get("DINO_API_KEY")

def get_dinox_output(api_token, image_path, prompt_text, threshold=0.25):
    """
    Query the DINOX cloud API for bounding boxes given an image and text prompt.
    Returns bounding boxes in [center_x, center_y, w, h] format along with associated scores and phrases.
    """
    # Initialize the client with the provided API token
    config = Config(api_token)
    client = Client(config)

    # Upload the local image to the server and obtain the URL
    image_url = client.upload_file(image_path)

    # If no prompt is provided, use an empty string
    if prompt_text is None:
        prompt = {
            "type": "universal",
            "universal": 1
        }
    else:
        prompt = {
            "type": "text",
            "text": prompt_text
        }

    # Create the detection task with the appropriate parameters
    task = V2Task(
        api_path="/v2/task/dinox/detection",
        api_body={
            "model": "DINO-X-1.0",
            "image": image_url,
            "prompt": prompt,
            "targets": ["bbox"],
            "bbox_threshold": threshold,
            "iou_threshold": 0.8
        }
    )

    # Execute the task
    client.run_task(task)

    # Retrieve the results from the task
    print(task.result)
    result = task.result
    objects = result["objects"]  # List of detection results

    boxes_centerwh = []
    scores = []
    pred_phrases = []

    # Process each detected object to extract bounding box and associated information
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]  # Bounding box in xyxy format
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        cat = obj["category"]
        score = obj["score"]
        boxes_centerwh.append([cx, cy, w, h])
        scores.append(obj["score"])
        pred_phrases.append(f"{cat}({score:.3f})")

    return (
        torch.tensor(boxes_centerwh, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        pred_phrases
    )


def mask_and_save(image_path, output_dir, tags, box_threshold=0.4, text_threshold=0.25, iou_threshold=0.5,
                  device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    image_pil, image = load_image(image_path)

    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
    # normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # transform = TS.Compose([
    #     TS.Resize((384, 384)),
    #     TS.ToTensor(), normalize
    # ])

    boxes_filt, scores, pred_phrases = get_dinox_output(
        api_token=API_TOKEN,
        image_path=image_path,
        prompt_text=tags,      # same string you used for `tags`
        threshold=box_threshold
    )


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        # boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    print("phrases:", pred_phrases)
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    print("phrases:", pred_phrases)

    print("boxes_filt:", boxes_filt)

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    print("masks result shape:", masks.shape)
    print("input image shape:", size)

    save_mask_data(output_dir, tags, masks, boxes_filt, pred_phrases)
    return masks, pred_phrases, boxes_filt


import time

checkpoint = "prs-eth/marigold-v1-0"
denoise_steps = 50
ensemble_size = 10
processing_res = 1024
resample_method = "bilinear"
match_input_res = True
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained(checkpoint, variant=None, torch_dtype=dtype)
try:
        pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    pass
pipe = pipe.to(device)

def pred_depth(image_dir, max_depth):
    # seed = int(time.time())
    input_image = Image.open(image_dir)
    color_map = "Spectral"
    # generator = torch.Generator(device=device)
    # generator.manual_seed(seed)
    pipe_out = pipe(
        input_image,
        denoising_steps=denoise_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        match_input_res=match_input_res,
        batch_size=1,
        color_map=color_map,
        show_progress_bar=True,
        resample_method=resample_method,
        generator=None
    )
    depth_pred: np.ndarray = pipe_out.depth_np
    depth_colored: Image.Image = pipe_out.depth_colored
    scale = max_depth / np.max(depth_pred)
    depth_pred *= scale
    print("scale:", scale)
    return depth_pred, depth_colored


def filter_bbox(idx, bbox_list, pred_phrases):
    filtered = False
    box = bbox_list[idx]
    name = pred_phrases[idx].split('(')[0]
    for j in range(len(bbox_list)):
        if idx == j:
            continue
        box2 = bbox_list[j]
        name1 = pred_phrases[j].split('(')[0]
        if get_box_iou(box, box2) > 0.9 and get_area(box) > get_area(box2) and name == name1:
            filtered = True
            break
    return filtered


def get_bbox(image_dir, output_dir, tags, descriptions, on_floors, camera_pose=None, intrinsic_K=None, max_depth=None,
             reference_depth=None, reference_mask=None, binary_mask=None, name_dict={}, filter=False, object_mask=None, mask_path=None, depth_path=None):
    # print("done mask and save")

    depth_pred, depth_colored = pred_depth(image_dir, max_depth=1.0)
    print("shape of reference", reference_depth.shape)
    print("shape of depth predict" , depth_pred.shape)
    depth_colored.save(os.path.join(output_dir, 'depth_color.jpg'))
    # print("shape of reference mask", reference_mask.shape)
    range_pred = depth_pred[reference_mask].max() - depth_pred[reference_mask].min()
    print("shape of combined", depth_pred[reference_mask].shape)
    range_real = reference_depth[reference_mask].max() - reference_depth[reference_mask].min()
    depth_pred = depth_pred * range_real / range_pred
    offset = reference_depth[reference_mask].mean() - depth_pred[reference_mask].mean()
    print(offset)
    depth_pred = depth_pred + offset
    # if object_mask is not None:

    # maybe use object_mask & inpaint mask to do further correctness

    # print("done predict depth")

    pc, valid_mask = backproject_depth_to_pointcloud(intrinsic_K, reference_depth, camera_pose)
    save_point_cloud_as_ply(pc[valid_mask].reshape(-1, 3), os.path.join(output_dir, f'pc_real.ply'))
    pc, valid_mask = backproject_depth_to_pointcloud(intrinsic_K, depth_pred, camera_pose)
    save_point_cloud_as_ply(pc[valid_mask].reshape(-1, 3), os.path.join(output_dir, f'pc.ply'))
    # if rotation is None:
    # pc = predict_pos(image_dir, output_dir, fov, max_depth)
    tags_list = []
    if tags is not None:
        print("tags: ", tags)
        tags_list = tags.split('.')
        for i in range(len(tags_list)):
            tags_list[i] = tags_list[i].strip()

    masks, pred_phrases, box_list = mask_and_save(image_dir, output_dir, tags)

    image = cv2.imread(image_dir)
    masked_images_dir = os.path.join(output_dir, "masked_images")
    if not os.path.exists(masked_images_dir):
        os.makedirs(masked_images_dir)
    padding = 20
    name_dict_mask = {}
    for i in range(len(masks)):
        # filter out large boxes
        if filter:
            if filter_bbox(i, box_list, pred_phrases):
                continue

        mask = masks[i, 0].cpu().numpy()
        mask_pos = np.where(mask)
        top, down = np.min(mask_pos[0]), np.max(mask_pos[0])
        left, right = np.min(mask_pos[1]), np.max(mask_pos[1])
        pred_phrase = pred_phrases[i]
        mask_expanded = mask[:, :, None]
        inverse_mask = 1 - mask_expanded
        white_image = np.ones_like(image) * 255

        # masked_image = image * mask_expanded + white_image * inverse_mask
        masked_image = image
        masked_image = masked_image[max(top - padding, 0): min(down + padding, mask_expanded.shape[0]),
                       max(left - padding, 0): min(right + padding, mask_expanded.shape[1])]

        name = pred_phrases[i].split("(")[0]
        if not name in tags_list and tags is not None:
            for tag in tags_list:
                if name in tag:
                    name = tag
                    break

        if not name in name_dict_mask:
            name_dict_mask[name] = 0
        cnt = name_dict_mask[name]
        name_dict_mask[name] += 1
        cv2.imwrite(os.path.join(masked_images_dir, f"{name}.png"), masked_image)
        cv2.imwrite(os.path.join(masked_images_dir, f"{name}-{cnt}.png"), masked_image)

    # print(f"pred phrases length: {len(pred_phrases)}, content: {pred_phrases}")
    # print(f"masks shape: {masks.shape}")

    x_min, y_min, z_min = np.min(pc, axis=(0, 1))
    x_max, y_max, z_max = np.max(pc, axis=(0, 1))

    bbox = []
    result = []
    result.append({'room_bbox': [[x_min, y_min, z_min], [x_max, y_max, z_max]]})
    for idx, mask in enumerate(masks):
        if filter:
            if filter_bbox(idx, box_list, pred_phrases):
                continue
        mask = mask.cpu().numpy()[0]
        if binary_mask is not None and np.sum(np.logical_and(mask, binary_mask)) / np.sum(mask) > 0.3:
            continue
        pcs = np.copy(pc[mask == True]).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs)

        # Use DBSCAN clustering to remove noise and outliers
        labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
        max_label = labels.max()

        # Create a new point cloud containing only the points from the largest cluster
        # (assuming the largest cluster is the main object and the rest are outliers/noise)
        main_cluster = pcs[labels == np.argmax(np.bincount(labels[labels != -1]))]
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(main_cluster)
        bounding_box = cleaned_pcd.get_axis_aligned_bounding_box()
        min_bound = bounding_box.min_bound
        max_bound = bounding_box.max_bound
        pattern = r"(.*)\((\d+\.\d+)\)"
        cnt = 0
        name = pred_phrases[idx].split("(")[0]
        if name == "":
            continue
        if not name in tags_list and tags is not None:
            for tag in tags_list:
                if name in tag:
                    name = tag
                    break
        if name in name_dict:
            cnt = name_dict[name]
            name_dict[name] += 1
        else:
            name_dict[name] = 1

        print(pred_phrases[idx], bounding_box)

        if name not in tags_list:
            description = ""
            on_floor = None
        else:
            if descriptions is None:
                description = None
                on_floor = None
            else:
                description = descriptions[name]
                on_floor = on_floors[name]

        result.append(
            {
                "bbox": [[float(min_bound[0]), float(min_bound[1]), float(min_bound[2])],
                         [float(max_bound[0]), float(max_bound[1]), float(max_bound[2])]],
                "name": name + '-' + str(cnt),
                "type": "rigid mesh",
                "description": description,
                "on_floor": on_floor,
                # "assetId": select_objects(name),
                "confidence": pred_phrases[idx].split("(")[1].split(")")[0]
            }
        )
    return result

import numpy as np

def euler_to_rotation_matrix_xyz(euler_deg):
    """
    Returns the rotation matrix for intrinsic X->Y->Z Euler angles (in degrees).
    """
    alpha, beta, gamma = np.deg2rad(euler_deg)

    # Rotation about X
    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(alpha), -np.sin(alpha)],
        [0,  np.sin(alpha),  np.cos(alpha)]
    ])

    # Rotation about Y
    Ry = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [           0, 1,           0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # Rotation about Z
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [          0,            0,     1]
    ])

    # Combined rotation: apply X, then Y, then Z => R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

def rotated_bbox_extents(bbox_extents, euler_deg):
    """
    Given an axis-aligned bbox extents (sx, sy, sz),
    rotate via Euler angles (intrinsic X->Y->Z) and return the new AABB extents.
    """
    hx, hy, hz = 0.5 * bbox_extents

    # 8 corners of the original AABB
    corners = np.array([
        [ hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy,  hz],
        [ hx, -hy, -hz],
        [-hx,  hy,  hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [-hx, -hy, -hz]
    ])

    # Apply rotation
    R = euler_to_rotation_matrix_xyz(euler_deg)
    rotated_corners = corners @ R.T

    # Compute axis-aligned extents
    min_xyz = rotated_corners.min(axis=0)
    max_xyz = rotated_corners.max(axis=0)
    return max_xyz - min_xyz  # shape (3,)

def fit_euler_and_scale_simplified(object_bbox, target_bbox, angle_candidates, ignore_dims=None):
    """
    Only try Euler angles of {0, 90} degrees about each axis (8 total).
    Compute a single, uniform scale for the included dimensions.
    If ignore_dims is specified, those dimension indices are not used
    in the scale calculation or the error metric.

    Parameters
    ----------
    object_bbox : np.array, shape (3,)
        The extents (width, height, depth) of the object.
    target_bbox : np.array, shape (3,)
        The extents we want to match.
    ignore_dims : list or set of dimension indices, optional
        For example, [1] means ignore Y dimension during scale & error.

    Returns
    -------
    best_euler : list of [x_angle, y_angle, z_angle]
    best_scale : float
        Uniform scale factor that best matches the included dimensions.
    best_error : float
        The final minimal error across included dimensions.
    """
    if ignore_dims is None:
        ignore_dims = []

    best_error = float('inf')
    best_euler = (0, 0, 0)
    best_scale = 1.0

    for angle_candidate in angle_candidates:
        euler_deg = angle_candidate
        rot_ext = rotated_bbox_extents(object_bbox, euler_deg)
        
        # Collect indices we actually use
        used_dims = [idx for idx in range(3) if idx not in ignore_dims]
        
        # If any used dimension is nearly zero in rot_ext => skip
        if any(rot_ext[idx] < 1e-8 for idx in used_dims):
            continue

        # Single scale factor across included dims
        # e.g. mean of (target_dim / rot_dim) for all included dims
        scale_factors = [target_bbox[idx] / rot_ext[idx] for idx in used_dims]
        uniform_scale = np.min(scale_factors)

        # Compute scaled extents
        scaled_ext = rot_ext * uniform_scale

        # Compute error only on included dims => sum of squared diff
        error = 0.0
        for idx in used_dims:
            diff = scaled_ext[idx] - target_bbox[idx]
            error += diff * diff

        if error < best_error:
            best_error = error
            best_euler = euler_deg
            best_scale = uniform_scale

    return best_euler, best_scale, best_error

