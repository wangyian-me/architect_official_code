import subprocess
import requests
import json
import time
import os
import trimesh
import base64
from mimetypes import guess_type
# import openai  # for GPT calls

from openai import OpenAI
# If you have a custom OpenAI client, adapt accordingly.
# Example:
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

small_cate_list = [
    "Book",
    "Remote control",
    "Pen",
    "Smartphone",
    "Charger",
    "Key",
    "Wallet",
    "Eyeglass",
    "Necklace",
    "Photo frame",
    "Figurine",
    "Candle",
    "Vase",
    "Coaster",
    "Tissue box",
    "Lip balm",
    "Cosmetic",
    "Bandage",
    "Tool",
    "Earbud",
    "Cup",
    "Plate",
    "Fork",
    "Organizer",
    "Spice",
    "Snack",
    "Water bottle",
    "Toy",
    "DVD",
    "Paper clip"
]

large_furniture_list = [
    "Bed",
    "Headboard",
    "Bed frame",
    "Bunk bed",
    "Loft bed",
    "Four-poster bed",
    "Canopy bed",
    "Daybed",
    "Murphy bed",
    "Nightstand",
    "Dresser",
    "Chest of drawers",
    "Wardrobe",
    "Armoire",
    "Vanity",
    "Chaise lounge",
    "Desk",
    "Chair",
    "Bench",
    "Blanket chest",
    "Bookcase",
    "Shelving unit",
    "Accent cabinet",
    "Linen cabinet",
    "Dressing table",
    "Storage ottoman",
    "Room divider",
    "Tallboy",
    "Trundle bed",
    "Clothes rack",
    # "houseplant",
    # "Bookshelf",
    # "Floor lamp",
    # "Sofa",
    # "Loveseat",
    # "Armchair",
    # "Recliner",
    # "Coffee table",
    # "TV stand",
    # "TV panel",
    # "TV set",
    # "Side table",
    # "Console table",
    # "Entertainment center",
    # "Fireplace",
    # "Ottoman",
    # "Paint",
    # "Stool",
    # "Shelf",
    # "Bean bag",
    # "Futon",
    # "Bookcase",
    # "Cabinet",
    # "Hutch",
    # "Storage ottoman",
    # "Display cabinet",
    # "Room divider",
    # "Bar cart",
]



def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def identify_front_view_one_by_one(asset_base_id, image_dir):
    """
    We have 4 images for the same object: index 0..3. 
    We feed them one by one to GPT, asking "Is this the front view?" 
    We stop once GPT says 'yes' (or some threshold of confidence).
    """
    image_paths = [
        os.path.join(image_dir, f"{asset_base_id}_0.png"),
        os.path.join(image_dir, f"{asset_base_id}_1.png"),
        os.path.join(image_dir, f"{asset_base_id}_2.png"),
        os.path.join(image_dir, f"{asset_base_id}_3.png")
    ]
    
    # Define a standard rotation per index if you want to store it
    # e.g. index 0 => [0,0,0], index 1 => [0,180,0], etc.

    # ours vs holodeck
    # "x+" 0 270
    # "y-" 270 180
    # "x-" 180 90
    # "y+" 90 0

    standard_rotations = [
        [0, 0, 0],
        [0, 180, 0],
        [0, -90, 0],
        [0, 90, 0]
    ]
    
    for idx in range(len(image_paths) - 1, -1, -1):
        path = image_paths[idx]
        if not os.path.exists(path):
            print(f"[Warning] Image not found: {path}")
            continue

        # Read image and encode to base64
        encoded_img = local_image_to_data_url(path)
        temp = dict()
        temp["type"] = "image_url"
        temp["image_url"] = {"url": encoded_img}
        
        # Create your prompt for GPT
        messages = [
            {
                "role": "user",
                "content": [
                    temp, 
                    {
                        "type": "text",
                        "text": (
                            "You are be given a single image (base64-encoded) of a 3D object "
                            "from a certain viewpoint (front, back, left, or right)."
                            "Please respond with a single word: 'yes' if this image is the front view, "
                            "or 'no' if it is not the front view."
                        )
                    }
                ]
            }
        ]
        
        # Make your GPT call
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # or your 'gpt4o' endpoint
                messages=messages,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip().lower()
            
            print(f"Image index {idx} => GPT says: {content}")
            
            # Check if GPT recognized it as front
            if content.startswith("yes"):
                messages_second = [
                    {
                        "role": "system",
                        "content": (
                            "You are a vision assistant. You will be given the same image "
                            "that was identified as the front view of a 3D object. "
                            "We only want to keep it if it's a textureful object, or if it's "
                            "a shelf/table that is empty. If the object is textureless/weird, or "
                            "a shelf/table with items on it, we do NOT want it.\n\n"
                            "Answer with a single word: 'yes' if it is a valid asset, "
                            "'no' if it is invalid."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            temp,
                            {
                                "type": "text",
                                "text": (
                                    "Is this object valid? Respond only 'yes' or 'no'. "
                                    "It's invalid if textureless/weird, or if shelf/table is not empty."
                                )
                            }
                        ]
                    }
                ]
                
                try:
                    response2 = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages_second,
                        temperature=0.0
                    )
                    valid_answer = response2.choices[0].message.content.strip().lower()
                    print(f"Second check for index {idx} => GPT says (valid asset?): {valid_answer}")
                    
                    if valid_answer.startswith("yes"):
                        # Great, it's front + valid => return
                        print(f"Identified FRONT VIEW => index: {idx}, GPT validated asset.")
                        return idx, standard_rotations[idx]
                    else:
                        print("GPT flagged the asset as invalid. Skipping.")
                        return 0, None
                except Exception as e2:
                    print(f"Error calling GPT second question: {e2}")
                    continue
        
        except Exception as e:
            print(f"Error calling GPT: {e}")
            # fallback or keep going
    
    # If we never got a 'yes', it's an error
    return 0, None


def find_smallest_dim_rotation(extents):
    """
    extents = [dx, dy, dz] (the bounding box extents).
    """
    dx, dy, dz = extents
    
    # Identify which axis is smallest
    # 0 -> x, 1 -> y, 2 -> z
    i_min = min(range(3), key=lambda i: extents[i])
    
    if i_min == 0:
        return [
            [0,   0, 0],
            [90,  0, 0],
            [180, 0, 0],
            [270, 0, 0],
        ]
    elif i_min == 1:
        return [
            [0,   0, -90],
            [0,  90, -90],
            [0, 180, -90],
            [0, 270, -90],
        ]
    else:
        return [
            [0.0, 90.0, 0.0],
            [90.0, 0.0, -90.0],
            [0.0, 270.0, 0.0],
            [180.0, 0.0, -90.0],
        ]
        
    
def identify_front_view_codimension(asset_base_id, image_dir, glb_dir):
    mesh = trimesh.load(os.path.join(glb_dir, f'{asset_base_id}.glb'), force='mesh')
    bounding_box_extents = mesh.bounding_box.extents.tolist()
    rotations = find_smallest_dim_rotation(bounding_box_extents)
    for rotation in rotations:
        subprocess.run([
                'python', 'render_codimension.py', 
                '--uid', asset_base_id, 
                '--image_dir', image_dir, 
                '--load_dir', glb_dir, 
                '--rot_x', str(rotation[0]),
                '--rot_y', str(rotation[1]),
                '--rot_z', str(rotation[2])
            ])
        image_paths = [
            os.path.join(image_dir, f"{asset_base_id}_0.png"),
            os.path.join(image_dir, f"{asset_base_id}_1.png"),
        ]

        for idx in range(2):
            path = image_paths[idx]
            if not os.path.exists(path):
                print(f"[Warning] Image not found: {path}")
                continue

            # Read image and encode to base64
            encoded_img = local_image_to_data_url(path)
            temp = dict()
            temp["type"] = "image_url"
            temp["image_url"] = {"url": encoded_img}
            
            # Create your prompt for GPT
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a vision assistant. You will be given a single image (base64-encoded) of a 3D object "
                        "from a certain viewpoint (front, back, left, or right)."
                        "Please respond with a single word: 'yes' if this image is the front view, "
                        "or 'no' if it is not the front view."
                    )
                },
                {
                    "role": "user",
                    "content": [temp]
                }
            ]
            
            # Make your GPT call
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",  # or your 'gpt4o' endpoint
                    messages=messages,
                    temperature=0.0
                )
                content = response.choices[0].message.content.strip().lower()
                
                print(f"Image index {idx} => GPT says: {content}")
                
                # Check if GPT recognized it as front
                if content.startswith("yes"):
                    messages_second = [
                        {
                            "role": "system",
                            "content": (
                                "You are a vision assistant. You will be given the same image "
                                "that was identified as the front view of a 3D object. "
                                "We only want to keep it if it's a textureful object, or if it's "
                                "a shelf/table that is empty. If the object is textureless/weird, or "
                                "a shelf/table with items on it, we do NOT want it.\n\n"
                                "Answer with a single word: 'yes' if it is a valid asset, "
                                "'no' if it is invalid."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                temp,
                                {
                                    "type": "text",
                                    "text": (
                                        "Is this object valid? Respond only 'yes' or 'no'. "
                                        "It's invalid if textureless/weird, or if shelf/table is not empty."
                                    )
                                }
                            ]
                        }
                    ]
                    
                    try:
                        response2 = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages_second,
                            temperature=0.0
                        )
                        valid_answer = response2.choices[0].message.content.strip().lower()
                        print(f"Second check for index {idx} => GPT says (valid asset?): {valid_answer}")
                        
                        if valid_answer.startswith("yes"):
                            # Great, it's front + valid => return
                            print(f"Identified FRONT VIEW => index: {idx}, GPT validated asset.")
                            return idx, rotation
                        else:
                            print("GPT flagged the asset as invalid. Skipping.")
                            return idx, None
                    except Exception as e2:
                        print(f"Error calling GPT second question: {e2}")
                        continue
            
            except Exception as e:
                print(f"Error calling GPT: {e}")

    return 0, None
        



class BlenderkitDataset():
    def __init__(self, data_dir):
        """
        data_dir: Directory where .glb files will be downloaded and 
                  where blenderkit_database.json resides.
        """
        self.data_dir = data_dir
        self.json_dir = os.path.join(self.data_dir, "blenderkit_database.json")
        
        if not os.path.exists(self.json_dir):
            # initialize an empty JSON if it doesn't exist
            with open(self.json_dir, "w") as f:
                json.dump({}, f)
                
        self.database = json.load(open(self.json_dir, "r"))
        self.free = False

    def online_retrieve(self, query):
        """
        Retrieves search results from BlenderKit API for the given query, downloads new .glb files 
        (up to 20 matches), and returns their assetBaseIds.
        """
        q = query.replace(' ', '+')
        search_url = f"https://www.blenderkit.com/api/v1/search/?query=asset_type:model+{q}+order:_score"
        
        # if self.free:
        #    search_url += "+is_free:True"
        
        result = requests.get(search_url)
        time.sleep(1)  # be polite to the API
        result = json.loads(result.content.decode())
        
        uids = []
        
        # Process only up to 20 results
        for obj in result.get('results', [])[:50]:
            # print(obj)
            asset_base_id = obj['assetBaseId']
            
            # skip if we already have it in our database
            if asset_base_id in self.database and "is_large" in self.database[asset_base_id]:
                continue
            
            if not os.path.exists(os.path.join(self.data_dir, f"{asset_base_id}.glb")):
                success = self.download_glb(asset_base_id, 'glb', self.data_dir)
                if not success:
                    continue
            
            # You could store the entire object entry, or store minimal data
            self.database[asset_base_id] = {
                "name": obj.get('name'),
                "description": obj.get('description'),
                "category": obj.get('category'),
            }
            print(asset_base_id)
            uids.append(asset_base_id)
        
        # Optionally save after retrieving
        self.save_database()
        
        return uids

    def download_glb(self, asset_base_id, file_type, save_dir):
        """
        Downloads a BlenderKit asset in .glb format by calling an external Python script 
        in a conda environment. Returns True if successful, False otherwise.
        """
        target_env = os.environ.get("BLENDER_ENV")
        target_code = 'download_blenderkit.py'
        
        new_env = os.environ.copy()
        banned_path = 'LuisaRender/build/bin:'
        if 'LD_LIBRARY_PATH' in new_env and banned_path in new_env['LD_LIBRARY_PATH']:
            new_env['LD_LIBRARY_PATH'] = new_env['LD_LIBRARY_PATH'].replace(banned_path, '')
        
        subprocess.run([
            'conda', 'run', 
            '--prefix', target_env, 
            'python', target_code,
            '--uid', asset_base_id,
            '--type', file_type,
            '--save_dir', save_dir
        ], env=new_env)
        
        return os.path.exists(os.path.join(save_dir, f"{asset_base_id}.glb"))

    def annotate_glb(self, asset_base_id, image_dir, glb_dir, is_large=False, codimension=False):
        """
        Annotates the .glb file with bounding box info, approximate scale, etc.
        If is_large=True, optionally calls a script to render multiple directions 
        and uses GPT to identify the front view.
        """
        glb_path = os.path.join(glb_dir, f"{asset_base_id}.glb")
        if not os.path.exists(glb_path):
            print(f"Warning: {glb_path} not found.")
            return
        
        # Optionally, render from multiple directions if is_large
        front_view_rotation = None
        if is_large:
            # 1) Render from multiple directions
            if codimension:
                front_view_idx, front_view_rotation = identify_front_view_codimension(asset_base_id, image_dir, glb_dir)
                if front_view_idx == 1 and front_view_rotation is not None:
                    front_view_rotation[2] += 180
            else:
                subprocess.run([
                    'python', 'render_directions.py', 
                    '--uid', asset_base_id, 
                    '--image_dir', image_dir, 
                    '--load_dir', glb_dir
                ])
                # 2) Use GPT to identify front view
                front_view_idx, front_view_rotation = identify_front_view_one_by_one(asset_base_id, image_dir)


            if front_view_rotation is None:
                print(f"Error for {asset_base_id}")
                del self.database[asset_base_id]
                file_path = os.path.join(self.data_dir, f"{asset_base_id}.glb")
                if os.path.exists(file_path):
                    # Delete the file
                    os.remove(file_path)
                return
        else:
            subprocess.run([
                'python', 'render_directions.py', 
                '--uid', asset_base_id, 
                '--image_dir', image_dir, 
                '--load_dir', glb_dir, 
                '--front_only'
            ])
            front_view_idx = 0
        # Use trimesh to load and inspect geometry
        try:
            mesh = trimesh.load(glb_path, force='mesh')
        except Exception as e:
            print(f"Error loading mesh for {asset_base_id}: {e}")
            return
        
        # Basic bounding box / scale data
        bounding_box_extents = mesh.bounding_box.extents.tolist()
        center = mesh.bounding_box.centroid.tolist()
        # A naive scale measure - the longest dimension of the bounding box
        scale = max(bounding_box_extents)
        
        # Save data in the database
        if asset_base_id not in self.database:
            self.database[asset_base_id] = {}
        
        self.database[asset_base_id].update({
            "bounding_box_extents": bounding_box_extents,
            "center": center,
            "front_view_rotation": front_view_rotation,
            "is_large": is_large,
            "codimension": codimension
        })

        self.refine_asset_description(asset_base_id, os.path.join(image_dir, f"{asset_base_id}_{front_view_idx}.png"))
        
        # Optionally, write the database after each annotation
        self.save_database()

    def save_database(self):
        """
        Persists the self.database dict to the local JSON file.
        """
        with open(self.json_dir, "w") as f:
            json.dump(self.database, f, indent=2)

    def encode_large_and_small_objects(self):
        """
        Splits your database objects into large vs. small based on 'is_large',
        encodes their descriptions using SBERT, and saves to two separate .pt files.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        def save_embeddings(assets, model, batch_size, save_path):
            """
            Encodes the descriptions of a list of assets (id, description)
            in batches and saves them, along with their IDs, to `save_path`.
            """
            descriptions = [asset_desc for _, asset_desc in assets]
            
            # Encode in batches
            all_embeddings = []
            for start_idx in range(0, len(descriptions), batch_size):
                batch = descriptions[start_idx:start_idx + batch_size]
                # Returns a torch.Tensor if convert_to_tensor=True
                batch_embeddings = model.encode(batch, convert_to_tensor=True)
                all_embeddings.append(batch_embeddings)

            if len(all_embeddings) == 0:
                return

            # Concatenate all batch embeddings
            all_embeddings = torch.cat(all_embeddings, dim=0)

            # Create a dictionary with IDs + embeddings
            id_list = [asset_id for asset_id, _ in assets]
            save_dict = {
                "ids": id_list,
                "embeddings": all_embeddings
            }
            torch.save(save_dict, save_path)
            print(f"Saved {len(assets)} embeddings to {save_path}")

        # 1. Load SBERT model
        sbert_model = SentenceTransformer('all-mpnet-base-v2').to('cuda:0')

        # 2. Separate your assets into large vs. small
        large_assets = []
        small_assets = []

        for asset_base_id, asset_info in self.database.items():
            description = asset_info.get("description", "")
            if description == "":
                description = asset_info.get("name", "")
            is_large = asset_info.get("is_large", False)

            if is_large:
                large_assets.append((asset_base_id, description))
            else:
                small_assets.append((asset_base_id, description))

        # 3. Encode and save each group separately
        batch_size = 64

        # Large objects
        large_save_path = os.path.join(self.data_dir, "descriptions_embeddings_large.pt")
        save_embeddings(large_assets, sbert_model, batch_size, large_save_path)

        # Small objects
        small_save_path = os.path.join(self.data_dir, "descriptions_embeddings_small.pt")
        save_embeddings(small_assets, sbert_model, batch_size, small_save_path)

    def test_orientation(self, image_dir, glb_dir, uid=None):
        for key in self.database.keys():
            if uid is None or key == uid:
                subprocess.run([
                    'python', 'render_directions.py', 
                    '--uid', key, 
                    '--image_dir', image_dir, 
                    '--load_dir', glb_dir,
                    '--test',
                    '--rotation', str(self.database[key]["front_view_rotation"][1])
                ])

    def refine_asset_description(self, asset_base_id, front_image_path):
        """
        Calls GPT-4 (or gpt4o) to refine the object's name+description 
        into a short, more accurate summary for retrieval.
        
        Assumes:
        - The 'front' image is stored at image_dir/asset_base_id_0.png 
            (or adapt if you store front_view_idx differently).
        - The database entry has .get('name') and .get('description').

        Saves the result as self.database[asset_base_id]['refined_description'].
        """
        # 1. Check if the front-view image exists
        if not os.path.exists(front_image_path):
            print(f"[Warning] No front-view image for asset {asset_base_id} at {front_image_path}")
            return
        
        # 2. Retrieve name and description from database
        asset_info = self.database.get(asset_base_id, {})
        original_name = asset_info.get("name", "Unknown object")
        original_desc = asset_info.get("description", "No description available.")
        
        # 3. Convert the front image to a data URL (base64)
        front_image_data_url = local_image_to_data_url(front_image_path)
        
        # 4. Construct the prompt for GPT
        #    We'll ask for a short descriptive text to help with retrieval.
        #    You can adjust the style, length, or constraints as needed.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that specializes in concise object descriptions. "
                    "You will be given:\n"
                    " - An object's name.\n"
                    " - Its existing (possibly inaccurate) textual description.\n"
                    " - A front-view image (base64-encoded).\n\n"
                    "Please provide a short, clear description that would be helpful for searching or retrieval. "
                    "Focus on major identifying features, materials, or shape, and keep it under ~100 characters. "
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Object Name: {original_name}\n"
                            f"Original Description: {original_desc}\n\n"
                            "Please refine the description using the front-view image below."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": front_image_data_url}
                    }
                ]
            }
        ]
        
        # 5. Make the GPT API call
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # or whatever your vision-capable model is called
                messages=messages,
                temperature=0.0
            )
            refined_text = response.choices[0].message.content.strip()
            
            print(f"[Info] Refined description for {asset_base_id}: {refined_text}")
            
            # 6. Save it back into the database
            self.database[asset_base_id]['description'] = refined_text
            self.save_database()
        
        except Exception as e:
            print(f"Error calling GPT for refine_asset_description: {e}")

    def get_img(self, uid, image_dir):
        rot_list = [[0,   0, 0],
            [90,  0, 0],
            [180, 0, 0],
            [270, 0, 0],
            [0,   0, -90],
            [0,  90, -90],
            [0, 180, -90],
            [0, 270, -90],
            [0.0, 90.0, 0.0],
            [90.0, 0.0, -90.0],
            [0.0, 270.0, 0.0],
            [180.0, 0.0, -90.0],
            ]
        direction = {0: 0, 180: 1, -90: 2, 90: 3}
        idx = 0
        if self.database[uid]["is_large"]:
            if self.database[uid]["codimension"]:
                if self.database[uid]["front_view_rotation"] in rot_list:
                    idx = 0
                else:
                    idx = 1
            else:
                idx = direction[self.database[uid]["front_view_rotation"][1]]

        imgpath = os.path.join(image_dir, f"{uid}_{idx}.png")
        return imgpath

    def encode_dino_and_clip_features_for_large_and_small_objects(self, image_dir, batch_size=32):
        """
        Splits your database objects into large vs. small using 'is_large',
        then for each group (large/small) encodes their images using both
        DINOv2 and CLIP. Saves 4 .pt files:
            - dino_embeddings_large.pt
            - dino_embeddings_small.pt
            - clip_embeddings_large.pt
            - clip_embeddings_small.pt
        """
        import torch
        from PIL import Image
        import torch.nn.functional as F
        from transformers import (
            AutoImageProcessor, 
            AutoModel,          # for DINO
            AutoProcessor, 
            CLIPModel           # for CLIP
        )
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 1) Load DINO model & processor
        dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        dino_model.eval()
        
        # 2) Load CLIP model & processor
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_model.eval()
        
        # 3) Separate your UIDs into large vs. small
        large_uids = []
        small_uids = []
        
        for uid, asset_info in self.database.items():
            if asset_info.get("is_large", False):
                large_uids.append(uid)
            else:
                small_uids.append(uid)
        
        # Helper function to batch-encode images with DINO
        def encode_images_dino(uids):
            """
            Returns: 
              embeddings (torch.Tensor, shape=(num_images, hidden_dim)),
              same order as input `uids`.
            """
            # We'll collect images in memory, then do batched forward passes.
            all_features = []
            batch_imgs = []
            valid_uids = []
            
            for idx, uid in enumerate(uids):
                img_path = self.get_img(uid, image_dir)
                if not os.path.exists(img_path):
                    # If image path doesn't exist, skip it or handle differently
                    continue
                
                image = Image.open(img_path).convert("RGB")
                batch_imgs.append(image)
                valid_uids.append(uid)
                
                # If we reached batch_size, or it's the last iteration => encode
                if len(batch_imgs) == batch_size or idx == (len(uids) - 1):
                    with torch.no_grad():
                        inputs = dino_processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
                        outputs = dino_model(**inputs)
                        
                        # outputs.last_hidden_state is shape (bs, seq_len, hidden_dim)
                        # We'll take the mean across the sequence dimension
                        feats = outputs.last_hidden_state.mean(dim=1)  # shape (bs, hidden_dim)
                        # Optionally normalize
                        # feats = F.normalize(feats, dim=1)
                        
                    all_features.append(feats.cpu())
                    batch_imgs = []
            
            # Concatenate everything
            if len(all_features) > 0:
                all_features = torch.cat(all_features, dim=0)
            else:
                all_features = torch.empty(0)
            
            return all_features, valid_uids
        
        # Helper function to batch-encode images with CLIP
        def encode_images_clip(uids):
            """
            Returns: 
              embeddings (torch.Tensor, shape=(num_images, clip_embed_dim)),
              same order as input `uids`.
            """
            all_features = []
            batch_imgs = []

            valid_uids = []
            
            for idx, uid in enumerate(uids):
                img_path = self.get_img(uid, image_dir)
                if not os.path.exists(img_path):
                    continue

                valid_uids.append(uid)
                
                image = Image.open(img_path).convert("RGB")
                batch_imgs.append(image)
                
                if len(batch_imgs) == batch_size or idx == (len(uids) - 1):
                    with torch.no_grad():
                        inputs = clip_processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
                        # get_image_features outputs shape (bs, embed_dim)
                        feats = clip_model.get_image_features(**inputs)
                        # Optionally normalize
                        # feats = F.normalize(feats, dim=1)
                        
                    all_features.append(feats.cpu())
                    batch_imgs = []
            
            if len(all_features) > 0:
                all_features = torch.cat(all_features, dim=0)
            else:
                all_features = torch.empty(0)
            
            return all_features, valid_uids
        
        # 4) Encode large + small with DINO
        #    We'll store them in a dict with "ids" and "embeddings"
        print("Encoding LARGE objects with DINO ...")
        dino_large_embeds, dino_valid_uids = encode_images_dino(large_uids)
        dino_large_dict = {
            "ids": dino_valid_uids,
            "embeddings": dino_large_embeds
        }
        
        print("Encoding SMALL objects with DINO ...")
        dino_small_embeds, dino_valid_uids_small = encode_images_dino(small_uids)
        dino_small_dict = {
            "ids": dino_valid_uids_small,
            "embeddings": dino_small_embeds
        }
        
        # 5) Encode large + small with CLIP
        print("Encoding LARGE objects with CLIP ...")
        clip_large_embeds, clip_valid_uids = encode_images_clip(large_uids)
        clip_large_dict = {
            "ids": clip_valid_uids,
            "embeddings": clip_large_embeds
        }
        
        print("Encoding SMALL objects with CLIP ...")
        clip_small_embeds, clip_valid_uids_small = encode_images_clip(small_uids)
        clip_small_dict = {
            "ids": clip_valid_uids_small,
            "embeddings": clip_small_embeds
        }
        
        # 6) Save to four .pt files
        dino_large_path = os.path.join(self.data_dir, "dino_embeddings_large.pt")
        dino_small_path = os.path.join(self.data_dir, "dino_embeddings_small.pt")
        clip_large_path = os.path.join(self.data_dir, "clip_embeddings_large.pt")
        clip_small_path = os.path.join(self.data_dir, "clip_embeddings_small.pt")
        
        torch.save(dino_large_dict, dino_large_path)
        torch.save(dino_small_dict, dino_small_path)
        torch.save(clip_large_dict, clip_large_path)
        torch.save(clip_small_dict, clip_small_path)
        
        print(f"DINO large embeddings: {dino_large_embeds.shape}, saved to {dino_large_path}")
        print(f"DINO small embeddings: {dino_small_embeds.shape}, saved to {dino_small_path}")
        print(f"CLIP large embeddings: {clip_large_embeds.shape}, saved to {clip_large_path}")
        print(f"CLIP small embeddings: {clip_small_embeds.shape}, saved to {clip_small_path}")