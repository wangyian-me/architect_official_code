from gpt_4.prompts.prompt_inpainting_prompt_generation import prompt_inpainting_prompt_generation
from gpt_4.prompts.prompt_tags_and_descriptions import prompt_tags
from mimetypes import guess_type
from openai import AzureOpenAI
import base64
import os
from openai import OpenAI

dataset_dir = "/work/pi_chuangg_umass_edu/yianwang_umass_edu-data/blenderkit_data_annotated"

llm = AzureOpenAI(
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2024-02-15-preview",
    # "https://qiuxw.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
    azure_endpoint="https://qiuxw.openai.azure.com/",
)


# Function to encode a local image into data URL
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


def generate_content(image_url, filenames, text):
    content = []

    for name in filenames:
        encoded_img = local_image_to_data_url(os.path.join(image_url, name))
        temp = dict()
        temp["type"] = "image_url"
        temp["image_url"] = {"url": encoded_img}
        content.append(temp)
    temp = dict()
    temp["type"] = "text"
    temp["text"] = text
    content.append(temp)
    return content

def generate_content_single(image_dir, text):
    content = []

    #for name in filenames:
    encoded_img = local_image_to_data_url(image_dir)
    temp = dict()
    temp["type"] = "image_url"
    temp["image_url"] = {"url": encoded_img}
    content.append(temp)
    temp = dict()
    temp["type"] = "text"
    temp["text"] = text
    content.append(temp)
    return content


# def query_llm(query):
    # response = llm.chat.completions.create(
    #     model="gpt-4o",  # e.g. gpt-35-instant
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": query,
    #         },
    #     ],
    #     max_tokens=2000,
    # )
    # result = response.choices[0].message.content
    # return result

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def query_llm(query):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="gpt-4",
    )
    res1 = response.choices[0].message.content
    return res1

def query_llm_vision(image_url, text, temperature=1.0):
    filenames = os.listdir(image_url)
    content = generate_content(image_url, filenames, text)
    response = client.chat.completions.create(
        # engine=self.lm_id,
        model="gpt-4o",
        messages=[
            # {"role": "user", "content": ""},
            {"role": "user",
             "content": content,
             },
        ],
    )
    result = response.choices[0].message.content
    return result


def query_llm_vision_ya(images_list, text, temperature=1.0):
    content = []

    for path in images_list:
        encoded_img = local_image_to_data_url(path)
        temp = dict()
        temp["type"] = "image_url"
        temp["image_url"] = {"url": encoded_img}
        content.append(temp)
    temp = dict()
    temp["type"] = "text"
    temp["text"] = text
    content.append(temp)

    response = client.chat.completions.create(
        # engine=self.lm_id,
        model="gpt-4o",
        messages=[
            # {"role": "user", "content": ""},
            {"role": "user",
             "content": content,
             },
        ],
    )
    result = response.choices[0].message.content
    return result

def parse_tags_and_descriptions(response):
    tags = []
    descriptions = []
    on_floors = []
    lines = response.split('\n')
    for line in lines:
        if ":" in line:
            tag, description = line.split(":")
            description, classification = description.split("|")
            if "top" in classification: continue
            tags.append(tag)
            descriptions.append(description)
            on_floors.append("floor" in classification)

    return tags, descriptions, on_floors

def get_tags_and_descriptions(image_url, room_type):
    image_list = [image_url]
    result = query_llm_vision_ya(image_list, prompt_tags.format(room_type=room_type))
    print(f"gpt query response: {result}")
    tags, descriptions, on_floors = parse_tags_and_descriptions(result)
    return tags, descriptions, on_floors

def get_inpainting_prompt(current_objects: list[str], room_type="living room"):
    '''
    current_objects: list of object names. eg: ['sofa', 'sofa', 'chair', 'plant']
    '''
    counting_objects = {object_name: current_objects.count(object_name) for object_name in current_objects}
    counting_objects_str = ', '.join([f"{count} {object_name}" for object_name, count in counting_objects.items()])
    prompt = prompt_inpainting_prompt_generation.format(current_scene=room_type, counting_objects=counting_objects_str)
    result = query_llm(prompt)
    result = result.split('\n')
    print(result)
    negative_objects = [line for line in result if line.startswith('reached limit')][0]
    positive_objects = [line for line in result if line.startswith('lacking')][0]
    negative_objects = negative_objects[len('reached limit: '):].split(', ')
    positive_objects = positive_objects[len('lacking: '):].split(', ')
    return negative_objects, positive_objects

import re

def parse_response(response: str) -> list:
    """
    Parses a string containing a bracketed, comma-separated list of objects.
    Returns a list of object names as strings.

    Example:
        Input: "Here are the objects: [book, vase, pen]"
        Output: ["book", "vase", "pen"]
    """
    pattern = re.compile(r'\[([^]]*)\]')
    match = pattern.search(response)
    if not match:
        return []
    
    # Extract the comma-separated contents inside the brackets
    items_str = match.group(1)
    # Split by commas and strip whitespace
    items = [item.strip() for item in items_str.split(',') if item.strip()]
    return items

def get_tags_small(image_path, kind):
    """
    Detects and returns the small objects placed on a table or shelf in the provided image.
    This function:
      1. Encodes the image.
      2. Queries the LLM with a refined prompt to list objects on top of the table or shelf.
      3. Parses the response to extract the list of object names.
    
    :param image_path: The local path to the image file.
    :return: A list of object names (strings).
    """
    # Refined prompt
    prompt = (
        f"I have an image showing a {kind} of objects. "
        f"List only the objects on the {kind}. "
        f"Exclude the {kind} itself. "
        "Provide each object's name in one or two words, in a single list format: [obj1, obj2, obj3, ...]."
    )
    
    # We reuse query_llm_vision_ya to send the image + text prompt
    # If you want to send multiple images, just wrap the single path in a list
    response = query_llm_vision_ya([image_path], prompt)
    
    # Use the parse_response function to extract the bracketed list of objects
    objects = parse_response(response)
    
    return objects

import json
        
with open(os.path.join(dataset_dir, "blenderkit_database.json"), "r") as f:
    database = json.load(f)

def find_large(all_large_path, current_scene_name):
    """
    1) Reads the JSON data from `all_large_path`.
    2) Finds the relevant scene by name.
    3) For all furniture in that scene, packs the descriptions into
       a single GPT-4 request.
    4) GPT-4 returns a JSON array, each element describing if we can place items,
       the furniture kind, and a short stable-diffusion-xl inpainting prompt.
    5) Returns parallel lists:
       - furniture_names: names of the furniture to place small items on
       - prompts: the GPT-4 generated inpainting prompts
       - kinds: "table" or "shelf"
    """

    # --- Load the JSON file ---
    with open(all_large_path, "r") as f:
        data = json.load(f)

    furniture_list = []
    furniture_descriptions = []
    for obj in data["objects"]:
        furniture_list.append(obj["name"])
        furniture_descriptions.append(database[obj["assetId"]]["description"])

    # --- Build a single GPT-4 request with all descriptions ---
    # System message gives GPT-4 context/instructions.
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that will classify furniture items in bulk. "
            "You will receive multiple furniture descriptions in one request."
        )
    }

    # Example stable-diffusion-xl inpainting prompts you might mention:
    #   "a messy desk with scattered papers"
    #   "a shelf brimming with books"
    #   "a glass coffee table with a vase of flowers"
    #   "a wide kitchen table with plates and cutlery"
    #   "a rustic wooden shelf displaying potted plants"
    #   etc.


    prompt = f"""You are given multiple furniture descriptions placed in a {current_scene_name}.
For each description, do the following steps:
1. Determine if the furniture has potential to hold small items on its surface (answer "YES" or "NO").
   - If the furniture is purely decorative with no usable surface, answer "NO".
2. If your answer is "YES", identify whether it is a:
   - "table": items can only go on the top surface.
   - "shelf": has multiple levels (like a bookshelf or a shelving unit).
   Note that TV stand is not a table or a shelf.
3. Provide a concise Stable Diffusion XL inpainting prompt describing the small items you'd add,
   ensuring:
   - They match the furniture
   - They are short and descriptive. For example:
     - "A neat arrangement of vintage books and a small potted plant on the wooden shelf, photorealistic style, consistent shadows."
     - "A messy office desk with scattered papers, a coffee mug, and a small desk lamp, photorealistic style."
     - "An organized coffee table with coffee mug, decorative plants, and books, photorealistic style, consistent shadows."
   - You can add brief style hints (photorealistic, modern, minimalist, etc.) but keep it under one sentence.

Return your answer strictly as a JSON array, where each element corresponds to one furniture description in the order provided. For each furniture item, your output MUST match one of these two formats, with no extra keys or text outside the JSON array:

[
  {{
    "answer": "YES",
    "kind": "shelf",
    "prompt": "A rustic wooden shelf displaying potted plants, photorealistic, consistent shadows."
  }},
  {{
    "answer": "NO"
  }}
]

Where the required keys are:
- "answer": Either "YES" or "NO"
- "kind": If "YES", then either "table" or "shelf"
- "prompt": A short Stable Diffusion XL inpainting prompt describing the items

Furniture descriptions (in order):

""" + "\n".join(
    [f"{i+1}) {desc}" for i, desc in enumerate(furniture_descriptions)]
)

    user_message = {
        "role": "user",
        "content": (
            prompt
        )
    }

    # print(user_message)

    # Edge case: if no descriptions, we can just return empty
    if not furniture_descriptions:
        return [], [], []

    try:
        # Send the chat request to GPT-4
        response = client.chat.completions.create(
            model="o1-preview",
            messages=[user_message],
        )

        # Extract GPT-4's response
        content = response.choices[0].message.content.strip()
        print("content:", content)
        # Attempt to parse the JSON
        results = json.loads(content)

        print("results:", results)

        # We expect results to be a list of the same length as furniture_list
        if not isinstance(results, list):
            # If GPT-4 didn't return a list, fallback
            return [], [], []

        # Prepare final return lists
        furniture_names = []
        prompts = []
        kinds = []

        # Iterate over each GPT-4 result (should match each item in furniture_list by index)
        for idx, item in enumerate(results):
            # If GPT-4 said "NO", skip
            if item.get("answer") == "YES":
                # Then retrieve the name, kind, prompt
                # We match the same index in furniture_list
                furniture_obj = data["objects"][idx]
                name = furniture_obj.get("name", "")
                kind = item.get("kind", "")
                prompt = item.get("prompt", "")

                furniture_names.append(name)
                prompts.append(prompt)
                kinds.append(kind)

        return furniture_names, prompts, kinds

    except Exception as e:
        print("Error calling GPT-4 or parsing response:", e)
        # In case of failure, just return empty
        return [], [], []

