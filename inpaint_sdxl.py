from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import numpy as np
import torch
from PIL import Image, ImageFilter
import PIL
import cv2
import os
from scipy.ndimage import morphology
import time

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, \
    StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from ram_utils.utils import mask_and_save

# init_image = load_image(
#    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
# )

# load sdxl
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                 torch_dtype=torch.float16, variant="fp16").to("cuda")

generator = torch.Generator(device="cpu").manual_seed(int(time.time()) % 10000)

# load seg to image controlnet and sd1.5
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
# )


# pipe_sd15 = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
# )
# pipe_sd15.scheduler = UniPCMultistepScheduler.from_config(pipe_sd15.scheduler.config)
# pipe_sd15.enable_model_cpu_offload()

def blur(image: PIL.Image.Image, blur_factor: int = 4) -> PIL.Image.Image:
    """
    Applies Gaussian blur to an image.
    """
    image = image.filter(ImageFilter.GaussianBlur(blur_factor))
    return image


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[
                               0:1], f"image {image.shape} and image_mask {image_mask.shape} must have the same image size."
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


pipe.mask_processor.blur = blur

def inpainting(image_dir, mask_image_dir, save_dir, prompt, negative_prompt, erosion_steps=5):
    print("##################### start inpainting #############################")
    init_image = load_image(
        image_dir
    )
    mask_image = load_image(
        mask_image_dir
    )

    mask_np = np.array(mask_image)
    mask_np = (mask_np > 50).astype(np.int32)[:, :, 0]
    eroded_mask = mask_np
    for i in range(erosion_steps):
        eroded_mask = morphology.binary_erosion(eroded_mask)
    eroded_mask = np.stack([eroded_mask, eroded_mask, eroded_mask], axis=-1)
    mask_image = Image.fromarray(eroded_mask.astype(np.uint8) * 255)

    blurred_mask = pipe.mask_processor.blur(mask_image, blur_factor=30)
    mask_image = blurred_mask
    print("image_size: ", init_image.size)
    print("mask_image_size: ", mask_image.size)

    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=30,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
        negative_prompt=negative_prompt
    ).images[0]
    print("save_image_size: ", image.size)
    image.save(save_dir)
    print("##################### end inpainting #############################")


def inpainting_shelf(image_dir, mask_image_dir, save_dir, prompt, negative_prompt, erosion_steps=4, blur_factor=25):
    init_image = load_image(
        image_dir
    )
    mask_image = load_image(
        mask_image_dir
    )

    mask_np = np.array(mask_image)
    mask_np = (mask_np > 50).astype(np.int32)[:, :, 0]
    eroded_mask = mask_np
    for i in range(erosion_steps):
        eroded_mask = morphology.binary_erosion(eroded_mask)
    eroded_mask = np.stack([eroded_mask, eroded_mask, eroded_mask], axis=-1)
    mask_image = Image.fromarray(eroded_mask.astype(np.uint8) * 255)

    parent_dir = os.path.dirname(save_dir)
    print("parent_dir:", parent_dir)
    save_eroded_mask = os.path.join(parent_dir, 'eroded_mask.jpg')
    save_blurred_mask = os.path.join(parent_dir, 'blurred_mask.jpg')

    mask_image.save(save_eroded_mask)

    blurred_mask = pipe.mask_processor.blur(mask_image, blur_factor=blur_factor)
    mask_image = blurred_mask

    mask_image.save(save_blurred_mask)

    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=30,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
        negative_prompt=negative_prompt
    ).images[0]
    image.save(save_dir)

