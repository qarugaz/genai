from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
from google.cloud import storage

from lib.upload import upload_multiple_images


def sdxl():
    prompt = ("woman wearing a bright dress "
              "while on a ski station surrounded by snow fields")
    negative_prompt = "low quality, bad quality, sketches"

    # download an image
    image = load_image(
        "https://storage.googleapis.com/crescis-testing/128464536273.png"
    )

    # initialize the models and pipeline
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    )

    # get canny image
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # generate image
    images = pipe(
        prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        image=canny_image,
        num_images_per_prompt=4
    ).images

    bucket_name = "sd-flux"
    return upload_multiple_images(images, bucket_name)