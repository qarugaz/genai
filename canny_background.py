import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

from lib.upload import upload_image


def create_canny_background_mask(image, low_threshold=100, high_threshold=200):
    """
    Create a background mask using Canny edge detection.

    Args:
        image_path (str): Path to the input image
        low_threshold (int): Lower threshold for Canny edge detection
        high_threshold (int): Higher threshold for Canny edge detection

    Returns:
        tuple: (PIL Image mask, original image)
        :param high_threshold:
        :param low_threshold:
        :param image:
    """
    # Read the image
    image = cv2.imread(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Create a mask by inverting and dilating edges
    mask = cv2.bitwise_not(edges)

    # Dilate the mask to cover more background area
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=3)

    # Convert mask to PIL Image
    pil_mask = Image.fromarray(dilated_mask)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return pil_mask, pil_image


def change_image_background(image, prompt, negative_prompt=None):
    """
    Change the background of an input image using Stable Diffusion inpainting.

    Args:
        input_image_path (str): Path to the input image
        prompt (str): Desired background description
        negative_prompt (str, optional): Description of what to avoid in the background

    Returns:
        Image: Modified image with new background
        :param negative_prompt:
        :param prompt:
        :param image:
    """
    # Load the Stable Diffusion Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    # Create mask using Canny edge detection
    mask, image = create_canny_background_mask(image)

    # Set default negative prompt if not provided
    if negative_prompt is None:
        negative_prompt = "low quality, blurry, unclear, distorted, unnatural"

    # Generate the new image with changed background
    output = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        strength=0.75
    ).images[0]

    return output


def main():
    image = load_image(
        "https://storage.googleapis.com/crescis-testing/128464536273.png"
    )

    # Desired background description
    background_prompt = "serene mountain landscape with misty morning light"

    # Change the background
    result_image = change_image_background(image, background_prompt)

    # Save the result
    upload_image(result_image,"sd-flux")


if __name__ == "__main__":
    main()