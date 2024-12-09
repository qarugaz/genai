import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from diffusers.utils import load_image
import numpy as np
import cv2

from lib.upload import upload_image


def change_image_background(image, prompt, negative_prompt=None):
    """
    Change the background of an input image using Stable Diffusion inpainting.

    Args:
        input_image_path (str): Path to the input image
        prompt (str): Desired background description
        negative_prompt (str, optional): Description of what to avoid in the background

    Returns:
        Image: Modified image with new background
    """
    # Load the Stable Diffusion Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    # Convert image to RGB if it's not already
    image = image.convert("RGB")

    # Create a mask to identify the background
    # This is a simple method and might need refinement for complex images
    def create_background_mask(image):
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Apply threshold to create a binary mask
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Optional: Apply some morphological operations to refine the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(thresh, kernel, iterations=2)

        return Image.fromarray(mask)

    # Generate the mask
    mask = create_background_mask(image)

    # Set default negative prompt if not provided
    if negative_prompt is None:
        negative_prompt = "low quality, blurry, unclear, distorted"

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


# Example usage
def main():
    # Path to your input image
    image = load_image(
        "https://storage.googleapis.com/crescis-testing/128464536273.png"
    )

    # Desired background description
    background_prompt = "sunny forest with green trees and soft sunlight"

    # Change the background
    result_image = change_image_background(image, background_prompt)

    # Save the result
    upload_image(result_image,"sd-flux")


if __name__ == "__main__":
    main()