import torch
from diffusers import FluxPipeline

from lib.upload import upload_multiple_images


def main():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    prompt = ("A luxurious sofa set placed in a spacious living room with "
              "large windows offering a serene view of a tranquil lake. "
              "The sofa features plush cushions in a neutral tone, with elegant "
              "throw pillows in soft fabrics adding a touch of color. "
              "The furniture is arranged around a stylish coffee table, with warm, "
              "ambient lighting casting a soft glow across the room. "
              "The background showcases the lake's calm waters, "
              "framed by lush greenery, and distant mountains, creating a peaceful "
              "and sophisticated atmosphere. "
              "The scene emphasizes relaxation, comfort, and a seamless connection "
              "to nature.")

    images = pipe(
        prompt,
        height=1024,
        width=1024,
        num_images_per_prompt=2,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images

    bucket_name = "sd-flux"
    return upload_multiple_images(images, bucket_name)

if __name__ == "__main__":
    main()

