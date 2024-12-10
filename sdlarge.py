import torch
from diffusers import StableDiffusion3Pipeline

from lib.upload import upload_multiple_images

def main():
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                    torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    prompt = ("Two durable, high-performance hiking boots displayed "
              "on a rocky outdoor surface. The boots feature a rugged leather "
              "and mesh design with reinforced toe caps, sturdy ankle support, "
              "and aggressive tread patterns for superior grip. "
              "The laces are tightly secured, and the boots are designed for "
              "comfort and stability on rugged terrain. The background includes a "
              "breathtaking mountain landscape with soft, natural lighting that "
              "highlights the boots' detailed craftsmanship and rugged durability, "
              "evoking a sense of adventure and exploration")

    images = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        num_images_per_prompt=2
    ).images

    bucket_name = "sd-flux"
    return upload_multiple_images(images, bucket_name)

if __name__ == "__main__":
    main()