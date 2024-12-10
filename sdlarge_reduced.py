from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
from lib.upload import upload_image


def main():
    model_id = "stabilityai/stable-diffusion-3.5-large"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()

    prompt = ("Two durable, high-performance hiking boots displayed "
                  "on a rocky outdoor surface. The boots feature a rugged leather "
                  "and mesh design with reinforced toe caps, sturdy ankle support, "
                  "and aggressive tread patterns for superior grip. "
                  "The laces are tightly secured, and the boots are designed for "
                  "comfort and stability on rugged terrain. The background includes a "
                  "breathtaking mountain landscape with soft, natural lighting that "
                  "highlights the boots' detailed craftsmanship and rugged durability, "
                  "evoking a sense of adventure and exploration")

    image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=4.5,
        max_sequence_length=512,
    ).images[0]
    upload_image(image,"sd-flux")

if __name__ == "__main__":
    main()
