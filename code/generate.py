import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition_pair", type=str, required=True, choices=["canny_depth", "openpose_canny", "scribble_depth"])
    parser.add_argument("--weight_ratio", type=str, required=True, help="e.g., 40:60")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    # Load ControlNet models
    if args.condition_pair == "canny_depth":
        controlnet = [
            ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
        ]
    elif args.condition_pair == "openpose_canny":
        controlnet = [
            ControlNetModel.from_pretrained("diffusers/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        ]
    elif args.condition_pair == "scribble_depth":
        controlnet = [
            ControlNetModel.from_pretrained("diffusers/controlnet-scribble-sdxl-1.0", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
        ]

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    # Parse weight ratio
    w1, w2 = map(float, args.weight_ratio.split(":"))
    scale1 = w1 / (w1 + w2)
    scale2 = w2 / (w1 + w2)

    # Load prompts
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    generator = torch.Generator("cuda").manual_seed(42)

    for i, prompt in enumerate(prompts):
        # You need to provide reference condition images here.
        # This is a placeholder; replace with actual reference loading.
        ref_cond1 = Image.new("RGB", (1024, 1024), "white")
        ref_cond2 = Image.new("RGB", (1024, 1024), "white")

        image = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, ugly, bad anatomy",
            image=[ref_cond1, ref_cond2],
            controlnet_conditioning_scale=[scale1, scale2],
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        image.save(f"{args.output_dir}/{args.condition_pair}_{args.weight_ratio}_{i:03d}.png")

if __name__ == "__main__":
    main()
