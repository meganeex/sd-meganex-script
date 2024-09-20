import os
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import re

class WatermarkRemover:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.pipe = None

    def parse_position(self, position_str, size):
        pattern = r'^(\d*\.?\d+)(?:px)?-(\d*\.?\d+)(?:px)?$'
        match = re.match(pattern, position_str)
        if not match:
            raise ValueError(f"Invalid position format: {position_str}")
        start, end = match.groups()
        
        def parse_value(value, size):
            if value.endswith('px'):
                return int(float(value[:-2]))
            elif '.' in value:
                return int(float(value) * size)
            else:
                return int(value)
        
        start_val = parse_value(start, size)
        end_val = parse_value(end, size)
        
        if start_val < 0 or start_val >= size or end_val <= start_val or end_val > size:
            raise ValueError(f"Invalid position values: {position_str}")
        
        return start_val, end_val

    def create_mask(self, start_x, start_y, width, height, img_width, img_height, invert=False):
        mask = Image.new('L', (img_width, img_height), 0)
        mask_draw = Image.new('L', (width, height), 255)
        mask.paste(mask_draw, (start_x, start_y))
        if invert:
            mask = Image.eval(mask, lambda x: 255 - x)
        return mask

    def remove_watermark(self, image, watermark_x, watermark_y, invert_mask, model_choice):
        if self.pipe is None:
            if not model_choice:
                raise ValueError("Please select an inpainting model before removing watermark.")
            model_path = os.path.join(self.model_dir, model_choice)
            self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
            ).to("cuda")

        width, height = image.size
        x_start, x_end = self.parse_position(watermark_x, width)
        y_start, y_end = self.parse_position(watermark_y, height)
        mask_width = x_end - x_start
        mask_height = y_end - y_start
        mask_image = self.create_mask(x_start, y_start, mask_width, mask_height, width, height, invert_mask)

        prompt = "clean image without watermark, high quality, detailed"
        negative_prompt = "watermark, text, logo, poor quality, blurry"

        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.enable_attention_slicing(1)

        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=height,
            width=width,
        ).images[0]

        return result