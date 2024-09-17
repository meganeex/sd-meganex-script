import modules.scripts as scripts
import gradio as gr
from modules.processing import process_images, Processed
from PIL import Image
import os
from datetime import datetime
import torch
import re
from diffusers import StableDiffusionInpaintPipeline

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.model_dir = os.path.join(scripts.basedir(), "models", "Stable-diffusion")
        self.model_list = self.get_model_list()
        self.pipe = None

    def get_model_list(self):
        return [f for f in os.listdir(self.model_dir) if f.endswith('.safetensors')]

    def title(self):
        return "Meganex SD Helper"

    def ui(self, is_img2img):
        with gr.Row():
            enable_resize = gr.Checkbox(label="Enable Resize", value=False)
            resize_factor = gr.Slider(minimum=0.25, maximum=4.0, step=0.25, label="Resize Factor", value=1.0)
        with gr.Row():
            enable_watermark_removal = gr.Checkbox(label="Enable Watermark Removal", value=False)
            watermark_x = gr.Textbox(label="Watermark X (start-end)", value="0.8-1.0")
            watermark_y = gr.Textbox(label="Watermark Y (start-end)", value="0.8-1.0")
        with gr.Row():
            model_choice = gr.Dropdown(label="Inpainting Model", choices=[""] + self.model_list, value="")
            invert_mask = gr.Checkbox(label="Invert Mask", value=False)
        return [enable_resize, resize_factor, enable_watermark_removal, watermark_x, watermark_y, model_choice, invert_mask]

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

    def remove_watermark(self, image, watermark_x, watermark_y, invert_mask):
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

    def run(self, p, enable_resize, resize_factor=1.0, enable_watermark_removal=False, watermark_x="0.8-1.0", watermark_y="0.8-1.0", model_choice='', invert_mask=False):
        if enable_watermark_removal:
            if not model_choice:
                raise gr.Error("Please select an inpainting model before removing watermark.")
            if self.pipe is None:
                model_path = os.path.join(self.model_dir, model_choice)
                self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                ).to("cuda")

        proc = process_images(p)
        
        current_time = datetime.now()
        date_str = current_time.strftime("%Y%m%d")
        time_str = current_time.strftime("%H%M%S")
                
        for i, image in enumerate(proc.images):
            img = image.copy()
            
            if enable_watermark_removal:
                img = self.remove_watermark(img, watermark_x, watermark_y, invert_mask)
            
            if enable_resize:
                width, height = img.size
                new_width = int(width * resize_factor)
                new_height = int(height * resize_factor)
                resample_filter = Image.LANCZOS if resize_factor < 1 else Image.BICUBIC
                img = img.resize((new_width, new_height), resample_filter)
            
            action = []
            if enable_resize:
                action.append(f"resized_{resize_factor:.2f}x")
            if enable_watermark_removal:
                action.append("nowm")
            action_str = "_".join(action) if action else "original"
            
            if enable_resize or enable_watermark_removal:
                processed_dir = os.path.join(p.outpath_samples, f'processed_images_{date_str}_{time_str}')
                os.makedirs(processed_dir, exist_ok=True)
                filename = f"{action_str}_{str(p.seed)}_{i:05}.png"
                save_path = os.path.join(processed_dir, filename)
                img.save(save_path)
                print(f"Processed image {i+1}/{len(proc.images)}: saved to {save_path}")
        
        if enable_resize or enable_watermark_removal:
            gr.Info(f"Processing completed. Saved :\n{processed_dir}")
        
        return Processed(p=p, images_list=proc.images)
