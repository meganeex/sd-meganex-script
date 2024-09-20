import modules.scripts as scripts
import gradio as gr
from modules.processing import process_images, Processed
from PIL import Image
import os
from datetime import datetime
from .modules.watermark_remover import WatermarkRemover
from .modules.background_remover import BackgroundRemover

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.model_dir = os.path.join(scripts.basedir(), "models", "Stable-diffusion")
        self.model_list = self.get_model_list()
        self.watermark_remover = WatermarkRemover(self.model_dir)
        self.background_remover = BackgroundRemover()

    def get_model_list(self):
        return [f for f in os.listdir(self.model_dir) if f.endswith('.safetensors')]

    def title(self):
        return "Meganex SD Helper with Background Removal"

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
        with gr.Row():
            enable_bg_removal = gr.Checkbox(label="Enable Background Removal", value=False)
            bg_mode = gr.Radio(["Transparent", "Color"], label="Background Mode", value="Transparent")
            bg_color = gr.ColorPicker(label="Background Color", visible=False)

        def update_bg_color_visibility(bg_mode):
            return gr.update(visible=bg_mode == "Color")

        bg_mode.change(fn=update_bg_color_visibility, inputs=[bg_mode], outputs=[bg_color])

        return [enable_resize, resize_factor, enable_watermark_removal, watermark_x, watermark_y, 
                model_choice, invert_mask, enable_bg_removal, bg_mode, bg_color]

    def run(self, p, enable_resize, resize_factor=1.0, enable_watermark_removal=False, 
            watermark_x="0.8-1.0", watermark_y="0.8-1.0", model_choice='', invert_mask=False,
            enable_bg_removal=False, bg_mode="Transparent", bg_color=None):
        
        proc = process_images(p)
        
        current_time = datetime.now()
        date_str = current_time.strftime("%Y%m%d")
        time_str = current_time.strftime("%H%M%S")
                
        for i, image in enumerate(proc.images):
            img = image.copy()
            
            if enable_watermark_removal:
                img = self.watermark_remover.remove_watermark(img, watermark_x, watermark_y, invert_mask, model_choice)
            
            if enable_resize:
                width, height = img.size
                new_width = int(width * resize_factor)
                new_height = int(height * resize_factor)
                resample_filter = Image.LANCZOS if resize_factor < 1 else Image.BICUBIC
                img = img.resize((new_width, new_height), resample_filter)
            
            if enable_bg_removal:
                img = self.background_remover.remove_background(img, bg_mode, bg_color)
            
            action = []
            if enable_resize:
                action.append(f"resized_{resize_factor:.2f}x")
            if enable_watermark_removal:
                action.append("nowm")
            if enable_bg_removal:
                action.append("bgremoved")
            action_str = "_".join(action) if action else "original"
            
            if enable_resize or enable_watermark_removal or enable_bg_removal:
                processed_dir = os.path.join(p.outpath_samples, f'processed_images_{date_str}_{time_str}')
                os.makedirs(processed_dir, exist_ok=True)
                filename = f"{action_str}_{str(p.seed)}_{i:05}.png"
                save_path = os.path.join(processed_dir, filename)
                img.save(save_path)
                print(f"Processed image {i+1}/{len(proc.images)}: saved to {save_path}")
        
        if enable_resize or enable_watermark_removal or enable_bg_removal:
            gr.Info(f"Processing completed. Saved to:\n{processed_dir}")
        
        return Processed(p=p, images_list=proc.images)