from PIL import Image
import numpy as np
import cv2
import onnxruntime as rt
from huggingface_hub import hf_hub_download
import os

# Initialize ONNX Runtime session for background removal
model_path = hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
rmbg_model = rt.InferenceSession(model_path, providers=providers)

def get_mask(img, s=1024):
    img = img.astype(np.float32) / 255.0
    h, w = h0, w0 = img.shape[:2]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        print(f"Error: Unable to read the image file: {image_path}")
        print(f"Exception: {str(e)}")
        return None

def remove_background(img, background_color=None):
    # Get mask
    mask = get_mask(img)
    
    # Apply mask
    img_rgba = np.concatenate([img, np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)], axis=-1)
    
    if background_color is None:
        # If no background color is specified, make it transparent
        img_rgba[:, :, 3] = (mask[:, :, 0] * 255).astype(np.uint8)
    else:
        # If background color is specified, replace background with that color
        r, g, b = background_color
        img_rgba[mask[:, :, 0] < 0.5] = [r, g, b, 255]  # Set specified color where mask is < 0.5
    
    return Image.fromarray(img_rgba)

if __name__ == "__main__":

    # 使用例
    input_image = "D:\\生成AIムービー＆音楽\\0015.Brooklyn Bookstore\\iamges\\00029-628369987.png"
    output_image = "00029-628369987_transparent.png"

    if not os.path.exists(input_image):
        print(f"Error: Input image file not found: {input_image}")
    else:
        print(f"Processing image: {input_image}")
        print(f"Current working directory: {os.getcwd()}")
        img = load_image(input_image)

        # 背景透過の場合
        removed = remove_background(img, background_color=None)
        removed.save(output_image)
        
        # 背景を白に塗りつぶす場合（例: 白 (255, 255, 255)）
        output_image_with_white_bg = "00029-628369987_white_bg.png"
        removed_with_bg = remove_background(img, background_color=(255, 255, 255))
        removed_with_bg.save(output_image_with_white_bg)

        print(f"Processed images saved to {output_image} and {output_image_with_white_bg}")

    print("Script execution completed.")
