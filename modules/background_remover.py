import numpy as np
import cv2
import onnxruntime as rt
from huggingface_hub import hf_hub_download
from PIL import Image

class BackgroundRemover:
    def __init__(self):
        self.rmbg_model = None

    def load_model(self):
        if self.rmbg_model is None:
            model_path = hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.rmbg_model = rt.InferenceSession(model_path, providers=providers)

    def get_mask(self, img, s=1024):
        img = img.astype(np.float32) / 255.0
        h, w = h0, w0 = img.shape[:2]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        mask = self.rmbg_model.run(None, {'img': img_input})[0][0]
        mask = np.transpose(mask, (1, 2, 0))
        mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
        return mask

    def remove_background(self, img, bg_mode, bg_color=None):
        self.load_model()
        img_array = np.array(img)
        mask = self.get_mask(img_array)
        img_rgba = np.concatenate([img_array, np.full((img_array.shape[0], img_array.shape[1], 1), 255, dtype=np.uint8)], axis=-1)
        
        if bg_mode == "Transparent":
            img_rgba[:, :, 3] = (mask[:, :, 0] * 255).astype(np.uint8)
        else:
            r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
            img_rgba[mask[:, :, 0] < 0.5] = [r, g, b, 255]
        
        return Image.fromarray(img_rgba)