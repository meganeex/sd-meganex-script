import launch

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "diffusers")

if not launch.is_installed("PIL"):
    launch.run_pip("install Pillow", "Pillow")

if not launch.is_installed("onnxruntime"):
    launch.run_pip("install onnxruntime", "onnxruntime")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "opencv-python")
    