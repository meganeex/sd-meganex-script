import launch

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "diffusers")

if not launch.is_installed("PIL"):
    launch.run_pip("install Pillow", "Pillow")
