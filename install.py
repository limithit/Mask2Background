import launch

if not launch.is_installed("numpy"):
    try:
        launch.run_pip("install numpy", "requirements for numpy")
    except:
        print("Can't install numpy. Please follow the readme to install manually")

if not launch.is_installed("cv2"):
    try:
        launch.run_pip("install opencv-python", "requirements for cv2")
    except:
        print("Can't install opencv-python. Please follow the readme to install manually")

if not launch.is_installed("PIL"):
    try:
        launch.run_pip("install Pillow", "requirements for PIL")
    except:
        print("Can't install Pillow. Please follow the readme to install manually")


if not launch.is_installed("lama_cleaner"):
    try:
        launch.run_pip("install lama-cleaner", "requirements for lama_cleaner")
    except:
        print("Can't install lama-cleaner. Please follow the readme to install manually")

if not launch.is_installed("ultralytics"):
    try:
        launch.run_pip("install ultralytics", "requirements for ultralytics")
    except:
        print("Can't install ultralytics. Please follow the readme to install manually")

if not launch.is_installed("packaging"):
    try:
        launch.run_pip("install packaging", "requirements for packaging")
    except:
        print("Can't install packaging. Please follow the readme to install manually")
