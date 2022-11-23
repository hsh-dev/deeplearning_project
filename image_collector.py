import os
from PIL import Image

DATASET_PATHS = ['./cartoon_face']
SAVE_PATH = './cartoon_resized'

if __name__ == "__main__":

    file_paths = []
    for dataset_path in DATASET_PATHS:
        file_list = os.listdir(dataset_path)

        for file_ in file_list:
            if "jpg" in file_ or "png" in file_ or "jpeg" in file_ or "JPG" in file_:
                file_paths.append(os.path.join(dataset_path, file_))

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    idx = 0
    for file_path in file_paths:
        image = Image.open(file_path)
        image = image.resize(( 128, 128 ))
        save_path_str = os.path.join(SAVE_PATH, "cartoon_" + str(idx) + ".png")
        image.save(save_path_str, "png")
        idx += 1