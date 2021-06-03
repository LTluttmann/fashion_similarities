from fashion_similarities.utils import GetSimpleData, Preprocessing
from argparse import ArgumentParser
from keras.preprocessing.image import load_img
import os

argparser = ArgumentParser()
argparser.add_argument('-p', '--path', type=str, required=True)

args = argparser.parse_args()

if __name__ == "__main__":
    path_to_img = args.path
    new_path = "/".join(path_to_img.replace('\\', '/').split("/")[:-1]) + "/preprocessed_images/"
    os.makedirs(new_path)
    images = GetSimpleData.list_files(path_to_img)
    for i, _file in enumerate(images):
        img = load_img(_file)  # this is a PIL image
        img = Preprocessing.resize_img_with_borders(img, (224, 224))
        img.save(new_path + _file.split("\\")[-1])
        if i % 5000 == 0:
            print(f"Done preprocessing {i} images")