# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# This notebook can be used to train the convolutional autoencoder from the fashion-similarities package on a GPU-enable google colab notebook. The .whl of the package has to be placed on google drive and the path to it must be specified. 

# + colab={"base_uri": "https://localhost:8080/"} id="pAw6rkv64evG" executionInfo={"status": "ok", "timestamp": 1612288592345, "user_tz": -60, "elapsed": 23840, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="db43cab4-b956-434a-8abe-bf2196e223d7"
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# + id="VEWuxX-948DB"
PKG_PATH = "/content/gdrive/MyDrive/colab_notebooks/packages"

# + colab={"base_uri": "https://localhost:8080/"} id="mPEz2OBZ4_mR" executionInfo={"status": "ok", "timestamp": 1612288656006, "user_tz": -60, "elapsed": 5228, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="b67be79b-ddb2-42b0-9210-6a12c51c5246" magic_args="-s $PKG_PATH" language="bash"
# cd $1
# pip install FashionSimilarities-0.5.0-py3-none-any.whl

# + id="RdGQ9B2l5Szv"
# standard modules
import os
from PIL import Imaspec
import subprocess

# other data science modules
import numpy as np

# fashion similarities package
from fashion_similarities.lsh import LSH
from fashion_similarities.utils import GetSimpleData

# + id="PLMOr6eN4yFp"
ZIP_PATH = "/content/gdrive/MyDrive/colab_notebooks/deep_learning/data/preprocessed_images.zip"
DIR="/content/img/"

# + colab={"base_uri": "https://localhost:8080/"} id="wgD7BoPC4ywo" executionInfo={"status": "ok", "timestamp": 1612288679773, "user_tz": -60, "elapsed": 8794, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="3d356e22-c9be-43cc-a1e0-bae0724abae5"
# ! echo "unzip images from ${ZIP_PATH} to ${DIR}"; \
# if ! [ -d "$DIR" ]; \
# then ( \
#     cp $ZIP_PATH . ; \
#     mkdir $DIR ; \
#     unzip -q preprocessed_images.zip -d $DIR ; \
#     rm preprocessed_images.zip ; \
# ) ; \
# else ( \
#     echo "folder already exists. Skipping execution" ; \
# ) ; \
# fi

# + id="94bwJdKK44z7"
images = GetSimpleData.list_files(DIR)

# + id="Uz7uoma45MXp" language="bash"
#
# DIR=/content/img_train/
# if ! [ -d "$DIR" ]; then
#     mkdir img_train
# fi
#
# DIR=/content/img_test/
# if ! [ -d "$DIR" ]; then
#     mkdir img_test
# fi
#
# DIR=/content/img_validation/
# if ! [ -d "$DIR" ]; then
#     mkdir img_validation
# fi

# + colab={"base_uri": "https://localhost:8080/", "height": 604} id="Q2m2E3cl5k2O" executionInfo={"status": "error", "timestamp": 1612289026089, "user_tz": -60, "elapsed": 209714, "user": {"displayName": "Laurin Luttmann", "photoUrl": "", "userId": "02947349670055126044"}} outputId="e19a06f6-3713-4121-e747-b2945bb7c057"
np.random.seed(12345)
test_frac = .2
for i, _file in enumerate(images[:]):
    filename = "_".join(_file.split("/")[1:])
    if i % 1000 == 0:
        print(f"Processing the {i}th image")
    if not (
        os.path.exists(f'img_test/{filename}') or 
        os.path.exists(f'img_train/{filename}') or 
        os.path.exists(f'img_validation/{filename}')
        ):
        if np.random.random() < test_frac:
            if np.random.random() < .5:
                subprocess.call(f"cp {_file} ./img_test/{filename}", shell=True)
            else:
                subprocess.call(f"cp {_file} ./img_validation/{filename}", shell=True)
        else:
            subprocess.call(f"cp {_file} ./img_train/{filename}", shell=True)

# + id="reIRD7gN5mNf"

