"""
script with utility functions which are used again and again
"""
import os
import gzip
import numpy as np
import urllib.request
import io
import sqlite3
from PIL import Image, ImageOps
import copy
from keras.preprocessing.image import load_img, img_to_array
import bisect
from shutil import copyfile
import subprocess

URL = "https://raw.github.com/zalandoresearch/fashion-mnist/master/data/fashion/"


class DBConnect:

    @staticmethod
    def adapt_array(arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def __init__(self, file='sqlite.db'):
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convert_array)

        self.file = file

    def __enter__(self):
        self.conn = sqlite3.connect(self.file, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()

    def __exit__(self, type, value, traceback):
        self.conn.commit()
        self.conn.close()


class GetSimpleData:
    @staticmethod
    def load_from_github(kind="train"):
        if kind == "test":
            kind = "t10k"
        response_label = urllib.request.urlopen(URL + '%s-labels-idx1-ubyte.gz' % kind)
        compressed_label_file = io.BytesIO()
        compressed_label_file.write(response_label.read())
        compressed_label_file.seek(0)

        response_images = urllib.request.urlopen(URL + '%s-images-idx3-ubyte.gz' % kind)
        compressed_images_file = io.BytesIO()
        compressed_images_file.write(response_images.read())
        compressed_images_file.seek(0)

        with gzip.GzipFile(fileobj=compressed_label_file, mode='rb') as decompressed_file:
            labels = np.frombuffer(decompressed_file.read(), dtype=np.uint8, offset=8)

        with gzip.GzipFile(fileobj=compressed_images_file, mode='rb') as decompressed_file:
            images = np.frombuffer(decompressed_file.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

    @staticmethod
    def list_files(dir, sort=True):
        r = []
        subdirs = [x[0] for x in os.walk(dir)]
        for subdir in subdirs:
            files = os.walk(subdir).__next__()[2]
            if len(files) > 0:
                for file in files:
                    if sort:
                        bisect.insort(r, os.path.join(subdir, file))
                    else:
                        r.append(os.path.join(subdir, file))
        return r

    @staticmethod
    def load_img_from_path(path, subset_size=None, return_id=False):
        images = GetSimpleData.list_files(path)
        subset_size = subset_size if subset_size else len(images)
        images = images[:subset_size]
        dataset = np.zeros((len(images), *img_to_array(load_img(images[0])).shape), dtype="uint8")
        row_id_dict = dict()
        for i, _file in enumerate(images):
            id = _file.replace('\\', '/').split("/")[-1].split(".")[0]
            img = load_img(_file)
            img = img_to_array(img).astype("uint8")
            dataset[i] = img
            row_id_dict[i] = id
            if i % 10000 == 0:
                print(f"currently at {i}")
        if return_id:
            return dataset, row_id_dict
        else:
            return dataset

    def __init__(self, path=None):
        self.path = path

    def load_mnist(self, kind='train'):
        if kind == "test":
            kind = "t10k"
        """Load MNIST data from github"""
        if not self.path:
            images, labels = self.load_from_github(kind)
        else:
            images, labels = self._load_from_path(kind)

        return images, labels

    def _load_from_path(self, kind):
        labels_path = os.path.join(self.path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(self.path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels


class Preprocessing:
    @staticmethod
    def resize_img_with_borders(img: Image, new_size: tuple) -> Image:
        """
        this function resizes an image to a requested size which must be lower than the current
        size. This function uses thumbnail to preserve the aspect ratio of the image in order not to
        stretch it in any direction. It than fills up the missing pixels with a white border..
        :param img: img of class Pillow image to be resized
        :param new_size: requested size in format (width, height)
        :return: resized img
        """
        def add_borders(img_to_add, new_shape):
            old_shape = img_to_add.width, img_to_add.height
            deltaw = new_shape[0] - old_shape[0]
            deltah = new_shape[1] - old_shape[1]
            brd = (deltaw//2, deltah//2, deltaw-(deltaw//2), deltah-(deltah//2))
            img_exp = ImageOps.expand(img_to_add, brd, fill="white")
            return img_exp

        new_img = copy.deepcopy(img)
        new_img.thumbnail(new_size)
        if new_img.width != new_size[0] or new_img.height != new_size[1] :
            new_img = add_borders(new_img, new_size)
        return new_img

    @staticmethod
    def train_test_split(img_file_path: list, data_dst_path):
        def copy_func(src, dst):
            """
            this function copies data from a source to a destination path and does that by using a function
            that is appropriate for the respective operating system
            :param src: path to file source
            :param dst: path to destination
            :return: executes the copy process
            """
            if os.name == "nt":  # windows case
                copyfile(_file, dst)
            elif os.name == "posix":
                subprocess.call(f"cp {src} {dst}", shell=True)
        np.random.seed(12345)
        test_frac = .2
        for i, _file in enumerate(img_file_path[:]):
            # extract filename
            filename = _file.replace("\\", "/").split("/")[-1]
            if i % 1000 == 0:
                print(f"Processing the {i}th image")
            if not (
                    os.path.exists(os.path.join(data_dst_path, "img_train", filename)) or
                    os.path.exists(os.path.join(data_dst_path, "img_test", filename)) or
                    os.path.exists(os.path.join(data_dst_path, "img_validation", filename))
            ):
                if np.random.random() < test_frac:
                    if np.random.random() < .5:
                        copy_func(src=_file, dst=(os.path.join(data_dst_path, "img_test", filename)))
                    else:
                        copy_func(src=_file, dst=(os.path.join(data_dst_path, "img_validation", filename)))
                else:
                    copy_func(src=_file, dst=(os.path.join(data_dst_path, "img_train", filename)))


if __name__ == "__main__":
    data = GetSimpleData.load_img_from_path("../../../data/archive/images_prep", None, False)
    print(f"Got {len(data)} data points with memory consumption: {data.nbytes // 1000000} MB")
    # data_getter = GetSimpleData()
    # x_train, y_train = data_getter.load_mnist()
    # x_test, y_test = data_getter.load_mnist(kind="test")

