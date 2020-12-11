"""
script with utility functions which are used again and again
"""
import os
import gzip
import numpy as np
import urllib.request
import io
import sqlite3

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
    def __init__(self, path=None):
        self.path = path

    def load_mnist(self, kind='train'):
        if kind == "test":
            kind = "t10k"
        """Load MNIST data from github"""
        if not self.path:
            images, labels = self.load_from_github(kind)
        else:
            images, labels = self._loaf_from_path(kind)

        return images, labels

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

    def _loaf_from_path(self, kind):
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


if __name__ == "__main__":
    data_getter = GetSimpleData()
    x_train, y_train = data_getter.load_mnist()
    x_test, y_test = data_getter.load_mnist(kind="test")

