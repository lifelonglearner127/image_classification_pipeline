import json
import os

import cv2
import h5py
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import BaseLogger
from tensorflow.keras.utils import to_categorical


class CustomLog:
    @classmethod
    def print_log(cls, message, level="INFO"):
        print(f"[{level}]: {message}")

    @classmethod
    def print_error_log(cls, message, step, level="ERROR"):
        print(f"[{step}] - [{level}]: {message}")

    @classmethod
    def print_info_log(cls, message, step, level="INFO"):
        print(f"[{step}] - [{level}]: {message}")


"""
HDF5 is binary data format created by the HDF5 group to store gigantic numerical datasets on
disk (far too large to store in memory) while facilitating easy access and computation on the rows of
the datasets. Data in HDF5 is stored hierarchically, similar to how a file system stores data. Data is
first defined in groups, where a group is a container-like structure which can hold datasets and other
groups. Once a group has been defined, a dataset can be created within the group. A dataset can be
thought of as a multi-dimensional array (i.e., a NumPy array) of a homogeneous data type (integer,
float, unicode, etc.)
HDF5 is written in C; however, by using the h5py module (h5py.org), we can gain access to
the underlying C API using the Python programming language. What makes h5py so awesome
is the ease of interaction with data. We can store huge amounts of data in our HDF5 dataset and
manipulate the data in a NumPy-like fashion. For example, we can use standard Python syntax to
access and slice rows from multi-terabyte datasets stored on disk as if they were simple NumPy
arrays loaded into memory. Thanks to specialized data structures, these slices and row accesses are
lighting quick. When using HDF5 with h5py, you can think of your data as a gigantic NumPy array
that is too large to fit into main memory but can still be accessed and manipulated just the same.
Perhaps best of all, the HDF5 format is standardized, meaning that datasets stored in HDF5
format are inherently portable and can be accessed by other developers using different programming
languages such as C, MATLAB, and Java.

This will allow us to generate HDF5 datasets from raw images to facilitate faster training.
"""
class HDF5DatasetWriter:
    def __init__(self, dims, output_path, force=False, data_key="images", buf_size=1000):
        """HDF5DatasetWriter Constructor
        Key Arguments:
        dims: dimension or shape of the data we will be storing in the dataset.
        output_path: path to where our output HDF5 file will be stored
        data_key: name of the dataset
        buf_size: size of our in-memory buffer
        """
        if os.path.exists(output_path) and force:
            os.remove(output_path)

        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0], ), dtype="int")
        self.buf_size = buf_size
        self.buffer = {
            "data": [],
            "labels": []
        }
        self.idx = 0

    def add(self, rows, labels):
        # We leaverage buffer/flush workflow, which means we store the data in buffer(memory)
        # once the bufer is full, we write the buffer to file data
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.buf_size:
            self.flush()

    def flush(self):
        # write the data to hdf5 file
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {
            "data": [],
            "labels": []
        }

    def store_class_labels(self, class_labels):
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset(
            "label_names", (len(class_labels), ), dtype=dt
        )
        label_set[:] = class_labels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(db_path, "r")
        self.num_images = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.num_images, self.batch_size):
                images = self.db["images"][i:i + self.batch_size]
                labels = self.db["labels"][i:i + self.batch_size]

                if self.binarize:
                    labels = to_categorical(labels, self.classes)

                if self.preprocessors is not None:
                    proc_images = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        proc_images.append(image)

                    images = np.array(proc_images)

                if self.aug is not None:
                    (images, labels) = next(
                        self.aug.flow(images, labels,
                                      batch_size=self.batch_size)
                    )

                yield (images, labels)

            epochs += 1

    def close(self):
        self.db.close()


class AspectAwarePreProcessor:
    """
    For basic benchmark datasets, resizing images without maintaining aspect so is acceptable.
    However, for more challenging datasets we should still seek to resize to a fixed size, but
    maintain the aspect ratio. This is a common technique
    """
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        d_w = 0
        d_h = 0

        if w < h:
            image = imutils.resize(image, width=self.width,
                                   inter=self.interpolation)
            d_h = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height,
                                   inter=self.interpolation)
            d_w = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[d_h:h - d_h, d_w:w - d_w]

        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.interpolation)


class MeanPreProcessor:
    def __init__(self, r_mean, g_mean, b_mean):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean

        return cv2.merge([B, G, R])


class ImageToArrayPreProcessor:
    def __init__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)


class PatchPreProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        return extract_patches_2d(
            image, (self.height, self.width), max_patches=1
        )[0]


class ResizePreProcessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        return cv2.resize(
            image, (self.width, self.height), interpolation=self.interpolation
        )


class TrainingMonitor(BaseLogger):

    def __init__(self, fig_path, json_path=None, start_at=0):
        super().__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at

    def on_train_begin(self, logs={}):
        self.H = {}

        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                if self.start_at > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():
            log = self.H.get(k, [])
            log.append(float(v))
            self.H[k] = log

        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                len(self.H["loss"]))
            )
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self.fig_path)
            plt.close()
