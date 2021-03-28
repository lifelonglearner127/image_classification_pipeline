from keras.applications import (
    ResNet50,
    ResNet50V2,
    ResNet101,
    ResNet101V2,
    ResNet152,
    ResNet152V2,
)

BASE_NETWORKS_MAPPING = {
    "ResNet50": {
        "backbone": ResNet50,
        "image_size": 224,
    },
    "ResNet50V2": {
        "backbone": ResNet50V2,
        "image_size": 224,
    },
    "ResNet101": {
        "backbone": ResNet101,
        "image_size": 224,
    },
    "ResNet101V2": {
        "backbone": ResNet101V2,
        "image_size": 224,
    },
    "ResNet152": {
        "backbone": ResNet152,
        "image_size": 224,
    },
    "ResNet152V2": {
        "backbone": ResNet152V2,
        "image_size": 224
    },
}
NETWORK = "ResNet50"
TRAINING_IMAGES_PATH = "images/training"
TEST_IMAGES_PATH = "images/test"
TRAIN_HDF5 = "hdf5/train.hdf5"
VAL_HDF5 = "hdf5/val.hdf5"
TEST_HDF5 = "hdf5/test.hdf5"
MODEL_PATH = "output/result.model"
OUTPUT_PATH = "output"
EPOCHS = 1
BATCH_SIZE = 8
CLASS_NUMS = 2
