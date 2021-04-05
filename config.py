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
# Specify the name of network that you can use as a base model
NETWORK = "ResNet50"

# The path of directory containig images that can be used for training
# You can split those into 3 sub datasets called; training dataset, validation dataset, test dataset.
TRAINING_IMAGES_PATH = "images/training"

# The path of directory containig images that can be used for testing the trained model
# You need to understand that this is the different from test dataset
TEST_IMAGES_PATH = "images/test"

# The path of hdf5 file that manage training dataset
TRAIN_HDF5 = "hdf5/train.hdf5"

# The path of hdf5 file that manage validation dataset
VAL_HDF5 = "hdf5/val.hdf5"

# The path of hdf5 file that manage test dataset
TEST_HDF5 = "hdf5/test.hdf5"

# Trained model will be saved to this 
MODEL_PATH = "output/result.model"

# You can set traing curve in this folder
OUTPUT_PATH = "output"

# Epochs 
EPOCHS = 1

# We use SGD. so this specify the number of batch size.
# Please specify this number depending on your p
BATCH_SIZE = 8

# This speicify the numer of classes. I set this to 2 because my demo dataset consist of dog & cat.
# If you want to build the digit recognition, you can specify this to 10.
CLASS_NUMS = 2
