import os
from enum import Enum

import cv2
import numpy as np
import progressbar
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import config
import errors
import utils


class PipeLineStep(Enum):
    CONFIG_SETTINGS = 0
    PRE_PROCESS = 1
    LOADING_PRE_TRAINED_NETWORK = 2
    BUILDING_DATASET = 3
    TRANING_DATASET = 4
    EVALUATE = 5


def pre_process():
    """Removing Image Duplication or something others"""
    pass


def load_network(network):
    utils.CustomLog.print_info_log(
        message=f"Loading {network_name} network...",
        step=PipeLineStep.LOADING_PRE_TRAINED_NETWORK.name
    )
    try:
        network_config = config.BASE_NETWORKS_MAPPING[network]
    except KeyError:
        raise errors.NetworkNotSupported
    else:
        utils.CustomLog.print_info_log(
            message=f"Successfully loaded {network_name} network...",
            step=PipeLineStep.LOADING_PRE_TRAINED_NETWORK.name
        )
        return network_config["backbone"], network_config["image_size"]


def build_dataset(image_size):
    utils.CustomLog.print_info_log(
        message=f"Building dataset...",
        step=PipeLineStep.BUILDING_DATASET.name
    )

    train_paths = list(paths.list_images(config.TRAINING_IMAGES_PATH))
    train_labels = [p.split(os.path.sep)[-2].split(".")[0] for p in train_paths]
    total_count = len(train_paths)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)

    split = train_test_split(train_paths, train_labels,
                             test_size=int(total_count * 0.1),
                             stratify=train_labels, random_state=42)
    (train_paths, test_paths, train_labels, test_labels) = split

    split = train_test_split(train_paths, train_labels,
                             test_size=int(total_count * 0.1),
                             stratify=train_labels, random_state=42)
    (train_paths, val_paths, train_labels, val_labels) = split

    aap = utils.AspectAwarePreProcessor(image_size, image_size)
    datasets = [
        ("train", train_paths, train_labels, config.TRAIN_HDF5),
        ("val", val_paths, val_labels, config.VAL_HDF5),
        ("test", test_paths, test_labels, config.TEST_HDF5)
    ]
    g_r = g_g = g_b = 0
    for (dataset_type, image_paths, labels, output_path) in datasets:
        utils.CustomLog.print_info_log(
            message=f"Building {output_path}...",
            step=PipeLineStep.BUILDING_DATASET.name
        )

        force = False
        if os.path.exists(output_path):
            decision = input(
                f"You already generated hdf5 file for {dataset_type} data. "
                "Press Y if you want to build again or "
                "If you want to use existing data press any key: "
            )
            if decision.lower() != "y":
                continue
            force = True

        writer = utils.HDF5DatasetWriter(
            dims=(len(image_paths), image_size, image_size, 3),
            output_path=output_path,
            force=force,
        )

        widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
                   progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets)
        pbar.start()

        for (i, (image_path, label)) in enumerate(zip(image_paths, labels)):
            image = cv2.imread(image_path)
            image = aap.preprocess(image)

            if dataset_type == "train":
                (b, g, r) = cv2.mean(image)[:3]
                g_r += r
                g_g += g
                g_b += b

            writer.add([image], [label])
            pbar.update(i)

        pbar.finish()
        writer.close()

    utils.CustomLog.print_info_log(
        message=f"Successfully Finished building dataset...",
        step=PipeLineStep.BUILDING_DATASET.name
    )

    g_r = int(g_r / total_count)
    g_g = int(g_g / total_count)
    g_b = int(g_b / total_count)
    utils.CustomLog.print_info_log(
        message=f"Mean (R, G, B) = ({g_r}, {g_g}, {g_b})...",
        step=PipeLineStep.BUILDING_DATASET.name
    )
    return {"R": g_r, "G": g_g, "B": g_b}, le.classes_


def train(network_class, image_size, means):
    utils.CustomLog.print_info_log(
        message=f"Start training...",
        step=PipeLineStep.TRANING_DATASET.name,
    )
    mp = utils.MeanPreProcessor(means["R"], means["G"], means["B"])
    iap = utils.ImageToArrayPreProcessor()
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    train_generator = utils.HDF5DatasetGenerator(
        db_path=config.TRAIN_HDF5,
        batch_size=config.BATCH_SIZE,
        preprocessors=[mp, iap],
        aug=aug,
        classes=config.CLASS_NUMS,
    )
    val_generator = utils.HDF5DatasetGenerator(
        db_path=config.VAL_HDF5,
        batch_size=config.BATCH_SIZE,
        preprocessors=[mp, iap],
        classes=config.CLASS_NUMS,
    )

    base_model = network_class(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3)
    )

    head_model = base_model.output
    head_model = Dense(256, activation="relu")(head_model)
    head_model = GlobalAveragePooling2D()(head_model)
    head_model = Dropout(0.2)(head_model)
    predictions = Dense(config.CLASS_NUMS, activation='softmax')(head_model)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = True

    opt = Adam(lr=1e-3)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
    callbacks = [utils.TrainingMonitor(path)]

    model.fit(
        train_generator.generator(),
        steps_per_epoch=train_generator.num_images // config.BATCH_SIZE,
        validation_data=val_generator.generator(),
        validation_steps=val_generator.num_images // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        max_queue_size=0,
        callbacks=callbacks,
        verbose=1
    )
    utils.CustomLog.print_info_log(
        message=f"Finished training...",
        step=PipeLineStep.TRANING_DATASET.name,
    )
    train_generator.close()
    val_generator.close()
    model.save(config.MODEL_PATH, overwrite=True)
    return model


def evaluate(model, image_size, means, classes):
    rp = utils.ResizePreProcessor(image_size, image_size)
    mp = utils.MeanPreProcessor(means["R"], means["G"], means["B"])
    iap = utils.ImageToArrayPreProcessor()
    utils.CustomLog.print_info_log(
        message=f"Predicting on test data ...",
        step=PipeLineStep.EVALUATE.name,
    )
    test_generator = utils.HDF5DatasetGenerator(
        db_path=config.TEST_HDF5,
        batch_size=config.BATCH_SIZE,
        preprocessors=[rp, mp, iap],
        classes=config.CLASS_NUMS,
    )
    predictions = model.predict(
        test_generator.generator(),
        steps=test_generator.num_images // config.BATCH_SIZE,
        max_queue_size=10
    )

    for (pred, label) in zip(predictions, test_generator.db["labels"]):
        pred = np.argsort(pred)[::-1]
        utils.CustomLog.print_info_log(
            message=f"Original class is {classes[label]}, "
                    f"Predicted as {classes[pred[0]]}...",
            step=PipeLineStep.EVALUATE.name,
        )

    test_generator.close()


def predict(model, image_size, means, classes):
    rp = utils.ResizePreProcessor(image_size, image_size)
    mp = utils.MeanPreProcessor(means["R"], means["G"], means["B"])
    iap = utils.ImageToArrayPreProcessor()
    images_path = list(paths.list_images(config.TEST_IMAGES_PATH))
    for image_path in images_path:
        origin_image = cv2.imread(image_path)
        image = origin_image.copy()
        for p in (rp, mp, iap):
            image = p.preprocess(image)

        image = np.expand_dims(image, axis=0)
        preds = model.predict(image)[0]
        pred = np.argsort(preds)[::-1]
        label = classes[pred[0]]
        cv2.putText(origin_image, label, (0, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("aaa", origin_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    network_name = config.NETWORK
    try:
        pre_process()

        network, image_size = load_network(network_name)
        means, classes = build_dataset(image_size)
        model = train(network, image_size, means)
        evaluate(model, image_size, means, classes)
        predict(model, image_size, means, classes)

    except errors.NetworkNotSupported as e:
        utils.CustomLog.print_error_log(
            message=f"{network_name} is not supported",
            step=PipeLineStep.CONFIG_SETTINGS.name,
        )
    except errors.NetworkImproperlyConfigured as e:
        utils.CustomLog.print_error_log(
            message=f"{network_name} is improperly configured",
            step=PipeLineStep.CONFIG_SETTINGS.name,
        )
    except Exception as e:
        print(e)

