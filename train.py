"""
train
"""

import os
import numpy as np
import tensorflow as tf

from sensse.utils.seed import set_seed
from sensse.utils.config import load_config
from sensse.utils.logger import get_logger
from sensse.utils.io import ensure_dir, save_history

from sensse.datasets.loaders import load_all_slices
from sensse.datasets.generators import get_dataset

from sensse.models.sensse import build_sensse

from sensse.losses.multitask import (
    EvidentialSegmentationLoss
)

from sensse.losses.evidential_regression import (
    EvidentialRegressionLoss
)


import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    required=True
)

args = parser.parse_args()

config = load_config(
    args.config
)

def main():

    

    set_seed(
        config["seed"]
    )

    logger = get_logger(
        config["output_dir"]
    )

    logger.info(
        "Loading dataset..."
    )

    patients = sorted([
        p
        for p in os.listdir(
            config["data_dir"]
        )
        if os.path.isdir(
            os.path.join(
                config["data_dir"],
                p
            )
        )
    ])

    patients = np.array(
        patients
    )

    np.random.shuffle(
        patients
    )

    split = int(
        config["train_split"]
        *
        len(patients)
    )

    train_patients = patients[:split]

    val_patients = patients[split:]


    train_dataset = get_dataset(
        config["data_dir"],
        train_patients,
        config["num_classes"],
        batch_size=config["batch_size"],
        num_aug=config["num_aug"],
        num_slices=config["num_slices"]
    )

    val_dataset = get_dataset(
        config["data_dir"],
        val_patients,
        config["num_classes"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_aug=0,
        num_slices=config["num_slices"]
    )

    logger.info(
        "Building model..."
    )

    model = build_sensse(
        input_shape=(
            config["height"],
            config["width"],
            config["num_slices"]
        ),
        num_classes=config["num_classes"],
        interaction=config["interaction"]
    )

    seg_loss = EvidentialSegmentationLoss(
        num_classes=config["num_classes"],
        max_kl_weight=config["max_kl_weight"],
        warmup_epochs=config["warmup_epochs"]
    )

    syn_loss = EvidentialRegressionLoss(
        evidential_weight=config[
            "evidential_weight"
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config["lr"]
        ),
        loss=[
            seg_loss,
            syn_loss
        ]
    )

    warmup_callback = (
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs:
            seg_loss.set_epoch(epoch)
        )
    )

    checkpoint_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                config["output_dir"],
                "weights",
                "best.h5"
            ),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss"
        )
    )

    # X, (Yseg, Yct) = next(iter(train_dataset))

    # print("\nTRAIN BATCH")
    # print("X:", X.shape)
    # print("Yseg:", Yseg.shape)
    # print("Yct:", Yct.shape)

    # print("\nMODEL INPUT")
    # print(model.input_shape)

    

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        validation_steps=config["validation_steps"],
        callbacks=[
            warmup_callback,
            checkpoint_callback
        ]
    )
    save_history(
        history,
        os.path.join(
            config["output_dir"],
            "history.json"
        )
    )


if __name__ == "__main__":
    main()
