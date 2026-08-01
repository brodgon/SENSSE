"""
Ablation:
Effect of Eq. 6 evidential regularization.
"""

import os
import sys

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, ROOT)


import json
import argparse
import numpy as np
import tensorflow as tf

from sensse.utils.config import load_config
from sensse.utils.seed import set_seed

from sensse.datasets.loaders import load_all_slices
from sensse.datasets.generators import get_dataset

from sensse.models.sensse import build_sensse

from sensse.losses.multitask import (
    EvidentialSegmentationLoss
)

from sensse.losses.evidential_regression import (
    EvidentialRegressionLoss
)


def run_ablation(
    train_dataset,
    val_dataset,
    config
):

    results = {}

    os.makedirs(
        config["output_dir"],
        exist_ok=True
    )

    for eta in config[
        "regularization_values"
    ]:

        print("\n" + "=" * 60)
        print(f"Running eta={eta}")
        print("=" * 60)

        model = build_sensse(
            input_shape=(
                config["height"],
                config["width"],
                5
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
            evidential_weight=eta
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

        model_name = (
            f"eta_{str(eta).replace('.','p')}"
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                config["output_dir"],
                f"{model_name}_best.h5"
            ),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss"
        )

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config["epochs"],
            callbacks=[checkpoint],
            verbose=1, 
            steps_per_epoch=config["steps_per_epoch"],
            validation_steps=config["validation_steps"]
            )


        results[str(eta)] = {

            "best_val_loss":
            float(
                np.min(
                    history.history["val_loss"]
                )
            ),

            "final_val_loss":
            float(
                history.history["val_loss"][-1]
            )
        }

    results_path = os.path.join(
        config["output_dir"],
        "ablation_regularization.json"
    )

    with open(
        results_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    set_seed(
        config["seed"]
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

    np.random.shuffle(
        patients
    )

    split = int(
        0.8 * len(patients)
    )

    train_patients = patients[:split]

    val_patients = patients[split:]


    train_dataset = get_dataset(
    config["data_dir"],
    train_patients,
    config["num_classes"],
    batch_size=config["batch_size"]
    )

    val_dataset = get_dataset(
        config["data_dir"],
        val_patients,
        config["num_classes"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_aug=0
    )
    

    train_dataset = get_dataset(
        config["data_dir"],
        train_patients,
        config["num_classes"],
        batch_size=config["batch_size"]
    )

    val_dataset = get_dataset(
        config["data_dir"],
        val_patients,
        config["num_classes"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_aug=0
    )

    run_ablation(
        train_dataset,
        val_dataset,
        config
    )