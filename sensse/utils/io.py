"""
Input / Output utilities.
"""

import os
import json
import numpy as np
import tensorflow as tf

def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )

    return path

def save_numpy(
    array,
    path
):

    np.save(
        path,
        array
    )


def load_numpy(
    path
):

    return np.load(
        path,
        allow_pickle=True
    )

def save_json(
    data,
    path
):

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )


def load_json(path):

    with open(path, "r") as f:

        return json.load(f)

def save_history(
    history,
    path
):

    if hasattr(history, "history"):
        history = history.history

    save_json(
        history,
        path
    )

def save_weights(
    model,
    path
):

    model.save_weights(path)


def load_weights(
    model,
    path
):

    model.load_weights(path)

    return model

def save_model(
    model,
    path
):

    model.save(path)


def load_model(path):

    return tf.keras.models.load_model(
        path,
        compile=False
    )
