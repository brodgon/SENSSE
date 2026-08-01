"""
Attention gate used in Attention UNet.
"""

import tensorflow as tf


def attention_gate(
    x,
    g,
    filters
):

    theta_x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(x)

    phi_g = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(g)

    # --------------------------------------------------
    # Fix for odd image sizes (e.g. width = 275)
    # --------------------------------------------------

    phi_g = tf.keras.layers.Resizing(
        theta_x.shape[1],
        theta_x.shape[2],
        interpolation="bilinear"
    )(phi_g)

    add = tf.keras.layers.Add()([
        theta_x,
        phi_g
    ])

    relu = tf.keras.layers.Activation(
        "relu"
    )(add)

    psi = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        activation="sigmoid",
        padding="same"
    )(relu)

    return tf.keras.layers.Multiply()([
        x,
        psi
    ])