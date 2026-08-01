"""
Building blocks used across SENSSE.
"""

import tensorflow as tf


def residual_block(x, filters, use_bn=True):

    shortcut = x

    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same"
        )(shortcut)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([shortcut, x])

    x = tf.keras.layers.Activation("relu")(x)

    return x
