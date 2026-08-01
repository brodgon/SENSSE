"""
Dirichlet evidential segmentation loss.
"""

import numpy as np
import tensorflow as tf

def dirichlet_kl(alpha):

    num_classes = alpha.shape[-1]

    beta = tf.ones(
        (1, num_classes),
        dtype=tf.float32
    )

    S_alpha = tf.reduce_sum(
        alpha,
        axis=-1,
        keepdims=True
    )

    S_beta = tf.reduce_sum(
        beta,
        axis=-1,
        keepdims=True
    )

    lnB = (
        tf.math.lgamma(S_alpha)
        -
        tf.reduce_sum(
            tf.math.lgamma(alpha),
            axis=-1,
            keepdims=True
        )
    )

    lnB_uni = (
        tf.reduce_sum(
            tf.math.lgamma(beta),
            axis=-1,
            keepdims=True
        )
        -
        tf.math.lgamma(S_beta)
    )

    dg0 = tf.math.digamma(S_alpha)

    dg1 = tf.math.digamma(alpha)

    kl = (
        tf.reduce_sum(
            (alpha - beta) *
            (dg1 - dg0),
            axis=-1,
            keepdims=True
        )
        +
        lnB
        +
        lnB_uni
    )

    return kl

def expected_squared_error(
    y,
    alpha
):

    S = tf.reduce_sum(
        alpha,
        axis=-1,
        keepdims=True
    )

    p = alpha / S

    error = tf.reduce_sum(
        (y - p) ** 2,
        axis=-1,
        keepdims=True
    )

    variance = tf.reduce_sum(
        alpha * (S - alpha)
        /
        (
            S * S * (S + 1)
        ),
        axis=-1,
        keepdims=True
    )

    return error + variance

def dirichlet_evidential_loss(
    y,
    alpha,
    kl_weight=1e-2
):

    alpha_hat = (
        y
        +
        (1 - y) * alpha
    )

    ese = expected_squared_error(
        y,
        alpha
    )

    kl = dirichlet_kl(
        alpha_hat
    )

    return tf.reduce_mean(
        ese +
        kl_weight * kl
    )

def dirichlet_cross_entropy(
    y_true,
    alpha,
    epsilon=1e-6
):

    probs = alpha / tf.reduce_sum(
        alpha,
        axis=-1,
        keepdims=True
    )

    probs = tf.clip_by_value(
        probs,
        epsilon,
        1.0
    )

    ce = -tf.reduce_sum(
        y_true * tf.math.log(probs),
        axis=-1
    )

    return tf.reduce_mean(
        ce
    )
