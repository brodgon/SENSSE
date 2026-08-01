"""
NIG uncertainty utilities.

Synthesis branch.
"""

import tensorflow as tf


def split_nig(y_pred):

    mu, v, alpha, beta = tf.split(
        y_pred,
        4,
        axis=-1
    )

    return (
        mu,
        v,
        alpha,
        beta
    )


def predictive_mean(y_pred):

    mu, _, _, _ = split_nig(
        y_pred
    )

    return mu


def aleatoric_uncertainty(y_pred):
    """
    Data uncertainty.
    """

    _, _, alpha, beta = split_nig(
        y_pred
    )

    return beta / (
        alpha - 1.0
    )


def epistemic_uncertainty(y_pred):
    """
    Model uncertainty.
    """

    _, v, alpha, beta = split_nig(
        y_pred
    )

    return beta / (
        v *
        (alpha - 1.0)
    )


def total_uncertainty(y_pred):

    return (
        aleatoric_uncertainty(y_pred)
        +
        epistemic_uncertainty(y_pred)
    )


def evidence(y_pred):
    """
    Useful visualization.

    Related to Eq. 6.
    """

    _, v, alpha, _ = split_nig(
        y_pred
    )

    return (
        2.0 * alpha
        +
        v
    )


def predictive_std(y_pred):

    return tf.sqrt(
        total_uncertainty(y_pred)
    )


def nig_parameters(y_pred):

    return split_nig(y_pred)
