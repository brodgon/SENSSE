"""
Dirichlet uncertainty utilities.

Segmentation branch.
"""

import tensorflow as tf
import numpy as np


def evidence_from_logits(logits):
    """
    Convert segmentation logits
    to evidential support.
    """

    return tf.nn.softplus(logits)


def alpha_from_logits(logits):

    evidence = evidence_from_logits(
        logits
    )

    return evidence + 1.0


def dirichlet_strength(alpha):
    """
    S = sum(alpha)
    """

    return tf.reduce_sum(
        alpha,
        axis=-1,
        keepdims=True
    )


def expected_probabilities(alpha):

    S = dirichlet_strength(alpha)

    return alpha / S


def total_uncertainty(alpha):
    """
    Eq. 10

    u = K / S
    """

    K = alpha.shape[-1]

    S = dirichlet_strength(alpha)

    return K / S


def evidence(alpha):

    return alpha - 1.0


def belief_mass(alpha):
    """
    Subjective logic belief masses.
    """

    e = evidence(alpha)

    S = dirichlet_strength(alpha)

    return e / S


def class_variance(alpha):
    """
    Eq. 11
    """

    S = dirichlet_strength(alpha)

    return (
        alpha *
        (S - alpha)
        /
        (
            S * S * (S + 1.0)
        )
    )


def predictive_entropy(
    alpha,
    eps=1e-8
):
    """
    Entropy of expected probabilities.
    """

    p = expected_probabilities(
        alpha
    )

    return -tf.reduce_sum(
        p *
        tf.math.log(
            p + eps
        ),
        axis=-1
    )


def confidence(alpha):

    p = expected_probabilities(
        alpha
    )

    return tf.reduce_max(
        p,
        axis=-1
    )


def predicted_class(alpha):

    p = expected_probabilities(
        alpha
    )

    return tf.argmax(
        p,
        axis=-1
    )
