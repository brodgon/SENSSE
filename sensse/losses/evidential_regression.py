"""
Evidential regression loss.

Normal Inverse Gamma formulation.

Reference:
Amini et al.
Deep Evidential Regression (NeurIPS 2020)
"""

import tensorflow as tf
import evidential_deep_learning as edl

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

def evidential_regularization(
    y_true,
    mu,
    v,
    alpha
):
    """
    Eq. 6 from manuscript.
    """

    return tf.abs(
        y_true - mu
    ) * (
        2.0 * alpha + v
    )

class EvidentialRegressionLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        evidential_weight=1e-2,
        name="evidential_regression"
    ):

        super().__init__(name=name)

        self.evidential_weight = evidential_weight

    def call(
        self,
        y_true,
        y_pred
    ):

        return edl.losses.EvidentialRegression(
            y_true,
            y_pred,
            coeff=self.evidential_weight
        )
