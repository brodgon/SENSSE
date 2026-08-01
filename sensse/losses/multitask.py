"""
multitask.py

Segmentation loss used in SENSSE.

Implements:

L_seg = L_CE + L_EDL

with KL warm-up as described in the manuscript.
"""

import tensorflow as tf

from .evidential_seg import (
    dirichlet_evidential_loss,
    dirichlet_cross_entropy
)

from .warmup import KLWarmup


class EvidentialSegmentationLoss(
    tf.keras.losses.Loss
):
    """
    Evidential segmentation loss.

    Implements Eq. 17:

        L_seg = L_CE + L_EDL

    where:

        L_EDL = ESE + lambda * KL

    and lambda is progressively increased
    through a warm-up schedule.
    """

    def __init__(
        self,
        num_classes,
        max_kl_weight=1e-2,
        warmup_epochs=50,
        ce_weight=1.0,
        name="evidential_segmentation"
    ):

        super().__init__(name=name)

        self.num_classes = num_classes
        self.ce_weight = ce_weight

        self.current_epoch = 0

        self.scheduler = KLWarmup(
            max_weight=max_kl_weight,
            warmup_epochs=warmup_epochs
        )

    def set_epoch(
        self,
        epoch
    ):
        """
        Called from callback at the end of
        each training epoch.
        """

        self.current_epoch = epoch

    def call(
        self,
        y_true,
        logits
    ):
        """
        Parameters
        ----------
        y_true:
            One-hot encoded labels.
            Shape: [B,H,W,C]

        logits:
            Raw network output.
            Shape: [B,H,W,C]

        Returns
        -------
        Scalar loss.
        """

        # ---------------------------------------
        # Evidence
        # ---------------------------------------

        evidence = tf.nn.softplus(
            logits
        )

        alpha = evidence + 1.0

        # ---------------------------------------
        # KL warm-up
        # ---------------------------------------

        kl_weight = self.scheduler(
            self.current_epoch
        )

        # ---------------------------------------
        # CE
        # ---------------------------------------

        ce_loss = dirichlet_cross_entropy(
            y_true,
            alpha
        )

        # ---------------------------------------
        # Evidential Loss
        # ---------------------------------------

        edl_loss = dirichlet_evidential_loss(
            y_true,
            alpha,
            kl_weight=kl_weight
        )

        # ---------------------------------------
        # Final Segmentation Loss
        # ---------------------------------------

        total_loss = (
            self.ce_weight * ce_loss
            +
            edl_loss
        )

        return total_loss

    def get_config(self):

        config = super().get_config()

        config.update({
            "num_classes": self.num_classes,
            "ce_weight": self.ce_weight,
            "current_epoch": self.current_epoch
        })

        return config
