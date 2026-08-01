"""
variants.py

Decoder interaction strategies for SENSSE.

Available variants:

- none
- syn2seg
- seg2syn
- bidirectional
"""

import tensorflow as tf


def feature_fusion(
    target_feature,
    source_feature,
    filters,
    name=None
):
    """
    Channel-wise fusion of decoder features.

    Paper:
        Concatenation + 1x1 projection.
    """

    x = tf.keras.layers.Concatenate(
        name=f"{name}_concat"
    )([
        target_feature,
        source_feature
    ])

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        activation="relu",
        name=f"{name}_proj"
    )(x)

    return x


class DecoderInteraction:

    def __init__(self, mode="none"):
        self.mode = mode

    def __call__(
        self,
        seg_feature,
        syn_feature,
        filters,
        level
    ):

        if self.mode == "none":

            return seg_feature, syn_feature

        elif self.mode == "syn2seg":

            seg_feature = feature_fusion(
                seg_feature,
                syn_feature,
                filters,
                name=f"syn2seg_l{level}"
            )

            return seg_feature, syn_feature

        elif self.mode == "seg2syn":

            syn_feature = feature_fusion(
                syn_feature,
                seg_feature,
                filters,
                name=f"seg2syn_l{level}"
            )

            return seg_feature, syn_feature

        elif self.mode == "bidirectional":

            seg_new = feature_fusion(
                seg_feature,
                syn_feature,
                filters,
                name=f"segfusion_l{level}"
            )

            syn_new = feature_fusion(
                syn_feature,
                seg_feature,
                filters,
                name=f"synfusion_l{level}"
            )

            return seg_new, syn_new

        else:

            raise ValueError(
                f"Unknown interaction mode: {self.mode}"
            )
