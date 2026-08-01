"""
sensse/models/sensse.py

SENSSE:
Uncertainty-Aware Simultaneous Synthesis and Segmentation
via Evidential Deep Learning

Author: Blanca Rodriguez-Gonzalez et al.

Supported interaction modes:

    none
    syn2seg
    seg2syn
    bidirectional
"""

import tensorflow as tf
import evidential_deep_learning as edl

from .blocks import residual_block
from .attention import attention_gate
from .variants import DecoderInteraction


# ============================================================
# Decoder Stage
# ============================================================

def decoder_stage(
    decoder_input,
    encoder_skip,
    filters,
    name="decoder"
):
    """
    Attention UNet decoder stage.
    """

    up = tf.keras.layers.Conv2DTranspose(
    filters,
    kernel_size=2,
    strides=2,
    padding="same",
    name=f"{name}_up"
    )(decoder_input)

    up = tf.keras.layers.Resizing(
        encoder_skip.shape[1],
        encoder_skip.shape[2],
        interpolation="bilinear"
    )(up)

    gated = attention_gate(
        encoder_skip,
        up,
        filters
    )

    x = tf.keras.layers.Concatenate(
        name=f"{name}_concat"
    )([
        up,
        gated
    ])

    x = residual_block(
        x,
        filters
    )

    return x


# ============================================================
# Segmentation Head
# ============================================================

def segmentation_head(
    x,
    num_classes,
    activation=None
):
    """
    EDL segmentation head.

    IMPORTANT:
    We output logits, not probabilities.

    Evidence is obtained later using:

        evidence = softplus(logits)

    This is more stable than ReLU.
    """

    logits = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        padding="same",
        activation=activation,
        name="segmentation_logits"
    )(x)

    return logits


# ============================================================
# Synthesis Head
# ============================================================

def synthesis_head(x):
    """
    NIG head.

    Outputs:

        μ
        v
        α
        β

    Packed as:

        [H,W,4]
    """


    return edl.layers.Conv2DNormalGamma(
        filters=1,
        kernel_size=1,
        name="nig_head"
    )(x)


# ============================================================
# Full Model
# ============================================================

def build_sensse(
    input_shape,
    num_classes=10,
    base_filters=32,
    interaction="syn2seg"
):
    """
    Parameters
    ----------

    interaction:

        none
        syn2seg
        seg2syn
        bidirectional

    Returns
    -------

    tf.keras.Model
    """

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=input_shape,
        name="input"
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    c1 = residual_block(
        inputs,
        base_filters
    )

    p1 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c1)

    c2 = residual_block(
        p1,
        base_filters * 2
    )

    p2 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c2)

    c3 = residual_block(
        p2,
        base_filters * 4
    )

    p3 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c3)

    c4 = residual_block(
        p3,
        base_filters * 8
    )

    p4 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c4)

    bottleneck = residual_block(
        p4,
        base_filters * 16
    )

    # --------------------------------------------------------
    # Interaction mechanism
    # --------------------------------------------------------

    interaction_layer = DecoderInteraction(
        mode=interaction
    )

    # ========================================================
    # DECODER LEVEL 4
    # ========================================================

    seg4 = decoder_stage(
        bottleneck,
        c4,
        base_filters * 8,
        name="seg4"
    )

    syn4 = decoder_stage(
        bottleneck,
        c4,
        base_filters * 8,
        name="syn4"
    )

    seg4, syn4 = interaction_layer(
        seg4,
        syn4,
        filters=base_filters * 8,
        level=4
    )

    # ========================================================
    # DECODER LEVEL 3
    # ========================================================

    seg3 = decoder_stage(
        seg4,
        c3,
        base_filters * 4,
        name="seg3"
    )

    syn3 = decoder_stage(
        syn4,
        c3,
        base_filters * 4,
        name="syn3"
    )

    seg3, syn3 = interaction_layer(
        seg3,
        syn3,
        filters=base_filters * 4,
        level=3
    )

    # ========================================================
    # DECODER LEVEL 2
    # ========================================================

    seg2 = decoder_stage(
        seg3,
        c2,
        base_filters * 2,
        name="seg2"
    )

    syn2 = decoder_stage(
        syn3,
        c2,
        base_filters * 2,
        name="syn2"
    )

    seg2, syn2 = interaction_layer(
        seg2,
        syn2,
        filters=base_filters * 2,
        level=2
    )

    # ========================================================
    # DECODER LEVEL 1
    # ========================================================

    seg1 = decoder_stage(
        seg2,
        c1,
        base_filters,
        name="seg1"
    )

    syn1 = decoder_stage(
        syn2,
        c1,
        base_filters,
        name="syn1"
    )

    seg1, syn1 = interaction_layer(
        seg1,
        syn1,
        filters=base_filters,
        level=1
    )

    # --------------------------------------------------------
    # Segmentation output
    # --------------------------------------------------------

    seg_output = segmentation_head(
        seg1,
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # Synthesis output
    # --------------------------------------------------------

    syn_output = synthesis_head(
        syn1
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            seg_output,
            syn_output
        ],
        name=f"SENSSE_{interaction}"
    )

    return model


# ============================================================
# Factory methods
# ============================================================

def build_no_interaction(
    input_shape,
    num_classes=10
):
    return build_sensse(
        input_shape=input_shape,
        num_classes=num_classes,
        interaction="none"
    )


def build_syn2seg(
    input_shape,
    num_classes=10
):
    return build_sensse(
        input_shape=input_shape,
        num_classes=num_classes,
        interaction="syn2seg"
    )


def build_seg2syn(
    input_shape,
    num_classes=10
):
    return build_sensse(
        input_shape=input_shape,
        num_classes=num_classes,
        interaction="seg2syn"
    )


def build_bidirectional(
    input_shape,
    num_classes=10
):
    return build_sensse(
        input_shape=input_shape,
        num_classes=num_classes,
        interaction="bidirectional"
    )


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    model = build_sensse(
        input_shape = (512,512,5),
        num_classes=10,
        interaction="syn2seg"
    )

    model.summary()
