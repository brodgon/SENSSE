"""
Seed utilities.

Ensures reproducibility.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    """
    Set seeds for:
        - Python
        - NumPy
        - TensorFlow
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    print(f"[INFO] Seed set to {seed}")
