"""
YAML configuration utilities.
"""

import yaml


def load_config(path):
    """
    Load YAML configuration file.
    """

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config, path):
    """
    Save configuration dictionary.
    """

    with open(path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False
        )


def print_config(config):

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    for key, value in config.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 60)

