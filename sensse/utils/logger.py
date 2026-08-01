"""
Logging utilities.
"""

import logging
import os


def get_logger(
    save_dir,
    name="SENSSE"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    logger = logging.getLogger(
        name
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        os.path.join(
            save_dir,
            "experiment.log"
        )
    )

    file_handler.setFormatter(
        formatter
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )

    return logger


def log_metrics(
    logger,
    epoch,
    metrics
):

    msg = f"Epoch {epoch} | "

    msg += " | ".join([
        f"{k}: {v:.4f}"
        for k, v in metrics.items()
    ])

    logger.info(msg)
