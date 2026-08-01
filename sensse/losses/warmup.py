"""
KL warmup scheduler.
"""

class KLWarmup:

    def __init__(
        self,
        max_weight=1e-2,
        warmup_epochs=50
    ):

        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs

    def __call__(
        self,
        epoch
    ):

        return (
            self.max_weight
            *
            min(
                1.0,
                epoch /
                self.warmup_epochs
            )
        )
