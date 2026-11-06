"""Early stopping and best-model tracking."""


class EarlyStopping:
    """Track the best score and stop after `patience` epochs without improvement.

    The best score is tracked from the first epoch, while patience only counts
    from `start_epoch` onward.
    """

    def __init__(self, patience: int, start_epoch: int) -> None:
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def update(self, score: float, epoch: int) -> bool:
        """Record an epoch's score and return whether it is a new best."""
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        elif epoch >= self.start_epoch:
            self.epochs_without_improvement += 1
        return improved
