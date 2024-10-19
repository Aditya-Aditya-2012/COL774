class ReduceLROnPlateau:
    def __init__(self, optimizer, patience: int = 5, factor: float = 0.2, min_lr: float = 1e-6, threshold: float = 1e-4, verbose: bool = True):
        """
        Args:
            optimizer: The optimizer for which to adjust the learning rate.
            patience: Number of epochs with no improvement after which learning rate will be reduced.
            factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
            min_lr: Lower bound on the learning rate.
            threshold: Minimum change in the monitored quantity to qualify as an improvement.
            verbose: If True, prints a message for each learning rate update.
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self.verbose = verbose

        # Internal state
        self.best_loss = None
        self.num_bad_epochs = 0

    def __call__(self, val_loss):
        """Make the class callable, checks if validation loss has plateaued and reduces the learning rate if needed."""
        self.step(val_loss)

    def step(self, val_loss):
        """Checks if validation loss has plateaued and reduces the learning rate if needed."""
        if self.best_loss is None:
            self.best_loss = val_loss  # Initialize with the first loss

        elif val_loss > self.best_loss - self.threshold:  # No significant improvement
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self._reduce_lr()
                self.num_bad_epochs = 0  # Reset counter after LR reduction
        else:
            self.best_loss = val_loss  # Update the best loss
            self.num_bad_epochs = 0  # Reset the counter if improvement

    def _reduce_lr(self):
        """Reduces the learning rate by the specified factor."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr > self.min_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

    def lr(self) -> float:
        """Returns the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
