class OverfittingDetector:
    """
    Detecta overfitting simple:
    train loss baixa mentre validation loss puja.
    """

    def __init__(self, min_points=3):
        self.min_points = min_points

    def detect(self, train_losses, validation_losses):
        if len(train_losses) < self.min_points or len(validation_losses) < self.min_points:
            return False

        recent_train = train_losses[-self.min_points:]
        recent_val = validation_losses[-self.min_points:]

        train_is_decreasing = recent_train[-1] < recent_train[0]
        val_is_increasing = recent_val[-1] > recent_val[0]

        return train_is_decreasing and val_is_increasing
