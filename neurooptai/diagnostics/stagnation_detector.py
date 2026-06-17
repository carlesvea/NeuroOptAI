class StagnationDetector:
    """
    Detecta estancament:
    validation loss no millora durant diverses epochs.
    """

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = float(min_delta)

    def detect(self, validation_losses):
        if len(validation_losses) <= self.patience:
            return False

        recent = validation_losses[-(self.patience + 1):]
        best_before = recent[0]
        best_recent = min(recent[1:])

        improvement = best_before - best_recent

        return improvement <= self.min_delta
