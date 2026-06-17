class GradientClippingAction:
    """
    Acció simple: recomana aplicar gradient clipping.
    """

    def __init__(self, max_norm=1.0):
        self.max_norm = float(max_norm)

    def execute(self):
        return {
            "action": "gradient_clipping",
            "recommendation": f"Apply gradient clipping with max_norm={self.max_norm}.",
            "max_norm": self.max_norm,
        }
