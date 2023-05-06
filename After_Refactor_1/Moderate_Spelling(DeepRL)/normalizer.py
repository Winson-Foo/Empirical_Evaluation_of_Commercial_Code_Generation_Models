# normalizer.py
class RescaleNormalizer:
    def __init__(self):
        self.mean = 0
        self.std = 1

    def normalize(self, x):
        return (x - self.mean) / self.std

    def update(self, x):
        pass