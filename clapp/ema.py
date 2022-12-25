from typing import Dict


class EMA:  # EMA a dictionary of values
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.cache = {}

    def update(self, values: Dict[str, float]):
        for k, v in values.items():
            if k not in self.cache:
                self.cache[k] = v
            else:
                self.cache[k] = self.decay * self.cache[k] + (1 - self.decay) * v
        return self.cache
