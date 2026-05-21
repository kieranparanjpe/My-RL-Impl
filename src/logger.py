from copy import deepcopy
from typing import Any, Dict, Optional

import wandb

from algorithms import Hyperparameters


class Logger:
    episode_return: int = 0
    episode_length: int = 0

    def __init__(self, run : Optional[wandb.Run], elements : Dict[str, Any]):
        self.elements_start = elements
        self.elements = deepcopy(self.elements_start)
        self.run = run

    def reset_fully(self):
        self.elements = deepcopy(self.elements_start)

    def reset_fields(self, *fields : str):
        for field in fields:
            self.elements[field] = self.elements_start[field]

    def set_log_data(self, kvps : Dict[str, Any]):
        self.elements.update(kvps)

    def add_log_data(self, kvps : Dict[str, Any]):
        for k, v in kvps.items():
            self.elements[k] += v

    def log_data(self):
        if self.run:
            self.run.log(self.elements)
