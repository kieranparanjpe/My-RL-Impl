from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    import wandb

class Logger:
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

    def sum_log_data(self, kvps : Dict[str, Any]):
        for k, v in kvps.items():
            self.elements[k] += v

    def log_data(self):
        if self.run is not None:
            self.run.log(data=self.elements)
