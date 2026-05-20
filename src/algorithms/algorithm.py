from abc import ABC, abstractmethod
from policy import Policy

class Algorithm(ABC):

    def __init__(self, policy : Policy):
        super().__init__()

