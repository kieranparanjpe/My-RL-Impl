from enum import Enum

class MdpTerminationState(Enum):
    IN_PROGRESS = 1
    TERMINATED = 2
    TRUNCATED = 3