from enum import Enum

class SampleType(Enum):
    FRONT = 0
    CENTER = 1
    RANDOM = 2

class DataMode(Enum):
    POSITION = 0
    VELOCITY = 1
    BOTH = 2