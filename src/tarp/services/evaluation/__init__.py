from enum import Enum


class Extremum(Enum):
    MIN = "min"
    MAX = "max"


class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"


class Thresholding(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
