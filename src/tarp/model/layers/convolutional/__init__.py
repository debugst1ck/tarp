from enum import Enum


class Autopad(Enum):
    SAME = "same"
    VALID = "valid"
    CAUSAL = "causal"


class PaddingMode(Enum):
    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
