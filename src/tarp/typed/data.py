from collections.abc import Mapping
from typing import TypeVar

from tarp.typed.core import KnownT

RowT = TypeVar("RowT", bound=Mapping[str, KnownT])
BatchT = TypeVar("BatchT")
