import time
from functools import wraps
from typing import Callable

from tarp.cli.logging import Console


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        Console.debug(
            f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds"
        )
        return result

    return timeit_wrapper
