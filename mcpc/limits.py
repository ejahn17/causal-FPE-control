"""Optional cooperative memory budget; experiment workers also have a watchdog."""
import os
import time

_budget = None
_process = None
_next_check = 0.0


def set_memory_budget(gb=None):
    global _budget, _process, _next_check
    import psutil
    _budget = None if gb is None else gb * 1024**3
    _process = psutil.Process(os.getpid())
    _next_check = 0.0


def check_memory():
    global _next_check
    if _budget is not None and time.monotonic() >= _next_check:
        _next_check = time.monotonic() + 0.1
        if _process.memory_info().rss > _budget:
            raise MemoryError("worker memory budget exceeded")
