"""`prefetch` runs a producer ahead on a thread; order and errors must survive."""
import threading
import time
from contextlib import closing

import pytest

from handball_cv.video import prefetch


def test_items_arrive_in_order():
    assert list(prefetch(iter(range(100)), depth=3)) == list(range(100))


def test_an_error_arrives_where_it_happened():
    def items():
        yield 1
        yield 2
        raise OSError("decode failed")

    got = []
    with pytest.raises(OSError, match="decode failed"):
        for item in prefetch(items()):
            got.append(item)
    assert got == [1, 2]


def test_closing_early_stops_the_producer():
    """A consumer that fails mid-sequence must not leave the producer holding its source."""
    finished = threading.Event()

    def items():
        try:
            for i in range(10_000):
                yield i
        finally:
            finished.set()

    with closing(prefetch(items(), depth=2)) as ahead:
        assert next(ahead) == 0
    assert finished.wait(timeout=5)
    time.sleep(0.3)
    assert not any(t.name == "prefetch" and t.is_alive() for t in threading.enumerate())
