"""`prefetch` runs a producer ahead on a thread; order and errors must survive."""
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
