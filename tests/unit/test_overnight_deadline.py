"""The queue's deadline must survive an evening start."""
from datetime import datetime

from scripts.overnight_render_queue import next_occurrence


def test_evening_start_defers_a_morning_deadline_to_tomorrow():
    # The string compare this replaced read "22:55" >= "07:30" as true and quit
    # before rendering anything.
    assert next_occurrence(datetime(2026, 9, 10, 22, 55), "07:30") == \
        datetime(2026, 9, 11, 7, 30)


def test_deadline_later_today_stays_today():
    assert next_occurrence(datetime(2026, 9, 10, 6, 0), "07:30") == \
        datetime(2026, 9, 10, 7, 30)


def test_passing_midnight_does_not_disarm_the_deadline():
    # Resolved once at startup, so a 02:00 clock is still before the 07:30 that
    # was chosen the previous evening.
    deadline = next_occurrence(datetime(2026, 9, 10, 22, 55), "07:30")
    assert datetime(2026, 9, 11, 2, 0) < deadline
    assert datetime(2026, 9, 11, 8, 0) > deadline
