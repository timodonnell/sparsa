"""Regression coverage for incomplete Iris cancellation timestamps."""

import pytest

from scripts.job_accounting import attempt_timing


def test_cancelled_attempt_stops_accruing_time():
    # Actual original Sparsa run: Iris recorded a job finish but no attempt finish.
    args = {
        "start_ms": 1789107607852,
        "finish_ms": None,
        "state": "killed",
        "job_finish_ms": 1789108575265,
    }
    first = attempt_timing(**args, observed_ms=1789112260039)
    later = attempt_timing(**args, observed_ms=1789198660039)
    assert first == later
    assert first["running_seconds"] == 967.413
    assert first["end_source"] == "job_finish_fallback"


def test_unfinished_running_attempt_accrues_time():
    args = {
        "start_ms": 1000,
        "finish_ms": None,
        "state": "running",
        "job_finish_ms": None,
    }
    assert attempt_timing(**args, observed_ms=3000)["running_seconds"] == 2
    assert attempt_timing(**args, observed_ms=5000)["running_seconds"] == 4


def test_stopped_attempt_without_end_is_unknown():
    result = attempt_timing(
        start_ms=1000,
        finish_ms=None,
        state="preempted",
        job_finish_ms=None,
        observed_ms=9000,
    )
    assert result["running_seconds"] is None
    assert result["end_source"] == "unknown_finish"


def test_explicit_finish_takes_precedence():
    result = attempt_timing(
        start_ms=1000,
        finish_ms=2000,
        state="succeeded",
        job_finish_ms=4000,
        observed_ms=9000,
    )
    assert result["running_seconds"] == 1
    assert result["end_source"] == "attempt_finish"


def test_queued_attempt_has_no_running_time():
    result = attempt_timing(
        start_ms=None,
        finish_ms=2000,
        state="worker_failed",
        job_finish_ms=4000,
        observed_ms=9000,
    )
    assert result["running_seconds"] == 0


def test_inconsistent_timestamps_are_rejected():
    with pytest.raises(ValueError, match="end precedes"):
        attempt_timing(
            start_ms=2000,
            finish_ms=1000,
            state="succeeded",
            job_finish_ms=4000,
            observed_ms=9000,
        )
