"""Circuit breaker: stop calling a provider that is clearly down.

Like the fuse in a house: after too many failures it "trips" so we stop wasting time.

  CLOSED     normal. Count consecutive failures. BREAKER_FAILS in a row -> OPEN.
  OPEN       skip this provider instantly. After BREAKER_COOLDOWN_S -> HALF_OPEN.
  HALF_OPEN  let exactly ONE test request through.
             success -> CLOSED (healthy again); failure -> OPEN (wait another cooldown).
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, fail_threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic):
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self.clock = clock  # injectable so tests can fast-forward time
        self.state = State.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._trial_in_flight = False
        self._trial_started = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """May we send a request to this provider right now?"""
        with self._lock:
            if self.state == State.CLOSED:
                return True
            if self.state == State.OPEN:
                if self.clock() - self.opened_at >= self.cooldown_s:
                    self.state = State.HALF_OPEN
                    self._start_trial()
                    return True  # this caller is the single test request
                return False
            # HALF_OPEN: a test request is already out; everyone else waits.
            # Safety net: if that test never reported back (e.g. it was cancelled),
            # allow a new one after another cooldown instead of waiting forever.
            if not self._trial_in_flight or self.clock() - self._trial_started >= self.cooldown_s:
                self._start_trial()
                return True
            return False

    def _start_trial(self) -> None:
        self._trial_in_flight = True
        self._trial_started = self.clock()

    def record_success(self) -> None:
        with self._lock:
            self.state = State.CLOSED
            self.failures = 0
            self._trial_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            if self.state == State.HALF_OPEN:
                self._trip()
                return
            self.failures += 1
            if self.failures >= self.fail_threshold:
                self._trip()

    def _trip(self) -> None:
        self.state = State.OPEN
        self.opened_at = self.clock()
        self._trial_in_flight = False

    def retry_after_s(self) -> float:
        if self.state != State.OPEN:
            return 0.0
        return max(0.0, self.cooldown_s - (self.clock() - self.opened_at))
