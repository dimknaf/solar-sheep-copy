"""Contract status values — match CONTRACTS.md."""

from enum import Enum


class Status(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
