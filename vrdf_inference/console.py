"""Logging for the inference pipeline.

One prefix on every line makes a run's own output greppable apart from PyTorch and decord noise.
"""

import sys

LOG = "[inference]"


def log(msg):
    print(f"{LOG} {msg}", flush=True)


def warn(msg):
    print(f"{LOG} WARNING: {msg}", flush=True)


def die(msg):
    print(f"{LOG} ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)
