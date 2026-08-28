"""Pytest configuration to make `src` importable during tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so tests can import `src` package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
