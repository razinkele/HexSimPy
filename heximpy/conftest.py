"""Ensure the parent directory is on sys.path so ``import heximpy`` works."""
import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
