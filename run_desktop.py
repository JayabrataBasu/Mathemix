#!/usr/bin/env python3
"""Launch the MatheMixX desktop application."""
import sys
from pathlib import Path

# Add the python directory to the path so we can import mathemixx_desktop
sys.path.insert(0, str(Path(__file__).parent / "python"))

from mathemixx_desktop import launch

if __name__ == "__main__":
    launch()
