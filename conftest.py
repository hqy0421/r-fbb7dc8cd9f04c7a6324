import os
import sys

# Under `python -I` the current directory is not placed on sys.path, so make
# the workspace root importable explicitly before pytest collects anything.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
