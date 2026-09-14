#!/usr/bin/env python3
"""Run the analysis stages belonging to this numbered study section."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from run import main

if __name__ == "__main__":
    main(allowed_stages=['priority'])
