"""Compatibility entry point; gate stability is measured on validation only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluate.gate_prior_sweep import main

if __name__ == "__main__":
    main()
