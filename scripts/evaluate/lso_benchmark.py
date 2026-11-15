"""Compatibility entry point for the validation-only paper pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluate.paper_pipeline import main

if __name__ == "__main__":
    main(["--split", "site", *sys.argv[1:]])
