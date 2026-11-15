"""Compatibility entry point: retrained, validation-selected model comparisons."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluate.paper_pipeline import main, NETWORKS

if __name__ == "__main__":
    main(["--models", *NETWORKS, *sys.argv[1:]])
