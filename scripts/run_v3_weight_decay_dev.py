"""Fixed M5 weight-decay comparison on DEVELOPMENT validation only.

Only --phase fit|score is accepted; both configured decay variants use the full
configured development grid and ensemble. Default is score-only. Fit requires
separately successful project preflight. Reuse the gate runner's caches and
immutable reports; never score held-out rows or locked generators.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_v3_gate_dev as runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=("fit", "score"), default="score")
    args = parser.parse_args(argv)
    return runner.main(["--phase", args.phase, "--variants", *runner.cfg.V3_DECAY_VARIANTS])


if __name__ == "__main__":
    main()
