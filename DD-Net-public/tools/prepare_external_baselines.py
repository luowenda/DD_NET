"""Prepare exact non-vendored baselines only after explicit user acknowledgment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddnet_revision.external_sources import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("motionrnn", "trajgru"))
    parser.add_argument("--acknowledge-unlicensed-upstream", action="store_true")
    parser.add_argument("--source-directory", type=Path,
                        help="Optional local checkout at the pinned commit, instead of downloading")
    args = parser.parse_args()
    print(prepare(args.model, acknowledge=args.acknowledge_unlicensed_upstream,
                  source_directory=args.source_directory))


if __name__ == "__main__":
    main()
