import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from neuroevolution.manager import run_neat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NEAT network for Ticket to Ride.")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a neat-checkpoint file to resume training from '
                             '(e.g. neat-checkpoint-5).')
    args = parser.parse_args()
    run_neat(resume_checkpoint=args.resume)
