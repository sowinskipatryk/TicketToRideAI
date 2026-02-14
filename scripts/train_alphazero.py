"""CLI entry point for AlphaZero training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from alphazero.trainer import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero for Ticket to Ride')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--games', type=int, default=50, help='Self-play games per iteration')
    parser.add_argument('--mcts-iters', type=int, default=100, help='MCTS iterations per move')
    parser.add_argument('--train-steps', type=int, default=200, help='Training steps per iteration')
    parser.add_argument('--batch-size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Network hidden size')
    parser.add_argument('--res-blocks', type=int, default=4, help='Number of residual blocks')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/alphazero',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    args = parser.parse_args()

    trainer = AlphaZeroTrainer(
        hidden_size=args.hidden_size,
        num_res_blocks=args.res_blocks,
        lr=args.lr,
    )

    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load_checkpoint(args.resume)

    print(f'Starting AlphaZero training:')
    print(f'  Iterations: {args.iterations}')
    print(f'  Games/iter: {args.games}')
    print(f'  MCTS iters: {args.mcts_iters}')
    print(f'  Train steps/iter: {args.train_steps}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Network: {args.hidden_size}x{args.res_blocks} ResBlocks')
    print()

    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        mcts_iterations=args.mcts_iters,
        train_steps_per_iteration=args.train_steps,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Save final checkpoint
    path = trainer.save_checkpoint(args.checkpoint_dir)
    print(f'\nTraining complete. Final checkpoint: {path}')


if __name__ == '__main__':
    main()
