"""CLI entry point for AlphaZero training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch

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
    parser.add_argument('--eval-interval', type=int, default=0,
                        help='Evaluate every N iterations (0 = disabled)')
    parser.add_argument('--eval-games', type=int, default=10,
                        help='Number of evaluation games')
    parser.add_argument('--eval-opponent', type=str, default='TicketFocused',
                        help='Opponent for evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cpu, cuda, cuda:0, etc. (default: auto-detect)')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        hidden_size = checkpoint.get('hidden_size', args.hidden_size)
        num_res_blocks = checkpoint.get('num_res_blocks', args.res_blocks)
        trainer = AlphaZeroTrainer(
            hidden_size=hidden_size,
            num_res_blocks=num_res_blocks,
            lr=args.lr,
            device=args.device,
        )
        print(f'Resuming from {args.resume}')
        trainer.load_checkpoint(args.resume)
        trainer.set_lr(args.lr)
        print(f'  Learning rate set to {args.lr}')
    else:
        trainer = AlphaZeroTrainer(
            hidden_size=args.hidden_size,
            num_res_blocks=args.res_blocks,
            lr=args.lr,
            device=args.device,
        )

    print(f'Starting AlphaZero training:')
    print(f'  Device: {args.device}')
    print(f'  Iterations: {args.iterations}')
    print(f'  Games/iter: {args.games}')
    print(f'  MCTS iters: {args.mcts_iters}')
    print(f'  Train steps/iter: {args.train_steps}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Network: {args.hidden_size}x{args.res_blocks} ResBlocks')
    if args.eval_interval > 0:
        print(f'  Eval: every {args.eval_interval} iters, {args.eval_games} games vs {args.eval_opponent}')
    print()

    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        mcts_iterations=args.mcts_iters,
        train_steps_per_iteration=args.train_steps,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_opponent=args.eval_opponent,
    )

    # Save final checkpoint, log, and plot
    path = trainer.save_checkpoint(args.checkpoint_dir)
    trainer._save_log(args.checkpoint_dir)
    trainer._plot_training(args.checkpoint_dir)
    print(f'\nTraining complete. Final checkpoint: {path}')
    print(f'Training log: {args.checkpoint_dir}/training_log.json')
    print(f'Training plot: {args.checkpoint_dir}/training_progress.png')


if __name__ == '__main__':
    main()
