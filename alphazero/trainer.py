"""AlphaZero training orchestrator: self-play + neural network training."""
import logging
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from alphazero.network import AlphaZeroNet
from alphazero.replay_buffer import ReplayBuffer
from alphazero.self_play import self_play_game


class AlphaZeroTrainer:
    """Orchestrates self-play data generation and network training.

    Training loop per iteration:
    1. Generate self-play games using current network
    2. Store samples in replay buffer
    3. Train network on minibatches from buffer
    4. Checkpoint periodically
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_res_blocks: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        buffer_size: int = 100_000,
        device: str = 'cpu',
    ):
        self.device = device
        self.network = AlphaZeroNet(
            hidden_size=hidden_size,
            num_res_blocks=num_res_blocks,
        ).to(device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.iteration = 0

    def train(
        self,
        num_iterations: int = 100,
        games_per_iteration: int = 50,
        mcts_iterations: int = 100,
        train_steps_per_iteration: int = 200,
        batch_size: int = 256,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        eval_interval: int = 0,
        eval_games: int = 10,
        eval_opponent: str = 'TicketFocused',
        verbose: bool = True,
    ):
        """Run the full training loop.

        Args:
            num_iterations: Total training iterations.
            games_per_iteration: Self-play games per iteration.
            mcts_iterations: MCTS iterations per move during self-play.
            train_steps_per_iteration: Gradient steps per iteration.
            batch_size: Minibatch size.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_interval: Save every N iterations.
            eval_interval: Evaluate every N iterations (0 = disabled).
            eval_games: Number of evaluation games.
            eval_opponent: Opponent type for evaluation.
            verbose: Print progress.
        """
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for iteration in range(self.iteration, self.iteration + num_iterations):
            iter_start = time.time()

            # 1. Self-play
            self.network.eval()
            total_samples = 0
            for g in range(games_per_iteration):
                samples = self_play_game(
                    self.network, device=self.device,
                    mcts_iterations=mcts_iterations,
                )
                self.replay_buffer.add(samples)
                total_samples += len(samples)

            # 2. Train
            self.network.train()
            total_policy_loss = 0.0
            total_value_loss = 0.0
            actual_steps = 0

            if len(self.replay_buffer) >= batch_size:
                for _ in range(train_steps_per_iteration):
                    states, policies, values = self.replay_buffer.sample(batch_size)
                    states = states.to(self.device)
                    policies = policies.to(self.device)
                    values = values.to(self.device)

                    policy_logits, value_pred = self.network(states)

                    # Policy loss: cross-entropy with MCTS visit distribution
                    policy_loss = -(
                        policies * F.log_softmax(policy_logits, dim=1)
                    ).sum(dim=1).mean()

                    # Value loss: MSE
                    value_loss = F.mse_loss(value_pred.squeeze(-1), values)

                    loss = policy_loss + value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    actual_steps += 1

            self.iteration = iteration + 1
            elapsed = time.time() - iter_start

            if verbose:
                avg_ploss = total_policy_loss / max(actual_steps, 1)
                avg_vloss = total_value_loss / max(actual_steps, 1)
                print(
                    f'Iter {self.iteration:3d} | '
                    f'Games: {games_per_iteration} | '
                    f'Samples: {total_samples} (buf: {len(self.replay_buffer)}) | '
                    f'Policy loss: {avg_ploss:.4f} | '
                    f'Value loss: {avg_vloss:.4f} | '
                    f'Time: {elapsed:.1f}s'
                )

            # 3. Checkpoint
            if checkpoint_dir and self.iteration % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir)

            # 4. Evaluate
            if eval_interval > 0 and self.iteration % eval_interval == 0:
                self.evaluate(eval_games, eval_opponent, mcts_iterations)

    def evaluate(self, num_games: int, opponent: str, mcts_iterations: int) -> dict:
        """Play games against a baseline opponent and report results."""
        from game.core import Game

        self.network.eval()
        prev_level = logging.getLogger().level
        logging.disable(logging.CRITICAL)

        wins = 0
        total_score_az = 0
        total_score_opp = 0

        try:
            for i in range(num_games):
                g = Game(['AlphaZero', opponent], 'USA')
                g.players[0].network.load_state_dict(self.network.state_dict())
                g.players[0].network.to(self.device)
                g.players[0].network.eval()
                g.players[0].az_mcts.iterations = mcts_iterations

                stats = g.play()
                s_az, s_opp = stats['score'][0], stats['score'][1]
                total_score_az += s_az
                total_score_opp += s_opp
                if s_az > s_opp:
                    wins += 1
        finally:
            logging.disable(prev_level)

        win_rate = wins / num_games
        avg_az = total_score_az / num_games
        avg_opp = total_score_opp / num_games
        print(
            f'  EVAL vs {opponent} ({num_games}g): '
            f'Win rate: {win_rate:.0%} | '
            f'Avg score: {avg_az:.1f} vs {avg_opp:.1f}'
        )
        return {'win_rate': win_rate, 'avg_score_az': avg_az, 'avg_score_opp': avg_opp}

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        path = os.path.join(checkpoint_dir, f'alphazero_iter{self.iteration}.pt')
        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer_size': len(self.replay_buffer),
            'hidden_size': self.network.input_fc.out_features,
            'num_res_blocks': len(self.network.res_blocks),
        }, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']

    def set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
