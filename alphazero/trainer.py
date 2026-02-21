"""AlphaZero training orchestrator: self-play + neural network training."""
import json
import logging
import os
import time
from typing import Optional, List, Dict

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
        self.training_log: List[Dict] = []

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

            # 1. Self-play (always on CPU: batch=1 inference is slower on GPU)
            self.network.eval()
            self.network.to('cpu')
            total_samples = 0
            for g in range(games_per_iteration):
                samples = self_play_game(
                    self.network, device='cpu',
                    mcts_iterations=mcts_iterations,
                )
                self.replay_buffer.add(samples)
                total_samples += len(samples)

            # 2. Train (move back to target device for batched gradient updates)
            self.network.to(self.device)
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
                    value_loss = F.mse_loss(value_pred.view(-1), values)

                    loss = policy_loss + value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    actual_steps += 1

            self.iteration = iteration + 1
            elapsed = time.time() - iter_start

            avg_ploss = total_policy_loss / max(actual_steps, 1)
            avg_vloss = total_value_loss / max(actual_steps, 1)

            entry = {
                'iteration': self.iteration,
                'policy_loss': round(avg_ploss, 4),
                'value_loss': round(avg_vloss, 4),
                'samples': total_samples,
                'buffer_size': len(self.replay_buffer),
                'time': round(elapsed, 1),
            }

            if verbose:
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
                eval_result = self.evaluate(eval_games, eval_opponent, mcts_iterations)
                entry['eval_win_rate'] = eval_result['win_rate']
                entry['eval_avg_score'] = eval_result['avg_score_az']
                entry['eval_opp_score'] = eval_result['avg_score_opp']

            self.training_log.append(entry)

            # Save log and plot
            if checkpoint_dir and self.iteration % checkpoint_interval == 0:
                self._save_log(checkpoint_dir)
                self._plot_training(checkpoint_dir)

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
                az = g.players[0]
                az.network.load_state_dict(self.network.state_dict())
                az.network.to(self.device)
                az.network.eval()
                az.az_mcts.network = az.network
                az.az_mcts.device = self.device
                az.az_mcts.iterations = mcts_iterations

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
        # Try to load training log from same directory
        checkpoint_dir = os.path.dirname(path)
        self._load_log(checkpoint_dir)

    def set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _save_log(self, checkpoint_dir: str) -> None:
        path = os.path.join(checkpoint_dir, 'training_log.json')
        with open(path, 'w') as f:
            json.dump(self.training_log, f, indent=2)

    def _load_log(self, checkpoint_dir: str) -> None:
        path = os.path.join(checkpoint_dir, 'training_log.json')
        if os.path.exists(path):
            with open(path) as f:
                self.training_log = json.load(f)

    def _plot_training(self, checkpoint_dir: str) -> None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if len(self.training_log) < 2:
            return

        iters = [e['iteration'] for e in self.training_log]
        policy_loss = [e['policy_loss'] for e in self.training_log]
        value_loss = [e['value_loss'] for e in self.training_log]

        eval_entries = [e for e in self.training_log if 'eval_win_rate' in e]
        has_eval = len(eval_entries) > 0

        num_plots = 3 if has_eval else 2
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

        # Policy loss
        axes[0].plot(iters, policy_loss, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Policy Loss')
        axes[0].set_title('AlphaZero Training Progress')
        axes[0].grid(True, alpha=0.3)

        # Value loss
        axes[1].plot(iters, value_loss, 'r-', linewidth=1.5)
        axes[1].set_ylabel('Value Loss')
        axes[1].set_xlabel('Iteration' if not has_eval else '')
        axes[1].grid(True, alpha=0.3)

        # Win rate + scores
        if has_eval:
            eval_iters = [e['iteration'] for e in eval_entries]
            win_rates = [e['eval_win_rate'] for e in eval_entries]
            az_scores = [e['eval_avg_score'] for e in eval_entries]
            opp_scores = [e['eval_opp_score'] for e in eval_entries]

            ax_wr = axes[2]
            ax_sc = ax_wr.twinx()

            ln1 = ax_wr.plot(eval_iters, win_rates, 'g-o', linewidth=2, markersize=4, label='Win Rate')
            ax_wr.set_ylabel('Win Rate', color='g')
            ax_wr.set_ylim(-0.05, 1.05)
            ax_wr.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax_wr.grid(True, alpha=0.3)

            ln2 = ax_sc.plot(eval_iters, az_scores, 'b--', linewidth=1, label='AZ Score')
            ln3 = ax_sc.plot(eval_iters, opp_scores, 'r--', linewidth=1, label='Opp Score')
            ax_sc.set_ylabel('Avg Score')

            lns = ln1 + ln2 + ln3
            labels = [l.get_label() for l in lns]
            ax_wr.legend(lns, labels, loc='upper left', fontsize=8)
            axes[2].set_xlabel('Iteration')

        plt.tight_layout()
        path = os.path.join(checkpoint_dir, 'training_progress.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
