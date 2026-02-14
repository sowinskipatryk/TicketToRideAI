"""Circular replay buffer for AlphaZero training samples."""
import random
from collections import deque
from typing import List, Tuple

import torch

from alphazero.self_play import TrainingSample


class ReplayBuffer:
    """Fixed-size circular buffer storing training samples."""

    def __init__(self, max_size: int = 100_000):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, samples: List[TrainingSample]) -> None:
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random minibatch.

        Returns:
            (states, policies, values) as tensors.
        """
        n = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), n)

        states = torch.stack([s.state for s in batch])
        policies = torch.stack([s.policy for s in batch])
        values = torch.tensor([s.value for s in batch], dtype=torch.float32)

        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)
