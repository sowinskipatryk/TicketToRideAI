"""AlphaZero neural network: dual-head MLP with residual blocks.

Architecture:
    Input(STATE_SIZE) → shared trunk (4 residual blocks)
                      → policy head (ACTION_SPACE_SIZE logits)
                      → value head (scalar in [-1, 1])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.encoding import STATE_SIZE, ACTION_SPACE_SIZE


class ResBlock(nn.Module):
    """Residual block: Linear→LN→ReLU→Linear→LN + skip → ReLU."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroNet(nn.Module):
    """Dual-head network for AlphaZero: policy + value.

    Args:
        state_size: Input feature dimension.
        action_size: Policy output dimension.
        hidden_size: Width of hidden layers.
        num_res_blocks: Number of residual blocks in the shared trunk.
    """

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
        hidden_size: int = 256,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Input projection
        self.input_fc = nn.Linear(state_size, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        # Shared trunk
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_size) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_fc1 = nn.Linear(hidden_size, hidden_size)
        self.policy_fc2 = nn.Linear(hidden_size, action_size)

        # Value head
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, STATE_SIZE).

        Returns:
            policy_logits: Tensor of shape (batch, ACTION_SPACE_SIZE). Raw logits.
            value: Tensor of shape (batch, 1). Scalar in [-1, 1].
        """
        # Shared trunk
        h = F.relu(self.input_ln(self.input_fc(x)))
        for block in self.res_blocks:
            h = block(h)

        # Policy head
        p = F.relu(self.policy_fc1(h))
        policy_logits = self.policy_fc2(p)

        # Value head
        v = F.relu(self.value_fc1(h))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
