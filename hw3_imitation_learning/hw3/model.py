"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int, 
        # d_model: int, depth: int
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # define linear layer
        #self.linear = nn.Linear(self.state_dim, self.action_dim*self.chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        d_model = 512

        self.mlp = nn.Sequential(
        nn.Linear(self.state_dim, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, self.action_dim * self.chunk_size)
    )

    def forward(
        self, states: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""

        flat_actions = self.mlp(states)
        return flat_actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        
        # predict action based on state
        predicted_action = self.forward(state)

        # return mse loss (pred_action - action)
        return nn.functional.mse_loss(predicted_action, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        
        # sample new actions based on current model
        return self.forward(state)

class MultiTaskPolicy(BasePolicy):
    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int, 
        # d_model: int, depth: int
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.state_dim = state_dim - 6
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        d_model = 512

        self.mlp = nn.Sequential(
        nn.Linear(self.state_dim, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, self.action_dim * self.chunk_size)
        )
        


    def forward(
        self, states: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""

        # Select cube which has to be placed in box and zero other cube values
        state_cpu = states.clone()
        new_states = []
        for item in state_cpu:          
            red = item[0].item()
            green = item[1].item()
            # blue = item[0, 2].item()

            if (red > 0):
                keep = item[3:6]
                item[6:9] = 0.0
                item[9:12] = 0.0

            elif (green > 0):
                item[3:6] = 0.0
                keep = item[6:9]
                item[9:12] = 0.0
            else:
                item[3:6] = 0.0
                item[6:9] = 0.0
                keep = item[9:12]
 
            # only return important cube (delete other two cubes)
            final_item = torch.cat((item[:3], keep, item[12:]), dim=-1)
            new_states.append(final_item)

     
        states = torch.stack(new_states)

        flat_actions = self.mlp(states)
        
        return flat_actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:

        predicted_action = self.forward(state)

        # return mse loss (pred_action - action)
        return nn.functional.mse_loss(predicted_action, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        
        # sample new actions based on current model
        return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    # d_model: int,
    # depth: int,
    d_model: None,
    depth: None,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            # d_model=d_model,
            # depth=depth,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            # d_model=d_model,
            # depth=depth,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

