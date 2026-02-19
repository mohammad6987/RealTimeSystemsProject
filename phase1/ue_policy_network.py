from __future__ import annotations

import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class UEPolicyNetwork(TorchModelV2, nn.Module):
    """
    Shared UE policy/value model.

    RLlib asks this class for two things:
    - policy logits used to sample UE actions,
    - state-value estimate for PPO critic loss.

    The architecture uses RLlib's default fully-connected backbone.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Initialize RLlib model wrapper and PyTorch module internals.
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Single backbone shared by actor (policy logits) and critic (value function).
        self.backbone = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, f"{name}_backbone")
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Compute action logits for one forward pass and cache value output."""
        logits, _ = self.backbone.forward(input_dict, state, seq_lens)
        self._value_out = self.backbone.value_function()
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        """Return cached scalar value estimates for PPO critic training."""
        return self._value_out


# Register model name so policy configs can reference it via `custom_model`.
ModelCatalog.register_custom_model("ue_policy_network", UEPolicyNetwork)


try:
    from stable_baselines3.common.policies import ActorCriticPolicy

    class UEPolicySB3(ActorCriticPolicy):
        """
        Stable-Baselines3 PPO policy matching the UE network layout.

        Uses a shared MLP for actor and critic with [256, 256] hidden units.
        """

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("net_arch", [256, 256])
            kwargs.setdefault("activation_fn", nn.ReLU)
            super().__init__(*args, **kwargs)

except Exception:  # pragma: no cover - optional SB3 dependency
    UEPolicySB3 = None
