from __future__ import annotations

import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class ServerSchedulerNetwork(TorchModelV2, nn.Module):
    """
    Server-side policy/value model.

    Output logits are interpreted by RLlib as continuous actions, where each
    action dimension corresponds to one UE priority score for server scheduling.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.backbone = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, f"{name}_backbone")
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Produce server action logits and cache critic output for value_function()."""
        logits, _ = self.backbone.forward(input_dict, state, seq_lens)
        self._value_out = self.backbone.value_function()
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        """Expose critic estimate required by PPO loss."""
        return self._value_out


ModelCatalog.register_custom_model("server_scheduler_network", ServerSchedulerNetwork)
