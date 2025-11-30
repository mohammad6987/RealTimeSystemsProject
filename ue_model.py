import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override

class UEModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # use RLlib's FCNet backbone to flatten dict obs
        self.backbone = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name + "_back")
        hidden = self.backbone._model_out
        # heads: offload logits (2), beta/theta continuous (2)
        self.offload_logits = nn.Linear(hidden, 2)
        self.beta_theta = nn.Linear(hidden, 2)
        self.value_head = nn.Linear(hidden, 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x, _ = self.backbone.forward(input_dict, state, seq_lens)
        logits = self.offload_logits(x)
        beta_theta = torch.sigmoid(self.beta_theta(x))  # [0,1]
        out = torch.cat([logits, beta_theta], dim=1)
        self._value_out = self.value_head(x).squeeze(1)
        return out, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out

try:
    ModelCatalog.register_custom_model("ue_model", UEModel)
except Exception:
    pass
