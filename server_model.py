import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override

class ServerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.backbone = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name + "_back")
        hidden = self.backbone._model_out
        # accept logits (N), beta alloc (N), theta alloc (N) -> outputs concatenated
        self.accept_logits = nn.Linear(hidden, obs_space["req_offload"].shape[0])
        self.beta_alloc = nn.Linear(hidden, obs_space["s"].shape[0])
        self.theta_alloc = nn.Linear(hidden, obs_space["s"].shape[0])
        self.value_head = nn.Linear(hidden, 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x, _ = self.backbone.forward(input_dict, state, seq_lens)
        accept_logits = self.accept_logits(x)
        beta = torch.sigmoid(self.beta_alloc(x))
        theta = torch.sigmoid(self.theta_alloc(x))
        out = torch.cat([accept_logits, beta, theta], dim=1)
        self._value_out = self.value_head(x).squeeze(1)
        return out, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out

try:
    ModelCatalog.register_custom_model("server_model", ServerModel)
except Exception:
    pass
