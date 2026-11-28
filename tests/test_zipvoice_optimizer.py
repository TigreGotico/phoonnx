"""The ZipVoice Lightning module gives ScaledAdam its parameter names."""
import torch

from phoonnx_train.engines.zipvoice import _QUALITY_PRESETS
from phoonnx_train.zipvoice.lightning import ZipVoiceModule


def test_scaled_adam_survives_a_clipped_step():
    """A gradient far above the clipping threshold makes ScaledAdam sort the
    parameters by their gradient ratio and name. Parameters with no gradient
    share the ratio 0, so only distinct names keep the sort off the tensors."""
    torch.manual_seed(0)
    module = ZipVoiceModule(num_symbols=12, model_params=dict(_QUALITY_PRESETS["low"]))
    opt = module.configure_optimizers()["optimizer"]
    params = list(module.model.parameters())
    # the threshold is set from the gradient norms at step 10
    for _ in range(12):
        opt.zero_grad()
        sum((p * torch.randn_like(p)).sum() for p in params).backward()
        opt.step()
    opt.zero_grad()
    (params[0] * 1e6).sum().backward()
    for p in params[1:]:
        p.grad = torch.zeros_like(p)
    opt.step()
    assert opt.parameters_names == [[n for n, _ in module.model.named_parameters()]]
