import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def make_modernbert_config(**params):
    """ModernBertConfig for a phoneme vocabulary.

    transformers 5.x's rope standardization writes back a ``rope_parameters``
    dict that its own strict validation rejects (legacy top-level
    rope_type/rope_theta keys), so validation is bypassed during
    construction. Default token ids point outside a 178-symbol vocab, so
    they are remapped ("$" pad = 0, per the StyleTTS2 symbol table).
    """
    from transformers import ModernBertConfig
    params.setdefault("pad_token_id", 0)
    for key in ("bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id"):
        params.setdefault(key, None)
    orig_setattr = ModernBertConfig.__setattr__
    ModernBertConfig.__setattr__ = object.__setattr__
    try:
        return ModernBertConfig(**params)
    finally:
        ModernBertConfig.__setattr__ = orig_setattr


def _build_bert(plbert_config):
    """Build the encoder for a plbert_dir; the optional ``backbone`` key
    (written by the styletts2-plbert training engine) selects the
    architecture — absent means upstream ALBERT."""
    backbone = plbert_config.get('backbone', 'albert')
    params = dict(plbert_config['model_params'])
    if backbone == 'albert':
        return CustomAlbert(AlbertConfig(**params))
    if backbone == 'modernbert':
        from transformers import ModernBertModel

        class CustomModernBert(ModernBertModel):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs).last_hidden_state

        params.pop('dropout', None)
        return CustomModernBert(make_modernbert_config(**params))
    raise ValueError(f"Unknown PL-BERT backbone: {backbone!r}")


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))

    bert = _build_bert(plbert_config)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    from phoonnx_train.torch_compat import trusting_torch_load

    with trusting_torch_load():
        checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    # older transformers exposed position_ids as a buffer; drop if present
    new_state_dict.pop("embeddings.position_ids", None)
    bert.load_state_dict(new_state_dict, strict=False)

    return bert
