import pytest
import torch
import torch.nn as nn

from phoonnx_train.vits.lora import LoRALinear, LoRAConv1d, LoRAConvTranspose1d
from phoonnx_train.vits.lora_config import LoRAConfig, SCOPE_PRESETS, VALID_TARGET_MODULES
from phoonnx_train.vits.apply_lora import (
    apply_lora,
    merge_lora,
    get_lora_state_dict,
    load_lora_adapter,
    count_parameters,
)


class TestLoRAConfig:
    def test_from_preset_generator_only(self):
        config = LoRAConfig.from_preset("generator-only")
        assert config.rank == 4
        assert config.target_modules == ("dec",)

    def test_from_preset_full_acoustic(self):
        config = LoRAConfig.from_preset("full-acoustic")
        assert config.rank == 8
        assert config.target_modules == ("dec", "enc_q", "flow", "dp")

    def test_from_preset_aggressive(self):
        config = LoRAConfig.from_preset("aggressive")
        assert config.rank == 16
        assert "enc_p" in config.target_modules

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown LoRA scope preset"):
            LoRAConfig.from_preset("nonexistent")

    def test_invalid_target_module_raises(self):
        with pytest.raises(ValueError, match="Invalid target module"):
            LoRAConfig(target_modules=("fake_module",))

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="LoRA rank must be"):
            LoRAConfig(rank=0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="LoRA alpha must be"):
            LoRAConfig(alpha=-1.0)

    def test_custom_config(self):
        config = LoRAConfig(rank=16, alpha=32.0, target_modules=("dec", "enc_q"))
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.target_modules == ("dec", "enc_q")


class TestLoRALinear:
    def test_init_shape(self):
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        assert lora.lora_A.shape == (4, 64)
        assert lora.lora_B.shape == (32, 4)
        assert lora.scaling == 2.0

    def test_zero_init_b(self):
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        assert torch.all(lora.lora_B == 0)

    def test_forward_no_change_at_init(self):
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(2, 64)
        with torch.no_grad():
            original_out = linear(x)
            lora_out = lora(x)
        assert torch.allclose(original_out, lora_out, atol=1e-5)

    def test_merge_produces_correct_weight(self):
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        nn.init.kaiming_uniform_(lora.lora_A)
        with torch.no_grad():
            lora.lora_B.normal_(0, 0.01)
        merged = lora.merge()
        expected_weight = linear.weight.data + lora.scaling * (lora.lora_B @ lora.lora_A)
        assert torch.allclose(merged.weight.data, expected_weight, atol=1e-5)

    def test_freeze_original_weights(self):
        linear = nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        assert not linear.weight.requires_grad
        assert not linear.bias.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad


class TestLoRAConv1d:
    def test_init_shape(self):
        conv = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        lora = LoRAConv1d(conv, rank=4, alpha=8.0)
        assert lora.lora_A.shape == (4, 64, 3)
        assert lora.lora_B.shape == (32, 4, 1)

    def test_forward_no_change_at_init(self):
        conv = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        lora = LoRAConv1d(conv, rank=4, alpha=8.0)
        x = torch.randn(2, 64, 50)
        with torch.no_grad():
            original_out = conv(x)
            lora_out = lora(x)
        assert torch.allclose(original_out, lora_out, atol=1e-5)

    def test_merge_shape(self):
        conv = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        lora = LoRAConv1d(conv, rank=4, alpha=8.0)
        merged = lora.merge()
        assert isinstance(merged, nn.Conv1d)
        assert merged.weight.shape == conv.weight.shape


class TestApplyLoRA:
    def _make_model(self):
        from phoonnx_train.vits.models import SynthesizerTrn
        model = SynthesizerTrn(
            n_vocab=100,
            spec_channels=513,
            segment_size=32,
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            resblock="1",
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            upsample_rates=(8, 8, 2, 2),
            upsample_initial_channel=512,
            upsample_kernel_sizes=(16, 16, 4, 4),
        )
        return model

    def test_apply_lora_generator_only(self):
        model = self._make_model()
        config = LoRAConfig.from_preset("generator-only")
        replaced = apply_lora(model, config)

        trainable, total, pct = count_parameters(model)
        assert trainable > 0
        assert trainable < total

        any_lora = False
        for name, module in model.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)):
                any_lora = True
                break
        assert any_lora

    def test_apply_lora_full_acoustic(self):
        model = self._make_model()
        config = LoRAConfig.from_preset("full-acoustic")
        replaced = apply_lora(model, config)

        trainable, total, pct = count_parameters(model)
        assert trainable > 0
        assert pct < 50.0

    def test_merge_lora_roundtrip(self):
        model = self._make_model()
        config = LoRAConfig.from_preset("generator-only")
        apply_lora(model, config)

        lora_state = get_lora_state_dict(model)
        assert len(lora_state) > 0

        merge_lora(model)

        any_lora = False
        for name, module in model.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)):
                any_lora = True
                break
        assert not any_lora

    def test_freeze_check(self):
        model = self._make_model()
        config = LoRAConfig.from_preset("generator-only")
        apply_lora(model, config)

        trainable, total, pct = count_parameters(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "lora_A" in name or "lora_B" in name, f"Unexpected trainable param: {name}"

    def test_load_lora_adapter(self):
        model = self._make_model()
        config = LoRAConfig.from_preset("generator-only")
        apply_lora(model, config)

        lora_state = get_lora_state_dict(model)

        model2 = self._make_model()
        config2 = LoRAConfig.from_preset("generator-only")
        apply_lora(model2, config2)

        load_lora_adapter(model2, lora_state)

        for key in lora_state:
            module_name = key.rsplit(".", 1)[0]
            attr_name = key.rsplit(".", 1)[1]
            from phoonnx_train.vits.apply_lora import _get_submodule
            mod = _get_submodule(model2, module_name)
            if mod is not None and hasattr(mod, attr_name):
                param = getattr(mod, attr_name)
                assert torch.allclose(param.data.cpu(), lora_state[key].cpu(), atol=1e-6)