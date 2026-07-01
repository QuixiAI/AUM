"""M1: the AUM-Ø evidence core runs forward+backward via the reference backend, on CPU and MPS.

Silence is disabled (the ~76.5M evidence-core baseline). Validates the Appendix-A parameter
count and that the whole model trains end-to-end with no Triton/Metal dependency.
"""

import pytest
import torch

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel


def _counts(model):
    total = sum(p.numel() for p in model.parameters())
    silence = sum(p.numel() for p in model.backbone.silence.parameters())
    return total, total - silence  # (full, evidence-core)


def test_param_counts():
    model = AumLMHeadModel(AumConfig())
    full, core = _counts(model)
    # Appendix-A reference: core ~76.5M (silence-ablated baseline), full ~78M, silence ~1.77M.
    assert abs(core - 76.5e6) < 0.5e6, f"core={core:,}"
    assert abs(full - 78.25e6) < 0.5e6, f"full={full:,}"
    assert full - core == 1_769_408, f"silence={full - core:,}"   # v6: pressure_in [128,515] (§9)


def _smoke(device):
    torch.manual_seed(0)
    cfg = AumConfig(n_layer=2, vocab_size=512, d_intermediate=256)  # tiny for speed
    model = AumLMHeadModel(cfg).to(device)
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    out = model(ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    loss = torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    return loss.item()


def test_forward_backward_cpu():
    assert _smoke("cpu") > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_forward_backward_mps():
    assert _smoke("mps") > 0


if __name__ == "__main__":
    test_param_counts()
    print("param counts OK")
    print("cpu loss:", _smoke("cpu"))
    if torch.backends.mps.is_available():
        print("mps loss:", _smoke("mps"))
