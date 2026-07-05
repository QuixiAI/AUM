"""Decode B1: single-token generation matches the full forward (the decode oracle).

Prefill the prompt, then step one token at a time carrying the A-phase KV cache, the U-phase
recurrent state (S, phi) and conv state, (and the silence sigma carry when enabled). The
per-position logits must equal a single full-forward pass. CPU + MPS, silence off and on."""

import pytest
import torch

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel
from aum_ssm.utils.generation import InferenceParams


def _decode_logits(model, x, prompt_len):
    B, T = x.shape
    ip = InferenceParams(max_seqlen=T, max_batch_size=B)
    ip.key_value_memory_dict = model.allocate_inference_cache(B, T)
    parts = [model(x[:, :prompt_len], inference_params=ip, num_last_tokens=1).logits]  # -> ref[:, prompt-1]
    ip.seqlen_offset = prompt_len
    for t in range(prompt_len, T - 1):                      # feed x[:,t] at position t -> ref[:, t]
        parts.append(model(x[:, t:t + 1], inference_params=ip, num_last_tokens=1).logits)
        ip.seqlen_offset += 1
    return torch.cat(parts, dim=1)                          # (B, T-prompt, V) == ref[:, prompt-1:T-1]


def _run(device, silence):
    torch.manual_seed(0)
    cfg = AumConfig(n_layer=2, vocab_size=128, d_intermediate=128, silence_enabled=silence)
    model = AumLMHeadModel(cfg).to(device).eval()
    T, B, prompt = 16, 2, 8
    x = torch.randint(0, 128, (B, T), device=device)
    with torch.no_grad():
        ref = model(x).logits
        dec = _decode_logits(model, x, prompt)
    diff = (dec.float() - ref[:, prompt - 1:T - 1].float()).abs().max().item()
    return diff


def test_decode_matches_forward_silence_off_cpu():
    assert _run("cpu", silence=False) < 2e-3


def test_decode_matches_forward_silence_on_cpu():
    assert _run("cpu", silence=True) < 2e-3


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_decode_matches_forward_mps():
    assert _run("mps", silence=False) < 5e-3


def test_generate_api_matches_forward():
    # the .generate() driver (cg=False, lazy cache) end-to-end, silence on
    torch.manual_seed(0)
    cfg = AumConfig(n_layer=2, vocab_size=128, d_intermediate=128, silence_enabled=True)
    m = AumLMHeadModel(cfg).eval()
    T, B, prompt = 16, 2, 8
    x = torch.randint(0, 128, (B, T))
    with torch.no_grad():
        ref = m(x).logits
        out = m.generate(input_ids=x[:, :prompt], max_length=T, cg=False,
                         output_scores=True, return_dict_in_generate=True, teacher_outputs=x)
    scores = torch.stack(out.scores, dim=1)
    assert tuple(out.sequences.shape) == (B, T)
    assert (scores.float() - ref[:, prompt - 1:T - 1].float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 inference regression is CUDA-only")
def test_bf16_generate_without_external_autocast_cuda():
    torch.manual_seed(0)
    cfg = AumConfig(
        n_layer=1,
        vocab_size=128,
        d_model=64,
        d_intermediate=128,
        attn_num_heads=2,
        attn_num_heads_kv=1,
        attn_head_dim=32,
        u_num_heads=2,
        u_head_dim=32,
        d_sigma=32,
        d_phase=8,
        silence_enabled=True,
        kernel_backend="reference",
    )
    model = AumLMHeadModel(cfg, device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randint(0, 128, (1, 6), device="cuda")
    with torch.inference_mode():
        result = model(x)
        out = model.generate(x[:, :3], max_length=6, teacher_outputs=x, cg=False)
    assert result.logits.dtype == torch.bfloat16
    assert tuple(out.shape) == (1, 6)


if __name__ == "__main__":
    print("silence off, cpu diff:", _run("cpu", False))
    print("silence on,  cpu diff:", _run("cpu", True))
