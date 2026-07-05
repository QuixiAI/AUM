#!/usr/bin/env python3
"""Generate text from an AUM checkpoint."""

import argparse

import torch
from transformers import TextStreamer

from aum_ssm.models.aum_lm import AumLMHeadModel
from train.tokenizer import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from an AUM checkpoint.")
    parser.add_argument(
        "--ckpt",
        default="train/checkpoints/aum-tiny-v6-cuda-1b/step-002036",
        help="Checkpoint directory containing config.json and pytorch_model.bin.",
    )
    parser.add_argument("--prompt", default="Once upon a time", help="Prompt text.")
    parser.add_argument("--new-tokens", type=int, default=128, help="Number of tokens to generate.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling; use 1 for greedy.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling threshold.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Device to run generation on.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="Model dtype. bf16 is the default on CUDA.",
    )
    return parser.parse_args()


def resolve_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    if args.device == "cpu" and args.dtype != "fp32":
        print("CPU generation uses fp32; ignoring --dtype.")
        dtype = torch.float32
    else:
        dtype = resolve_dtype(args.dtype)

    tok = load_tokenizer()
    model = AumLMHeadModel.from_pretrained(args.ckpt, device=args.device, dtype=dtype).eval()

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    max_length = input_ids.shape[1] + args.new_tokens
    streamer = TextStreamer(tok, skip_special_tokens=True)

    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            max_length=max_length,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            eos_token_id=tok.eos_token_id,
            cg=False,
            streamer=streamer,
        )


if __name__ == "__main__":
    main()
