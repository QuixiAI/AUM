#!/usr/bin/env python
"""SmolLM2 tokenizer for AUM-Ø (vocab 49152, byte-level BPE) — load + verify against the config.

AUM-Ø.md fixes ``vocab_size = 49152`` (``embed_tokens [49152, 512]``). That IS the SmolLM2 /
StarCoder2 byte-level BPE vocabulary, so we use it as-is: exact match (no dead embedding rows, the
tied 25M table stays right-sized), no UNKs, handles code + unicode, HF-native fast tokenizer.

    from train.tokenizer import load_tokenizer
    tok = load_tokenizer()                       # HuggingFaceTB/SmolLM2-135M ; len(tok) == 49152

CLI:
    python train/tokenizer.py --config train/checkpoints/aum-tiny-v5.3-init/config.json

Requires ``transformers`` (+ ``tokenizers``) at run time — install on the training box.
"""

import argparse
import json

DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"   # 49152-vocab byte-level BPE (== StarCoder2)


def load_tokenizer(name: str = DEFAULT_TOKENIZER, set_pad: bool = True):
    """Load the HF fast tokenizer. Defines pad_token = eos_token if unset (packed training never
    pads, but keeps a valid pad id for any batched/eval path)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - environment dependent
        raise SystemExit("transformers is required: pip install transformers tokenizers") from e
    tok = AutoTokenizer.from_pretrained(name)
    if set_pad and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenizer_info(tok) -> dict:
    return {
        "name_or_path": getattr(tok, "name_or_path", None),
        "len": len(tok),                     # ids the model must be able to embed
        "vocab_size": tok.vocab_size,        # base vocab (may exclude added specials)
        "bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id,
        "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
    }


def verify(tok, vocab_size: int):
    """The model embedding must cover every token id: len(tok) <= vocab_size. Returns (exact, n)."""
    n = len(tok)
    if n > vocab_size:
        raise SystemExit(
            f"tokenizer has {n} tokens > model vocab_size {vocab_size}: token ids would index past "
            f"the [{vocab_size}, d] embedding. Use a {vocab_size}-vocab tokenizer or grow the model.")
    return n == vocab_size, n


def main():
    ap = argparse.ArgumentParser(description="Load + verify the AUM-Ø tokenizer.")
    ap.add_argument("--name", default=DEFAULT_TOKENIZER)
    ap.add_argument("--config", default=None, help="model config.json to read vocab_size from")
    ap.add_argument("--vocab-size", type=int, default=49152, help="used if --config is not given")
    args = ap.parse_args()

    vocab_size = args.vocab_size
    if args.config:
        vocab_size = json.load(open(args.config))["vocab_size"]

    tok = load_tokenizer(args.name)
    print(json.dumps(tokenizer_info(tok), indent=2))
    exact, n = verify(tok, vocab_size)
    print(f"vocab check: tokenizer len={n} vs model vocab_size={vocab_size} -> "
          + ("EXACT match" if exact else f"fits, {vocab_size - n} spare embedding rows"))

    sample = "AUM-Ø learns when to stay silent."
    ids = tok(sample)["input_ids"]
    print(f"sample: {sample!r} -> {ids[:16]}{' ...' if len(ids) > 16 else ''}")
    print(f"decode: {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
