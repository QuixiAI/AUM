#!/usr/bin/env python
"""Tokenize a corpus into packed uint16 token shards for AUM-Ø training.

Reads documents (a HuggingFace dataset id, or local ``.txt`` / ``.jsonl`` files), tokenizes them
with the AUM-Ø tokenizer (SmolLM2, vocab 49152 -> fits uint16), separates documents with the EOS
id, and writes a flat token stream split into fixed-size ``<split>_NNNNN.bin`` shards plus a
``manifest.json``. Flat packing (no fixed context baked in) lets the training loader sample any
``seq_len`` window; ``--seq-len`` here is only a manifest hint / sequence-count estimate.

    # HuggingFace streaming corpus
    python train/prepare_data.py --source HuggingFaceFW/fineweb-edu --split train \
        --streaming --out-dir train/data/fineweb-edu --shard-size-tokens 100_000_000

    # local files (one document per .txt line; jsonl reads --text-column)
    python train/prepare_data.py --source 'corpus/*.jsonl' --out-dir train/data/mine --val-fraction 0.01

    # validate the packing/sharding mechanics with no external deps
    python train/prepare_data.py --self-test

A shard reads back as:  np.memmap(path, dtype=np.uint16, mode="r")  -> a 1-D token stream.
Requires ``transformers`` (always) and ``datasets`` (only for HF sources) at run time.
"""

import argparse
import gzip
import glob
import json
import os
import sys
import time

import numpy as np

# Make `train.tokenizer` importable when run as `python train/prepare_data.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DTYPE = np.uint16          # vocab 49152 < 65536; token ids (and eos) fit in uint16
_MAXID = np.iinfo(DTYPE).max


# --------------------------------------------------------------------------- documents
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def iter_local(paths, text_column):
    """Yield strings from local files: each non-empty line of .txt; obj[text_column] of .jsonl."""
    for path in paths:
        is_jsonl = ".jsonl" in path or ".ndjson" in path
        with _open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line).get(text_column, "") if is_jsonl else line


def iter_hf(source, split, text_column, streaming):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("datasets is required for HF sources: pip install datasets") from e
    ds = load_dataset(source, split=split, streaming=streaming)
    for ex in ds:
        yield ex[text_column]


def iter_documents(source, split, text_column, streaming, limit):
    matches = glob.glob(source)
    it = iter_local(sorted(matches), text_column) if matches else \
        iter_hf(source, split, text_column, streaming)
    for i, doc in enumerate(it):
        if limit and i >= limit:
            return
        if doc:
            yield doc


# --------------------------------------------------------------------------- sharding
class ShardWriter:
    """Accumulate token arrays and flush flat uint16 shards of ~shard_size tokens each."""

    def __init__(self, out_dir, split, shard_size):
        self.out_dir, self.split, self.shard_size = out_dir, split, shard_size
        self.buf, self.buf_len, self.shards, self.total = [], 0, [], 0

    def add(self, arr):
        self.buf.append(arr)
        self.buf_len += arr.size
        self.total += arr.size
        if self.buf_len >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buf_len:
            return
        arr = np.concatenate(self.buf).astype(DTYPE)
        name = f"{self.split}_{len(self.shards):05d}.bin"
        arr.tofile(os.path.join(self.out_dir, name))
        self.shards.append({"name": name, "n_tokens": int(arr.size)})
        self.buf, self.buf_len = [], 0

    def close(self):
        self._flush()


def pack(doc_iter, encode_fn, eos_id, writer_for, batch_docs):
    """Tokenize documents (batched), append EOS to each, and route them to a ShardWriter."""
    idx_buf, txt_buf, n_docs = [], [], 0

    def flush():
        nonlocal n_docs
        if not txt_buf:
            return
        for i, ids in zip(idx_buf, encode_fn(txt_buf)):
            if not len(ids):
                continue
            a = np.asarray(ids, dtype=np.int64)
            if int(a.max()) >= _MAXID:
                raise ValueError(f"token id {int(a.max())} does not fit uint16")
            a = np.append(a, eos_id)                     # EOS document separator
            writer_for(i).add(a.astype(DTYPE))
            n_docs += 1
        idx_buf.clear(); txt_buf.clear()

    for i, doc in enumerate(doc_iter):
        idx_buf.append(i); txt_buf.append(doc)
        if len(txt_buf) >= batch_docs:
            flush()
    flush()
    return n_docs


def write_manifest(out_dir, tokenizer, vocab_size, eos_id, seq_len, source, writers):
    splits = {}
    for split, w in writers.items():
        splits[split] = {"shards": w.shards, "total_tokens": w.total,
                         "approx_sequences": w.total // seq_len if seq_len else None}
    manifest = {
        "tokenizer": tokenizer, "vocab_size": vocab_size, "dtype": "uint16",
        "eos_id": int(eos_id), "seq_len": seq_len, "source": source,
        "splits": splits, "created_unix": int(time.time()),
        "read_hint": "np.memmap(<out_dir>/<shard>, dtype=np.uint16, mode='r') -> 1-D token stream",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# --------------------------------------------------------------------------- driver
def run(args):
    from train.tokenizer import load_tokenizer, verify  # local import: only needed for real runs

    tok = load_tokenizer(args.tokenizer)
    vocab_size = args.vocab_size
    if args.config:
        vocab_size = json.load(open(args.config))["vocab_size"]
    verify(tok, vocab_size)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id

    def encode_fn(texts):
        return tok(texts, add_special_tokens=False)["input_ids"]

    os.makedirs(args.out_dir, exist_ok=True)
    writers = {"train": ShardWriter(args.out_dir, "train", args.shard_size_tokens)}
    if args.val_fraction > 0:
        writers["val"] = ShardWriter(args.out_dir, "val", args.shard_size_tokens)
        every = max(2, round(1.0 / args.val_fraction))
        writer_for = lambda i: writers["val" if i % every == 0 else "train"]
    else:
        writer_for = lambda _: writers["train"]

    docs = iter_documents(args.source, args.split, args.text_column, args.streaming, args.limit_docs)
    n_docs = pack(docs, encode_fn, eos_id, writer_for, args.batch_docs)
    for w in writers.values():
        w.close()
    m = write_manifest(args.out_dir, args.tokenizer, vocab_size, eos_id, args.seq_len,
                       args.source, writers)
    print(f"tokenized {n_docs} docs -> {args.out_dir}")
    for split, s in m["splits"].items():
        print(f"  {split:5s}: {s['total_tokens']:>14,} tokens  "
              f"~{s['approx_sequences']:>10,} x {args.seq_len}-seqs  ({len(s['shards'])} shards)")


def self_test():
    """Exercise pack/shard/manifest with a synthetic corpus + fake tokenizer (numpy only)."""
    import shutil
    import zlib

    vocab, eos = 49152, 0
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest_shards")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    docs = [("token sample document number %d " % d) * (3 + d % 7) for d in range(500)]
    enc = lambda texts: [[1 + zlib.crc32(w.encode()) % (vocab - 1) for w in t.split()] for t in texts]

    writers = {"train": ShardWriter(tmp, "train", shard_size=4000),
               "val": ShardWriter(tmp, "val", shard_size=4000)}
    every = 20
    n = pack(iter(docs), enc, eos, lambda i: writers["val" if i % every == 0 else "train"], batch_docs=64)
    for w in writers.values():
        w.close()
    m = write_manifest(tmp, "self-test-fake", vocab, eos, seq_len=128, source="synthetic", writers=writers)

    # checks: token accounting, uint16 round-trip, id range, EOS present, window extraction
    for split, w in writers.items():
        counted = sum(s["n_tokens"] for s in w.shards)
        assert counted == w.total, (split, counted, w.total)
        stream = np.concatenate([np.memmap(os.path.join(tmp, s["name"]), dtype=DTYPE, mode="r")
                                 for s in w.shards]) if w.shards else np.array([], DTYPE)
        assert stream.size == w.total
        assert stream.max(initial=0) < vocab and (stream == eos).any()
        win = stream[:128]
        assert win.size == 128 and win.dtype == DTYPE
    assert m["splits"]["train"]["total_tokens"] > m["splits"]["val"]["total_tokens"] > 0
    assert n == 500
    shutil.rmtree(tmp)
    print(f"self-test OK: packed {n} synthetic docs -> "
          f"train {writers['train'].total} + val {writers['val'].total} tokens, "
          f"uint16 shards round-trip, EOS-separated, windows readable")


def main():
    ap = argparse.ArgumentParser(description="Tokenize a corpus into packed uint16 shards.")
    ap.add_argument("--source", help="HF dataset id OR a local path/glob (.txt/.jsonl[.gz])")
    ap.add_argument("--out-dir", default=os.path.join("train", "data", "corpus"))
    ap.add_argument("--split", default="train", help="HF split name")
    ap.add_argument("--text-column", default="text", help="jsonl/HF field holding the document text")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--config", default=None, help="model config.json to verify vocab against")
    ap.add_argument("--vocab-size", type=int, default=49152)
    ap.add_argument("--streaming", action="store_true", help="stream the HF dataset (no full download)")
    ap.add_argument("--shard-size-tokens", type=int, default=100_000_000)
    ap.add_argument("--batch-docs", type=int, default=1000, help="documents per tokenizer batch")
    ap.add_argument("--val-fraction", type=float, default=0.0, help="hold out ~this fraction as val")
    ap.add_argument("--seq-len", type=int, default=2048, help="manifest hint / sequence-count estimate")
    ap.add_argument("--limit-docs", type=int, default=0, help="cap document count (0 = all)")
    ap.add_argument("--self-test", action="store_true", help="validate mechanics with synthetic data")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.source:
        ap.error("--source is required (or pass --self-test)")
    run(args)


if __name__ == "__main__":
    main()
