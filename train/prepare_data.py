#!/usr/bin/env python
"""Tokenize the AUM-Ø training corpus into packed uint16 token shards.

Default mode reads the HuggingFace dataset list from ``train/datasets`` (one ``org/repo`` or
``org/repo:config`` per line) and STREAMS documents from each until a per-dataset TOKEN budget
is met: ``--chunks-per-dataset`` (default 15 000) training chunks of ``--seq-len`` (default
4096) tokens, i.e. ~61M tokens per source regardless of how long its documents are. Streaming +
the budget means only the needed samples are ever fetched, never a full dataset. Documents are
tokenized with the AUM-Ø tokenizer (SmolLM2, vocab 49152 -> fits uint16), separated by the EOS
id, and written as a flat token stream into fixed-size ``<split>_NNNNN.bin`` shards + a
``manifest.json`` under ``train/data/`` (gitignored). Flat packing (no fixed context baked in)
lets the training loader sample any ``seq_len`` window.

The corpus is ENGLISH-ONLY by default, enforced twice: per-language repos must have their config
pinned in the datasets file (``org/repo:eng_Latn`` — auto-resolution accepts a match against the
English preferences but REFUSES to fall back to an arbitrary config), and any example carrying a
language-ish field (``language``/``lang``/...) that is not English is skipped (``--language all``
disables; skips are counted and printed, never silent). The text field is auto-detected
(``text``/``content``/first string column) and the split falls back to the first available one.

Chunking note: shards are a FLAT EOS-separated token stream — documents are NOT pre-chunked
here. The training loader (train/train.py) reads non-overlapping ``seq_len``-token windows, so a
book-length document is automatically split into consecutive 4k chunks without injecting fake
document boundaries inside it; ``approx_sequences`` in the manifest counts those chunks.

For faster downloads: ``pip install "huggingface_hub[hf_transfer]"`` — when the hf_transfer
package is importable, HF_HUB_ENABLE_HF_TRANSFER=1 is set automatically.

    # the default corpus: 15k samples from every dataset in train/datasets -> train/data/
    python train/prepare_data.py

    # a single HF dataset or local files instead
    python train/prepare_data.py --source HuggingFaceFW/finepdfs-edu:eng_Latn --out-dir train/data/finepdfs-edu
    python train/prepare_data.py --source 'corpus/*.jsonl' --out-dir train/data/mine --val-fraction 0.01

    # validate the packing/sharding mechanics with no external deps
    python train/prepare_data.py --self-test

A shard reads back as:  np.memmap(path, dtype=np.uint16, mode="r")  -> a 1-D token stream.
Requires ``transformers`` + ``datasets`` at run time (pip install "aum_ssm[data]").
"""

import argparse
import gzip
import glob
import itertools
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
    """Yield strings from local files: each non-empty line of .txt; the text field of .jsonl
    (explicit --text-column, else auto-detected)."""
    for path in paths:
        is_jsonl = ".jsonl" in path or ".ndjson" in path
        with _open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield _extract_text(json.loads(line), text_column) if is_jsonl else line


# Config / text-field preferences for heterogeneous HF repos (finepdfs-edu and friends are
# per-language configs; books/phrase corpora use different text keys).
_PREFERRED_CONFIGS = ("eng_Latn", "en", "eng", "english", "default", "sample-10BT")
_TEXT_KEYS = ("text", "content", "markdown", "raw_content", "document", "chapter")
# Example-level language metadata (fineweb: language="en"; finepdfs: eng_Latn; ...).
_LANG_KEYS = ("language", "lang", "language_code", "language_script")


def _require_datasets():
    try:
        import datasets  # noqa: F401
        return datasets
    except ImportError as e:  # pragma: no cover
        raise SystemExit('datasets is required for HF sources: pip install "aum_ssm[data]"') from e


def _enable_hf_transfer():
    """Turn on hf_transfer downloads when the package is present (no-op otherwise)."""
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass


def _lang_ok(example, language):
    """True unless the example carries a language-ish field that contradicts --language.
    Examples with no language metadata always pass (config pinning covers those repos)."""
    if language in (None, "", "all"):
        return True
    for k in _LANG_KEYS:
        v = example.get(k)
        if isinstance(v, str) and v:
            return v.lower().replace("-", "_").startswith(language.lower())
    return True


def _load_stream(name, split="train", config=None):
    """load_dataset with config + split auto-resolution, streaming. Returns (iterable, config).

    Streaming + the caller's sample limit means only the requested documents are ever fetched —
    no dataset is downloaded in full.
    """
    hf = _require_datasets()

    def _load(config):
        try:
            return hf.load_dataset(name, config, split=split, streaming=True)
        except ValueError as e:
            if "split" not in str(e).lower():
                raise
            avail = hf.get_dataset_split_names(name, config)
            pick = next((s for s in avail if "train" in s), avail[0])
            print(f"  [{name}] split {split!r} missing; using {pick!r} (available: {avail})")
            return hf.load_dataset(name, config, split=pick, streaming=True)

    if config is not None:                                  # explicit repo:config — no guessing
        return _load(config), config
    try:
        return _load(None), None
    except ValueError as e:
        if "Config name is missing" not in str(e) and "BuilderConfig" not in str(e):
            raise
    configs = hf.get_dataset_config_names(name)
    chosen = next((p for p in _PREFERRED_CONFIGS if p in configs), None)
    if chosen is None:
        # English-only corpus: never guess a language config. An arbitrary fallback silently
        # fills the corpus with another language — hard-fail and ask for a pin instead.
        raise SystemExit(
            f"[{name}] has {len(configs)} configs and none matched the English preferences "
            f"{_PREFERRED_CONFIGS}. Pin one explicitly as '{name}:<config>' in the datasets "
            f"file (e.g. from: {configs[:8]}{' ...' if len(configs) > 8 else ''}).")
    print(f"  [{name}] {len(configs)} configs; using {chosen!r}")
    return _load(chosen), chosen


def _extract_text(example, text_column=None):
    """The document string of a heterogeneous example: the given column, a known key, or the
    first (longest) string field."""
    if text_column and isinstance(example.get(text_column), str):
        return example[text_column]
    for k in _TEXT_KEYS:
        if isinstance(example.get(k), str):
            return example[k]
    strings = [v for v in example.values() if isinstance(v, str)]
    return max(strings, key=len) if strings else ""


def iter_hf(source, split, text_column, limit=0, config=None, language="en"):
    """Stream up to `limit` documents from a HF dataset. Streaming + the limit means only the
    requested samples are fetched — never the full dataset. Examples whose language metadata
    contradicts `language` are skipped BEFORE counting toward the limit (skips are reported)."""
    ds, _ = _load_stream(source, split, config)
    n = skipped = 0
    for ex in ds:
        if not _lang_ok(ex, language):
            skipped += 1
            continue
        yield _extract_text(ex, text_column)
        n += 1
        if limit and n >= limit:
            break
    if skipped:
        print(f"  [{source}] skipped {skipped:,} non-{language} documents (language filter)")


def iter_documents(source, split, text_column, limit, language="en"):
    matches = glob.glob(source)
    if matches:
        it = iter_local(sorted(matches), text_column)
    else:
        name, _, config = source.partition(":")
        it = iter_hf(name, split, text_column, limit, config=config or None, language=language)
    for i, doc in enumerate(it):
        if limit and i >= limit:
            return
        if doc:
            yield doc


def read_dataset_list(path):
    """One dataset per line: `org/repo` or `org/repo:config`. Blanks and #-comments skipped.
    Returns [(name, config_or_None)]."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            name, _, config = line.partition(":")
            entries.append((name, config or None))
    return entries


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


def pack(doc_iter, encode_fn, eos_id, writer_for, batch_docs, token_budget=0, progress=None):
    """Tokenize documents (batched), append EOS to each, and route them to a ShardWriter.

    token_budget > 0 stops consuming the iterator once that many tokens have been written for
    THIS source (checked after each flush; a flush is also forced when the buffered text grows
    large, so the overshoot stays small even for book-sized documents). progress, if given, is
    called with the token count of each flush (e.g. tqdm.update).
    """
    idx_buf, txt_buf, buf_chars, n_docs, n_tokens = [], [], 0, 0, 0
    max_buf_chars = 16_000_000                           # ~4M tokens per flush at ~4 chars/token

    def flush():
        nonlocal n_docs, n_tokens, buf_chars
        if not txt_buf:
            return
        before = n_tokens
        for i, ids in zip(idx_buf, encode_fn(txt_buf)):
            if not len(ids):
                continue
            a = np.asarray(ids, dtype=np.int64)
            if int(a.max()) >= _MAXID:
                raise ValueError(f"token id {int(a.max())} does not fit uint16")
            a = np.append(a, eos_id)                     # EOS document separator
            writer_for(i).add(a.astype(DTYPE))
            n_docs += 1
            n_tokens += a.size
        idx_buf.clear(); txt_buf.clear(); buf_chars = 0
        if progress is not None:
            progress(n_tokens - before)

    for i, doc in enumerate(doc_iter):
        idx_buf.append(i); txt_buf.append(doc); buf_chars += len(doc)
        if len(txt_buf) >= batch_docs or buf_chars >= max_buf_chars:
            flush()
            if token_budget and n_tokens >= token_budget:
                break
    flush()
    return n_docs, n_tokens


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
def _setup(args):
    """Tokenizer + vocab check + writers + the val-striping selector, shared by both modes."""
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
        counter = itertools.count()                      # GLOBAL stripe — never restarts per
        writer_for = lambda _i: writers["val" if next(counter) % every == 0 else "train"]  # source
    else:
        writer_for = lambda _i: writers["train"]
    return vocab_size, eos_id, encode_fn, writers, writer_for


def _finish(args, vocab_size, eos_id, writers, source, n_docs):
    for w in writers.values():
        w.close()
    m = write_manifest(args.out_dir, args.tokenizer, vocab_size, eos_id, args.seq_len,
                       source, writers)
    print(f"tokenized {n_docs:,} docs -> {args.out_dir}")
    for split, s in m["splits"].items():
        print(f"  {split:5s}: {s['total_tokens']:>14,} tokens  "
              f"~{s['approx_sequences']:>10,} x {args.seq_len}-seqs  ({len(s['shards'])} shards)")


def run(args):
    """Single-source mode: one HF dataset id (optionally `:config`) or a local path/glob."""
    vocab_size, eos_id, encode_fn, writers, writer_for = _setup(args)
    budget = args.chunks_per_dataset * args.seq_len if args.chunks_per_dataset else 0
    docs = iter_documents(args.source, args.split, args.text_column, args.limit_docs,
                          language=args.language)
    n_docs, _ = pack(docs, encode_fn, eos_id, writer_for, args.batch_docs, token_budget=budget)
    _finish(args, vocab_size, eos_id, writers, args.source, n_docs)


def run_mix(args):
    """Default mode: a TOKEN budget from every HF dataset listed in --datasets-file — 15 000
    seq_len-token training chunks each (docs are streamed until the budget is met or the stream
    ends) — one shard stream, with per-source accounting in the manifest."""
    names = read_dataset_list(args.datasets_file)
    if not names:
        raise SystemExit(f"no datasets listed in {args.datasets_file}")
    vocab_size, eos_id, encode_fn, writers, writer_for = _setup(args)
    budget = args.chunks_per_dataset * args.seq_len
    print(f"corpus: {args.chunks_per_dataset:,} x {args.seq_len}-token chunks "
          f"(~{budget / 1e6:.0f}M tokens) from each of {len(names)} datasets "
          f"({args.datasets_file}) -> {args.out_dir}")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    per_source, n_total = {}, 0
    for si, (name, config) in enumerate(names):
        label = f"{name}:{config}" if config else name
        print(f"[{si + 1}/{len(names)}] {label}")
        before = {s: w.total for s, w in writers.items()}
        docs = iter_hf(name, args.split, args.text_column,
                       limit=args.samples_per_dataset, config=config, language=args.language)
        bar = tqdm(total=budget, unit="tok", unit_scale=True, desc=label,
                   dynamic_ncols=True) if tqdm is not None else None
        n, n_tok = pack((d for d in docs if d), encode_fn, eos_id, writer_for, args.batch_docs,
                        token_budget=budget, progress=bar.update if bar else None)
        if bar:
            bar.close()
        tokens = {s: writers[s].total - before[s] for s in writers}
        per_source[label] = {"documents": n, "tokens": tokens}
        n_total += n
        short = f"  (stream ended below the {budget / 1e6:.0f}M-token budget)" \
            if budget and n_tok < budget else ""
        print(f"  {n:,} docs, {sum(tokens.values()):,} tokens{short}")

    source = {"datasets_file": args.datasets_file, "chunks_per_dataset": args.chunks_per_dataset,
              "token_budget_per_dataset": budget, "per_source": per_source}
    _finish(args, vocab_size, eos_id, writers, source, n_total)


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
    n, n_tok = pack(iter(docs), enc, eos, lambda i: writers["val" if i % every == 0 else "train"],
                    batch_docs=64)
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
    assert n == 500 and n_tok == sum(w.total for w in writers.values())

    # token-budget stop: a small budget must halt the stream early (post-flush check)
    wb = ShardWriter(tmp, "budget", shard_size=4000)
    nb, tb = pack(iter(docs), enc, eos, lambda i: wb, batch_docs=8, token_budget=500)
    wb.close()
    assert nb < 500 and tb >= 500, (nb, tb)
    shutil.rmtree(tmp)
    print(f"self-test OK: packed {n} synthetic docs -> "
          f"train {writers['train'].total} + val {writers['val'].total} tokens, "
          f"uint16 shards round-trip, EOS-separated, windows readable")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Tokenize the AUM-Ø corpus into packed uint16 shards.")
    ap.add_argument("--source", default=None,
                    help="a single HF dataset id OR a local path/glob (.txt/.jsonl[.gz]); "
                         "omit to use --datasets-file")
    ap.add_argument("--datasets-file", default=os.path.join(here, "datasets"),
                    help="file listing HF dataset ids, one per line (default mode)")
    ap.add_argument("--chunks-per-dataset", type=int, default=15_000,
                    help="TOKEN budget per dataset, expressed in seq-len-token training chunks "
                         "(default: 15000 x 4096 ~ 61M tokens each; 0 = no budget)")
    ap.add_argument("--samples-per-dataset", type=int, default=0,
                    help="optional additional DOCUMENT cap per dataset (0 = uncapped; the token "
                         "budget is what stops a stream)")
    ap.add_argument("--out-dir", default=os.path.join(here, "data"))
    ap.add_argument("--split", default="train", help="HF split name (auto-falls back)")
    ap.add_argument("--text-column", default=None,
                    help="document text field (default: auto-detect text/content/...)")
    ap.add_argument("--language", default="en",
                    help="skip examples whose language metadata does not match this prefix "
                         "(en matches en/eng/eng_Latn/english; 'all' disables the filter)")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--config", default=None, help="model config.json to verify vocab against")
    ap.add_argument("--vocab-size", type=int, default=49152)
    ap.add_argument("--shard-size-tokens", type=int, default=100_000_000)
    ap.add_argument("--batch-docs", type=int, default=1000, help="documents per tokenizer batch")
    ap.add_argument("--val-fraction", type=float, default=0.01, help="hold out ~this fraction as val")
    ap.add_argument("--seq-len", type=int, default=4096, help="manifest hint / sequence-count estimate")
    ap.add_argument("--limit-docs", type=int, default=0, help="single-source mode: cap docs (0 = all)")
    ap.add_argument("--self-test", action="store_true", help="validate mechanics with synthetic data")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    _enable_hf_transfer()
    if args.source:
        run(args)
    else:
        run_mix(args)


if __name__ == "__main__":
    main()
