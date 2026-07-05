import gzip
import hashlib
import json
import os

import pytest

from train.syn.alphabet import MODEL_TOKENIZER, build_alphabet
from train.syn.harness_readers import open_jsonl, prediction_pos_for_answer
from train.syn.pack import make_instance, run_generation
from train.syn.qa import run_qa
from train.syn.registry import Registry, split_for
from train.train import PackedWindows


@pytest.fixture(scope="module")
def alpha():
    transformers = pytest.importorskip("transformers")
    from train.tokenizer import load_tokenizer
    return build_alphabet(load_tokenizer(MODEL_TOKENIZER), seed=11)


def test_registry_splits_and_hash_stable():
    r1 = Registry(seed=7)
    r2 = Registry(seed=7)
    assert len(r1.rules["F1"]) == 512
    assert split_for(409) == "train"
    assert split_for(410) == "eval"
    assert r1.hash() == r2.hash()


def test_family_generators_emit_valid_sidecars(alpha):
    reg = Registry(seed=3)
    for fam in ("F1", "F2", "F3", "F4", "F5"):
        split = "eval" if fam == "F5" else "train"
        ids, rec = make_instance(alpha, reg, fam, split, 0, seed=3, target_len=384)
        assert 96 <= len(ids) <= 384
        assert rec["family"] == fam
        assert rec["token_hash"]
        assert rec["task_token_count"] + rec["filler_token_count"] == len(ids)
        assert rec["task_fraction"] >= 0.25
        assert all(0 <= x < 65536 for x in ids)
        for q in rec.get("queries", []):
            assert prediction_pos_for_answer(q) == q["answer_pos"] - 1


def test_offset_contract(alpha):
    reg = Registry(seed=13)
    for fam in ("F1", "F2", "F3", "F4", "F5"):
        split = "eval" if fam == "F5" else "train"
        ids, rec = make_instance(alpha, reg, fam, split, 7, seed=13, target_len=1536)
        for lab in rec["label_positions"]:
            decoded = alpha.tokenizer.decode([ids[lab["pos"]]], clean_up_tokenization_spaces=False)
            assert decoded.strip() == lab["expected"], (fam, lab, decoded)


def test_tiny_pack_qa_and_packed_windows(tmp_path, alpha):
    reg = Registry(seed=5)
    summary = run_generation(
        alpha, reg, str(tmp_path), MODEL_TOKENIZER, 49152,
        train_tokens=80_000, eval_tokens=32_768, shard_size_tokens=65_536, seed=5,
    )
    report = run_qa(str(tmp_path), alpha)
    assert report["ok"], json.dumps(report, indent=2)
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["manifest_tokenizer"]["ok"]
    assert checks["packing_policy"]["ok"]
    assert checks["task_density_train"]["ok"]
    assert checks["task_density_eval"]["ok"]
    assert checks["packed_window_roundtrip_train"]["ok"]
    assert checks["packed_window_roundtrip_eval"]["ok"]
    assert checks["window_background_scaffolding_train"]["bad_counts"] == {}
    assert checks["window_background_scaffolding_eval"]["bad_counts"] == {}
    assert checks["query_position_contract"]["ok"]
    ds = PackedWindows(str(tmp_path), "train", 4096)
    assert len(ds) > 0
    assert ds[0].shape[0] == 4096
    recs = list(open_jsonl(os.path.join(tmp_path, "sidecars", "train.jsonl.gz")))
    assert recs
    assert max(r["start_offset"] + r["token_len"] for r in recs) <= 4096
    assert summary["splits"]["train"]["sidecar_records"] == len(recs)


def _corpus_fingerprint(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    parts = []
    for split in ("train", "eval"):
        for shard in manifest["splits"][split]["shards"]:
            path = os.path.join(out_dir, shard["name"])
            with open(path, "rb") as f:
                parts.append((split, shard["name"], hashlib.sha256(f.read()).hexdigest()))
        sidecar_path = os.path.join(out_dir, "sidecars", f"{split}.jsonl.gz")
        with gzip.open(sidecar_path, "rt", encoding="utf-8") as f:
            parts.append((split, "sidecar", hashlib.sha256(f.read().encode()).hexdigest()))
    return parts


def test_generation_is_deterministic(tmp_path, alpha):
    for name in ("a", "b"):
        reg = Registry(seed=17)
        run_generation(
            alpha, reg, str(tmp_path / name), MODEL_TOKENIZER, 49152,
            train_tokens=40_000, eval_tokens=24_576, shard_size_tokens=65_536, seed=17,
        )
    assert _corpus_fingerprint(tmp_path / "a") == _corpus_fingerprint(tmp_path / "b")
