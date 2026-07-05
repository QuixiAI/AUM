"""Deterministic SYN-1B rule registries."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

from train.syn import N_RULES, TRAIN_RULE_CUTOFF
from train.syn.alphabet import stable_seed


@dataclass(frozen=True)
class Rule:
    family: str
    rule_id: int
    kind: str
    data: dict


def split_for(rule_id: int) -> str:
    return "train" if rule_id < TRAIN_RULE_CUTOFF else "eval"


def _perm(rng, n):
    vals = list(range(n))
    while True:
        rng.shuffle(vals)
        if all(i != v for i, v in enumerate(vals)):
            return vals


def _f1_rule(rule_id, rng):
    return Rule("F1", rule_id, rng.choice(["swap", "cycle", "mirror", "derange"]),
                {"perm": _perm(rng, 8)})


def _f2_rule(rule_id, rng):
    k = rng.randint(4, 8)
    return Rule("F2", rule_id, "binding", {"k": k, "assignment": _perm(rng, k)})


def _f3_rule(rule_id, rng):
    return Rule("F3", rule_id, "kv", {"m": rng.randint(3, 8)})


def _f4_rule(rule_id, rng):
    return Rule("F4", rule_id, rng.choices(["F1", "F2", "F3"], [40, 35, 25])[0], {})


def _f5_rule(rule_id, rng):
    m = rng.choice([11, 13, 17])
    a = rng.randrange(2, m)
    b = rng.randrange(0, m)
    return Rule("F5", rule_id, "mod", {"m": m, "a": a, "b": b})


_BUILDERS = {"F1": _f1_rule, "F2": _f2_rule, "F3": _f3_rule, "F4": _f4_rule, "F5": _f5_rule}


class Registry:
    def __init__(self, seed: int = 1337, corpus_version: str = "syn-1b-v1.1"):
        self.seed = seed
        self.corpus_version = corpus_version
        self.rules = {}
        for family, builder in _BUILDERS.items():
            self.rules[family] = []
            for rule_id in range(N_RULES):
                rng = random.Random(stable_seed(corpus_version, "registry", family, rule_id, seed))
                self.rules[family].append(builder(rule_id, rng))

    def rule_for(self, family: str, rule_id: int) -> Rule:
        return self.rules[family][rule_id]

    def ids_for_split(self, split: str):
        return range(0, TRAIN_RULE_CUTOFF) if split == "train" else range(TRAIN_RULE_CUTOFF, N_RULES)

    def hash(self) -> str:
        payload = {f: [asdict(r) for r in rs] for f, rs in self.rules.items()}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

