"""Tokenizer-verified alphabet, scaffolding, paraphrases, and BG-v1 filler."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from train.syn import CORPUS_VERSION

MODEL_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
MODEL_VOCAB_SIZE = 49152

WORD_CANDIDATES = (
    "red", "blue", "green", "yellow", "orange", "purple", "black", "white",
    "silver", "golden", "brown", "gray", "pink", "stone", "chair", "table",
    "river", "cloud", "field", "metal", "paper", "glass", "wood", "apple",
    "berry", "ocean", "wheat", "amber", "earth", "house", "iron", "magic",
    "night", "voice", "world", "dog", "cat", "horse", "bird", "fish",
    "whale", "tiger", "lion", "bear", "fox", "wolf", "sheep", "goat",
    "mouse", "rabbit", "duck", "eagle", "shark", "turtle", "snake", "frog",
    "tree", "flower", "grass", "leaf", "root", "seed", "bread", "cake",
    "honey", "lemon", "peach", "pear", "grape", "olive", "onion", "basil",
    "cedar", "maple", "pine", "oak", "boat", "train", "plane", "truck",
    "wheel", "clock", "phone", "book", "pen", "pencil", "cup", "bowl",
    "plate", "spoon", "fork", "knife", "door", "window", "bridge", "garden",
    "forest", "island", "mountain", "valley", "desert", "beach", "castle",
    "cabin", "basket", "blanket", "candle", "carpet", "bottle", "button",
    "camera", "guitar", "piano", "radio", "mirror", "pillow", "pocket",
    "rocket", "saddle", "bucket", "ladder", "marble", "magnet", "needle",
    "planet", "quartz", "ribbon", "shield", "ticket", "tunnel", "velvet",
    "wagon", "cotton", "coffee", "pepper", "sugar", "salt", "crown",
    "pearl", "shell", "coin", "rope", "chain", "bell", "drum", "flute",
    "flag", "kite", "lamp", "nail", "paint", "soap", "towel", "vase",
    "wire", "yarn", "zebra", "panda", "camel", "monkey", "donkey",
    "penguin", "parrot", "salmon", "coral",
)

SCAFFOLDING = (
    "rule", "now", "means", "is", "becomes", "note", "update", "query",
    "answer", ":", ".", ",", "->", "the", "a", "so", "then", "still",
    "same", "key", "box", "opens", "holds", "where", "switch", "next",
    "and", "were", "exchanged", "nothing", "was", "not",
)
STRUCTURAL_TOKENS = frozenset(SCAFFOLDING)
DIGIT_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
F5_WORDS = ("zero",) + DIGIT_WORDS + ("switch", "next", ":")

PARAPHRASES = {
    "rule": [
        ("the", "rule", ":"),
        ("rule", ":"),
        ("note", "the", "rule", ":"),
        ("now", "the", "rule", ":"),
        ("the", "same", "rule", ":"),
        ("rule", "now", ":"),
        ("note", ":", "rule"),
        ("the", "rule", "is", ":"),
    ],
    "correction": [
        ("update", ":"),
        ("note", ":"),
        ("now", ":"),
        ("then", ":"),
        ("so", "now", ":"),
        ("note", "now", ":"),
        ("update", "now", ":"),
        ("the", "rule", "now", ":"),
    ],
    "query": [
        ("query", ":"),
        ("answer", ":"),
        ("where", ":"),
        ("then", "query", ":"),
        ("now", "query", ":"),
        ("so", "answer", ":"),
        ("the", "answer", ":"),
        ("query", "now", ":"),
    ],
    "binding": [
        ("note", ":"),
        ("the", "key", ":"),
        ("the", "rule", ":"),
        ("now", ":"),
        ("same", "rule", ":"),
        ("note", "the", "key", ":"),
        ("rule", ":"),
        ("the", "key", "is", ":"),
    ],
    "distractor": [
        ("note", ":", "the", "rule", "still", "same", "."),
        ("update", ":", "still", "same", "."),
        ("same", "rule", "."),
        ("note", ":", "same", "."),
        ("the", "rule", "is", "still", "same", "."),
        ("now", ":", "still", "same", "."),
        ("update", ":", "the", "same", "rule", "."),
        ("note", ":", "the", "key", "still", "same", "."),
    ],
    "distractor_F1": [
        ("note", ":", "the", "rule", "still", "means", "same", "."),
        ("update", ":", "same", "means", "same", "."),
        ("the", "rule", "now", ":", "same", "means", "same", "."),
        ("note", "now", ":", "same", "rule", "means", "same", "."),
        ("then", ":", "the", "same", "rule", "means", "same", "."),
        ("so", "now", ":", "same", "means", "same", "."),
        ("update", "now", ":", "the", "rule", "means", "same", "."),
        ("note", ":", "rule", "means", "same", "."),
    ],
    "distractor_F2": [
        ("note", ":", "nothing", "was", "exchanged", "."),
        ("update", ":", "nothing", "was", "exchanged", "."),
        ("now", ":", "nothing", "was", "exchanged", "."),
        ("then", ":", "nothing", "was", "exchanged", "."),
        ("note", "now", ":", "same", "and", "same", "were", "not", "exchanged", "."),
        ("update", "now", ":", "same", "and", "same", "were", "not", "exchanged", "."),
        ("the", "rule", "now", ":", "nothing", "was", "exchanged", "."),
        ("so", "now", ":", "nothing", "was", "exchanged", "."),
    ],
    "distractor_F3": [
        ("update", ":", "the", "key", "still", "opens", "box", "same", "."),
        ("note", ":", "the", "key", "still", "opens", "box", "same", "."),
        ("now", ":", "the", "key", "still", "opens", "box", "same", "."),
        ("then", ":", "the", "key", "still", "opens", "box", "same", "."),
        ("so", "now", ":", "the", "key", "still", "opens", "box", "same", "."),
        ("note", "now", ":", "the", "key", "opens", "box", "same", "."),
        ("update", "now", ":", "the", "key", "opens", "box", "same", "."),
        ("the", "rule", "now", ":", "the", "key", "still", "opens", "box", "same", "."),
    ],
}


@dataclass(frozen=True)
class TokenizerIdentity:
    name_or_path: str
    tokenizer_class: str
    transformers_version: str
    vocab_size: int
    length: int
    backend_hash: str
    vocab_hash: str


@dataclass(frozen=True)
class SynAlphabet:
    tokenizer: object
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    sigma: tuple[str, ...]
    sigma_train: tuple[str, ...]
    sigma_eval: tuple[str, ...]
    bg_vocab: tuple[str, ...]
    bg_matrix: tuple[tuple[float, ...], ...]
    eos_id: int
    identity: TokenizerIdentity | None = None

    def single_id(self, token: str) -> int:
        return self.token_to_id[token]

    def variant_words(self, rng: random.Random, name: str) -> tuple[str, ...]:
        return tuple(rng.choice(PARAPHRASES[name]))

    def bg_words(self, rng: random.Random, n: int) -> list[str]:
        if n <= 0:
            return []
        idx = rng.randrange(len(self.bg_vocab))
        out = []
        for _ in range(n):
            word = self.bg_vocab[idx]
            out.append(word)
            row = self.bg_matrix[idx]
            u = rng.random()
            acc = 0.0
            for j, p in enumerate(row):
                acc += p
                if u <= acc:
                    idx = j
                    break
        return out


def stable_seed(*parts) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def tokenizer_identity(tok) -> TokenizerIdentity:
    try:
        import transformers
        version = transformers.__version__
    except Exception:
        version = "unknown"
    backend = tok.backend_tokenizer.to_str()
    vocab = getattr(tok, "get_vocab", lambda: {})()
    raw_vocab = repr(sorted(vocab.items())).encode()
    return TokenizerIdentity(
        name_or_path=getattr(tok, "name_or_path", ""),
        tokenizer_class=tok.__class__.__name__,
        transformers_version=version,
        vocab_size=int(getattr(tok, "vocab_size", -1)),
        length=len(tok),
        backend_hash=hashlib.sha256(backend.encode()).hexdigest(),
        vocab_hash=hashlib.sha256(raw_vocab).hexdigest(),
    )


def assert_model_tokenizer(tok):
    ident = tokenizer_identity(tok)
    if ident.name_or_path != MODEL_TOKENIZER or ident.vocab_size != MODEL_VOCAB_SIZE:
        raise SystemExit(
            "SYN tokenizer mismatch: "
            f"expected name={MODEL_TOKENIZER!r} vocab={MODEL_VOCAB_SIZE}, "
            f"got name={ident.name_or_path!r} vocab={ident.vocab_size}")
    return ident


def _single_leading_id(tok, word: str) -> int | None:
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return int(ids[0]) if len(ids) == 1 else None


def _validate_words(tok, words):
    bad = []
    token_to_id = {}
    for word in words:
        tid = _single_leading_id(tok, word)
        if tid is None:
            bad.append(word)
        else:
            token_to_id[word] = tid
    if bad:
        raise SystemExit(f"SYN tokenizer audit failed; not single-token in leading-space context: {bad}")
    return token_to_id


def _derive_sigma(tok, seed: int) -> tuple[str, ...]:
    reserved = set(SCAFFOLDING) | set(DIGIT_WORDS) | set(F5_WORDS)
    for bank in PARAPHRASES.values():
        for variant in bank:
            reserved.update(variant)
    valid = [w for w in WORD_CANDIDATES if w not in reserved and _single_leading_id(tok, w) is not None]
    if len(valid) < 64:
        raise SystemExit(f"SYN alphabet: only found {len(valid)} clean single-token words")
    rng = random.Random(stable_seed(CORPUS_VERSION, "alphabet", seed))
    valid = list(dict.fromkeys(valid))
    rng.shuffle(valid)
    return tuple(valid[:64])


def _derive_neutral_filler(tok, sigma: tuple[str, ...], n: int = 32) -> tuple[str, ...]:
    reserved = set(SCAFFOLDING) | set(DIGIT_WORDS) | set(F5_WORDS) | set(sigma)
    out = [w for w in WORD_CANDIDATES if w not in reserved and _single_leading_id(tok, w) is not None]
    if len(out) < n:
        raise SystemExit(f"SYN filler: only found {len(out)} neutral filler words")
    return tuple(out[:n])


def _bg_matrix(n: int, seed: int):
    rng = random.Random(stable_seed("BG-v1", seed))
    rows = []
    for i in range(n):
        raw = [0.05 + rng.random() for _ in range(n)]
        raw[i] += 1.0
        s = sum(raw)
        rows.append(tuple(x / s for x in raw))
    return tuple(rows)


def build_alphabet(tok, seed: int = 1337) -> SynAlphabet:
    ident = assert_model_tokenizer(tok)
    sigma = _derive_sigma(tok, seed)
    neutral = _derive_neutral_filler(tok, sigma)
    sigma_train = sigma[:52]
    sigma_eval = sigma[52:]
    bg_vocab = tuple(sigma_train) + neutral
    leaked = set(bg_vocab) & STRUCTURAL_TOKENS
    if leaked:
        raise SystemExit(f"SYN filler vocabulary contains structural tokens: {sorted(leaked)}")
    required = tuple(sorted(set(SCAFFOLDING) | set(DIGIT_WORDS) | set(F5_WORDS)
                            | set(sigma) | set(neutral)))
    token_to_id = _validate_words(tok, required)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    return SynAlphabet(
        tokenizer=tok,
        token_to_id=token_to_id,
        id_to_token={v: k for k, v in token_to_id.items()},
        sigma=sigma,
        sigma_train=sigma_train,
        sigma_eval=sigma_eval,
        bg_vocab=bg_vocab,
        bg_matrix=_bg_matrix(len(bg_vocab), seed),
        eos_id=int(eos_id),
        identity=ident,
    )
