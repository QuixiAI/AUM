"""Render-then-tokenize instance construction with offset integrity checks."""

from __future__ import annotations

from dataclasses import dataclass, field


class RenderRejected(RuntimeError):
    pass


@dataclass
class Renderer:
    alpha: object
    text: str = ""
    spans: list[dict] = field(default_factory=list)
    filler_runs: list[tuple[int, int]] = field(default_factory=list)
    n_tokens: int = 0

    def add(self, word: str, label: dict | None = None):
        start = len(self.text)
        self.text += " " + str(word)
        end = len(self.text)
        self.n_tokens += 1
        if label is not None:
            item = dict(label)
            item.setdefault("expected", str(word))
            item["char_span"] = (start, end)
            self.spans.append(item)

    def words(self, *words: str):
        for word in words:
            self.add(word)

    def variant(self, rng, name: str, label_first: dict | None = None):
        for i, word in enumerate(self.alpha.variant_words(rng, name)):
            self.add(word, label_first if i == 0 else None)

    def bg(self, rng, n: int):
        words = self.alpha.bg_words(rng, n)
        if words:
            self.filler_runs.append((self.n_tokens, len(words)))
        for word in words:
            self.add(word)

    def token_len(self) -> int:
        return self.n_tokens

    def _tokenize(self):
        enc = self.alpha.tokenizer(
            self.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return enc["input_ids"], enc["offset_mapping"]

    def _span_to_index(self, offsets, span):
        matches = [i for i, off in enumerate(offsets) if tuple(off) == tuple(span)]
        if len(matches) != 1:
            raise RenderRejected(f"span {span} maps to {matches}, not one token")
        return matches[0]

    def finalize(self, record: dict):
        ids, offsets = self._tokenize()
        labels = []
        for span in self.spans:
            idx = self._span_to_index(offsets, span["char_span"])
            expected = span["expected"]
            decoded = self.alpha.tokenizer.decode([ids[idx]], clean_up_tokenization_spaces=False)
            if decoded.strip() != expected:
                raise RenderRejected(
                    f"label token mismatch at {idx}: expected {expected!r}, decoded {decoded!r}")
            path = span.get("path")
            if path is not None:
                _set_path(record, path, idx)
            labels.append({
                "pos": idx,
                "role": span.get("role", "label"),
                "expected": expected,
                "decoded": decoded,
            })
        ids2 = self.alpha.tokenizer(self.text, add_special_tokens=False)["input_ids"]
        if ids2 != ids:
            raise RenderRejected("tokenize(text) was not stable across calls")
        filler_tokens = sum(n for _, n in self.filler_runs)
        task_tokens = len(ids) - filler_tokens
        record["label_positions"] = labels
        record["filler_rle"] = [[int(s), int(n)] for s, n in self.filler_runs if n > 0]
        record["task_token_count"] = int(task_tokens)
        record["filler_token_count"] = int(filler_tokens)
        record["task_fraction"] = float(task_tokens / max(1, len(ids)))
        record["text_hash"] = __import__("hashlib").sha256(self.text.encode()).hexdigest()
        return ids, record, self.text


def _set_path(obj, path, value):
    cur = obj
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value
