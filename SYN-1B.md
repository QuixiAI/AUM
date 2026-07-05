# AUM-Ø v6 — Synthetic Corpus Build Specification (SYN-1B v1.1)

Companion to AUM-Ø v6 §13–§16. Specifies the ~1B-token programmatic task corpus (5% of the
20B pretraining mix) completely enough to implement without further design decisions. Every
✅-gated metric in the v6 MVP table is computed on data produced by these generators; the
generators are therefore part of the experiment, not data preparation.

Relationship to existing code: `aum_ssm/training/tasks/synthetic.py` is the MICRO-scale
version of the same four latent structures (toy 24-token vocab, used by the §14 gate harness
for architecture-level checks). SYN-1B is the corpus-scale counterpart in the real 49,152-BPE
token space, mixed into pretraining. Family names and latent definitions match the micro
suite (`branch_reversal`, `latent_binding_swap`, `delayed_correction`, `flat_null`) so results
cross-reference cleanly; F5 exists only at corpus scale.

**v1.1 changes from v1.0** (review fixes): built for the actual **4096-token** training
context (v1.0 assumed 2048 — this truncated the evidence-age axis, the recency falsifier's
x-axis, at 1500 when 4096 supports ~3,500); packing rewritten around the repo's real
flat-stream pipeline (`train/prepare_data.py` + `PackedWindows`) with window-boundary-aware
placement; F5 probe given its own split; the b_t position convention stated; QA-1's null cut
point defined; QA-7 (window integrity) added; sidecar keyed to windows, not doc offsets.

---

## 0. Scope and Budget

| Family | Latent structure | Role | Token budget | Share |
|---|---|---|---|---|
| F1 Branch reversal | rule flips mid-sequence | train + eval | 300 M | 30% |
| F2 Binding swap | entity↔attribute swap | train + eval | 250 M | 25% |
| F3 Delayed correction / recall | old evidence, controlled age | train + eval | 280 M | 28% |
| F4 Flat null | rule never changes (+ distractors) | train + eval | 170 M | 17% |
| F5 Modular-stream switch | rule flips, alien surface | **eval only, zero training exposure** | 20 M | — |

F1–F4 total 1.0 B tokens mixed into pretraining at a **constant 5% rate across all four
stages** (no ramp — any behavioral change at a stage boundary must be attributable to the
stage, not the data). F5 never enters training; it answers "the register memorized your task
formats" and is evaluated only.

All generation is **deterministic**: seeded per-instance from `(family, rule_id,
instance_index, corpus_version)`, with a versioned manifest of content hashes. Corpus version
string: `syn-1b-v1.1`.

---

## 1. Shared Infrastructure

### 1.1 Alphabet and scaffolding vocabulary

All families draw from one shared token pool so families are surface-indistinguishable except
at events.

- **Symbol alphabet Σ:** 64 lowercase English words, each verified **single-token in
  leading-space context** under the SmolLM2 49,152-BPE tokenizer via
  `tok(" " + word, add_special_tokens=False)`. The generator renders text first, tokenizes
  the rendered text once, and derives all sidecar offsets from tokenizer offset mappings.
  Split: 52 **train symbols** Σ_tr, 12 **eval-only symbols** Σ_ev used only in held-out eval
  instances (tests transfer to unseen surface within known structure — mild, because the
  tokens themselves occur in web data).
- **Scaffolding set:** ~24 function words/punctuation used identically across families:
  `rule`, `now`, `means`, `is`, `becomes`, `note`, `update`, `query`, `answer`, `:`, `.`,
  `,`, `->`, `the`, `a`, `so`, `then`, `still`, `same`, `key`, `box`, `opens`, `holds`,
  `where`.
- **Template paraphrase banks.** Every semantic slot (rule statement, correction, query,
  confirmation) has **8 surface variants**, sampled per use. No single marker token may
  deterministically signal an event — the primary anti-shortcut constraint, checked by QA-3.

### 1.2 Background filler process

Filler between semantic segments is a **shared order-1 Markov stream over Σ plus neutral
whole-word filler tokens** with fixed transition matrix `BG-v1` (generated once, seeded,
checked into the repo). The filler vocabulary explicitly excludes every scaffolding token
(`update`, `query`, `answer`, `exchanged`, `->`, etc.); scaffolding appears only when a
family generator places real structure. Identical process in all five families and in null.
Filler is the medium in which evidence-age is swept; it must carry zero rule information
(QA-1).

### 1.3 Instance geometry (4096-token windows)

- **Instance length:** variable and compact. F1/F2/F4/F5 are mostly task-bearing tokens with
  only bounded local filler gaps; F3 alone emits long controlled filler gaps for the
  evidence-survival age sweep. No generator pads an instance to fill a requested length.
- **Loss:** plain LM loss over all tokens (pretraining data, not SFT). Eval metrics are
  computed at labeled positions by the harness (§10); no loss masking anywhere.
- **Hard constraint — no window straddling.** The evidence state S and the register σ reset
  at window boundaries (each 4096-token window is an independent training sample), so an
  instance split across windows loses its evidence before its query and becomes unlearnable
  noise. Placement (§8) must keep every instance inside one window; QA-7 verifies.

### 1.4 Rule registry and held-out split

Each family has a finite, versioned **rule registry of 512 rules**: `rules[family][0..511]`,
generated from the family's parameter space with the corpus seed. Split **410 train / 102
eval** by index (410–511 eval). Held-out evaluation means: eval-split rules × fresh instance
seeds × (optionally) Σ_ev symbols. All probe training (σ-decode classifier) uses train-split
rules; all gate metrics (corr(π,b), recency gradient, σ-decode accuracy, null firing) are
**reported on eval-split rules only**. F5's registry gets the same 410/102 split — not for
training (F5 never trains the model) but for the **probe**: the F5 σ-decode probe fits on
activations from rules 0–409 and reports on 410–511, so probe memorization cannot
masquerade as transfer.

### 1.5 Event-position and age distributions

- **Event position** (reversal/swap/switch): kept away from boundaries for F1/F2/F5. F3
  correction events may occur early because the controlled post-correction gap is the
  measured survival variable.
- **Evidence-age sweep** (the recency-gradient x-axis): distance from evidence write (or the
  correction, per family definition) to the query is a controlled input, not a side effect of
  window padding. The generator samples the age bin first and emits that many filler tokens
  only in the relevant F3 gap. The sidecar records the realized age exactly; the harness
  converts to per-head phase-distance Δφ from the trained model at eval time.

---

## 2. F1 — Branch Reversal (300 M tokens)

**Latent:** a token-mapping rule; one mid-sequence reversal to a different rule.

**Rule space:** bijections over a per-instance working set of 8 symbols from Σ. Four
templates: pairwise swaps, 3-cycles, mirror map (i↔7−i over the working set), arbitrary
derangement. Registry rule = (template, working set, parameters). Rule B (post-reversal) is a
different registry rule over the same working set.

**Instance format:**
```
[rule statement A: 8-variant template, e.g. "the rule : red means blue . green means stone . ..."]
[demonstration stream: alternating prompt→mapped-token pairs under A, interleaved with BG filler, 6–20 demonstrations]
[REVERSAL: 8-variant correction template — full or partial restatement of rule B (50/50)]
[demonstration stream under B, 6–20 demonstrations, BG interleaved]
[queries: "query : red -> " answered under the currently active rule; 3–8 queries, placed both pre- and post-reversal]
```
Partial restatement (rule B stated for only 3 of 8 symbols; the remainder inferred as
"unchanged" or per-template) creates instances where revision requires reading old evidence,
not just parsing the correction text — labeled `restatement=partial` for stratified analysis.

**Labels per token:** `active_rule_id`; events `{type: reversal, pos, old_rule, new_rule,
restatement}`; queries `{pos, answer_pos, queried_symbol, correct_answer, age_from_reversal,
age_from_original_statement}`.

**Registered signatures:** π spikes within a few tokens after the reversal; b_t > 0
concentrated at post-reversal queries; σ-decode flips from rule A to rule B across the event.

---

## 3. F2 — Binding Swap (250 M tokens)

**Latent:** entity→attribute bindings; a correction swaps two. Same latent structure as F1,
deliberately different surface, so a "reversal-token detector" cannot generalize between them.

**Rule space:** assignments of k∈[4,8] entities (symbols from Σ) to k attributes (disjoint
symbols from Σ). Registry rule = (entity set, attribute set, assignment). The swap picks 2
(or, 25% of the time, a 3-cycle of) entities.

**Instance format:**
```
[bindings: "alpha is red . bravo is green . ..." 8-variant phrasing, order shuffled]
[BG filler + periodic queries under original bindings]
[SWAP: "note : alpha and delta were exchanged ." 8 variants; never restates resulting bindings — the model must compute them]
[BG filler + queries under swapped bindings, including queries about UNSWAPPED entities]
```
The unswapped-entity queries are a **within-instance allocation control**: a model that
globally perturbs σ at the correction damages them; correct revision is *surgical*.
Registered: b_t positive at swapped-entity queries, ≈ 0 at unswapped-entity queries.

**Labels:** binding table per token (RLE); swap event `{pos, swapped_entities}`; queries
`{pos, entity, correct_answer, entity_was_swapped: bool, age_from_swap,
age_from_binding_statement}`.

---

## 4. F3 — Delayed Correction / Long-Range Recall (280 M tokens)

**Latent:** key–value associations written early; queried after controlled delay; half the
instances carry a late correction. This family *is* the evidence-age axis, now swept to
3,500 tokens (§1.5).

**Sub-modes (50/50):**
- **Recall:** write `["the key opens box four" ...]` for m∈[3,8] associations → BG filler of
  length = sampled age → one or more queries. No event. Measures whether evidence *survives
  and is readable* at age; feeds the evidence-survival probe.
- **Delayed correction:** writes → bounded local gap → correction ("update : the key now
  opens box nine", 8 variants) → controlled filler gap → one or more queries. Two ages
  recorded: write→query and correction→query.

**Values** come from small closed sets (digits one–nine as words, or 8 symbols) so answers
are single tokens (QA-5).

**Labels:** writes `{pos, key, value}`; corrections `{pos, key, old, new}`; queries `{pos,
key, correct_answer, age_write_to_query, age_correction_to_query, mode}`.

**Registered expectation (v6 §14–16):** base v6 is *predicted to degrade with age* here —
this family is **measured, not gated**. The evidence-survival probe (linear recoverability of
the association from S_t at query time, per age bin) is computed on F3-recall and
disambiguates decay (α story) from addressing failure (phase story).

---

## 5. F4 — Flat Null (170 M tokens)

**Latent:** one rule, never changes. Surface-matched to F1/F2/F3 including scaffolding — and
including **correction-like distractors**.

**Construction:** sample a base family shape (F1-shaped 40% / F2-shaped 35% / F3-shaped 25%);
generate identically **except no event occurs**. In place of events, insert distractor
segments from an 8-variant bank of event-*resembling* but semantically inert statements:
"note : the rule stays the same .", "update : still , red means blue ." (restating the
*existing* rule), "as before , alpha is red .". Distractor positions follow the same
[15%, 85%] distribution as real events; each instance records its matched **pseudo-event
position** (the distractor slot) for QA-1's cut point.

This is the null control's teeth: π must stay quiet not merely on eventless text, but on text
carrying the *surface trappings* of events. Registered prediction: π ≈ 0 and E[J] → 0
**including at distractor positions** (logged separately — π firing at distractors but not
elsewhere is the precise signature of surface-shortcut learning).

**Labels:** constant `active_rule_id`; distractors `{pos, mimics_family, variant}`;
`pseudo_event_pos`; queries per shape family.

---

## 6. F5 — Modular-Stream Switch (20 M tokens, EVAL ONLY)

**Latent:** a numeric stream x_{t+1} = (a·x_t + b) mod m, with one mid-sequence switch of
(a, b). Entirely different surface: digit tokens, no Σ, no scaffolding banks —
`7 3 1 5 ... switch ... 2 9 4 ...` with queries "next :".

**Purpose:** zero training exposure; same evidence/hypothesis structure (a rule governs the
stream; an event changes it; continuation requires revising the inferred rule). If π fires at
F5 switches and σ-decode reads (a,b)-class above chance — probe fit on F5 rules 0–409,
reported on 410–511 (§1.4) — the mechanism generalizes beyond memorized formats. Registered
as a **reported result, not a gate**: failure narrows the claim ("format-bound revision")
rather than falsifying the mechanism.

Parameters: m ∈ {11, 13, 17}; (a, b) registry of 512; switch position per §1.5; segment
lengths 40–200 tokens.

---

## 7. Sidecar Label Schema

One JSONL record per instance. Because instances are placed window-aware (§8), records key to
the **packed corpus by `(shard, window_index, start_offset)`** — `window_index` is the
0-based 4096-token window within the shard's flat stream, `start_offset` the instance's first
token within that window. Token positions in labels are instance-relative; the harness adds
`start_offset`.

```json
{
  "instance_id": "f1-r0173-s000482",
  "corpus_version": "syn-1b-v1.1",
  "family": "F1", "split": "train|eval",
  "shard": "train-syn-003.bin", "window_index": 18211, "start_offset": 1024,
  "rule_ids": {"A": 173, "B": 391},
  "token_len": 2847,
  "active_rule_rle": [[0, 173, 1512], [1512, 391, 1335]],
  "events": [{"type": "reversal", "pos": 1512, "old": 173, "new": 391, "restatement": "partial"}],
  "writes": [{"pos": 14, "key": "red", "value": "blue"}],
  "queries": [{"pos": 2799, "answer_pos": 2801, "key": "red", "answer": "chair",
               "age_from_event": 1287, "age_from_write": 2785, "age_bin": 7}],
  "distractors": [], "pseudo_event_pos": null
}
```
The token stream contains **no labels, no ids, no delimiters beyond natural scaffolding**. A
replay checker (QA-4) regenerates each instance from its seed and verifies stream↔label
consistency before shipping.

---

## 8. Packing into the 20B Corpus (window-boundary-aware)

The repo's pipeline (`train/prepare_data.py` → flat EOS-separated stream → `PackedWindows`
cuts fixed 4096-token windows, freely splitting documents) is kept for web/code — but
synthetic instances must NOT be split (§1.3), so synthetic mixing happens **inside window
assembly**, not upstream:

- `syn/pack.py` assembles standalone 4096-token synthetic windows by placing whole instances
  until the window reaches a sampled task-token fraction target. Instances are EOS-separated;
  leftover space is EOS padding, never Markov filler.
- **Task density is a construction rule.** Each packed window targets 45-60% task-bearing
  tokens and must pass a 35-70% QA band. This prevents the corpus from spending most of its
  synthetic budget training the filler Markov process.
- Mixture: **5.0% synthetic (F1–F4, §0 proportions) / 85% web-edu / 10% code**, constant
  across stages; document-level global shuffle for web/code; synthetic insertion points
  Poisson-spaced to hit the 5% rate.
- **Eval pack:** eval-split instances (F1–F4 eval rules + all of F5, ~40 M tokens) are
  excluded from the training corpus entirely and shipped separately with their sidecars,
  pre-packed one instance per window (eval never needs co-packing).

Implementation note: `train/gen_SYN-1B.py` emits a standalone SYN artifact rather than
directly splicing into the web/code stream. Its manifest records `packing_policy` including
the task-density target, EOS separators, EOS padding, and the lack of an instance-boundary
attention mask. Budget reports distinguish **family instance tokens**, **family task-bearing
tokens**, and **on-disk total tokens**.

---

## 9. Generator QA Suite (blocks shipping)

1. **Leak test:** logistic classifier on {1,2,3}-gram counts, null-vs-eventful, using only
   tokens **before the event position** — for F4 the cut is the matched
   `pseudo_event_pos`, so pre-event windows are length-matched: AUC ≤ 0.55. Also
   eventful-family-vs-family AUC on pre-event tokens ≤ 0.6.
2. **Distribution audit:** event-position histogram uniform over [15%, 85%] (KS test);
   age-bin counts equal per family within 2%; length distribution matches §1.3.
3. **Marker audit:** no single token whose presence predicts "event within ±4 tokens" with
   precision > 0.7 (scan the full vocabulary); paraphrase-bank usage roughly uniform.
4. **Replay consistency:** 100% of instances regenerate byte-identically from seeds and pass
   stream↔label verification.
5. **Tokenizer audit:** every Σ and scaffolding word single-token; every labeled `answer`
   single-token in leading-space context; generator tokenizer identity and hash must match
   the manifest/model tokenizer; report exact token lengths.
6. **Dedup:** no duplicate instance hashes; 13-gram collision rate with the web corpus
   < 1e-6.
7. **Window integrity (post-packing):** for every sidecar record, the instance's
   `[start_offset, start_offset + token_len)` lies within one 4096-token window of the named
   shard, the tokens there hash-match the regenerated instance, and no window holds more
   instances than the manifest's `max_synthetic_instances_per_window`.
8. **Packed-window roundtrip:** for every packed 4096-token synthetic window,
   `tokenizer.encode(tokenizer.decode(window_ids), add_special_tokens=False) == window_ids`.
   This catches BPE re-segmentation at packing boundaries, not just inside instances.
9. **Filler scaffolding exclusion:** scan both instance-local filler spans and window-level
   background outside sidecar spans; any scaffolding token in filler is build-blocking.
10. **Query-position contract:** for every query, the token at `answer_pos - 1` is the
    prediction token (`->` for F1-F4, `:` for F5), and the next token decodes to the sidecar
    answer.
11. **Task-density audit:** compute task-bearing token fraction per packed window and per
    split from sidecar `task_token_count`; every window must stay in the configured QA band,
    and corpus-level density must do the same.

---

## 10. Eval-Harness Contract (which metric reads which labels)

Position convention: the counterfactual benefit b (aum_ssm/training/counterfactual.py) is
defined at positions *predicting the next token* — b has shape (B, L−1) and the value
relevant to a query answer at `answer_pos` lives at **`answer_pos − 1`**. Off-by-one here
produces a plausible-looking wrong correlation; `syn/harness_readers.py` owns this shift so
no metric implements it twice.

- **corr(π, b), fixed-K (gate):** eval-split F1+F2; π and b read at all positions;
  event/query positions from the sidecar define the "fires where it should" analysis.
- **Null control (gate):** F4 eval-split; mean π and E[J] overall **and at distractor
  positions separately**.
- **σ-decode probe (gate):** classifier fit on `active_rule_rle` labels from **train-split**
  F1+F2 activations; accuracy reported on **eval-split** rules.
- **Recency gradient (gate):** F1+F3 queries; regress b at `answer_pos − 1` on Δφ (the
  harness computes accumulated per-head phase between evidence position and query from the
  trained model), with token-age as secondary covariate; report per age bin, noting the
  3,500-token ceiling (§1.5).
- **Evidence-survival probe:** F3-recall; linear readout of (key→value) from S_t at query
  time, per age bin.
- **σ-intervention:** paired eval instances differing only in `rule_ids`; registers stored
  at matched positions.
- **Unswapped-entity control:** F2 queries with `entity_was_swapped=false` — b ≈ 0
  registered.
- **F5 transfer:** all F5 metrics reported, none gated; probe split per §1.4.

---

## 11. Deliverables

1. `train/syn/alphabet.py` — tokenizer-verified Σ, scaffolding, paraphrase banks, `BG-v1`
   matrix (all seeded, checked in).
2. `train/syn/registry.py` — 512-rule registries × 5 families, train/eval split, version hash.
3. `train/syn/gen_f{1..5}.py` — instance generators emitting `(token_ids, sidecar_record)`.
4. `train/syn/pack.py` — window-aware packing + mixing per §8; emits shards + sidecar index
   compatible with `train/prepare_data.py`'s manifest format.
5. `train/syn/qa.py` — the seven QA checks; CI-style pass/fail report.
6. `train/syn/harness_readers.py` — sidecar readers implementing §10's metric↔label contract
   (including the `answer_pos − 1` shift).
7. `MANIFEST.syn-1b-v1.1.json` — seeds, hashes, counts, QA report.

Estimated effort: the five generators are 100–200 lines each; `pack.py` and the QA suite are
where the care goes — every gate metric inherits its validity from QA-1, QA-3, and QA-7.
