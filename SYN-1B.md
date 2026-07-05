# AUM-Ø v6 — Synthetic Corpus Build Specification (SYN-1B v1.3)

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

**v1.2 changes from v1.1** (surface + difficulty fixes): token stream construction is now
specified as tightly as the labels. Generation is render-then-tokenize with tokenizer identity
hashing; all symbols/scaffolding/answers must be single-token in leading-space context; BG
filler excludes scaffolding and can appear only as an explicitly sampled gap, never as generic
window filler. Difficulty is promoted to a first-class sampled axis: composition depth,
restatement mode, query target type, evidence positions, and minimal-sufficient-suffix length
are recorded in the sidecar and reported per stratum; beyond-window status is derived by the
eval harness from the run's attention window. F1 partial restatements are made
mechanism-forcing by querying unmentioned symbols. F3 adds chained-correction variants.
Standalone dry-run packing pads unused window space with EOS so decoded inspection is not
dominated by synthetic filler; the production packing rule remains web/code co-packing with
at most two synthetic instances per window. F1 partial restatements now use conditioned
transition entries so unmentioned symbols are truly unchanged and inferable; σ probing is
retargeted from rule-id
classification to active mapping/binding decoding, which remains coherent for composite
states.

**v1.3 changes from v1.2** (difficulty-label fixes): controlled age gaps are separated from
ordinary local filler in both the sidecar and density QA via `controlled_gap_tokens`.
Difficulty labels now count every answer-bearing span as possible evidence: rule statements,
corrections, demonstrations, and previous answered queries. Hard-stratum construction excludes
hard target symbols/entities from demonstrations and answered queries inside the target suffix,
and QA independently recomputes minimal sufficient suffixes before shipping. QA reports write
measured statistics into the manifest and missing/unrun checks fail closed. The sidecar does
not emit a summary `difficulty` label; hard/easy strata are derived by the harness and QA from
primitive fields such as `composition_depth`, `minimal_sufficient_suffix_len`, and
family-specific control flags.

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

All generation is **deterministic**: seeded per-instance from `(family, registry_ids,
instance_index, corpus_version)`, with a versioned manifest of content hashes. Corpus version
string: `syn-1b-v1.3`.

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
  Paraphrase variation deliberately does **not** move or replace the answer-prediction
  anchor: F1-F4 queries end at `->`, and F5 `next` queries end at `:`. This fixed anchor keeps
  b_t, QA-10, and the eval harness reading the same prediction position across variants.

### 1.2 Background filler process

Filler between semantic segments is a **shared order-1 Markov stream over Σ plus neutral
whole-word filler tokens** with fixed transition matrix `BG-v1` (generated once, seeded,
checked into the repo). The filler vocabulary explicitly excludes every scaffolding token
(`update`, `query`, `answer`, `exchanged`, `->`, etc.); scaffolding appears only when a
family generator places real structure. Identical process in all five families and in null.

Filler length is a **construction rule**, not a remainder sink. A generator may emit BG filler
only for one of two reasons:

1. A bounded local gap sampled by the family grammar.
2. A controlled evidence-age gap sampled from the age/difficulty axis.

No generator pads an instance to a requested length with BG filler, and no standalone dry-run
packer fills unused 4096-window space with BG filler. Unused standalone window space is EOS
padding; production co-packing uses web/code text as the non-synthetic context (§8). Filler is
the medium in which evidence-age is swept; it must carry zero rule information (QA-1).
Controlled gaps are recorded separately as `controlled_gap_tokens`; local filler is not.

### 1.3 Instance geometry (4096-token windows)

- **Instance length:** variable, bounded by the 4096-token window, and determined by sampled
  family structure plus controlled gaps. Length is a consequence of the chosen difficulty
  stratum, not an instruction to add filler until a target is met.
- **Loss:** plain LM loss over all tokens (pretraining data, not SFT). Eval metrics are
  computed at labeled positions by the harness (§10); no loss masking anywhere.
- **Hard constraint — no window straddling.** The evidence state S and the register σ reset
  at window boundaries (each 4096-token window is an independent training sample), so an
  instance split across windows loses its evidence before its query and becomes unlearnable
  noise. Placement (§8) must keep every instance inside one window; QA-7 verifies.

### 1.4 Rule registry and held-out split

Each family has a finite, versioned **registry of 512 entries**:
`registry[family][0..511]`, generated from the family's parameter space with the corpus seed.
For pure-state families an entry is a rule/state; for revision families an entry may be a
**transition** from a current state to a derived next state. Split **410 train / 102 eval** by
entry index (410–511 eval). Held-out evaluation means: eval-split entries × fresh instance
seeds × (optionally) Σ_ev symbols. For transition registries, the held-out object is the
transition entry (changed subset + delta), not necessarily the A-side state; an eval
transition may share its source/base state with a train transition.
For depth-2 eval instances, **all transition entries** in the composition are drawn from the
eval split; the base/source state may be shared with train.

Probe training uses train-split entries and reports on eval-split entries. The main σ probe is
not a 512-way active-rule classifier for revision families: it decodes the **active
mapping/binding content** per working-set symbol (§10), so composite states from partial or
multi-event revisions have a well-defined target. F5's registry gets the same 410/102 split —
not for training (F5 never trains the model) but for the probe, so probe memorization cannot
masquerade as transfer.

### 1.5 Event-position and age distributions

- **Event position** (reversal/swap/switch): uniform over the middle **[15%, 85%]** only for
  families/modes where event position is sampled independently. When an age or
  minimal-sufficient-suffix target is sampled first, event position is a consequence and is
  reported, not KS-gated.
- **Evidence-age sweep** (the recency-gradient x-axis): distance from evidence write (or the
  correction, per family definition) to the query is drawn **log-uniform over [8, 3500]
  tokens** for controlled F1/F3 modes, stratified into **10 logarithmic bins** with equal
  instance counts per controlled family/mode. The 3500 ceiling is set by the 4096 window minus
  scaffolding. The generator samples the age/suffix bin first and emits exactly the required
  controlled filler gap; age is never whatever length happened to fall out of window packing.
  The sidecar records raw evidence positions and realized ages exactly; the harness converts
  to per-head phase-distance Δφ from the trained model at eval time.

### 1.6 Difficulty axes and mechanism necessity

Difficulty axes are sampled explicitly and recorded as primitive sidecar fields. Easy
instances are kept so the model can learn the task grammar under plain LM loss; hard strata
are the primary evidence for mechanism necessity. Every gated metric in §10 must report both
aggregate and per-stratum numbers, with the hard stratum derived from primitive labels rather
than a serialized summary field.

Common sidecar fields:

- `composition_depth`: number of state-changing events whose effects must be composed.
- `target_age_bin`: per query, the sampled age/suffix bin for the instance/mode that controls
  long-range distance.
- `evidence_positions`: for each query, the minimal set of instance-relative token positions
  whose facts/events are sufficient to determine the answer.
- `minimal_sufficient_suffix_len`: for each query, the shortest suffix ending at the
  prediction position (`answer_pos - 1`) that contains a sufficient evidence set. Formally,
  over all sufficient evidence sets S, it is
  `min(prediction_pos - min(S) + 1)`.
- `hard_suffix`: per query, a generator-side boolean equal to
  `minimal_sufficient_suffix_len > max_attention_window_planned`. The eval harness still
  derives run-specific `beyond_window(W)` from the suffix length and the run's actual W;
  `hard_suffix` is only the planned-corpus stratum marker.
- `controlled_gap_tokens`: per instance, the total number of BG filler tokens emitted for
  controlled evidence-age gaps (§1.2 reason 2), excluding bounded local filler.

Evidence definition: rule statements, corrections, demonstrations, and previous answered
queries all count as evidence. A query's own answer token does not count as evidence for that
query.

Construction rules:

- **Composition depth:** F1/F2/F3 include multi-event variants. The active state at a hard
  query must sometimes be a composition of prior events, not a value copied from one span.
- **Minimal-sufficient-suffix primary stratum:** the data records
  `minimal_sufficient_suffix_len`; the eval harness derives `beyond_window(W)` for each run as
  `minimal_sufficient_suffix_len > W`, where W is that model's local attention window. Hard
  generated strata target suffix lengths above the maximum W in the planned config sweep so
  a windowed-attention-only solver lacks sufficient local evidence.
- **Query density:** all families gate density using
  `task_token_count / (token_len - controlled_gap_tokens) >= 0.25`, so the controlled age
  sweep is not penalized for being long. F1/F2/F4 require at least 4 answer-bearing queries;
  F3 requires at least 2.
- **Mechanism-necessity argument:** each family documents which span is insufficient and what
  state must be carried or composed. This is checked by per-stratum sidecar labels rather than
  inferred after training.

---

## 2. F1 — Branch Reversal (300 M tokens)

**Latent:** a token-mapping rule; one or more mid-sequence transitions to derived rules.

**Rule space:** bijections over a per-instance working set of 8 symbols from Σ. Base rules
use four templates: pairwise swaps, 3-cycles, mirror map (i↔7−i over the working set),
arbitrary derangement. F1 revision entries are **conditioned transitions**:
`(changed_subset, delta_permutation)`. For a partial transition A→B, `changed_subset` has
size 3, `delta_permutation` is a derangement over exactly those symbols' current images, and
B is identical to A outside `changed_subset`. Thus every unstated symbol is genuinely
unchanged and inferable from context, while B remains a bijection. Full restatement uses the
whole working set. Depth-2 applies the same construction to B→C.

**Instance format:**
```
[rule statement A: 8-variant template, e.g. "the rule : red means blue . green means stone . ..."]
[demonstration stream: alternating prompt→mapped-token pairs under A, interleaved with BG filler, 6–20 demonstrations]
[REVERSAL: 8-variant correction template — full or partial restatement of rule B]
[demonstration stream under B, 6–20 demonstrations, BG interleaved]
[queries: "query : red -> " answered under the currently active rule; 4–8 queries, placed both pre- and post-reversal]
```

Difficulty amendments:

- `restatement=partial` is sampled about 75% of the time; `restatement=full` remains as the
  easy stratum.
- Under partial restatement, a quota of post-reversal queries must target symbols **not
  mentioned** in the correction. Because the transition construction guarantees unmentioned
  symbols are unchanged, these queries have deterministic answers but require carrying the
  old active mapping forward; a copy of the correction span alone is insufficient.
- Partial-restatement instances contain at least 2 post-reversal unmentioned-symbol queries.
  When the sampled age/suffix bin targets the hard stratum, at least 1 query is designated
  hard: the queried symbol is excluded from demonstrations and answered queries inside the
  target suffix, and `minimal_sufficient_suffix_len` must exceed
  `max_attention_window_planned`. Low-bin partial instances still query unmentioned symbols,
  but they are not counted toward the hard-suffix quota.
- `composition_depth` is 1 or 2. Depth-2 instances contain two sequential reversals over the
  same working set, and post-event queries are labeled by which event prefix determines the
  answer.

**Labels per token:** `active_map_rle`; events `{type: reversal, pos, transition_id,
source_state, target_state, restatement, changed_symbols, mentioned_symbols}`; queries `{pos,
answer_pos, key, answer, age_from_reversal,
age_from_original_statement, mentioned_in_latest_correction, evidence_positions,
minimal_sufficient_suffix_len, target_age_bin, composition_depth, prediction_anchor,
hard_suffix}`. `answer` is the canonical answer field; generated v1.3 sidecars do not emit
synonymous `correct_answer` aliases.

**Registered signatures:** π spikes within a few tokens after the reversal; b_t > 0
concentrated at post-reversal queries; active-map decode flips only the changed mappings
across each event while preserving unchanged mappings.

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

Difficulty amendment: `composition_depth` is 1 or 2. Depth-2 instances contain sequential
swaps; the active binding table is the composition of both swaps, and hard-stratum queries
include entities affected by one swap, both swaps, and neither swap.
For hard F2 queries, the target entity is excluded from periodic answered queries inside the
target suffix, so a local copy of a recent answer cannot satisfy the hard label.

**Labels:** binding table per token (RLE); swap event `{pos, swapped_entities}`; queries
`{pos, entity, answer, entity_was_swapped: bool, age_from_swap,
age_from_binding_statement, evidence_positions, minimal_sufficient_suffix_len,
target_age_bin, composition_depth, prediction_anchor, hard_suffix}`.

---

## 4. F3 — Delayed Correction / Long-Range Recall (280 M tokens)

**Latent:** key–value associations written early; queried after controlled delay; half the
instances carry a late correction. This family *is* the evidence-age axis, now swept to
3,500 tokens (§1.5).

**Sub-modes (50/50):**
- **Recall:** write `["the key opens box four" ...]` for m∈[3,8] associations → BG filler of
  length = sampled age → two or more queries. No event. Measures whether evidence *survives
  and is readable* at age; feeds the evidence-survival probe.
- **Delayed correction:** writes → bounded local gap → one or more corrections ("update :
  the key now opens box nine", 8 variants) → controlled filler gap → two or more queries.
  Two ages recorded: write→query and latest-correction→query.

Difficulty amendment: correction mode samples `composition_depth` 1 or 2. Depth-2 instances
include chained corrections before the query; hard-stratum queries require the final active
table, not merely the original write or the first correction span. The controlled gap is
placed so the oldest necessary evidence in the minimal sufficient set, not merely the latest
event, can exceed the planned hard-suffix threshold. Recall mode remains the clean
evidence-survival probe.

**Values** come from small closed sets (digits one–nine as words, or 8 symbols) so answers
are single tokens (QA-5).

**Labels:** writes `{pos, key, value}`; corrections `{pos, key, old, new, event_index}`;
queries `{pos, key, answer, age_write_to_query, age_correction_to_query, mode,
evidence_positions, minimal_sufficient_suffix_len, target_age_bin, composition_depth,
hard_suffix}`.

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

**Labels:** constant `active_map_rle`/binding table matching the sampled base shape;
distractors `{pos, mimics_family, variant}`;
`pseudo_event_pos`; queries per shape family.

---

## 6. F5 — Modular-Stream Switch (20 M tokens, EVAL ONLY)

**Latent:** a numeric stream x_{t+1} = (a·x_t + b) mod m, with one mid-sequence switch of
(a, b). Entirely different surface: digit tokens, no Σ, no scaffolding banks —
`7 3 1 5 ... switch ... 2 9 4 ...` with queries "next :".

**Purpose:** zero training exposure; same evidence/hypothesis structure (a rule governs the
stream; an event changes it; continuation requires revising the inferred rule). If π fires at
F5 switches and the parameter decode probe reads (a,b)-class above chance — probe fit on F5
rules 0–409, reported on 410–511 (§1.4) — the mechanism generalizes beyond memorized
formats. Registered as a **reported result, not a gate**: failure narrows the claim
("format-bound revision") rather than falsifying the mechanism.

Parameters: m ∈ {11, 13, 17}; (a, b) registry of 512; switch position per §1.5; segment
lengths 40–200 tokens.

---

## 7. Sidecar Label Schema

One JSONL record per instance. Because instances are placed window-aware (§8), records key to
the **packed corpus by `(shard, window_index, start_offset)`** — `window_index` is the
0-based 4096-token window within the shard's flat stream, `start_offset` the instance's first
token within that window. Token positions in labels are instance-relative; the harness adds
`start_offset`. The schema below is canonical for shared fields; family sections define the
family-specific event/write/query payloads.

```json
{
  "instance_id": "f1-r0173-s000482",
  "corpus_version": "syn-1b-v1.3",
  "family": "F1", "split": "train|eval",
  "shard": "train-syn-003.bin", "window_index": 18211, "start_offset": 1024,
  "registry_ids": {"base_rule": 173, "transitions": [391]},
  "composition_depth": 1,
  "token_len": 2847,
  "task_token_count": 812,
  "filler_token_count": 2035,
  "controlled_gap_tokens": 1200,
  "density_denominator": 1647,
  "controlled_gap_adjusted_task_fraction": 0.493,
  "active_map_rle": [
    {"start": 0, "end": 1512,
     "map": {"red": "blue", "green": "stone", "shell": "basil", "chair": "table",
             "peach": "grape", "river": "cloud", "paper": "oak", "cat": "fish"}},
    {"start": 1512, "end": 2847,
     "map": {"red": "stone", "green": "grape", "shell": "basil", "chair": "table",
             "peach": "blue", "river": "cloud", "paper": "oak", "cat": "fish"}}
  ],
  "events": [{"type": "reversal", "pos": 1512, "transition_id": 391,
              "source_state": "A", "target_state": "B",
              "restatement": "partial", "changed_symbols": ["red", "green", "peach"],
              "mentioned_symbols": ["red", "green", "peach"]}],
  "writes": [{"pos": 14, "key": "shell", "value": "basil"}],
  "queries": [{"pos": 2799, "answer_pos": 2801, "key": "shell", "answer": "basil",
               "prediction_anchor": "->", "age_from_event": 1287,
               "age_from_write": 2785, "target_age_bin": 7,
               "evidence_positions": [14, 1512],
               "minimal_sufficient_suffix_len": 2787,
               "hard_suffix": true,
               "mentioned_in_latest_correction": false}],
  "distractors": [], "pseudo_event_pos": null
}
```
The token stream contains **no labels, no ids, no delimiters beyond natural scaffolding**. A
replay checker (QA-4) regenerates each instance from its seed and verifies stream↔label
consistency before shipping.
Endpoint conventions: token ages such as `age_from_write` are `answer_pos - evidence_pos`;
`minimal_sufficient_suffix_len` is `prediction_pos - min(evidence_positions) + 1`, where
`prediction_pos = answer_pos - 1`. `task_token_count` counts tokens in generator-emitted
semantic segments (statements, corrections, demonstrations, queries, answers, distractors)
and excludes all BG filler. `filler_token_count` counts all generator-emitted BG filler,
including controlled gaps. `density_denominator = token_len - controlled_gap_tokens`, and
`controlled_gap_adjusted_task_fraction = task_token_count / density_denominator`.
`controlled_gap_tokens` is the sum of controlled age-gap filler.
Every scaffolded answer-bearing query must have a corresponding entry in `queries`; per-query
position, answer, evidence, suffix, anchor, stratum, and family-control fields are mandatory.
The machine-readable canonical schema lives in `train/syn/schema.py`; QA-11 imports that file
instead of maintaining a second field list.

---

## 8. Packing into the 20B Corpus (window-boundary-aware)

The repo's pipeline (`train/prepare_data.py` → flat EOS-separated stream → `PackedWindows`
cuts fixed 4096-token windows, freely splitting documents) is kept for web/code — but
synthetic instances must NOT be split (§1.3), so synthetic mixing happens **inside window
assembly**, not upstream:

- `syn/pack.py` assembles the mixed stream window by window: it fills from the web/code
  stream and, at the sampled synthetic-insertion points, places a synthetic instance only if
  it fits entirely within the current window's remainder (else defers it and continues with
  web text). Web documents still split freely across windows, exactly as today.
- **≤ 2 synthetic instances per window, always co-packed with web text** in production.
  Purpose — this belongs with QA-3, not a packing footnote: a window is a fresh S/σ context,
  so synthetic-pure windows would let the model learn a *window-level* "task window" prior
  and pre-arm π from token 1, quietly defeating the F4 null control. Embedding every instance
  in ordinary web text forces event detection to be *local* evidence integration.
- Mixture: **5.0% synthetic (F1–F4, §0 proportions) / 85% web-edu / 10% code**, constant
  across stages; document-level global shuffle for web/code; synthetic insertion points
  Poisson-spaced to hit the 5% rate.
- **Eval pack:** eval-split instances (F1–F4 eval rules + all of F5, ~40 M tokens) are
  excluded from the training corpus entirely and shipped separately with their sidecars,
  pre-packed one instance per window (eval never needs co-packing).

Implementation note: `train/gen_SYN-1B.py` can emit a standalone dry-run artifact for QA and
human inspection. Because that mode has no web/code stream, it pads unused 4096-window space
with EOS rather than BG filler. Dry-run window task fraction is therefore not the production
mixture; QA-8 measures task density **inside synthetic instances**, where the generator has
control. In dry-run mode `--dry-run-tokens` targets synthetic instance tokens before EOS
padding, not final stream tokens, so the on-disk shard token count can be much larger than the
flag when many windows contain one or two short inspection instances plus padding.
Human inspection must not repeatedly decode deterministic shard heads as a proxy for corpus
coverage. The generator's `--inspect-strata` mode samples decoded instances by seeded audit
strata: controlled top age-bin gap, F3 depth-2 correction, F4 with F2-shaped distractor, and
eval/Σ_ev usage. When `--inspect-out-dir` is set, the inspection output writes full decoded
text and full sidecars with `elided: []`; stdout excerpts are explicitly non-normative.

---

## 9. Generator QA Suite (blocks shipping)

Every QA check writes measured statistics into `MANIFEST.syn-1b-v1.3.json`, not just pass/fail.
A check that cannot run because data or fields are missing reports FAIL, never SKIP.

1. **Leak test:** logistic classifier on {1,2,3}-gram counts, null-vs-eventful, using only
   tokens **before the event position** — for F4 the cut is the matched
   `pseudo_event_pos`, so pre-event windows are length-matched: AUC ≤ 0.55. Also
   eventful-family-vs-family AUC on pre-event tokens ≤ 0.6.
2. **Controlled-distribution audit:** gate only variables sampled independently by the
   generator. For F1/F3 controlled-age gaps, every controlled filler gap realizes its sampled
   length exactly, and the realized length falls inside the recorded target bin.
   `controlled_gap_tokens`
   must equal the sum of controlled gap lengths in the sidecar. For non-age-controlled
   eventful modes, event-position histogram is uniform over [15%, 85%] (KS test). Instance
   length distribution is reported, not gated. The age-bin uniformity gate has two explicit
   branches, reported in the manifest: if expected count per bin is ≥ 1000, each bin must be
   within 2% of expected; otherwise dry-run/small-sample artifacts use a Pearson χ² uniformity
   smoke test with df=9 and p>0.01.
3. **Marker audit:** no single token whose presence predicts "event within ±4 tokens" with
   precision > 0.7 (scan the full vocabulary); paraphrase-bank usage roughly uniform.
4. **Replay consistency:** 100% of instances regenerate byte-identically from seeds and pass
   stream↔label verification, including independent recomputation of active mapping/binding
   state from the rendered stream.
5. **Tokenizer audit:** every Σ and scaffolding word single-token; every labeled `answer`
   single-token in leading-space context; generator tokenizer identity and hash must match
   the manifest/model tokenizer; report exact token lengths.
6. **Dedup:** no duplicate instance hashes; 13-gram collision rate with the web corpus
   < 1e-6.
7. **Window integrity (post-packing):** for every sidecar record, the instance's
   `[start_offset, start_offset + token_len)` lies within one 4096-token window of the named
   shard, the tokens there hash-match the regenerated instance, and no production window
   holds > 2 synthetic instances.
8. **Instance-span and seam roundtrip:** for every sidecar instance span,
   `tokenizer.encode(instance_text, add_special_tokens=False)` equals the packed token IDs at
   `[start_offset, start_offset + token_len)`. Also retokenize the instance text together with
   its immediate packed left/right context and assert the synthetic instance subsequence is
   unchanged. Whole-window decode→encode roundtrip is required only for eval windows after
   excluding EOS padding.
9. **Filler scaffolding exclusion:** scan generator-emitted filler spans only; any scaffolding
   token in filler is build-blocking. Production web/code context outside sidecar spans is not
   subject to this check.
10. **Query-position contract:** for every query, the token at `answer_pos - 1` is the
    fixed prediction anchor (`->` for F1-F4, `:` for F5), and the next token decodes to the
    sidecar answer.
11. **Split-integrity audit:** F5 is eval-only. Any F5 sidecar record with `split=train`, or
    any F5 instance placed in a train shard, is build-blocking. The manifest reports
    per-family split counts.
12. **Schema-completeness audit:** every sidecar record and every query entry contains the
    mandatory shared fields plus the family-specific control fields used by §10 metrics.
    Missing fields are build-blocking.
13. **Instance task-density audit:** compute task-bearing fraction as
    `task_token_count / (token_len - controlled_gap_tokens)`. All families require fraction
    ≥ 0.25. F1/F2/F4 require at least 4 answer-bearing queries per instance; F3 requires at
    least 2.
14. **Difficulty-stratum audit:** hard stratum is a fraction, not "nonzero." Eval instances
    require primitive-derived hard examples ≥ 15% for F1/F2/F3, where hard means
    `composition_depth > 1` or at least one query with
    `minimal_sufficient_suffix_len > max_attention_window_planned`. Queries with suffix above
    the planned attention window must be ≥ 10% of F1 and F3 query positions. Multi-event
    composition-depth examples must be ≥ 15% of eval instances for F1/F2/F3.
15. **Minimal-sufficient-suffix audit:** recompute, from the rendered stream under each
    symbolic family semantics (F1-F4), the true shortest sufficient suffix for every query,
    counting rule statements, corrections, demonstrations, and previous answered queries as
    evidence. Assert recomputed suffix length equals `minimal_sufficient_suffix_len`, and
    `evidence_positions` is sufficient. For every query in every family, assert
    `hard_suffix == (minimal_sufficient_suffix_len > max_attention_window_planned)`. F5 is
    explicitly exempt from symbolic replay and counted in
    `skipped_by_family`: its evidence sufficiency is recurrence-parameter inference over the
    modular stream rather than finite symbolic replay, and F5 remains a reported transfer
    probe rather than a mechanism gate. Any non-exempt mismatch is build-blocking.

---

## 10. Eval-Harness Contract (which metric reads which labels)

Position convention: the counterfactual benefit b (aum_ssm/training/counterfactual.py) is
defined at positions *predicting the next token* — b has shape (B, L−1) and the value
relevant to a query answer at `answer_pos` lives at **`answer_pos − 1`**. Off-by-one here
produces a plausible-looking wrong correlation; `syn/harness_readers.py` owns this shift so
no metric implements it twice. The fixed query anchors in §1.1 are intentional: the harness
always reads b at `->` for F1-F4 and at `:` for F5.

For any evaluated model with local attention window W, the harness derives
`beyond_window(W)` per query as `minimal_sufficient_suffix_len > W`. This is not stored as a
generator truth label because W is run-specific.

- **corr(π, b), fixed-K (gate):** eval-split F1+F2; π and b read at all positions;
  event/query positions from the sidecar define the "fires where it should" analysis.
  Report aggregate plus hard-stratum subsets (`beyond_window(W)=true`,
  `composition_depth>1`, and F1 `mentioned_in_latest_correction=false`) with the hard stratum
  as the primary mechanism claim.
- **Null control (gate):** F4 eval-split; mean π and E[J] overall **and at distractor
  positions separately**.
- **Active-map decode probe (gate):** readout fit on train-split F1+F2 activations predicts,
  for each working-set key/entity, the current mapped symbol/attribute from the model state.
  Accuracy is reported on eval-split registry entries and hard strata. This replaces
  512-way `active_rule_id` classification so partial and depth-2 composite states remain
  valid probe targets.
- **Recency gradient (gate):** F1+F3 queries; regress b at `answer_pos − 1` on Δφ (the
  harness computes accumulated per-head phase between evidence position and query from the
  trained model), with token-age as secondary covariate. Report per age bin and split primary
  results by `beyond_window(W)`.
- **Evidence-survival probe:** F3-recall; linear readout of (key→value) from S_t at query
  time, per age bin.
- **σ-intervention:** paired eval instances differing only in active mapping/binding content;
  registers stored at matched positions.
- **Unswapped-entity control:** F2 queries with `entity_was_swapped=false` — b ≈ 0
  registered.
- **F5 transfer:** all F5 metrics reported, none gated; probe split per §1.4.

---

## 11. Deliverables

1. `train/syn/alphabet.py` — tokenizer-verified Σ, scaffolding, paraphrase banks, `BG-v1`
   matrix (all seeded, checked in).
2. `train/syn/registry.py` — 512-entry state/transition registries × 5 families,
   train/eval split, version hash.
3. `train/syn/schema.py` — machine-readable canonical sidecar schema used by QA-11.
4. `train/syn/gen_f{1..5}.py` — instance generators emitting `(token_ids, sidecar_record)`.
5. `train/syn/pack.py` — window-aware packing + mixing per §8; emits shards + sidecar index
   compatible with `train/prepare_data.py`'s manifest format.
6. `train/syn/qa.py` — the fifteen QA checks; CI-style pass/fail report.
7. `train/syn/harness_readers.py` — sidecar readers implementing §10's metric↔label contract
   (including the `answer_pos − 1` shift).
8. `train/syn/forward_probe.py` — model-side plumbing probe: verifies packed-window
   anchor/answer alignment through the real model input path and reports padding-excluded
   loss diagnostics.
9. `train/gen_SYN-1B.py --inspect-strata --inspect-out-dir <dir>` — seeded human-audit
   sample exporter covering top-age, F3 depth-2, F4/F2-distractor, and eval/Σ_ev strata with
   full sidecars and no silent field elision.
10. `MANIFEST.syn-1b-v1.3.json` — seeds, hashes, counts, `max_attention_window_planned`, and
   per-check QA statistics.

Estimated effort: the five generators are 100–200 lines each; `pack.py` and the QA suite are
where the care goes — every gate metric inherits its validity from QA-4, QA-5, QA-7, QA-8,
QA-10, QA-11, QA-12, QA-14, and QA-15.
