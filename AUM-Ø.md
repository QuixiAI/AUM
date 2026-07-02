# AUM-Ø v6: Attentive Unfolding Modulation with Silence

## An Affine Resonant Evidence Core with a Benefit-Gated Global Hypothesis Register

**AUM-Ø**, pronounced **Aum-nought**, is a recurrent sequence architecture built on one principle:

$$
\text{Continuation arises from temporary configuration.}
$$

and one structural commitment:

$$
\text{Separate evidence from interpretation, and spend extra computation only where revising the interpretation pays.}
$$

The architecture maintains two kinds of state on two different clocks. An **evidence state** $S_t$ — a phase-addressed associative memory, updated once per token by an affine recurrence — records *what has been observed*. A **hypothesis register** $\sigma_t$ — a small nonlinear state, revised zero or more times per token by an inner "silence" loop — holds *how the evidence is currently interpreted*: which rule is active, which branch is live, which binding holds, whether a correction has invalidated a prior assumption. The token clock writes evidence; the silence clock revises hypothesis; a learned **integration pressure** $\pi_t$, trained against measured counterfactual benefit, decides when revision is worth the compute.

The base model performs **recent-evidence hypothesis revision**: its silent read is phase-locked to the current token, so it preferentially reinterprets recently written evidence. This is a designed property, not a defect — it yields the sharpest available mechanistic falsifier (§14). Extending the read to older evidence (**temporal hypothesis search**) is the first scoped follow-up (§17), deliberately excluded from v6 so the falsifier stays clean.

This document supersedes v5.3 and is written to be built from directly: architecture, training recipe, pre-registered evaluation, and tensor manifest are all here. Nothing in it is provisional.

---

## 0. What Changed from v5.3, and Why

Six structural corrections, found by re-deriving the mechanics rather than reviewing the prose. Each is now native to the design rather than an erratum.

**(1) The rotation operator gains a frequency ladder.** v5.3's single-frequency $R(\phi)$ made read–write alignment *periodic* in phase distance: evidence at $\Delta\phi = 2\pi k$ re-aligned perfectly, so "recency-selective retrieval" was actually aliased retrieval, and the registered recency gradient could fail for spec reasons rather than mechanism reasons. v6 uses a geometric multi-frequency ladder (RoPE-style), under which alignment decays quasi-monotonically with phase distance. The recency falsifier is correspondingly re-registered against **accumulated phase distance**, with token-age as the secondary axis. (§4)

**(2) Halting mixes losses, not states.** v5.3's $\bar\sigma_t=\sum_j w_j\sigma_t^j$ blended hypotheses — a convex mixture of "rule A" and "rule B" is not a hypothesis, it corrupted the σ-decode probe, and it created a train/test mismatch with hard halting. v6 trains the PonderNet-style expected loss over per-step outputs, keeps each $\sigma^j$ a coherent candidate, and carries forward a single register. (§8)

**(3) The degenerate loss basin is closed.** In v5.3, the precision-sparsity, consistency, and compute losses jointly admitted a low-loss solution with precision off, register frozen, and silence never firing — a dead mechanism with a healthy loss curve, indistinguishable from the null-control's intended behavior. v6 detaches $\mu$ inside the consistency functional and applies precision sparsity only to the per-layer fields, never the global one. (§7, §10)

**(4) The pressure label is policy-independent.** v5.3's benefit label $\ell_0-\ell_J$ was computed under the live halting policy, whose halting head consumes $\pi_t$ — the pressure head was chasing a target that moved with its own output. v6 computes the label **always at fixed depth $K$ on the exploration subset**, breaking the circularity; the on-policy gauge measures the residual gap. (§11)

**(5) The parallelism claim is stated truthfully.** The affine invariant makes the *evidence core* scan-parallel. The global block is a sequential nonlinear recurrence over tokens ($\sigma_{t-1}\to e_t \to \mu_t \to \sigma_t^0$) and does not scan. Its state is small ($d_\sigma{=}128$), so at reference scale it is a fused sequential kernel with negligible cost — but the document no longer advertises full-model scannability it does not have. (§2, §12)

**(6) Two evaluation fixes.** The Top-GRU baseline now keeps a pooled-evidence prediction head, so the ablated factor is *precisely* the silent evidence read (not prediction quality). Full-vocabulary predictive entropy is removed from the base pressure features — it put a 49k-softmax on the critical path twice per token — and demoted to a registered optional-feature ablation. (§9, §14)

Also folded in: the per-head silent read is defined (§8); an evidence-survival probe separates decay from addressing on old-evidence tasks (§16); a σ-relevance check guards against the prediction head learning to ignore the register (§16); the pressure-training gate is scale-free (§12); and the training recipe lives in the spec (§13, Appendix B).

---

## 1. Design Commitments

These forks are settled. They are the architecture's rationale, recorded so future work changes them knowingly or not at all.

**C1 — Top-only hypothesis register.** One global $\sigma$, one halting policy, one silent depth $J_t$ per token, attached as a single block above the evidence stack. Predictive grounding, prediction error, error-fed precision, and the register exist only there; the $L$ evidence layers run pure token-clock recurrence. The silence mechanism is ~2% of parameters, so the silence-ablated model is a nearly parameter-matched baseline and any measured benefit is the mechanism's, not capacity's.

**C2 — Affine evidence recurrence (invariant).** $S_t$ is affine in $S_{t-1}$: diagonal/block-diagonal gain plus input-dependent additive write. No nonlinearity inside the state recurrence, ever; normalization applies to write inputs and readouts only. Extensions requiring nonlinear mutation route through $\sigma$, a correction patch, or the readout — never through $S$. This buys scan-parallel training *of the core* (see C7 for the honest full-model statement).

**C3 — Stable evidence addressing.** Prediction error never perturbs the write key. Surprise may modulate write strength or value; the address is sacred, because address–surprise coupling destroys associative recall.

**C4 — Bottlenecked register.** $d_\sigma \ll d$ (128 in the reference model). The register is squeezed toward holding a low-dimensional interpretation rather than becoming a second readout buffer. Whether 128 is too small is answered by the σ-decode probe at the gate (§15), never assumed in advance.

**C5 — Hypothesis-conditioned predictive grounding.** The prediction head reads the evidence state *through the current hypothesis*, symmetric with the silent read. A wrong hypothesis reads the wrong evidence and mispredicts the next grounding — this is how $\sigma$ is held accountable by the world rather than by an auxiliary label.

**C6 — Phase-locked silent read with a frequency ladder.** The silent read uses the current phase under the multi-frequency rotation, giving graded recency-selective retrieval and the registered differential prediction of §14. Temporal search is excluded from v6 by design.

**C7 — Honest parallelism.** The evidence core trains as a parallel scan. The global block is a sequential nonlinear token recurrence whose working state is $O(d_\sigma + d + H_U d_h^2)$ per batch row — the register, the grounded summary, and the top layer's evidence state $S$, which the block must step alongside $\sigma$ to serve its reads (≈ 66 K floats at Tiny scale; SRAM-resident for a fused kernel, but not "small" and never claimed scannable). Training memory is handled by exact-gradient segment checkpointing: only segment-boundary carries $(S, \sigma)$ are stored and in-segment states are recomputed on backward, so the recurrence trains at full sequence length without materializing the per-token $S$ chain (§12). At reference scale the wall-clock cost is small; at larger scales it is a known, stated bottleneck — not a surprise.

**C8 — Benefit-gated silence with a policy-independent label.** Pressure is trained to predict measured counterfactual benefit on a fixed calibrated scale, with the label computed at fixed depth on an exploration subset and downstream silence frozen in both branches. Silence is allocated by learned usefulness, not by uncertainty.

**C9 — Mechanism-isolating evaluation is part of the architecture.** The no-op, no-read, phase-scrambled, and random-silence controls, the Top-GRU and evidence-core baselines, and the causal σ-intervention are first-class components. The design exists to be falsified crisply; that is its main methodological asset.

---

## 2. System Overview and State

Input $x_1,\dots,x_T$, embeddings $x_t\in\mathbb{R}^d$. The inference condition is

$$
\chi_t = (\phi_t,\ S_t,\ \sigma_t,\ \mu_t)
$$

with $\phi_t$ the per-head phase position, $S_t$ the evidence state, $\sigma_t$ the hypothesis register, $\mu_t$ the global precision field. Execution per token:

$$
\underbrace{\text{evidence core, } L \text{ layers}}_{\text{scan-parallel token clock}}
\;\to\; g_t
\;\to\;
\underbrace{\text{global block: predict, err, precision, pressure, silence}}_{\text{sequential nonlinear token clock + inner silence clock}}
\;\to\; o_t
$$

The evidence state is written only by the token clock. The register is revised only by the silence clock. The two clocks never write each other's state — that separation *is* the architecture.

---

## 3. Evidence Core: Per-Layer Structure

Each of the $L$ evidence layers contains bounded grounding (A), resonant unfolding (U), and error-free precision (M), around a standard SwiGLU MLP and pre-norms.

**Controller and projections.** With $\bar x_t=\operatorname{LN}(x_t)$: content projections $q_t=W_q\bar x_t$, $k_t=W_k\bar x_t$, $v_t=W_v\bar x_t$; output gate $z_t=W_z\bar x_t$; controller $c_t=W_c\bar x_t$ emitting per-head dynamics

$$
(\bar\tau_t,\ \bar\lambda_t,\ r_t,\ \theta_t,\ m_t,\ s_t) = W_p\, c_t
$$

— step size, dissolution, write strength, phase velocity, precision drive, pressure drive. Only the top layer's $s_t$ is consumed (§9).

**A — bounded grounding.** Grouped-query local attention over a sliding window $w$: $h_t^{A}=\operatorname{LocalAttn}(\bar x_t,\ x_{t-w:t})$. The A phase answers *what is present*; it is bounded by design so the recurrence, not attention, carries long range.

**M — error-free precision (lower layers).** With no error signal below the top:

$$
\mu_t^\ell=\sigma\!\big(W_\mu^\ell[\,h_t^{A,\ell},h_t^{U,\ell},m_t^\ell\,]\big),\quad
\Delta h_t^\ell=U_m^\ell\operatorname{diag}(\mu_t^\ell)V_m^\ell h_t^{U,\ell},\quad
h_t^{M,\ell}=h_t^{A,\ell}+h_t^{U,\ell}+\Delta h_t^\ell
$$

The residual stream accumulates $h^{M,\ell}$; the top layer emits the grounded summary $g_t\in\mathbb{R}^d$.

---

## 4. U Phase: Resonant Evidence Unfolding

**Dynamics.** Per U-head:

$$
\tau_t=\operatorname{softplus}(\bar\tau_t+b_\tau),\qquad
\lambda_t=\epsilon+f(\bar\lambda_t),\qquad
f(x)=\begin{cases}1+x,&x\ge 0\\[2pt]\tfrac{1}{1-x},&x<0\end{cases},\qquad
\alpha_t=e^{-\lambda_t\tau_t}
$$

**Phase position.** Each U-head $h$ carries a scalar phase position advanced by data-dependent velocity:

$$
\phi_t^{(h)}=\phi_{t-1}^{(h)}+\pi\tanh\!\big(\theta_t^{(h)}\big)\,\tau_t^{(h)}
$$

$\phi$ is *not* wrapped modulo $2\pi$; it is an unbounded accumulated position, exactly as a token index is in RoPE, and the rotation below consumes it at many frequencies.

**Multi-frequency rotation (the ladder).** Within each head of dimension $d_h$, partition into $B=d_h/2$ two-dimensional blocks and assign geometric frequencies

$$
\omega_b=\omega_{\max}\left(\frac{\omega_{\min}}{\omega_{\max}}\right)^{\frac{b-1}{B-1}},\qquad b=1,\dots,B,\qquad
(\omega_{\max},\omega_{\min})=(1,\ 10^{-3})
$$

and define $R(\phi)=\operatorname{blockdiag}\big(\operatorname{Rot}(\omega_1\phi),\dots,\operatorname{Rot}(\omega_B\phi)\big)$, parameter-free. Because rotations are orthogonal and both writes and reads are rotated, the retrieval score between a read at phase $\phi_t$ and evidence written at phase $\phi_s$ depends only on the **relative phase** $\Delta\phi=\phi_t-\phi_s$ — this is data-dependent relative position. Under the ladder, alignment across the $B$ frequencies interferes destructively as $|\Delta\phi|$ grows, so retrieval decays quasi-monotonically with phase distance instead of ringing with period $2\pi$. This monotonicity is what makes "recency-selective" a real, testable property (§14) rather than an aliased accident.

**Write.** Rotate and normalize:

$$
\tilde q_t=R(\phi_t)q_t,\quad \tilde k_t=R(\phi_t)k_t,\quad
\hat k_t=\frac{\tilde k_t}{\lVert\tilde k_t\rVert+\epsilon},\quad
\hat v_t=\frac{v_t}{\lVert v_t\rVert+\epsilon}
$$

Write gate $\rho_t=\sigma(r_t)$, write $W_t=\hat v_t\otimes\hat k_t$, and the **affine** update (invariant C2):

$$
S_t=\alpha_t\odot S_{t-1}+\rho_t\tau_t\odot W_t
$$

**Gated readout.** Per head:

$$
h_t^{U,(h)}=\operatorname{silu}\!\big(z_t^{(h)}\big)\odot\operatorname{RMSNorm}\!\big(S_t^{(h)}\tilde q_t^{(h)}+D^{(h)}v_t^{(h)}\big)
$$

with learned skip $D^{(h)}$. The U phase answers *what evidence has accumulated, and how it reads under the current phase*.

**Stable addressing (C3).** The base write key is never a function of prediction error. At the top layer, where $e_t$ exists, surprise may optionally modulate $\rho_t$ or $\hat v_t$ — never $\hat k_t$ — and that variant is evaluated separately, not in the reference model.

---

## 5. Global Block: Hypothesis-Conditioned Predictive Grounding

Before folding $g_t$ into the register, the block predicts it by reading the *previous* evidence through the *previous* hypothesis at the *previous* phase:

$$
q_{t-1}^{\text{pred}}=R(\phi_{t-1})\,W_q^{\text{pred}}\,\sigma_{t-1},\qquad
r_{t-1}^{\text{pred}}=S_{t-1}\,q_{t-1}^{\text{pred}}
$$

$$
\hat g_t=W_P\,\operatorname{LN}\!\big(W_R\,r_{t-1}^{\text{pred}}+W_\sigma^{P}\,\sigma_{t-1}+W_\phi\,\Phi(\phi_{t-1})\big),
\qquad e_t=g_t-\hat g_t
$$

$\Phi$ is a sinusoidal phase embedding; $W_q^{\text{pred}}$ is separate from the silent read's $W_q^\sigma$ (different jobs: predicting the next grounding vs. revising the current hypothesis). Accountability is structural: a register holding the wrong hypothesis addresses the wrong evidence and mispredicts $g_t$, raising $e_t$ — the world corrects $\sigma$, not a label. §16 registers the check that the head has not learned to bypass $\sigma$ ($W_q^{\text{pred}}\!\to\!0$ is a silent failure this design permits and must therefore be monitored).

---

## 6. Global Block: Error-Fed Precision

$$
\mu_t=\sigma\!\big(W_\mu[\,g_t,\ e_t,\ \sigma_{t-1},\ m_t\,]\big)\in[0,1]^{k},\qquad
\tilde e_t=\mu_t\odot W_e e_t
$$

The global $\mu_t$ is precision only: it weights how error and evidence drive revision and consistency. It carries no readout adapter, and — closing the degenerate basin — it receives **no sparsity penalty** (§10) and enters the consistency functional **detached** (§7). $g_t$ flows to the output and register paths directly.

---

## 7. Global Block: Register, Consistency

**Register initialization** (once per token, before any silent step):

$$
\sigma_t^{0}=\operatorname{LN}\!\big(W_{\sigma 0}[\,\sigma_{t-1},\ g_t,\ \tilde e_t,\ \mu_t\,]\big)
$$

**Precision-weighted consistency** (a measurable feature, not a solved objective), with $\bar\mu=\operatorname{stopgrad}(\mu_t)$:

$$
d_G(\sigma)=P_G g_t-Q_G\sigma,\qquad
d_R(\sigma)=P_R\,r^{\sigma}-Q_R\sigma,\qquad
r^{\sigma}=S_t R(\phi_t)W_q^\sigma\sigma
$$

$$
\mathcal{E}_t(\sigma)=\lVert W_G^\mu\bar\mu\odot d_G(\sigma)\rVert^2+\lVert W_R^\mu\bar\mu\odot d_R(\sigma)\rVert^2+\kappa\lVert\sigma-\sigma_{t-1}\rVert^2
$$

The stopgrad means precision *weights* the diagnostic but gradient descent cannot shrink $\mathcal{E}$ by turning precision off — the escape hatch that made v5.3's consistency loss an anti-revision penalty is welded shut.

---

## 8. Ø Phase: Silence as Hypothesis Revision

During silence, $S_t$ is **read, never written**. The read is phase-aligned and per-head: the query projection $W_q^\sigma:\mathbb{R}^{d_\sigma}\to\mathbb{R}^{d}$ produces a vector split across the $H_U$ heads; each head-slice is rotated by that head's ladder and applied to that head's state; the reads concatenate:

$$
q_{\sigma,t}^{j,(h)}=R\big(\phi_t^{(h)}\big)\big[W_q^\sigma\sigma_t^j\big]^{(h)},\qquad
r_t^{j}=\big\Vert_{h}\ S_t^{(h)}q_{\sigma,t}^{j,(h)}
$$

With $z_t^{\varnothing,j}=[\,\sigma_t^j,\ g_t,\ \tilde e_t,\ \mu_t,\ r_t^j\,]$, the revision is a nonlinear gated residual:

$$
\sigma_t^{j+1}=\operatorname{RMSNorm}\!\big(\sigma_t^j+\sigma(W_g z_t^{\varnothing,j})\odot\tanh(W_n z_t^{\varnothing,j})\big)
$$

**Halting: mix losses, never states.** At each step a halting probability

$$
p_j=H_\theta\big(\sigma_t^j,\ \pi_t,\ \mathcal{E}_t(\sigma_t^j)\big),\qquad p_{J_{\max}}\equiv 1,\qquad
w_j=p_j\prod_{i<j}(1-p_i),\qquad \textstyle\sum_j w_j=1
$$

$H_\theta$ deliberately does **not** consume raw $g_t$: the grounding already reaches the decision through all three of its inputs ($\sigma^j$ is initialized and revised from $g_t$; $\mathcal{E}$'s $d_G$ term compares against $g_t$; $\pi$'s features include $g_t$), and the tiny head keeps halting driven by hypothesis-quality signals rather than re-derived evidence.

Each candidate register produces its **own coherent output**:

$$
o_t^{(j)}=W_o\operatorname{LN}\!\big(g_t+W_\sigma\,\sigma_t^{j}\big),\qquad
p^{(j)}(x_{t+1})=\operatorname{softmax}\!\big(E^\top o_t^{(j)}\big)
$$

Training minimizes the expected loss over the halting distribution,

$$
\mathcal{L}_{\text{LM}}=\sum_{j=0}^{J_{\max}} w_j\,\big[-\log p^{(j)}(x_{t+1})\big],
\qquad \mathbb{E}[J_t]=\sum_j j\,w_j
$$

and the register **carried into $t{+}1$ is a single candidate**, $\sigma_t=\sigma_t^{j^*}$: during training $j^*\!\sim\!\operatorname{Categorical}(w)$ (the halting head receives gradient through the loss mixture, so no straight-through estimator is needed); at inference $j^*=\min\{j:p_j\ge\delta\}$, or under the stage-4 pressure-triggered policy $j^*=\mathcal{J}(\pi_t)$ — run to depth $K$ only when $\pi_t$ clears a threshold, else $j^*=0$ (§12). No convex blend of hypotheses ever exists in the state: every $\sigma^j$ the probe sees, the intervention edits, or the next token inherits is a coherent interpretation. This is what makes the σ-diagnostics of §16 mean what they claim to mean, and it removes the soft-train/hard-infer mismatch. With $J_{\max}=2$, the cost is at most three output-head evaluations on tokens where silence fires — cheap at reference scale and payable only where $\pi$ spends it.

---

## 9. Global Block: Integration Pressure

Disagreement features (all cheap — no full-vocabulary softmax on the critical path):

$$
\Delta_t^{e}=\lVert\tilde e_t\rVert,\qquad
\Delta_t^{\sigma R}=\lVert P_R r_t^{0}-Q_R\sigma_t^{0}\rVert,\qquad
r_t^0=S_tR(\phi_t)W_q^\sigma\sigma_t^0
$$

Summary $\zeta_t = g_t$ (parameter-free — the manifest carries no pooling weights; $\sigma^0$, $\tilde e$, and $\mu$ reach the pressure decision through $\Delta^{\sigma R}$, $\Delta^e$, and the halting head's $\mathcal{E}$); with the top layer's pressure drive $s_t$:

$$
\pi_t=\operatorname{softplus}\!\Big(w_\pi^\top\tanh\!\big(W_\pi[\,\zeta_t,\ \Delta_t^{e},\ \Delta_t^{\sigma R},\ s_t\,]\big)\Big)
$$

Full predictive entropy $H_t$ is **not** a base feature: it costs a 49k softmax before every halting decision. A registered week-one ablation (§14) tests whether adding an entropy signal (true $H_t$, or a small proxy head regressed onto it) improves $\operatorname{corr}(\pi,b)$ enough to pay for itself; the default assumption is that $\Delta^e$ and $\Delta^{\sigma R}$ carry the signal.

---

## 10. Training Objective

$$
\mathcal{L}=\mathcal{L}_{\text{LM}}+\mathcal{L}_{\text{pressure}}+\mathcal{L}_{\text{pred}}+\lambda_C\,\mathbb{E}[J_t]+\mathcal{L}_{\text{consistency}}+\lambda_\mu\textstyle\sum_\ell\lVert\mu_t^\ell\rVert_1+\lambda_S\lVert S_t\rVert^2
$$

with

$$
\mathcal{L}_{\text{pred}}=\lambda_P\big\lVert\hat g_t-\operatorname{stopgrad}(g_t)\big\rVert^2,
\qquad
\mathcal{L}_{\text{consistency}}=\lambda_E\sum_j\max\!\big(0,\ \mathcal{E}_t(\sigma_t^{j+1})-\mathcal{E}_t(\sigma_t^j)\big)
$$

Two deliberate asymmetries, both anti-degeneracy: the $\ell_1$ sparsity applies to **per-layer precision only** (the global $\mu$ may not be starved to zero to satisfy a regularizer), and $\mathcal{E}$ consumes **detached** precision (§7). Together with the fixed-$K$ pressure label below, the "precision off / register frozen / silence never fires" basin of v5.3 is no longer a low-loss solution.

---

## 11. Counterfactual Silence Benefit

For a token in the supervision set, run both branches with **paired determinism** — same batch, same teacher-forced continuation, shared dropout masks (or dropout disabled for the measurement), same precision path — differing *only* in whether silence fired at $t$:

$$
\ell_0=-\log p^{(0)}(x_{t+1}),\qquad \ell_K=-\log p^{(K)}(x_{t+1}),\qquad b_t=\ell_0-\ell_K
$$

Short-horizon variant for tasks whose payoff is downstream: $b_t^{(K,H)}=\sum_{r=1}^{H}\omega_r(\ell_{0,t+r}-\ell_{K,t+r})$, with $\omega_r$ fixed before training (uniform or exponentially decaying; never learned — the label scale must stay interpretable) and **downstream silence frozen off in both branches** over the window — the label is the marginal causal value of the single decision at $t$, uncontaminated by cascades.

> **Policy-independent label (committed).** The with-silence branch always runs **fixed depth $K$** (the forced-exploration depth), never the live halting policy. The halting head consumes $\pi_t$; if the label were computed under the policy, $\pi$ would be regressed onto a target that moves with $\pi$. Fixing $K$ makes the target a property of the *mechanism*, not the *policy*. The residual policy-gap is measured, not ignored: after training, report $\operatorname{corr}(\pi_t,\ b_t^{\text{on-policy}})$ beside the fixed-$K$ correlation.

**Fixed calibrated target.** No batch normalization of the label — that destroys the scale that makes $\delta$ and $\lambda_C$ interpretable. Instead a fixed monotone squash in nats, $\beta=0.02$:

$$
y_t=\log\!\Big(1+\frac{\max(b_t,0)}{\beta}\Big),\qquad
\mathcal{L}_{\text{pressure}}=\big(\pi_t-\operatorname{stopgrad}(y_t)\big)^2
$$

(Honest caveat: $b_t$ in nats shrinks as the model improves, so $\pi$ tracks a slowly drifting target even under a fixed transform. The transform reduces the drift; it cannot remove it.)

**Forced exploration.** Sample $z_t\sim\operatorname{Bernoulli}(p_{\text{explore}})$; if $z_t=1$, run $J=K$ regardless of $\pi_t$ and record the label. Anneal $p_{\text{explore}}\downarrow 0$ as calibration improves. Exploration is what lets the pressure head observe benefit on tokens its current policy would skip.

---

## 12. Training Schedule and Guards

**Stage 1 — evidence core** ($J_t=0$): train A/U/M and the prediction head with $\mathcal{L}_{\text{LM}}+\mathcal{L}_{\text{pred}}+\lambda_\mu\sum_\ell\lVert\mu^\ell\rVert_1+\lambda_S\lVert S\rVert^2$.

> **Scale-free pressure gate.** Enable $\mathcal{L}_{\text{pressure}}$ only when the prediction head beats the trivial predictor by a margin on held-out data:
> $$1-\frac{\lVert e_t\rVert^2}{\lVert g_t-\bar g\rVert^2} > \eta_{R^2}\qquad(\bar g=\text{running mean};\ \ \eta_{R^2}=0.15\ \text{reference})$$
> Before this, $b_t$ is computed through an untrained $\hat g$ and is noise; a pressure head trained on noise labels miscalibrates stickily. An $R^2$-style gate is scale-free across tasks and training stages, unlike an absolute loss threshold.

**Stage 2 — forced revision:** silence forced on a sparse subset at $J{=}K\in\{1,2\}$; add $\mathcal{L}_{\text{pressure}}$ (fixed-$K$ labels) and $\mathcal{L}_{\text{consistency}}$.

**Stage 3 — soft halting:** enable the loss-mixture halting of §8 with $p_{J_{\max}}{=}1$; add $\lambda_C\mathbb{E}[J_t]$ (from near zero — the compute penalty is the collapse knob and is watched jointly with $\operatorname{corr}(\pi,b)$); anneal $p_{\text{explore}}\to 0$ while keeping a small floor so fixed-$K$ labels never fully vanish.

**Stage 4 — event-triggered inference:** hard or pressure-triggered halting; silence fires only at high expected benefit.

**Parallelism profile (C7, stated once, honestly).** Stage-1 training of the core is a chunked parallel scan. From Stage 2 on, the global block introduces a strict sequential dependency across tokens ($\sigma_{t-1}\!\to\!\hat g_t\!\to\!e_t\!\to\!\mu_t\!\to\!\sigma_t^0\!\to\!\sigma_t$). Its working state is $[B,\ d_\sigma + d + H_U d_h^2]$ — the block steps the top layer's evidence state $S$ alongside $\sigma$ to serve the predictive read ($S_{t-1}$ at $\phi_{t-1}$) and the silent read ($S_t$ at $\phi_t$) from the per-token write tensors the core scan already produced. **Training memory:** the token loop runs under exact-gradient segment checkpointing (reference segment 64) — only segment-boundary carries $(S,\sigma)$ are stored, in-segment states are recomputed on backward, and the Categorical draw for $j^*$ uses pre-drawn uniforms so recomputation reproduces the same sample; gradients are exact, so seq-4096 training never materializes the per-token $S$ chain. **Wall-clock:** implement the loop as one fused sequential kernel after the core's scan completes. The sequential depth per optimizer step equals the sequence length (longer sequences buy the recurrence more range on the evidence-age axis at the price of serial time; 4096 is the modern main-pretraining norm alongside 8192). At $d_\sigma{=}128$ this remains minutes-per-epoch overhead at Tiny scale — acceptable, measured, and *declared*, rather than an undisclosed violation of a scannability claim.

---

## 13. Reference Configuration and Training Recipe: AUM-Ø-Tiny v6

| Field | Value |
|---|---|
| $d_{\text{model}}$ / layers $L$ | 512 / 12 evidence + 1 global block |
| Vocab (tied) | 49 152 |
| MLP $d_{\text{ff}}$ (SwiGLU) | 1408 |
| A: heads / kv / head-dim / window | 8 / 2 / 64 / 256 |
| U: heads $H_U$ / head-dim $d_h$ / ladder | 4 / 128 / $B{=}64$, $\omega\in[10^{-3},1]$ geometric |
| Controller $d_c$ / precision $k_\mu$ / register $d_\sigma$ / phase-embed $d_\phi$ | 128 / 32 / 128 / 32 |
| $J_{\max}$ / seq len | 2 / 4096 |
| Params: total / silence block / ablated baseline | ≈ 78 M / ≈ 1.8 M / ≈ 76.5 M |

**Data.** ~20 B tokens: filtered web/edu mix, ~10 % code, and **5 % synthetic structured tasks** generated programmatically with known latent hypotheses (branch reversal, binding swap, delayed correction, flat null — §14), with *held-out generators* so probe and calibration numbers are measured on unseen task structure. The synthetic fraction is not optional garnish; the σ-decode probe and the recency falsifier are computed on it.

**Optimization.** Muon (momentum 0.95, 5 Newton–Schulz steps, LR 0.02 in spectral-norm units, wd 0.1) on the 2D hidden weight matrices; AdamW $\beta{=}(0.9,0.95)$, peak LR $6\times10^{-4}$, no weight decay, for the tied embedding/classifier and all scalars (norms, gains, `A_log`, `dt_bias`, $D$, biases, the depthwise conv). Both groups share a 1500-step warmup and cosine to 10 %; grad-clip 1.0; batch ≈ 0.5 M tokens; BF16 with FP32 optimizer states, `A_log`, and state-norm accumulators; init $\mathcal{N}(0,0.02)$ with `A_log`, `dt_bias` set for $\alpha\approx0.99$, $\tau\approx1$ at init. Stage split over the 20 B: 60 / 20 / 15 / 5 %. Loss weights (reference): $\lambda_P{=}0.5$, $\lambda_E{=}0.1$, $\lambda_\mu{=}10^{-3}$, $\lambda_S{=}10^{-4}$, $\lambda_C: 0\!\to\!5\times10^{-3}$ ramped in Stage 3; $\beta{=}0.02$, $\eta_{R^2}{=}0.15$, $K{\in}\{1,2\}$, $p_{\text{explore}}:0.2\!\to\!0.02$ floor, $\delta{=}0.5$.

Wall-clock at 8×H100, ~40 % MFU: order of a day, dominated by the core scan; the global block's sequential kernel and the ≤3 output-head passes on silence-fired tokens are second-order.

---

## 14. Pre-Registered Evaluation

**Primary claim.** Benefit-gated hypothesis revision improves continuation at sparse interpretive events, and pressure allocates that revision where it pays.

**Task families** (synthetic, latent hypothesis known, evidence-age controlled): **branch reversal** (a rule holds; a reversal token flips it); **latent binding swap** (`A=red … Correction: A and C were swapped. What color is A?` — same structure, different surface, so a "reversal-token detector" cannot pass); **delayed correction / long-range recall** (old evidence must be reinterpreted — the *evidence-age axis*, measured, not gated); **flat null** (no interpretive events).

**Registered differential prediction — recency in phase distance.** For each instance, compute the accumulated phase distance $\Delta\phi=\phi_{t_{\text{reinterpret}}}-\phi_{t_{\text{evidence}}}$ between the evidence and the point of reinterpretation (per head, summarized by the mean over heads), and predict

$$
\operatorname{corr}\!\big(b_t,\ \Delta\phi\big)<0
$$

with token-age as the secondary covariate. Under the frequency ladder, retrieval decays with phase distance; if learned phase velocities are roughly constant the two axes coincide, and if they are not, phase distance is the mechanistically correct one — report the phase-velocity distribution alongside. Confirming the *gradient* is stronger than confirming benefit exists: it confirms retrieval works the way the architecture says.

**Registered null prediction.** On the flat task: $\pi_t\approx0$ and $\mathbb{E}[J_t]\to0$. Firing on the null means pressure learned surface uncertainty, not integration benefit.

**Named baselines** (parameter- and compute-matched):
**Evidence-core** (~76.5 M): silence ablated, $g_t$-only output — the capacity floor.
**Top-GRU adapter** (~1.8 M top block): identical access to $g_t,e_t,\mu_t$, identical halting machinery, **no evidence read** — $\sigma^{j+1}=\operatorname{GRU}(\sigma^j,[g_t,e_t,\mu_t])$ — and, critically, a **pooled-evidence prediction head** $\hat g_t = W_P\operatorname{LN}(W_S\operatorname{Pool}(S_{t-1})+W_\sigma\sigma_{t-1}+W_\phi\Phi(\phi_{t-1}))$, with $\operatorname{Pool}(S)=S\,q_{\text{pool}}$ per head — a **learned static query** ($q_{\text{pool}}\in\mathbb{R}^{d}$, phase-free, $\sigma$-free; mean pooling is its uniform special case). The pool is deliberately the strongest evidence summary available *without* hypothesis- or phase-conditioned addressing (C9: the baseline is generous to the null), so its $e_t$ is not handicapped and the single ablated factor is the silent read of $S$ through $\sigma$. If full AUM-Ø does not beat this per compute, the evidence-read mechanism has not earned its complexity.

**Mechanism-isolating controls** (on the full model): **no-op silence** (loop runs, register frozen at $\sigma^0$ — is it *revision* or compute?); **no-read silence** ($r^j{=}0$ — is *reading $S$* necessary?); **phase-scrambled silence** ($q^j_\sigma=R(\phi_t+\epsilon_t)W_q^\sigma\sigma^j$, $\epsilon_t$ shuffled across tokens — is *phase-aligned* reading, not $\alpha$-decay, the cause?); **random silence** (matched $\mathbb{E}[J]$, random tokens — does *allocation* matter?).

### Minimum-viable-proof table

| Test | Type | Expected | Proves | Gate |
|---|---|---|---|---|
| Full AUM-Ø v6 | reference | improves reversal **and** swap | mechanism can help | — |
| Evidence-core baseline | baseline | worse than full | silence adds value | ✅ |
| Top-GRU adapter | baseline | full beats it per compute | evidence read earns complexity | ✅ |
| No-op Ø | control | no gain | revision, not compute | ✅ |
| No-read Ø | control | reduced gain | reading $S$ matters | — |
| Phase-scrambled Ø | control | reduced gain | phase addressing matters | ✅ |
| Random silence | control | worse efficiency | allocation matters | — |
| No $\mathcal{L}_{\text{pressure}}$ | ablation | misallocation | benefit supervision matters | — |
| No $\mathcal{L}_{\text{pred}}$ | ablation | weak $\operatorname{corr}(\pi,b)$ | prediction error matters | — |
| Entropy feature on/off | ablation | little change expected | $H_t$ not worth its softmax | — |
| Flat null | control | $\pi\!\approx\!0$, $\mathbb{E}[J]\!\to\!0$ | not uncertainty-firing | ✅ |
| σ-decode probe (held-out) | diagnostic | above chance | register tracks hypothesis | ✅ |
| σ-intervention | diagnostic | causal output shift | register is *used* | — |
| Recency gradient (phase distance) | prediction | $\operatorname{corr}(b,\Delta\phi)<0$ | phase-addressed retrieval is the cause | ✅ |

---

## 15. Gate Before Scaling

All ✅ rows must pass on held-out branch-reversal **and** binding-swap generators. Delayed correction and long-range recall are **measured along the evidence-age axis, not gated** — base v6 is *predicted* to help them less; a weak number there is the design speaking, not failing.

**Failure triage — read the σ-decode probe first.** Probe fails → the register genuinely cannot represent the hypothesis; only now consider widening $d_\sigma$ or slots (§17). Probe passes, no benefit → σ holds the hypothesis but revision isn't landing; inspect no-op / no-read / phase-scrambled results, the prediction head, the silent update — not $d_\sigma$. No-op recovers the gain → the "benefit" was compute; mechanism unvalidated regardless of headline numbers. Recency gradient absent (with phase-scrambled *also* showing no gap) → retrieval is not phase-addressed; the benefit, if real, has a different cause and the mechanism claim must be withdrawn or reworked. Null fires → pressure is uncertainty-triggered; re-examine the label pipeline (paired determinism, fixed-$K$) before anything else.

---

## 16. Diagnostics

Continuous: $\phi$ per head and the **phase-velocity distribution** (needed to interpret the recency axis), $\alpha_t$, $\rho_t$, $\lVert e_t\rVert$, $\lVert\mu_t\rVert$, $\pi_t$, $J_t$, $b_t$, held-out $\operatorname{corr}(\pi,b)$ at fixed $K$, the on-policy correlation gauge, $\mathcal{E}(\sigma^0)-\mathcal{E}(\sigma^{j^*})$, and efficiency $b_t/(1+\mathbb{E}[J_t])$ (reported, never trained against in v6).

**Hypothesis inertia and the quartet.** $\Delta\sigma_t^{\text{silent}}=\lVert\sigma_t^{j^*}-\sigma_t^0\rVert$; register the co-firing $\pi\!\uparrow,J\!\uparrow,\Delta\sigma\!\uparrow,b\!\uparrow$ at interpretive events and its joint flatness on null — this separates "ran but didn't revise" from "revised without benefit."

**Per-step decomposition.** Attribute gains to $\sigma^0\!\to\!\sigma^1$ vs $\sigma^1\!\to\!\sigma^2$; if step 2 adds nothing, $J_{\max}{=}1$ is the honest setting and the spec says so in the paper.

**σ-decode probe** (held-out generators): linear decode of the active latent rule from $\sigma_t$.

**σ-intervention (causal).** Overwrite $\sigma_t$ with a stored register from a different-rule example and continue: predict the output shifts toward the injected rule, next-step $e$ rises when the injected rule is wrong, and reversal answers flip or degrade. Decodability shows correlation; intervention shows the register is *used as* the hypothesis.

**σ-relevance of the prediction head.** Measure $\hat g$'s degradation when the $\sigma$ input to $P_\theta$ is zeroed or shuffled across the batch. If degradation is negligible, the head has learned to predict from phase/state alone ($W_q^{\text{pred}}\!\to\!0$) and the accountability loop of C5 is silently severed — a failure no other diagnostic detects.

**Evidence-survival probe.** On old-evidence tasks, before blaming addressing: check whether the target association is still linearly recoverable from $S_t$ at query time. Not recoverable → the evidence *decayed* ($\alpha$ story) and no read policy — including future temporal search — can retrieve it; the remedy is retention (dissolution priors, or the correction patch), not search. Recoverable but unread → the addressing story stands and temporal search is the right extension. This probe is what keeps the recency result interpretable.

---

## 17. Out of Scope (ordered follow-ups, each contingent on the gate)

**Temporal hypothesis search** — learnable read-phase offset $q^j_\sigma=R(\phi_t+\delta^j)W_q^\sigma\sigma^j$, $\delta^j=\psi_\theta(\sigma^j)$, regularized for *spread across silent steps* (penalize $\delta^1\!\approx\!\delta^2$) so it cannot collapse to a fixed backward glance; turns silence into a temporal scan of $S$. Pursued only if the evidence-survival probe shows old evidence *survives but goes unread*. **Hypothesis-conditioned core read** — $\tilde q_{t+1}\leftarrow\tilde q_{t+1}\odot(1+W_h\sigma_t)$ on the readout path only; downstream reach without touching $S$. **Pooled cross-layer evidence read**, **slot register** $\sigma\in\mathbb{R}^{n_\sigma\times d_{\sigma s}}$, **correction patch** $\tilde S=S+C$ (nonlinear correction routed outside $S$, honoring C2), **layer-local registers**, **salience writes / MTP head**, and a **variational reading** of $\sigma$ as an amortized posterior $q_\theta(z\mid S,g)$ with a KL bottleneck — a theory direction that would make "hypothesis register" mathematically literal, not a Tiny change.

---

## 18. Relationship to Prior Work (positioning, to be cited properly at write-up)

The evidence core is a gated linear-attention / selective-SSM recurrence in the Mamba-2 / GLA family; the frequency-ladder rotation is RoPE generalized to a *data-dependent* position (accumulated phase); adaptive silent depth descends from ACT and PonderNet; per-token latent iteration relates to recurrent-depth latent reasoning; the error/precision vocabulary is predictive coding made architectural; test-time memory writes invite comparison with Titans, from which C3 (stable addressing) is a deliberate departure. The claimed delta over all neighbors is singular and testable: **counterfactual-benefit-supervised allocation of latent revision over an explicitly separated evidence store** — which is exactly what the Top-GRU baseline and the control set exist to isolate. Verify exact mechanisms against the literature before citing; the family resemblances are stated from memory.

---

## 19. Core Definition

AUM-Ø v6 maintains $\chi_t=(\phi_t,S_t,\sigma_t,\mu_t)$: a multi-frequency phase position, an **affine** phase-addressed evidence state written once per token, a small nonlinear hypothesis register revised only by benefit-gated silence, and a precision field weighting how error and evidence drive revision. Prediction reads evidence through the hypothesis; error funds precision and pressure; pressure — trained on fixed-depth, paired-deterministic, frozen-downstream counterfactual benefit on a fixed calibrated scale — decides when the register earns another revision step; halting mixes losses over coherent candidates and carries exactly one hypothesis forward.

$$
A \rightarrow U \rightarrow M \rightarrow \varnothing:\qquad
\text{observe}\ \rightarrow\ \text{accumulate}\ \rightarrow\ \text{weigh}\ \rightarrow\ \text{revise when it pays.}
$$

$$
\text{affine resonant evidence core}\ +\ \text{benefit-gated global hypothesis register}
$$

$$
\text{Continuation arises from temporary configuration.}
$$

---

## Appendix A. Physical Layout — AUM-Ø-Tiny v6 (78,255,136 params)

Format: `name,[shape],dtype` — these are the reference implementation's **actual state-dict keys** (the `backbone.` prefix written as `model.`; `lm_head.weight` is tied to the embedding and stored once), so the manifest is checkable against a checkpoint with one line of code. Evidence layers `[0-11]`; silence subsystem a single top-level block. Exact totals: **78,255,136** parameters; silence block **1,769,408**; silence-ablated evidence core **76,485,728**. The ladder buffer `rope_freqs` is fixed, non-trainable, and excluded from the counts.

```
model.embedding.weight,[49152,512],BF16                  # tied to lm_head.weight

# ---- A: bounded local GQA grounding (all evidence layers) ----
model.layers.[0-11].input_layernorm.weight,[512],BF16
model.layers.[0-11].ground_attn.q_proj.weight,[512,512],BF16
model.layers.[0-11].ground_attn.k_proj.weight,[128,512],BF16
model.layers.[0-11].ground_attn.v_proj.weight,[128,512],BF16
model.layers.[0-11].ground_attn.o_proj.weight,[512,512],BF16
model.layers.[0-11].ground_attn.q_norm.weight,[64],BF16
model.layers.[0-11].ground_attn.k_norm.weight,[64],BF16

# ---- U: resonant AFFINE evidence recurrence + output gate (all layers) ----
model.layers.[0-11].unfold.dt_bias,[4],BF16
model.layers.[0-11].unfold.A_log,[4],F32
model.layers.[0-11].unfold.D,[512],BF16
model.layers.[0-11].unfold.rope_freqs,[64],F32           # BUFFER: the fixed geometric ladder
model.layers.[0-11].unfold.controller.weight,[128,512],BF16
model.layers.[0-11].unfold.in_proj_qkv.weight,[1536,512],BF16
model.layers.[0-11].unfold.in_proj_z.weight,[512,512],BF16
model.layers.[0-11].unfold.in_proj_dyn.weight,[49,128],BF16
model.layers.[0-11].unfold.conv1d.weight,[1536,1,4],BF16
model.layers.[0-11].unfold.conv1d.bias,[1536],BF16
model.layers.[0-11].unfold.norm.weight,[128],F32
model.layers.[0-11].unfold.out_proj.weight,[512,512],BF16

# ---- M: error-free precision (all evidence layers) ----
model.layers.[0-11].modulate.in_proj_mu.weight,[32,1056],BF16
model.layers.[0-11].modulate.down.weight,[32,512],BF16
model.layers.[0-11].modulate.up.weight,[512,32],BF16

# ---- MLP + post-norm (all evidence layers; fc1 fuses [gate; up]) ----
model.layers.[0-11].post_attention_layernorm.weight,[512],BF16
model.layers.[0-11].mlp.fc1.weight,[2816,512],BF16       # SwiGLU: rows 0-1407 gate, 1408-2815 up
model.layers.[0-11].mlp.fc2.weight,[512,1408],BF16       # down projection

# ---- Global block: hypothesis-conditioned predictive grounding ----
model.silence.predict.query_proj.weight,[512,128],BF16   # W_q^pred: sigma -> state-key
model.silence.predict.read_proj.weight,[512,512],BF16    # W_R
model.silence.predict.hyp_proj.weight,[512,128],BF16     # W_sigma^P
model.silence.predict.phase_proj.weight,[512,32],BF16    # W_phi
model.silence.predict.out_proj.weight,[512,512],BF16     # W_P
model.silence.predict.norm.weight,[512],BF16             # LayerNorm (with bias)
model.silence.predict.norm.bias,[512],BF16

# ---- Global block: error-fed precision (precision only; no adapter, no L1) ----
model.silence.modulate.err_proj.weight,[32,512],BF16     # W_e
model.silence.modulate.in_proj_mu.weight,[32,1184],BF16  # W_mu

# ---- Global block: register + revision ----
model.silence.register.init_proj.weight,[128,704],BF16   # W_sigma0
model.silence.register.read_proj.weight,[512,128],BF16   # W_q^sigma (split per U-head)
model.silence.register.update_gate.weight,[128,1216],BF16
model.silence.register.update_cand.weight,[128,1216],BF16
model.silence.register.norm.weight,[128],BF16            # shared by init (LN) + revise
model.silence.register.norm.bias,[128],BF16

# ---- Global block: consistency ----
model.silence.consistency.P_G.weight,[128,512],BF16
model.silence.consistency.Q_G.weight,[128,128],BF16
model.silence.consistency.P_R.weight,[128,512],BF16
model.silence.consistency.Q_R.weight,[128,128],BF16
model.silence.consistency.prec_G.weight,[128,32],BF16
model.silence.consistency.prec_R.weight,[128,32],BF16

# ---- Global block: pressure + halting ----
model.silence.pressure_halt.pressure_in.weight,[128,515],BF16  # zeta(512)+d_e+d_sigmaR+s_t
model.silence.pressure_halt.pressure_out,[128],BF16      # w_pi (bare vector parameter)
model.silence.pressure_halt.halt_1.weight,[64,130],BF16  # sigma(128)+pi+E
model.silence.pressure_halt.halt_2.weight,[1,64],BF16

# ---- Global block: output / condition projection ----
model.silence.condition_out.weight,[512,512],BF16        # W_o
model.silence.condition_out_sigma.weight,[512,128],BF16  # W_sigma
model.silence.condition_norm.weight,[512],BF16
model.silence.condition_norm.bias,[512],BF16

model.norm_f.weight,[512],BF16
```

**Shape notes.** `rope_freqs,[64]` is the per-head geometric ladder ($B{=}64$ blocks of 2 over $d_h{=}128$), fixed. `in_proj_z` is the U output gate ($\operatorname{silu}(z)\odot\cdot$). `in_proj_dyn,[49,128]` packs $(\bar\tau,\bar\lambda,r,\theta)$ per U-head $(4{\times}4)$ + $m(32)$ + $s(1)$. `register.read_proj` output (512) splits into 4 head-slices of 128, each rotated by that head's ladder at that head's $\phi$. `register.init_proj` in: $704=\sigma(128){+}g(512){+}\tilde e(32){+}\mu(32)$; `register.update_*` in: $1216=704{+}r^j(512)$; `modulate.in_proj_mu` in: $1184=g(512){+}e(512){+}\sigma(128){+}m(32)$; `pressure_in` in: $515=\zeta(512){+}\Delta^e{+}\Delta^{\sigma R}{+}s_t$ (516 when the optional entropy feature is enabled for its ablation); `halt_1` in: $130=\sigma(128){+}\pi{+}\mathcal{E}$ (no raw $g$ — see §8). Embeddings tied to the classifier. Loss-mixture halting adds no parameters — only up to $J_{\max}{+}1$ output-head evaluations on silence-fired tokens. The Top-GRU baseline additionally carries `silence.gru.*`, `silence.pool_proj.weight,[512,512]`, and `silence.pool_query,[512]` (§14) — baseline-only tensors, not part of this reference manifest. In the BF16 training configuration, `A_log`, `unfold.norm.weight`, and `rope_freqs` stay F32.

## Appendix B. Run Checklist

**Baselines:** evidence-core (~76.5 M); Top-GRU adapter (~1.8 M top block, pooled-S prediction head, no evidence read). **Controls (full model):** no-op Ø; no-read Ø; phase-scrambled Ø; random silence (matched $\mathbb{E}[J]$). **Ablations:** no $\mathcal{L}_{\text{pressure}}$; no $\mathcal{L}_{\text{pred}}$; entropy feature on/off. **Diagnostics:** σ-decode (held-out generators); σ-intervention; σ-relevance of the prediction head; evidence-survival probe; Δσ quartet; per-step decomposition; phase-velocity distribution; on-policy correlation gauge. **Guards:** scale-free $R^2$ pressure gate before Stage 2; $\lambda_C$ ramped from ~0 in Stage 3, watched jointly with $\operatorname{corr}(\pi,b)$; $p_{\text{explore}}$ annealed to a floor, never zero, so fixed-$K$ labels persist. **The gate** is the ✅ subset of §14's table, evaluated on held-out branch-reversal and binding-swap; delayed correction is measured on the evidence-age axis, not gated.