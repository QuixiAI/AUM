# AUM-Ø v5.3 (final): Attentive Unfolding Modulation with Silence

## Recent-Evidence Hypothesis Revision — An Affine Resonant Evidence Core with a Benefit-Gated Global Hypothesis Register

**AUM-Ø**, pronounced **Aum-nought**, is a phase-typed recurrent architecture for adaptive test-time inference. Its core principle is:

$$
\text{Continuation arises from temporary configuration.}
$$

AUM-Ø separates an **evidence state** $S_t$ that accumulates observations across the sequence from a **hypothesis register** $\sigma_t$ that holds the current interpretation of that evidence. The token clock accumulates evidence; the silence clock revises hypothesis. The four phases are $A \rightarrow U \rightarrow M \rightarrow \varnothing$ — observe, accumulate evidence, assign precision, revise hypothesis when useful.

This is the **final** specification. Every choice below is committed. The base model performs **recent-evidence hypothesis revision**: the silent read is phase-locked to the current token and therefore preferentially reinterprets recent evidence. Reaching older evidence is the job of the **temporal hypothesis search** extension (§24), which is explicitly *not* part of this specification. The recency bias is a designed property that yields a falsifiable mechanistic prediction (§22), not a limitation to be patched before the first run.

The defining claim:

$$
\text{Adaptive inference requires separating evidence from interpretation, and spending extra computation only where revising the interpretation pays.}
$$

---

## 0. Design Commitments

These forks are settled. They are recorded here as the architecture's rationale, not as open choices.

1. **Top-only hypothesis register.** One global $\sigma$ and one shared halting policy, attached as a single block on top of the evidence stack. Predictive grounding, prediction error $e_t$, error-fed precision, and $\sigma$ exist only there. The $L$ evidence layers run pure token-clock recurrence. This makes the silence mechanism a ~2% add-on over a nearly parameter-matched evidence-core baseline, so any measured benefit is attributable to the mechanism, not to capacity.
2. **Affine evidence recurrence (invariant).** The $S$-update is affine in $S_{t-1}$; no nonlinearity inside the state recurrence. This preserves scan-parallel training and efficient recurrent inference, and is invariant across all variants and extensions.
3. **Stable evidence addressing.** Prediction error never perturbs the write key $\hat k_t$. Surprise may modulate write strength or value, never the address.
4. **Bottlenecked register.** $d_\sigma \ll d$, fixed at $128$ in the reference model. The register is forced toward a low-dimensional interpretation rather than a second readout buffer. Whether this is too small is answered by the σ-decode probe at the gate (§23), not assumed.
5. **Hypothesis-conditioned predictive grounding.** The prediction head reads the evidence state *through the current hypothesis*, symmetric with the silent read. A wrong hypothesis reads the wrong evidence and predicts the next grounding poorly — this is how $\sigma$ is held accountable.
6. **Phase-locked silent read (recent-evidence).** The silent read uses the current phase $\phi_t$, giving recency-selective retrieval and a registered differential prediction. Temporal search ($\delta_j$) is a scoped follow-up, deliberately excluded here so the recency falsifier stays clean.
7. **Frozen-downstream counterfactual benefit with a fixed calibrated target.** Benefit is the marginal value of a single silence decision, measured with downstream silence frozen and paired stochasticity; the pressure target is a fixed monotone transform of benefit in nats, not a batch-normalized scale.
8. **Mechanism-isolating controls are part of the spec.** No-op, no-read, phase-scrambled silence, a Top-GRU adapter baseline, and a causal σ-intervention are first-class evaluation components, not optional extras.

---

## 1. The Inference Condition

Let the input be $x_1,\dots,x_T$ with embeddings $x_t\in\mathbb{R}^d$. AUM-Ø maintains:

$$
\chi_t = (\phi_t,\, S_t,\, \sigma_t,\, \mu_t)
$$

with $\phi_t$ the resonance phase, $S_t$ the evidence state, $\sigma_t$ the hypothesis register, $\mu_t$ the precision field. Two clocks: the **token clock** advances once per token, $(x_t,\chi_{t-1})\mapsto\chi_t$; the **silence clock** advances internally, $\sigma_t^j\mapsto\sigma_t^{j+1}$. The evidence state is updated only by the token clock; the hypothesis register is revised only by the silence clock.

---

## 2. Evidence and Hypothesis

$S_t$ is an associative recurrent substrate carrying traces of what has been observed. $\sigma_t$ is an interpretive state — the model's current latent explanation of the evidence: which rule is active, which branch is followed, which binding is valid, which reading of an ambiguous instruction holds, whether a correction reversed a prior assumption. The evidence state carries *what* was observed; the hypothesis register carries *how* it is interpreted. Therefore $\varnothing = \text{hypothesis revision}$ — latent reinterpretation of accumulated evidence, not further accumulation.

---

## 3. Placement: A Single Global Silence Block

$$
\underbrace{\text{evidence core over } L \text{ layers}}_{\text{token clock, all layers}} \;\rightarrow\; \underbrace{\text{global hypothesis revision}}_{\text{silence clock, once}} \;\rightarrow\; \text{output}
$$

The lower stack carries no $\sigma$ and computes no $e_t$. The global block reads the **top evidence layer's** $S_t$ and $\phi_t$ (notation below refers to the top layer unless stated). There is one silent depth per token, $J_t$.

---

## 4. Evidence Core and Predictive Grounding

The evidence core processes $x_t$ through all $L$ layers and emits a top-of-stack grounded summary:

$$
g_t = \operatorname{EvidenceCore}(x_t, \chi_{t-1}) \in \mathbb{R}^d
$$

The A phase (bounded local grounding) occurs throughout the core; the global block receives only $g_t$, distinct from any layer's internal $h_t^{A,\ell}$.

**Hypothesis-conditioned predictive grounding.** Before $g_t$ is folded into the hypothesis, the global block predicts it by reading the previous evidence state *through the previous hypothesis*, under the previous phase:

$$
q_{t-1}^{\text{pred}} = R(\phi_{t-1})\, W_q^{\text{pred}}\, \sigma_{t-1}, \qquad r_{t-1}^{\text{pred}} = S_{t-1}\, q_{t-1}^{\text{pred}}
$$

$$
\hat{g}_t = W_P\, \operatorname{LN}\!\left( W_R\, r_{t-1}^{\text{pred}} + W_\sigma\, \sigma_{t-1} + W_\phi\, \Phi(\phi_{t-1}) \right)
$$

where $\Phi(\phi)$ is a learned or sinusoidal phase embedding. The prediction read uses a **separate** query projection $W_q^{\text{pred}}$ from the silent read's $W_q^\sigma$ (§12); the two serve different purposes — predicting the next grounding vs. revising the current hypothesis — and are not tied. The prediction error is:

$$
e_t = g_t - \hat{g}_t
$$

This makes $\sigma$ accountable through evidence: a wrong hypothesis reads the wrong slot of $S$ and predicts $g_t$ poorly. The A phase answers: *What is present, and how does it differ from what the current hypothesis expected?*

---

## 5. Compact Controller (Per Evidence Layer)

Each evidence layer forms $\bar{x}_t = \operatorname{LN}(x_t)$, a controller vector $c_t = W_c\bar{x}_t$, content projections $q_t=W_q\bar{x}_t,\; k_t=W_k\bar{x}_t,\; v_t=W_v\bar{x}_t$, and an output-gate projection $z_t = W_z\bar{x}_t$. The controller emits the recurrent dynamics:

$$
(\bar{\tau}_t,\, \bar{\lambda}_t,\, r_t,\, \theta_t,\, m_t,\, s_t) = W_p\, c_t
$$

with $\bar{\tau}_t$ step size, $\bar{\lambda}_t$ dissolution, $r_t$ write strength, $\theta_t$ phase velocity, $m_t$ precision drive, $s_t$ pressure drive. Only the top layer's $s_t$ is consumed (§14).

---

## 6. U Phase: Resonant Evidence Unfolding

Step size, dissolution, survival:

$$
\tau_t = \operatorname{softplus}(\bar{\tau}_t + b_\tau), \quad \lambda_t = \epsilon + f(\bar{\lambda}_t), \quad f(x)=\begin{cases}1+x,&x\geq 0\\[3pt]\tfrac{1}{1-x},&x<0\end{cases}, \quad \alpha_t = \exp(-\lambda_t\tau_t)
$$

Resonance phase and rotated, normalized factors:

$$
\phi_t = \left(\phi_{t-1} + \pi\tanh(\theta_t)\tau_t\right)\bmod 2\pi, \quad \tilde{q}_t=R(\phi_t)q_t,\ \tilde{k}_t=R(\phi_t)k_t,\ \hat{k}_t=\tfrac{\tilde{k}_t}{\lVert\tilde{k}_t\rVert+\epsilon},\ \hat{v}_t=\tfrac{v_t}{\lVert v_t\rVert+\epsilon}
$$

Write gate $\rho_t=\sigma(r_t)$, write $W_t=\hat{v}_t\otimes\hat{k}_t$, **affine** state update:

$$
S_t = \alpha_t\odot S_{t-1} + \rho_t\tau_t\odot W_t
$$

> **Standing invariant (affine evidence recurrence).** $S_t$ is affine in $S_{t-1}$: a per-step diagonal/block-diagonal gain $\alpha_t$ plus an input-dependent additive write. No nonlinearity inside the state recurrence; normalization applies to write inputs and readouts only. Any extension needing nonlinear mutation routes it through $\sigma$, the correction patch, or the readout — never through $S$.

**Gated readout.** The unfolding readout is normalized after the linear read and gated by $z_t$ in the Mamba/GLA lineage:

$$
h_t^{U,(h)} = \operatorname{silu}\!\big(z_t^{(h)}\big)\odot \operatorname{RMSNorm}\!\left(S_t^{(h)}\tilde{q}_t^{(h)} + D^{(h)}v_t^{(h)}\right)
$$

where $h$ indexes heads and $D^{(h)}$ is a learned skip coefficient. The U phase answers: *What evidence has accumulated, and how does it unfold under the current phase?*

---

## 7. Stable Addressing

The base write uses a stable key, $W_t=\hat{v}_t\otimes\hat{k}_t$; error never perturbs $\hat k_t$. Surprise may optionally modulate write strength or value at the top layer (where $e_t$ exists), never the key:

$$
\rho_t' = \sigma\!\left(r_t + w_e^\top\tanh(W_e e_t)\right), \qquad \hat{v}_t' = \hat{v}_t + B_v\tanh(W_e e_t)
$$

Salience-augmented writes are a scoped follow-up (§24); the reference model uses stable, error-free writes throughout the stack.

---

## 8. M Phase: Precision Modulation

**Lower-layer precision (error-free), every evidence layer:**

$$
\mu_t^\ell = \sigma\!\left(W_\mu^\ell[\,h_t^{A,\ell},\,h_t^{U,\ell},\,m_t^\ell\,]\right), \quad \Delta h_t^\ell = U_m^\ell\operatorname{diag}(\mu_t^\ell)V_m^\ell h_t^{U,\ell}, \quad h_t^{M,\ell}=h_t^{A,\ell}+h_t^{U,\ell}+\Delta h_t^\ell
$$

The residual stream accumulates $h_t^{M,\ell}$; the top layer emits $g_t$.

**Global precision (error-fed), once in the silence block — precision only, no readout adapter:**

$$
\mu_t = \sigma\!\left(W_\mu[\,g_t,\,e_t,\,\sigma_{t-1},\,m_t\,]\right)\in[0,1]^k, \qquad \tilde{e}_t = \mu_t\odot W_e e_t
$$

The global $\mu_t$ weights how evidence and error drive revision and consistency. It does **not** carry an up/down readout modulation; $g_t$ enters the output and hypothesis paths directly. M is a precision field, not a second recurrence and not an output adapter.

---

## 9. The Hypothesis Register

Initialized before silence:

$$
\sigma_t^0 = \operatorname{LN}\!\left(W_{\sigma0}[\,\sigma_{t-1},\,g_t,\,\tilde{e}_t,\,\mu_t\,]\right)
$$

It conditions future prediction (§4) and output (§16). Its width is bottlenecked, $d_\sigma=128$.

---

## 10. Ø Phase: Silence as Hypothesis Revision

The Ø phase revises $\sigma$ while keeping $S_t$ fixed — read, not rewritten. The read is **phase-aligned** to the current token:

$$
q_{\sigma,t}^j = R(\phi_t)\,W_q^\sigma\sigma_t^j, \qquad r_t^j = S_t\,q_{\sigma,t}^j
$$

With $z_t^{\varnothing,j}=[\,\sigma_t^j,\,g_t,\,\tilde{e}_t,\,\mu_t,\,r_t^j\,]$, the update is a nonlinear gated residual:

$$
\sigma_t^{j+1} = \operatorname{RMSNorm}\!\left(\sigma_t^j + \sigma(W_g z_t^{\varnothing,j})\odot\tanh(W_n z_t^{\varnothing,j})\right)
$$

The Ø phase answers: *How should the hypothesis be revised in light of the evidence?*

> **Recency property (registered, §22).** The read uses the current phase $\phi_t$, aligning with recently written evidence — right for recent triggers (branch reversal, binding swap), weaker for old evidence (long-range recall, delayed correction). This yields the registered differential prediction $\operatorname{corr}(b_t,\text{evidence-age})<0$, tested against the phase-scrambled control to confirm phase addressing — not mere $\alpha_t$ decay — is the cause.

---

## 11. Precision-Weighted Consistency

$$
d_G(\sigma)=P_G g_t - Q_G\sigma, \qquad d_R(\sigma)=P_R r_t^\sigma - Q_R\sigma, \qquad r_t^\sigma = S_t\,R(\phi_t)\,W_q^\sigma\sigma
$$

With $\mu_G=W_G^\mu\mu_t,\ \mu_R=W_R^\mu\mu_t$:

$$
\mathcal{E}_t(\sigma) = \lVert\mu_G\odot d_G(\sigma)\rVert^2 + \lVert\mu_R\odot d_R(\sigma)\rVert^2 + \kappa\lVert\sigma-\sigma_{t-1}\rVert^2
$$

A trainable, measurable consistency feature and regularizer; the silent transition stays nonlinear.

---

## 12. Integration Pressure

$$
o_t^0 = W_o\operatorname{LN}\!\left(g_t + W_\sigma\sigma_t^0\right), \quad p_0(x_{t+1})=\operatorname{softmax}(E^\top o_t^0), \quad H_t = -\sum_x p_0(x)\log p_0(x)
$$

$$
\Delta_t^e = \lVert\tilde{e}_t\rVert, \qquad \Delta_t^{\sigma R} = \lVert P_R r_t^0 - Q_R\sigma_t^0\rVert, \qquad r_t^0 = S_t\,R(\phi_t)\,W_q^\sigma\sigma_t^0
$$

With $\zeta_t=\operatorname{Pool}_\zeta(g_t,\sigma_t^0,\tilde{e}_t,\mu_t)\in\mathbb{R}^d$ and the top-layer pressure drive $s_t$:

$$
\pi_t = \operatorname{softplus}\!\left(w_\pi^\top\tanh\!\left(W_\pi[\,\zeta_t,\,H_t,\,\Delta_t^e,\,\Delta_t^{\sigma R},\,s_t\,]\right)\right)
$$

$\pi_t$ estimates expected benefit of revision, trained against realized counterfactual benefit (§17).

---

## 13. Soft Halting

$$
p_j = H_\theta\!\left(\sigma_t^j,\,g_t,\,\pi_t,\,\mathcal{E}_t(\sigma_t^j)\right)\in[0,1], \qquad \boxed{p_{J_{\max}}=1}
$$

The forced final halt makes the weights a proper distribution:

$$
w_j = p_j\prod_{i<j}(1-p_i), \quad w_{J_{\max}}=\prod_{i<J_{\max}}(1-p_i), \quad \sum_{j=0}^{J_{\max}} w_j = 1
$$

$$
\bar{\sigma}_t = \sum_{j=0}^{J_{\max}} w_j\sigma_t^j, \qquad \mathbb{E}[J_t] = \sum_{j=0}^{J_{\max}} j\,w_j
$$

At inference, hard halting $J_t=\min\{j:p_j\geq\delta\}$ or pressure-triggered $J_t=\mathcal{J}(\pi_t)$.

---

## 14. Output

$$
o_t = W_o\operatorname{LN}\!\left(g_t + W_\sigma\bar{\sigma}_t\right), \qquad p(x_{t+1}\mid x_{\leq t})=\operatorname{softmax}(E^\top o_t)
$$

Updated condition $\chi_t=(\phi_t,S_t,\bar{\sigma}_t,\mu_t)$.

---

## 15. Compact Reference (token + silence step)

$$
g_t=\operatorname{EvidenceCore}(x_t,\chi_{t-1}),\quad \hat{g}_t=W_P\operatorname{LN}(W_R S_{t-1}R(\phi_{t-1})W_q^{\text{pred}}\sigma_{t-1}+W_\sigma\sigma_{t-1}+W_\phi\Phi(\phi_{t-1})),\quad e_t=g_t-\hat{g}_t
$$

$$
S_t=\alpha_t S_{t-1}+\rho_t\tau_t(\hat{v}_t\otimes\hat{k}_t),\quad h_t^U=\operatorname{silu}(z_t)\odot\operatorname{RMSNorm}(S_t\tilde{q}_t+Dv_t)
$$

$$
\mu_t=\sigma(W_\mu[g_t,e_t,\sigma_{t-1},m_t]),\quad \tilde{e}_t=\mu_t\odot W_e e_t,\quad \sigma_t^0=\operatorname{LN}(W_{\sigma0}[\sigma_{t-1},g_t,\tilde{e}_t,\mu_t])
$$

$$
r_t^j=S_t R(\phi_t)W_q^\sigma\sigma_t^j,\quad \sigma_t^{j+1}=\operatorname{RMSNorm}(\sigma_t^j+\sigma(W_g z_t^{\varnothing,j})\odot\tanh(W_n z_t^{\varnothing,j})),\quad o_t=W_o\operatorname{LN}(g_t+W_\sigma\bar{\sigma}_t)
$$

with $J_{\max}=2$.

---

## 16. Prediction-Head Objective

$$
\mathcal{L}_{\text{pred}} = \lambda_P\left\lVert\hat{g}_t - \operatorname{stopgrad}(g_t)\right\rVert^2
$$

or contrastively with $\operatorname{sim}(\hat g_t,g_t)/\tau_c$ against negatives. $e_t$ is informative only once $\hat g_t$ predicts.

---

## 17. Counterfactual Silence Benefit

No-silence and with-silence losses give one-step benefit $b_t^{(1)}=\ell_0-\ell_J$, with short-horizon $b_t^{(K)}=\sum_{r=1}^K\omega_r(\ell_{0,t+r}-\ell_{J,t+r})$, $\sum_r\omega_r=1$.

> **Rollout policy.** Silence is frozen off in $t{+}1,\dots,t{+}K$ for both branches; the only difference is whether silence fired once, at $t$. $\pi_t$ thus learns the marginal causal value of the single decision.

> **Paired determinism.** Both branches share batch, teacher-forced continuation, dropout masks (or run dropout-disabled), and numerical-precision path; the benefit label is stop-gradient. Otherwise $b_t$ is noisy for reasons unrelated to silence.

**Fixed calibrated target.** Instead of batch normalization, map benefit through a fixed monotone transform in nats with constant $\beta$ (reference: $\beta=0.02$):

$$
y_t = \log\!\left(1 + \frac{\max(b_t,0)}{\beta}\right), \qquad \mathcal{L}_{\text{pressure}} = \left(\pi_t - \operatorname{stopgrad}(y_t)\right)^2
$$

This keeps $\pi_t$ on a stable, task-comparable scale so the halting threshold $\delta$ and compute penalty $\lambda_C$ remain interpretable. ($b_t$ in nats still drifts as the model improves across training; $\pi_t$ tracks a moving target by construction — the fixed transform reduces, not removes, that drift.)

> **On-policy gauge (reported).** After training, recompute realized benefit *with* the full downstream cascade and report $\operatorname{corr}(\pi_t,b_t^{\text{on-policy}})$ beside the frozen-target correlation, to quantify the train/test mismatch.

---

## 18. Training Objective

$$
\mathcal{L}_{\text{LM}}=-\log p(x_{t+1}\mid x_{\leq t}),\ \ \mathcal{L}_{\text{compute}}=\lambda_C\mathbb{E}[J_t],\ \ \mathcal{L}_{\text{precision}}=\lambda_\mu\lVert\mu_t\rVert_1,\ \ \mathcal{L}_{\text{state}}=\lambda_S\lVert S_t\rVert^2
$$

$$
\mathcal{L}_{\text{consistency}}=\lambda_E\sum_j\max\!\left(0,\ \mathcal{E}_t(\sigma_t^{j+1})-\mathcal{E}_t(\sigma_t^j)\right)
$$

$$
\mathcal{L} = \mathcal{L}_{\text{LM}}+\mathcal{L}_{\text{pressure}}+\mathcal{L}_{\text{pred}}+\mathcal{L}_{\text{compute}}+\mathcal{L}_{\text{consistency}}+\mathcal{L}_{\text{precision}}+\mathcal{L}_{\text{state}}
$$

---

## 19. Forced Silence Exploration

Sample $z_t\sim\operatorname{Bernoulli}(p_{\text{explore}})$; if $z_t=1$, run $J=K$ silent steps regardless of $\pi_t$, so benefit is observed on tokens the policy would skip. Anneal $p_{\text{explore}}\downarrow 0$.

---

## 20. Training Schedule

**Stage 1 — Evidence core.** $J_t=0$. Train $A,U,M$, prediction head with $\mathcal{L}_{\text{LM}}+\mathcal{L}_{\text{pred}}+\mathcal{L}_{\text{precision}}+\mathcal{L}_{\text{state}}$.

> **Pressure-training gate.** Do not enable $\mathcal{L}_{\text{pressure}}$ until held-out $\mathcal{L}_{\text{pred}}^{\text{val}}<\eta$. Before that, $b_t$ is computed from an untrained $\hat g_t$ and is noise; training $\pi_t$ on noise labels causes sticky miscalibration.

**Stage 2 — Forced hypothesis revision.** Once the gate clears, force silence on a sparse subset, $K\in\{1,2\}$; add $\mathcal{L}_{\text{pressure}}+\mathcal{L}_{\text{consistency}}$.

**Stage 3 — Soft halting.** Enable $w_j$ with $p_{J_{\max}}=1$; add $\mathcal{L}_{\text{compute}}$; anneal $p_{\text{explore}}\to 0$.

**Stage 4 — Event-triggered inference.** Fire silence only at high $\pi_t$.

---

## 21. Diagnostics

Track $\phi_t,\alpha_t,\rho_t,\lVert e_t\rVert,\lVert\mu_t\rVert,\pi_t,J_t$; benefit $b_t$; calibration $\operatorname{corr}(\pi_t,b_t)$ held-out; consistency improvement $\mathcal{E}_t(\sigma_t^0)-\mathcal{E}_t(\bar\sigma_t)$.

**Hypothesis inertia.** $\Delta\sigma_t^{\text{silent}}=\lVert\bar\sigma_t-\sigma_t^0\rVert$, $\Delta\sigma_t=\lVert\bar\sigma_t-\sigma_{t-1}\rVert$. Register the co-firing quartet at events, flat on null: $\pi_t\uparrow,J_t\uparrow,\Delta\sigma_t^{\text{silent}}\uparrow,b_t\uparrow$. Separates "silence ran but didn't change $\sigma$" from "changed $\sigma$ without benefit."

**Efficiency (reported).** $\text{efficiency}_t=b_t/(1+\mathbb{E}[J_t])$.

**Per-step decomposition.** Attribute the gain to $\sigma^0\!\to\!\sigma^1$ vs $\sigma^1\!\to\!\sigma^2$; if all value is in step 1, $J_{\max}=1$ is the honest setting.

**On-policy gauge.** $\operatorname{corr}(\pi_t,b_t^{\text{on-policy}})$ vs frozen-target (§17).

**σ-decode probe.** On synthetic tasks with known latent rule, linearly decode the active rule from $\sigma_t$; report held-out accuracy. Primary instrument for interpreting a gate failure (§23).

**σ-intervention (causal).** Train a classifier $c(\sigma_t)\to\text{rule}$, then overwrite $\sigma_t \leftarrow \sigma_t^{\text{other}}$ with a stored/averaged register from a different-rule example and continue. Predict: output shifts toward the injected rule; next-step prediction error rises if the injected rule is wrong; reversal answers flip or degrade. A probe shows $\sigma$ is *decodable*; the intervention shows $\sigma$ is *causally used as* the hypothesis.

---

## 22. Evaluation: Claims, Tasks, Controls

**Primary claim.** Benefit-gated hypothesis revision improves continuation at sparse interpretive events, and silence is allocated where it pays.

**Task families.** Synthetic, with known latent hypotheses and controlled evidence-age:
1. **Branch reversal** — a rule holds, a reversal token flips it.
2. **Latent binding swap** — e.g. `A=red, B=blue, C=green … Correction: A and C were swapped. What color is A?` Same evidence/hypothesis structure, different surface form, so the register cannot pass by detecting a "reversal" token.
3. **Delayed correction / long-range recall** — old evidence must be reinterpreted; the evidence-age axis along which the recency gradient is measured.
4. **Flat null** — no interpretive events.

**Registered differential prediction (recency).** Sort instances by evidence-age and predict $\operatorname{corr}(b_t,\text{evidence-age})<0$. Confirming the *gradient* is stronger than confirming silence merely helps.

**Registered null prediction.** On the flat task, $\pi_t\approx 0$, $\mathbb{E}[J_t]\to 0$.

**Named baselines** (each parameter- and compute-comparable):
- **Evidence-core baseline** (~76.5 M): silence ablated, $g_t$-only output. The capacity-matched floor.
- **Top-GRU adapter** (~1.8 M top block): same $g_t$/$e_t$/$\mu_t$ access and adaptive halting, but **no $S$ read** — $\sigma_t^{j+1}=\operatorname{GRU}(\sigma_t^j,[g_t,e_t,\mu_t])$. Tests whether the evidence-read mechanism earns its complexity vs. a generic top-level recurrent adapter.

**Mechanism-isolating controls:**
- **No-op silence** — run the loop, freeze $\bar\sigma_t=\sigma_t^0$. Tests whether *revision* (not compute) helps.
- **No-read silence** — set $r_t^j=0$. Tests whether *reading $S$* is necessary.
- **Phase-scrambled silence** — $q_{\sigma,t}^j=R(\phi_t+\epsilon_t)W_q^\sigma\sigma_t^j$ with $\epsilon_t$ shuffled across tokens, compute unchanged. Tests whether *phase-aligned* reading (not $\alpha_t$ decay) drives the recency benefit.
- **Random silence** — same $\mathbb{E}[J_t]$, fired at random tokens. Tests whether *pressure allocation* helps.

### Minimum-viable-proof table

| Test | Type | Expected | Proves | Gate |
|---|---|---|---|---|
| Full AUM-Ø v5.3 | reference | improves reversal **and** swap | mechanism can help | — |
| Evidence-core baseline | baseline | worse than full | silence adds value | ✅ |
| Top-GRU adapter | baseline | full beats it per compute | evidence read earns complexity | ✅ |
| No Ø | ablation | worse than full | silence matters | — |
| No-op Ø (frozen $\bar\sigma$) | ablation | no gain | revision, not compute, matters | ✅ |
| No-read Ø ($r{=}0$) | ablation | reduced gain | reading $S$ matters | — |
| Phase-scrambled Ø | ablation | reduced gain | phase addressing matters | ✅ |
| Random silence | ablation | worse efficiency | pressure allocation matters | — |
| No $\mathcal{L}_{\text{pressure}}$ | ablation | misallocated silence | benefit supervision matters | — |
| No $\mathcal{L}_{\text{pred}}$ | ablation | weak $\operatorname{corr}(\pi,b)$ | prediction error matters | — |
| Flat null | control | $\pi\approx 0$, $\mathbb{E}[J]\to 0$ | not generic-uncertainty firing | ✅ |
| σ-decode probe | diagnostic | above chance, held-out | register tracks hypothesis | ✅ |
| σ-intervention | diagnostic | causal output shift | register is *used* | — |
| Recency gradient | prediction | $\operatorname{corr}(b,\text{age})<0$ | phase-addressed retrieval is the cause | ✅ |

The defining experiment: *Can AUM-Ø revise hypothesis by reading $S$ without corrupting it, and does $\pi$ spend that revision only where it pays?*

---

## 23. Training Gate Before Scaling

**Binary gates** (all must pass, on held-out branch-reversal **and** binding-swap): evidence-core baseline beaten; Top-GRU adapter beaten per compute; no-op silence recovers no gain; phase-scrambled silence underperforms real; null control passes; σ-decode above chance; recency gradient $\operatorname{corr}(b_t,\text{evidence-age})<0$.

Delayed-correction / long-range recall are **measured, not gated** — they are the evidence-age axis, and base v5.3 is expected to help them *less* (the recency property). A weak delayed-correction result is consistent with the design, not a failure.

**If a gate fails, read the σ-decode probe first:**
- Probe **fails** → the register is genuinely under-capacity. Only now widen $d_\sigma$, or move to slots / layer-local registers (§24).
- Probe **passes**, no benefit → σ holds the hypothesis but revision isn't helping. Inspect the no-op / no-read / phase-scrambled controls, the prediction head, and the silent update — not $d_\sigma$.
- No-op control **recovers** the gain → apparent benefit was compute; mechanism not validated.
- Recency gradient **absent** → headline benefit may be real but phase-addressed retrieval isn't the cause; investigate before claiming the mechanism.

---

## 24. Future Extensions (out of scope for v5.3)

Explicitly **not** part of this specification. Scoped follow-ups, ordered; each is taken only after the gate (§23) is cleared.

1. **Temporal hypothesis search.** Learnable read-phase offset $q_{\sigma,t}^j=R(\phi_t+\delta_t^j)W_q^\sigma\sigma_t^j$, $\delta_t^j=\psi_\theta(\sigma_t^j)$, regularized for *spread* across silent steps (penalize $\delta^1\approx\delta^2$) so it does not collapse to a fixed small backward look. Turns silence into a temporal scan; the registered fix for the old-evidence case.
2. **Hypothesis-conditioned evidence read in the core.** Let $\bar\sigma_t$ modulate the next token's read query, $\tilde q_{t+1}\leftarrow\tilde q_{t+1}\odot(1+W_h\bar\sigma_t)$, on the readout path only — downstream reach without touching $S$ or violating the affine invariant.
3. **Pooled cross-layer evidence read.** Read a weighted sum of $S_t^\ell$ over the final layers if top-layer retrieval is detail-poor.
4. **Slot-based register.** $\sigma_t\in\mathbb{R}^{n_\sigma\times d_{\sigma s}}$ for competing hypotheses or bindings.
5. **Correction patch.** Bounded $C_t$, $\tilde S_t=S_t+C_t$, routing correction outside $S$.
6. **Layer-local registers.** Per-integration-layer $\sigma^\ell$.
7. **Salience-augmented writes; multi-token-prediction head.**
8. **Variational framing.** Interpret $\sigma_t$ as an amortized posterior $q_\theta(z_t\mid S_t,g_t)$ with a KL/information-bottleneck term, making "hypothesis register" mathematically literal. A theory direction, not an implementation change.

---

## 25. Interpretation of the Four Phases

- **A — Attentive Grounding**: bounded grounding throughout the core; hypothesis-conditioned prediction error once, at the top.
- **U — Latent Evidence Unfolding**: observations written into a resonant **affine** recurrent state, read out through a phase rotation and an output gate.
- **M — Precision Modulation**: error-free below, error-fed precision at the top; weights what matters for revision.
- **Ø — Silence**: phase-aligned reading of $S$ and nonlinear revision of the register when expected benefit is high.

---

## 26. Core Definition

AUM-Ø v5.3 is a predictive recurrent architecture that separates evidence accumulation from hypothesis revision. It maintains $\chi_t=(\phi_t,S_t,\sigma_t,\mu_t)$ with $S_t$ a resonant **affine** evidence state, $\sigma_t$ a single global hypothesis register read and revised through the current phase, $\mu_t$ a precision field, $\phi_t$ a dynamic resonance phase. The token clock updates evidence across all layers; the silence clock revises hypothesis once, on top, by reading $S$ through the hypothesis. Integration pressure is trained against frozen-downstream counterfactual benefit on a fixed calibrated scale, so silence is allocated where revision improves continuation.

The defining transition is $A\rightarrow U\rightarrow M\rightarrow\varnothing$; the defining structure is

$$
\text{affine resonant evidence core} + \text{benefit-gated global hypothesis register};
$$

the defining principle is

$$
\text{Continuation arises from temporary configuration.}
$$

---

## Appendix A. Physical Layout — AUM-Ø-Tiny v5.3 (≈ 78 M)

Format: `name,[shape],dtype`. Evidence layers `[0-11]`; silence subsystem is a single top-level block. Total ≈ 78 M; silence block ≈ 1.8 M; silence-ablated baseline ≈ 76.5 M.

```
model.embed_tokens.weight,[49152,512],BF16

# ---- A phase: bounded local GQA grounding (all evidence layers) ----
model.layers.[0-11].ground_attn.q_proj.weight,[512,512],BF16
model.layers.[0-11].ground_attn.k_proj.weight,[128,512],BF16
model.layers.[0-11].ground_attn.v_proj.weight,[128,512],BF16
model.layers.[0-11].ground_attn.o_proj.weight,[512,512],BF16
model.layers.[0-11].ground_attn.q_norm.weight,[64],BF16
model.layers.[0-11].ground_attn.k_norm.weight,[64],BF16

# ---- U phase: resonant AFFINE evidence recurrence + output gate (all layers) ----
model.layers.[0-11].unfold.controller.weight,[128,512],BF16
model.layers.[0-11].unfold.in_proj_qkv.weight,[1536,512],BF16
model.layers.[0-11].unfold.in_proj_z.weight,[512,512],BF16
model.layers.[0-11].unfold.in_proj_dyn.weight,[49,128],BF16
model.layers.[0-11].unfold.conv1d.weight,[1536,1,4],BF16
model.layers.[0-11].unfold.A_log,[4],F32
model.layers.[0-11].unfold.dt_bias,[4],BF16
model.layers.[0-11].unfold.D,[512],BF16
model.layers.[0-11].unfold.norm.weight,[128],F32
model.layers.[0-11].unfold.out_proj.weight,[512,512],BF16

# ---- M phase: error-free precision (all evidence layers) ----
model.layers.[0-11].modulate.in_proj_mu.weight,[32,1056],BF16
model.layers.[0-11].modulate.up.weight,[512,32],BF16
model.layers.[0-11].modulate.down.weight,[32,512],BF16

# ---- MLP (SwiGLU, all evidence layers) ----
model.layers.[0-11].mlp.gate_proj.weight,[1408,512],BF16
model.layers.[0-11].mlp.up_proj.weight,[1408,512],BF16
model.layers.[0-11].mlp.down_proj.weight,[512,1408],BF16

# ---- Layer norms (all evidence layers) ----
model.layers.[0-11].input_layernorm.weight,[512],BF16
model.layers.[0-11].post_attention_layernorm.weight,[512],BF16

# ---- Global silence block: hypothesis-conditioned predictive grounding ----
model.silence.predict.query_proj.weight,[512,128],BF16
model.silence.predict.read_proj.weight,[512,512],BF16
model.silence.predict.hyp_proj.weight,[512,128],BF16
model.silence.predict.phase_proj.weight,[512,32],BF16
model.silence.predict.out_proj.weight,[512,512],BF16
model.silence.predict.norm.weight,[512],BF16

# ---- Global silence block: error-fed precision (precision only) ----
model.silence.modulate.err_proj.weight,[32,512],BF16
model.silence.modulate.in_proj_mu.weight,[32,1184],BF16

# ---- Global silence block: hypothesis register + revision ----
model.silence.init_proj.weight,[128,704],BF16
model.silence.read_proj.weight,[512,128],BF16
model.silence.update_gate.weight,[128,1216],BF16
model.silence.update_cand.weight,[128,1216],BF16
model.silence.norm.weight,[128],BF16

# ---- Global silence block: consistency functional ----
model.silence.consistency.P_G.weight,[128,512],BF16
model.silence.consistency.Q_G.weight,[128,128],BF16
model.silence.consistency.P_R.weight,[128,512],BF16
model.silence.consistency.Q_R.weight,[128,128],BF16
model.silence.consistency.prec_G.weight,[128,32],BF16
model.silence.consistency.prec_R.weight,[128,32],BF16

# ---- Global silence block: pressure + halting ----
model.silence.pressure_in.weight,[128,516],BF16
model.silence.pressure_out.weight,[128],BF16
model.silence.halt_1.weight,[64,130],BF16
model.silence.halt_2.weight,[1,64],BF16

# ---- Global silence block: output / condition projection ----
model.silence.condition_out.weight,[512,512],BF16
model.silence.condition_out.sigma_proj.weight,[512,128],BF16

model.norm.weight,[512],BF16
```

**Shape notes.** `unfold.in_proj_z,[512,512]` is the output gate $W_z$; the readout is $\operatorname{silu}(z_t)\odot\operatorname{RMSNorm}(S_t\tilde q_t+Dv_t)$. `in_proj_dyn,[49,128]` packs $(\bar\tau,\bar\lambda,r,\theta)$ per U-head $(4\times4)$ plus $m(32)$ and $s(1)$; rotation $R(\phi)$ is parameter-free. **Predictive grounding** reads $S$ through the hypothesis: `predict.query_proj` is $W_q^{\text{pred}}:\sigma(128)\to$ state-key space $(512)$, `predict.read_proj` is $W_R:r^{\text{pred}}(512)\to d(512)$ — separate from the silent read's `read_proj` $W_q^\sigma:\sigma(128)\to512$. Global `modulate` carries **only** `err_proj` $(W_e)$ and `in_proj_mu` $(W_\mu)$; there is no global up/down readout adapter. `init_proj` input $704=\sigma(128)+g(512)+\tilde e(32)+\mu(32)$. `update_*` input $1216=\sigma(128)+g(512)+\tilde e(32)+\mu(32)+r^j(512)$. `modulate.in_proj_mu` input $1184=g(512)+e(512)+\sigma(128)+m(32)$. `pressure_in` input $516=\zeta(512)+H_t+\Delta^e+\Delta^{\sigma R}+s_t$. `halt_1` input $130=\sigma(128)+\pi_t+\mathcal{E}$. Embeddings tied to the output classifier.

## Appendix B. Baselines and Ablations (one place)

Run all comparisons parameter- and compute-matched. **Baselines:** evidence-core (~76.5 M, silence ablated); Top-GRU adapter (~1.8 M top block, no $S$ read). **Ablations on the full model:** no-Ø; no-op Ø ($\bar\sigma=\sigma^0$); no-read Ø ($r=0$); phase-scrambled Ø; random silence (matched $\mathbb{E}[J]$); no $\mathcal{L}_{\text{pressure}}$; no $\mathcal{L}_{\text{pred}}$. **Diagnostics:** σ-decode probe; causal σ-intervention; Δσ quartet; per-step benefit decomposition; on-policy correlation gauge. The gate (§23) is the subset marked ✅ in §22.
