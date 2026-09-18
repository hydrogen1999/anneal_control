# Controlled acquisition for amortized annealing control

## Research claim and current boundary

The defensible question is whether a learned control generator benefits from
labels acquired on its own proposals, beyond the benefit of more labels and
beyond matching the decoder's feasible waveform family. Architectural superiority
is a separate hypothesis. The moment-summary encoder is the economical primary
model; hierarchy remains an ablation until a matched test supports its value.

The implementation supports this experiment. Its presence is not evidence that
policy acquisition wins. A small integration run cannot support a research claim.
The archived embedding results concern closed-system simulations of small
instances. Connectivity matching alone is not hardware validation.

## Problem and architecture

A task consists of a logical Ising instance, an embedding into a physical graph,
chain strength, programmed energy scale, and runtime. Data generation constructs
the physical Hamiltonian from logical problem, chain penalties, driver, and any
explicit catalyst terms. Physical embeddings of one logical parent remain in
the same split. The objective is decoded logical ground-state success, not merely
the physical ground-state population.

For fixed runtime $T$, let $s:[0,1]\to[0,1]$ satisfy $s(0)=0$, $s(1)=1$,
$0\le ds/d\tau\le v_{\max}$. The loss is
$\ell_x(s)=1-\langle\psi_{x,s}(T)|P_x|\psi_{x,s}(T)\rangle$,
where $P_x$ projects onto physical bitstrings decoded to a logical optimum.
The closed-system simulator uses matrix-free propagation and step-doubling
diagnostics. Those diagnostics are not certified true-error bounds.

The encoder maps physical/logical graph information to instance features. A
policy head produces $M$ schedules through capped-simplex increments. The critic
predicts the outcome of an instance–schedule pair and selects the lowest
predicted loss. A shared finite bank provides a second amortized deployment
mode: the same critic selects a bank waveform. Both deployment modes use zero
online simulator calls. Their quality, latency, offline labels, and training
cost must all be compared.

The reference summary model receives physical moments, chain statistics and
context. The hierarchical model uses physical message passing, chain pooling,
logical message passing and attention. The comparison tests representation;
it must not be conflated with whether embedding information is useful.

## Matched one-round algorithm

`annealctrl acquisition-study` freezes configuration, source, dataset identity,
training seeds and all four arms before opening test outcomes.

1. Train a baseline generator/critic on the original training bank. Select its
   checkpoint using original validation-bank regret.
2. Freeze that checkpoint. On training parents only, acquire exactly $M$ extra
   simulator outcomes per record for each non-control arm.
3. Retrain each arm from scratch using the same initialization seed, optimizer,
   stopping rule, model and original validation records. This avoids attributing
   extra optimizer exposure to the acquisition rule.
4. Complete every seed and arm before test evaluation. Test-time bank candidates
   remain identical, including for models trained on an augmented bank.
5. Select proposals without outcomes, then simulate all of them for diagnosis.
   Re-score the baseline's frozen proposals with both critics to distinguish
   changes in ranking from changes in generation.

| Arm | Added training waveforms | Purpose |
|---|---|---|
| CONTROL | None | Same-data retraining and deterministic implementation check |
| BANKEXT | Next entries of the original shared Sobol bank | Additional-label control |
| DECODER-RANDOM | Shared IID normal logits through the actual policy decoder | Feasible-family control |
| POLICY | Every proposal emitted by the frozen baseline | Policy-dependent acquisition |

The random logit standard deviation is fixed in the configuration. Matching a
decoder does not match the trained policy's marginal distribution. An observed
POLICY advantage would therefore identify value beyond this particular random
control, not optimality over all possible acquisition rules. Additional rounds
and tuned random priors need separately frozen studies; this runner is explicitly
one round.

BANKEXT and DECODER-RANDOM use the same prespecified waveforms across training
seeds. Seed uncertainty therefore concerns initialization/training conditional
on those fixed acquisition designs. To quantify variability across acquisition
designs, run additional independently prespecified random-design seeds as a
separate experimental factor; do not count timing repetitions as such seeds.

Collectors preserve the full measured outcome, uncertainty indicator, numerical
cost and source-record digest. A failed requested label fails the matched arm;
there is no silent rejection sampling, truncation, or substitution of average
diagnostics. Failed scorer work whose internal count is unavailable is marked
incomplete. Artifact hashes prevent resuming from modified labels/checkpoints.

## Head-isolated mechanism experiment

Enable `mechanism.enabled` in the study configuration to run a separate,
paired continuation experiment. The four primary from-scratch acquisition arms
above are unchanged. For each seed, all mechanism arms start from the *same
validation-selected baseline checkpoint and its exact training normalizer*.
The baseline's acquired POLICY labels are reused byte-for-byte, with the same
artifact hash. No mechanism arm requests a new acquisition label.

| Trainable scope | Parameters updated | Paired datasets |
|---|---|---|
| `critic` | Schedule encoder and critic head | Original bank; original bank plus frozen POLICY labels |
| `policy` | Policy query and policy head | Original bank; original bank plus frozen POLICY labels |
| `heads` | Both sets above | Original bank; original bank plus frozen POLICY labels |

The instance encoder, shared attention, response head, response-query encoder,
and feature normalization remain frozen. Freezing shared attention matters:
otherwise a nominally critic-only update could also change policy proposals.
Critic-only arms have identical generated waveforms to the baseline; policy-only
arms have an identical critic as a function of any fixed waveform. Regression
tests check both properties and exact unchanged shared parameter tensors.

Let $D$ denote the fixed training bank, $A_0$ the true simulator outcomes of the
frozen baseline's proposals, and $(\theta_s^0,\theta_c^0,\theta_\pi^0)$ its shared,
critic, and policy parameters. For $D'\in\{D,D\cup A_0\}$, fit each scope from
that same checkpoint:

$$
L_c(D')=L_{\mathrm{Huber}}(D')+\lambda_r L_{\mathrm{rank}}(D'),\qquad
L_\pi(D')=\lambda_\pi\operatorname{CE}(p_{D'},q_{\theta_\pi}),
$$

where $p_{D'}$ is the existing uncertainty-aware soft target derived from **true
fixed simulator losses**. The updated or frozen critic does not supply policy
pseudo-labels. Critic-only optimizes $L_c$, policy-only optimizes $L_\pi$, and
heads-joint optimizes their sum. No response auxiliary loss is optimized here.

The continuation epoch count is declared in `mechanism.epochs` before outcomes
are read. Every scope selects its **final fixed-budget epoch** and disables
early stopping. Applying validation-bank regret selection to policy-only
training would always observe an unchanged critic and would arbitrarily prefer
its first epoch. Validation banks remain unchanged and their metrics are logged
for diagnosis, never used to select these mechanism checkpoints. Original and
acquired pairs share initialization, seed, optimizer recipe, record order,
number of updates, and fixed epoch count. Their per-update FLOPs differ because
the acquired bank contains more candidates; measured training time is reported.

For each scope, the reported paired effect is loss(acquired) minus
loss(original), separately for bank and direct deployment, aggregated with
equal parent/seed-cell weights. The original-bank continuation is essential:
comparing only to the pre-continuation baseline would mix acquisition with
additional optimizer exposure. These within-scope effects and the frozen
proposal diagnosis identify which head can exploit the extra labels under a
fixed representation. They do **not** prove that critic distribution shift is
the unique cause of a general failure, or that this effect persists when the
shared representation is retrained. Comparing a head-only arm directly against
a primary from-scratch arm does not isolate a head effect.

The manifest records initialization checkpoint hashes, trainable scope, label
source, checkpoint-selection rule, and shared acquisition ownership. Acquisition
cost is charged once to POLICY; mechanism evaluation and training have their own
measured costs. All primary and mechanism fits for all seeds finish before any
test outcome is opened. Mechanism intervals are descriptive and unadjusted;
multiple head comparisons are not automatically confirmatory discoveries.

`configs/acquisition_smoke.json` enables all three scopes for integration checks;
`configs/acquisition_research.json` enables the same design across five seeds.
Neither configuration nor passing tests constitutes a scientific result. The
relevant hypotheses may fail, and negative results must remain in the report.

## Exact diagnosis and conditional bounds

For one task let $C$ be the generated proposal set, $\hat s$ the critic-selected
proposal, $s_C^*=\arg\min_{s\in C}\ell(s)$, and $s_B$ the critic-selected bank
schedule. The following identity is exact:

$$\ell(\hat s)-\ell(s_B)
=\underbrace{\ell(\hat s)-\ell(s_C^*)}_{\text{ranking regret}\ge0}
+\underbrace{\ell(s_C^*)-\ell(s_B)}_{\text{generation gap, possibly negative}}.$$

The second term may be negative. Calling both terms nonnegative regret would be
incorrect. `policy_diagnostics.py` computes the identity using true outcomes
after selection. The fixed-proposal comparison holds $C$ constant while changing
the critic, so a change in that diagnostic cannot be caused by a new proposal set.
It remains a diagnostic of a jointly retrained model, not a separately trained
critic-only ablation.

**Finite-set ranking bound.** Suppose $|\hat\ell(s)-\ell(s)|\le\epsilon$ for
every $s\in C$. Then
$\ell(\hat s)\le\ell(s_C^*)+2\epsilon$.
Indeed, $\ell(\hat s)\le\hat\ell(\hat s)+\epsilon
\le\hat\ell(s_C^*)+\epsilon\le\ell(s_C^*)+2\epsilon$.
Small error on the original bank alone does not imply this assumption on $C$.
Proposal-label acquisition targets this missing support, but no finite experiment
guarantees uniform error on unseen parents or future proposal sets.

**Control continuity bound.** Assume a finite-dimensional closed system, a fixed
normalized initial state, a fixed success projector, and an absolutely continuous
Hamiltonian path with $L_H=\sup_{u\in[0,1]}\|\partial H(u)/\partial u\|<\infty$.
For two schedules at the same runtime, Duhamel's identity and unitary invariance
give

$$\|\psi_s(T)-\psi_r(T)\|
\le\int_0^T\|H(s(t/T))-H(r(t/T))\|dt
\le T L_H\|s-r\|_\infty.$$

Expanding the two quadratic forms and using $\|P\|\le1$ yields
$|\ell(s)-\ell(r)|\le\min\{1,2TL_H\|s-r\|_\infty\}$.
If a proposal is within $\delta$ of the best bank waveform and the previous
uniform critic-error assumption holds, then

$$\ell(\hat s)-\min_{b\in B}\ell(b)\le2\epsilon+2TL_H\delta.$$

These are elementary conditional bounds, not a novel convergence theorem.
The norm constant may be loose, may grow with physical size/scale, and is not
estimated by nearest-neighbour distance alone. Numerical label ambiguity adds
an additional error term only if independently bounded; the stored heuristic
indicator is insufficient for a certified guarantee. The derivation does not
apply unchanged to open-system dynamics, calibration drift or hardware noise.

## Evaluation and decision rules

The primary attribution comparisons are POLICY minus BANKEXT and POLICY minus
DECODER-RANDOM for direct loss. Negative values favour POLICY. Report individual
training seeds and paired parent means, and examine the crossed parent/seed
bootstrap. With few seeds, its uncertainty estimate is fragile. Current study
intervals are descriptive and unadjusted; they do not automatically certify a
discovery. A confirmatory paper analysis must freeze its comparison family,
effect-size threshold, multiplicity procedure and target population in advance.
See [statistical inference](statistical_inference.md).

Also report direct minus bank and direct minus global. Closing a direct–bank gap
does not establish deployment value when bank selection is equally amortized.
Direct deployment should justify its use by a reproducible quality/cost benefit,
better out-of-distribution transfer, or an explicitly measured restriction on
bank deployment. Always include label generation, failed attempts, shared
baseline training, retraining, offline diagnostics and deployment latency.

Current checkpoint selection optimizes original validation-bank regret. This is
held fixed to isolate acquisition, but can underselect direct performance. A
later validation-based direct-selection experiment must label a validation
proposal panel, account for that cost, and freeze its choice before a fresh
test evaluation; it must not be retrofitted after examining test outcomes.

## Relation to prior work

[Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html)
motivate collecting data under a learner-induced distribution in imitation
learning. Here the oracle returns a scalar simulator outcome for a waveform,
not an expert action at a sequentially visited state. This is proposal-label
aggregation inspired by that idea; the DAgger no-regret result is not inherited.

[Trabucco et al. (2021)](https://proceedings.mlr.press/v139/trabucco21a.html)
study surrogate exploitation under distribution shift in offline model-based
optimization. That literature is central to the failure diagnosis. This project
allows additional simulator queries, unlike a strictly fixed-data setting.
Conservative objective models are relevant comparison work, not an implemented
baseline in this revision.

[Finžgar et al. (2024)](https://doi.org/10.1103/PhysRevResearch.6.023063)
optimize annealing schedules with Bayesian optimization. Our task-constrained
implementation and its deviations are listed in
[literature baseline](literature_baseline.md). Per-instance BO spends online
objective calls; comparing only its latency with a pretrained controller would
omit offline amortization costs. Original nonmonotonic schedules and model
settings must be distinguished from the monotone embedded-Ising adaptation.

## Commands

```bash
python -m annealctrl acquisition-study --config configs/acquisition_smoke.json --output runs/acquisition_smoke
python -m annealctrl acquisition-study --config configs/acquisition_research.json --data runs/research/data --output runs/acquisition_summary --stage train
python -m annealctrl acquisition-study --config configs/acquisition_research.json --data runs/research/data --output runs/acquisition_summary --stage evaluate --resume
python -m annealctrl acquisition-study --config configs/acquisition_research.json --data runs/research/data --output runs/acquisition_summary --stage report --resume
```

The supplied dataset must match the frozen dataset configuration. For a hierarchy
replication, copy the research configuration before training and change only the
encoder variant. Keep the same seeds, data, proposal count and budget. Do not
modify a study configuration after inspecting its test results.
