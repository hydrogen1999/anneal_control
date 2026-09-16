# ADR-0003: One declared factor per embedding intervention, with a scale-controlled arm for chain strength

## Status
Accepted

## Date
2026-09-16

## Context

The G3 claim is causal: changing embedding information changes which control is
preferred. `docs/paper_protocol.md` §7 requires holding logical coefficients,
decoder, driver family, runtime, candidate bank, numerical accuracy and budget
fixed while intervening on exactly one physical feature.

`compile_embedding` computes a common coefficient scale
`α = 1/max(1, max|h|/h_limit, max|J_total|/j_limit)`. Chain strength enters
`J_total` through the intra-chain penalty, so raising κ generally lowers α. A κ
intervention therefore changes two things at once: the penalty *and* the global
scale of `H_Z`. `IMPLEMENTATION_PLAN_VI.md` §G3 names this explicitly and
demands separate raw-fixed and scale-controlled interventions.

## Decision

1. `build_pair` accepts exactly one `factor ∈ {geometry, ports, field_allocation,
   chain_strength}`. A spec that differs in more than one is rejected with
   `ValueError`, not normalised.
2. Every pair carries `held_fixed` (the explicit list) and `physical_size_matched`.
   A factor that changes physical size must be declared; an undeclared size
   change is an error.
3. A `chain_strength` pair always produces **two arms**:
   - `total_compiled_effect` — each side uses its own α, as a device would;
   - `scale_controlled` — both sides compiled with one conservative common α,
     via the new `scale_override`, so the penalty moves and the global scale
     does not.
   Both are reported. Neither is called "the" effect.
4. `scale_override` scales `H_Z` only. The driver and the runtime are untouched,
   preserving the existing contract.

## Alternatives considered

**Normalise multi-factor specs into a sequence of single-factor pairs.** Rejected:
it hides from the caller that they asked an ambiguous question.

**Rescale the driver to compensate for α.** Rejected: that changes the physical
model, and `compile_embedding` documents that it scales `H_Z` only.

**Report only the scale-controlled arm.** Rejected: the total compiled effect is
what an operator actually experiences, and suppressing it would overstate how
clean the geometry story is.

## Consequences

- κ results are always two numbers, which is more honest and more work to
  present; Figure 4 must show both.
- `conservative_common_scale` must be computed from both arms before either is
  compiled, so pair construction is a two-pass operation.
- Geometry, port and field-allocation pairs get one arm, since α is unaffected
  when physical size and coefficient magnitudes are matched — verified per pair
  rather than assumed, and flagged when it does move.
