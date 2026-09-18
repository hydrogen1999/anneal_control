# Measured experiment results

Measured run results only. No automatic significance, quantum advantage or conference-readiness claim.

equal parent after averaging training seeds and within-parent variants.
Parent bootstrap conditional on observed training seeds; seed SD is separate, not a joint CI.

Lower loss is better; loss = 1 - decoded logical success.

| Method | Mode | Seeds | Parents | Loss | Seed SD | Δ vs global [parent 95% CI] |
|---|---|---:|---:|---:|---:|---| 
| hierarchy_outcome | bank | 3 | 12 | 0.63198 | 0.00226 | -0.01913 [-0.03453, -0.00693] |
| hierarchy_outcome | direct | 3 | 12 | 0.69464 | 0.01293 | +0.04353 [0.02354, 0.06502] |
| hierarchy_physics | bank | 3 | 12 | 0.63198 | 0.00226 | -0.01913 [-0.03453, -0.00693] |
| hierarchy_physics | direct | 3 | 12 | 0.69464 | 0.01293 | +0.04353 [0.02354, 0.06502] |
| logical | bank | 3 | 12 | 0.63987 | 0.00180 | -0.01124 [-0.01827, -0.00492] |
| logical | direct | 3 | 12 | 0.70718 | 0.01073 | +0.05607 [0.02849, 0.08409] |
| physical | bank | 3 | 12 | 0.63391 | 0.00169 | -0.01720 [-0.03239, -0.00501] |
| physical | direct | 3 | 12 | 0.68708 | 0.01715 | +0.03597 [0.01642, 0.05704] |
| summary | bank | 3 | 12 | 0.62821 | 0.00190 | -0.02291 [-0.03830, -0.01058] |
| summary | direct | 3 | 12 | 0.70667 | 0.00616 | +0.05556 [0.03007, 0.08269] |

## Reproducibility

Config hash: `3d7bc3dc8ef93dcd7c6ed920ad14da17b470d3dc3bdaa220f91cfef26443eafd`.
Source hash: `ab8b4e7ba9bd15fc225d893251e1657859d9c123f08d2f9b8d7a5f32bb197080`.
All predeclared methods/seeds are included; negative results are retained.
Best-bank outcomes are privileged references, not deployable methods or continuous optima.
Resumed training costs are flagged incomplete: observed attempts may omit work before a hard process kill.
Plots show observed values only, with no smoothing or manufactured runs.
