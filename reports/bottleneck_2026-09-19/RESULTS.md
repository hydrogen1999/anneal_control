# Measured experiment results

Measured run results only. No automatic significance, quantum advantage or conference-readiness claim.

equal parent after averaging training seeds and within-parent variants.
Parent bootstrap conditional on observed training seeds; seed SD is separate, not a joint CI.

Lower loss is better; loss = 1 - decoded logical success.

| Method | Mode | Seeds | Parents | Loss | Seed SD | Δ vs global [parent 95% CI] |
|---|---|---:|---:|---:|---:|---| 
| bottleneck_1 | bank | 3 | 48 | 0.55100 | 0.00139 | -0.01443 [-0.02051, -0.00901] |
| bottleneck_1 | direct | 3 | 48 | 0.59191 | 0.00859 | +0.02647 [0.01355, 0.03969] |
| bottleneck_8 | bank | 3 | 48 | 0.54445 | 0.00155 | -0.02099 [-0.02859, -0.01427] |
| bottleneck_8 | direct | 3 | 48 | 0.59971 | 0.00340 | +0.03427 [0.02278, 0.04654] |
| unconstrained | bank | 3 | 48 | 0.54535 | 0.00166 | -0.02009 [-0.02746, -0.01356] |
| unconstrained | direct | 3 | 48 | 0.59018 | 0.00441 | +0.02474 [0.01484, 0.03559] |

## Reproducibility

Config hash: `3fe7bfb3a82f103b892a050a160fa2e62f3a95f963c792cb0c49fd2875b43cc0`.
Source hash: `1d9a3bcd6a915e23a59f831a57a98ccaebad83302c8876f03e857d70589434c9`.
All predeclared methods/seeds are included; negative results are retained.
Best-bank outcomes are privileged references, not deployable methods or continuous optima.
Resumed training costs are flagged incomplete: observed attempts may omit work before a hard process kill.
Plots show observed values only, with no smoothing or manufactured runs.
