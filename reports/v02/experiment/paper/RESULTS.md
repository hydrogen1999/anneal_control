# Measured experiment results

Measured run results only. No automatic significance, quantum advantage or conference-readiness claim.

equal parent after averaging training seeds and within-parent variants.
Parent bootstrap conditional on observed training seeds; seed SD is separate, not a joint CI.

Lower loss is better; loss = 1 - decoded logical success.

| Method | Mode | Seeds | Parents | Loss | Seed SD | Δ vs global [parent 95% CI] |
|---|---|---:|---:|---:|---:|---| 
| hierarchy_outcome | bank | 2 | 4 | 0.69410 | 0.03578 | +0.02530 [0.00498, 0.04812] |
| hierarchy_outcome | direct | 2 | 4 | 0.66976 | 0.00324 | +0.00097 [-0.04871, 0.04245] |
| hierarchy_physics | bank | 2 | 4 | 0.69410 | 0.03578 | +0.02530 [0.00498, 0.04812] |
| hierarchy_physics | direct | 2 | 4 | 0.66977 | 0.00324 | +0.00098 [-0.04869, 0.04245] |
| logical | bank | 2 | 4 | 0.68822 | 0.00000 | +0.01943 [-0.02086, 0.05422] |
| logical | direct | 2 | 4 | 0.66410 | 0.01657 | -0.00469 [-0.05310, 0.03603] |
| physical | bank | 2 | 4 | 0.68100 | 0.01022 | +0.01220 [-0.03110, 0.04913] |
| physical | direct | 2 | 4 | 0.67494 | 0.01309 | +0.00615 [-0.03862, 0.04740] |
| summary | bank | 2 | 4 | 0.68822 | 0.00000 | +0.01943 [-0.02086, 0.05422] |
| summary | direct | 2 | 4 | 0.67555 | 0.00524 | +0.00676 [-0.03845, 0.04696] |

## Reproducibility

Config hash: `1275733d7baa56b799bd3e0f9039ab12c87c894cef9fcc480ecf7c00c420a62e`.
Source hash: `7ac72e2b8325a86a75c70a2f95be07255b3ec3649a3dd959cc72fa3df373f4b1`.
All predeclared methods/seeds are included; negative results are retained.
Best-bank outcomes are privileged references, not deployable methods or continuous optima.
Resumed training costs are flagged incomplete: observed attempts may omit work before a hard process kill.
Plots show observed values only, with no smoothing or manufactured runs.
