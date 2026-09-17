# Measured experiment results

Measured run results only. No automatic significance, quantum advantage or conference-readiness claim.

equal parent after averaging training seeds and within-parent variants.
Parent bootstrap conditional on observed training seeds; seed SD is separate, not a joint CI.

Lower loss is better; loss = 1 - decoded logical success.

| Method | Mode | Seeds | Parents | Loss | Seed SD | Δ vs global [parent 95% CI] |
|---|---|---:|---:|---:|---:|---| 
| hierarchy_outcome | bank | 5 | 48 | 0.54557 | 0.00129 | -0.01986 [-0.02689, -0.01340] |
| hierarchy_outcome | direct | 5 | 48 | 0.59066 | 0.00763 | +0.02522 [0.01494, 0.03637] |
| hierarchy_physics | bank | 5 | 48 | 0.54502 | 0.00131 | -0.02041 [-0.02737, -0.01377] |
| hierarchy_physics | direct | 5 | 48 | 0.59065 | 0.00452 | +0.02521 [0.01518, 0.03659] |
| logical | bank | 5 | 48 | 0.55318 | 0.00135 | -0.01226 [-0.01812, -0.00666] |
| logical | direct | 5 | 48 | 0.58981 | 0.00574 | +0.02438 [0.01366, 0.03565] |
| physical | bank | 5 | 48 | 0.54620 | 0.00095 | -0.01924 [-0.02655, -0.01242] |
| physical | direct | 5 | 48 | 0.59776 | 0.00524 | +0.03232 [0.02145, 0.04520] |
| summary | bank | 5 | 48 | 0.54466 | 0.00237 | -0.02077 [-0.02746, -0.01428] |
| summary | direct | 5 | 48 | 0.59053 | 0.00533 | +0.02510 [0.01637, 0.03468] |

## Reproducibility

Config hash: `998e70ed5dfe66554220da177cb906167afedbc2eb4f949bc0b69825f96c2f32`.
Source hash: `30d3eec7f7774d2442a323bf1a21b3ed2db637b27038d8f4711ec3ae551f8e9e`.
All predeclared methods/seeds are included; negative results are retained.
Best-bank outcomes are privileged references, not deployable methods or continuous optima.
Resumed training costs are flagged incomplete: observed attempts may omit work before a hard process kill.
Plots show observed values only, with no smoothing or manufactured runs.
