# Controlled acquisition study

Lower loss is better. Parent/seed cells receive equal weight.

| Arm | Bank | Direct | Global | Linear |
|---|---:|---:|---:|---:|
| control | 0.667667 | 0.682967 | 0.644433 | 0.681375 |
| bankext | 0.667667 | 0.682737 | 0.644433 | 0.681375 |
| decoder_random | 0.661933 | 0.676769 | 0.644433 | 0.681375 |
| policy | 0.671474 | 0.682969 | 0.644433 | 0.681375 |

Full paired contrasts, raw rows, diagnostics and costs: `summary.json.gz`.

This is an integration smoke run: only two test parents and four training epochs.
No comparative scientific conclusion is justified by this run.

These results do not establish architectural superiority, hardware performance, or conference readiness.
Primary attribution requires policy acquisition to improve over BOTH bank extension and decoder-random across independent parents and seeds.
All checkpoints use the original validation bank; test labels enter only the frozen evaluation stage.
Both bank and direct deployment require zero online simulator calls. Offline diagnostic scoring is separate.
