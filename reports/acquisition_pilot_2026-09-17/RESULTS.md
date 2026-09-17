# Controlled acquisition study

Lower loss is better. Parent/seed cells receive equal weight.

| Arm | Bank | Direct | Global | Linear |
|---|---:|---:|---:|---:|
| control | 0.564679 | 0.562287 | 0.568782 | 0.601700 |
| bankext | 0.551096 | 0.585768 | 0.568782 | 0.601700 |
| decoder_random | 0.552729 | 0.623681 | 0.568782 | 0.601700 |
| policy | 0.568782 | 0.589876 | 0.568782 | 0.601700 |

Full paired contrasts, raw rows, diagnostics and costs: `summary.json.gz`.

These results do not establish architectural superiority, hardware performance, or conference readiness.
Primary attribution requires policy acquisition to improve over BOTH bank extension and decoder-random across independent parents and seeds.
All checkpoints use the original validation bank; test labels enter only the frozen evaluation stage.
Both bank and direct deployment require zero online simulator calls. Offline diagnostic scoring is separate.
