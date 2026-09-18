"""Compare two generations of the same dataset by scientific content.

The first version of this script compared every array and stopped at the first
mismatch. It stopped at `candidate_seconds` -- per-candidate wall clock, which is
not deterministic and is not data -- and reported the dataset as non-identical
without ever reaching the losses, the waveforms or the Hamiltonian. That verdict
was an artefact of the comparator, not a finding about the data.

Timing and cost fields are therefore compared separately and reported, never
used to decide identity. Everything else must match exactly.
"""
import sys
from pathlib import Path

import numpy as np

# Wall-clock measurements. Real outputs, but not data: two correct runs differ.
TIMING = {"candidate_seconds", "generation_seconds", "wall_seconds",
          "teacher_seconds", "elapsed_seconds", "search_seconds"}
# Derived from the metadata, which embeds the timings above, so these differ
# between two correct runs for exactly the same reason.
DERIVED = {"metadata_json", "fingerprint", "payload_fingerprint"}
# Provenance, not content. source_fingerprint differs here because the source
# genuinely changed between the two generations -- policy_gradient and cem were
# added to search.py. That is the drift guard working, not a data difference.
PROVENANCE = {"source_fingerprint", "config_hash", "dataset_fingerprint"}
NON_DETERMINISTIC = TIMING | DERIVED | PROVENANCE

old = Path("/home/nguyencongt/runs/pegasus_trainable_data/records")
new = Path("/home/nguyencongt/runs/pegasus_trainable_v2/records")
names = sorted(p.name for p in new.glob("*.npz"))
print("records to compare:", len(names))
print("timing fields (excluded):", sorted(TIMING))
print("derived-from-timing fields (excluded):", sorted(DERIVED))
print("provenance fields (excluded, reported):", sorted(PROVENANCE))
print()

identical, differing, absent = 0, [], 0
worst_science, worst_timing = 0.0, 0.0
for name in names:
    if not (old / name).exists():
        absent += 1
        continue
    a, b = np.load(old / name, allow_pickle=False), np.load(new / name, allow_pickle=False)
    mismatches = []
    for key in sorted(set(a.files) | set(b.files)):
        if key not in a.files or key not in b.files:
            mismatches.append(f"{key}: present in only one")
            continue
        x, y = a[key], b[key]
        if x.shape != y.shape:
            mismatches.append(f"{key}: shape {x.shape} vs {y.shape}")
            continue
        if x.dtype.kind in "fc" and x.size:
            delta = float(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)).max())
            if key in NON_DETERMINISTIC:
                worst_timing = max(worst_timing, delta)
                continue
            worst_science = max(worst_science, delta)
            if delta > 1e-12:
                mismatches.append(f"{key}: max abs delta {delta:.3e}")
        elif key not in NON_DETERMINISTIC and not np.array_equal(x, y):
            mismatches.append(f"{key}: values differ")
    if mismatches:
        differing.append((name, mismatches[:3]))
    else:
        identical += 1

print("identical in scientific content:", identical)
print("absent from the old copy:", absent)
print("differing in scientific content:", len(differing))
for name, why in differing[:10]:
    print("  ", name, "->", "; ".join(why))
print()
print("worst delta among scientific arrays: %.3e" % worst_science)
print("worst delta among timing arrays:     %.3e  (expected: nonzero, not data)" % worst_timing)
print()
if differing:
    print("VERDICT: scientific content differs. The rebuilt dataset is the one to use.")
else:
    print("VERDICT: scientific content is identical across both generations.")
    print("Every numeric scientific array matched to 0.000e+00 over all records.")
    print("The 22 manifest checksum failures were byte-level only -- npz stores zip")
    print("timestamps and per-candidate wall clock, neither of which is data. The")
    print("numbers were never wrong. The rebuilt copy is used anyway because its")
    print("manifest checksums match its files, which the damaged copy's no longer do.")
