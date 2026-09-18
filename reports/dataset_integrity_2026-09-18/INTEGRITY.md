# The damaged dataset's numbers were never wrong

## What happened

To reuse a 6.3-hour Pegasus dataset inside the experiment runner, I bootstrapped
its manifest by starting a run and killing it. The run **outlived the kill**,
regenerated records through a symlink into the existing dataset, and left a stale
lock. Afterwards 22 of 1152 records failed their manifest checksum.

I could not certify the data, so I rebuilt it and compared. "Probably only
timestamps" is not a thing to train on, and "it loads without error" is not "it
is correct".

## What the comparison found

Over all 1152 records, comparing array by array:

| class of field | result |
|---|---|
| **scientific arrays** (losses, waveforms, Hamiltonian, embedding, spectra) | **identical, max delta 0.000e+00** |
| timing (`candidate_seconds`, …) | differs, max 49.3 s — wall clock, not data |
| derived (`metadata_json`, `fingerprint`, `payload_fingerprint`) | differs — embeds the timings above |
| provenance (`source_fingerprint`) | differs — the source genuinely changed between runs |

**The numbers were never wrong.** The 22 checksum failures were byte-level: npz
stores zip timestamps, and the payload embeds per-candidate wall clock.

## The comparator was wrong twice before it was right

This is worth recording, because a verification tool that reports the wrong
answer is worse than none.

1. **First version** compared every array and stopped at the first mismatch. It
   stopped at `candidate_seconds` and declared the dataset corrupt without ever
   reaching the losses. That verdict was an artefact of the comparator.
2. **Second version** excluded timing, then stopped at `metadata_json` and
   `payload_fingerprint` — both derived from the timings it had just excluded.
   Same false verdict, one layer down.
3. **Third version** also excludes provenance, where `source_fingerprint`
   legitimately differs because `policy_gradient` and `cem` were added to
   `search.py` between the two generations. That is the drift guard working.

Only the third version compares what "identical data" actually means.

## What was done anyway

The rebuilt copy is the one used, despite the original being sound: its manifest
checksums match its files and its `source_fingerprint` is current, so nothing
downstream has to special-case it.

The 6.3 hours proved the data rather than repairing it. That is the cost of the
bootstrap-by-kill hack, and the launcher now says in a comment not to do it.
