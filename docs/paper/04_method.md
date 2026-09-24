# 4. Method

The method is a learned, instance-conditioned **critic that ranks a pool of
candidate schedules** from a representation of the embedded problem. Its value
is set by two things that we measure separately: what the representation is
allowed to see (§4.1), and what the critic is allowed to choose between (§4.3).

## 4.1 Representation, and the ablation built into it

The encoder produces a token bank from the instance: physical-qubit tokens,
chain tokens, and logical tokens, with explicit membership relations tying each
physical qubit to its chain and each chain to its logical spin. Edges carry
**signed** weights — a sign-blind encoder cannot distinguish a frustrated loop
from an unfrustrated one — and the programmed scale enters as its own channel
rather than being folded into the couplings.

The central experiment is an ablation *of the representation, not of the
architecture*, so the variants are defined by what they may read:

| variant | physical graph | chain membership | logical graph | how |
|---|:--:|:--:|:--:|---|
| `logical` | ✗ | ✗ | ✓ | message passing on $G_{\mathrm L}$ only |
| `physical` | ✓ | ✗ | ✗ | message passing on the embedded graph only |
| `summary` | ✓ | ✓ | ✓ | **pooled statistics only, no message passing** |
| `hierarchical` | ✓ | ✓ | ✓ | full token bank with typed roles and attention |

`logical` is the embedding-blind control: it is given exactly the problem a
spectral rule would be computed from. The other three are embedding-aware.
Width, depth, training schedule, data, candidate set and evaluation are
identical across all four.

`summary` deserves emphasis because it is the crudest aware variant — it does
no message passing at all, only pooled statistics of both graphs — and in §5 it
is not separated from the full hierarchical token bank. That is what makes the
claim a claim about *information* rather than about representation machinery.

## 4.2 Heads

Three heads share the encoder, following the branch structure of the design
protocol:

- a **spectral auxiliary** head predicting response summaries at sampled path
  points (used only as a training signal; §7 shows it is not load-bearing);
- a **policy** head decoding a feasible schedule directly from the tokens;
- a **critic** $Q$ predicting the loss of a *given* schedule for a *given*
  instance.

The critic is the operative head. At deployment it scores each pool member and
the argmin is executed; **no outcome is observed during selection**. Nothing in
the loop requires a spectrum, and nothing requires a simulator call on the test
instance, so the method is `amortised`.

## 4.3 The pool

The critic ranks a pool containing the policy's own proposals together with
admitted baselines. Pool composition is the second axis of the method and we
walk it explicitly in §6:

| pool | members |
|---|---|
| proposals only | the policy head's decoded schedules |
| + fallback | proposals and the matched linear ramp |
| + designed library | proposals, linear, slow windows, pauses (8 fixed waveforms) |
| full library | proposals and all 64 stored candidates |

Every member of the library is a **fixed waveform, identical across
instances**. Carrying them costs no simulation at deployment: the critic
predicts their losses from the graph. What varies per instance is the ranking,
which is exactly the quantity the representation is supposed to inform.

Physics-derived schedules (normalized $D_2$, gap-only allocation) are a natural
pool member and are **deliberately excluded**, because computing them at
deployment requires the spectrum and would move the method into
`privileged_spectrum`. We report them in §7 as teachers, in their own class.

## 4.4 Training

Losses for *all* evaluated candidates are stored, and the critic is trained on
a soft target over them rather than on a one-hot best — a near-optimal set is
broader than its argmin and brittle endpoint labels discard most of the signal.
Training adds a ranking term and the spectral auxiliary term above. Selection
at evaluation time never sees an outcome; the supervised losses are offline
labels of the training split.
