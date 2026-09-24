# 2. Setup: the problem the device actually solves

## 2.1 From logical instance to executed Hamiltonian

A logical Ising instance is a graph $G_{\mathrm{L}}$ with fields $h$ and
couplings $J$. A device offers a fixed hardware graph, and almost never the one
the instance needs. **Minor embedding** maps each logical spin $v$ to a
connected *chain* $C_v$ of physical qubits, distributing $h_v$ across the chain
and placing a ferromagnetic **penalty coupling** of strength $\kappa$ on every
intra-chain edge so the chain prefers to act as one spin. Logical couplings are
routed onto whichever physical edges connect the two chains. Finally the whole
physical Hamiltonian is multiplied by a **programmed scale** $\alpha$ to fit the
device's range.

Three consequences matter for control, and all three are invisible to the
logical instance:

1. the executed problem has more spins, a different graph, and a different
   spectrum than $G_{\mathrm{L}}$;
2. $\kappa$ trades one failure mode for another — too weak and chains break,
   too strong and the penalty dominates the problem;
3. $\alpha$ rescales every energy, and therefore every gap, and therefore the
   timescale on which a schedule is or is not adiabatic.

## 2.2 The annealing path and the control

The device traverses

$$\bar H(s) \;=\; a(s)\,H_X \;+\; b(s)\,H_Z \;+\; c(s)\,H_{XX},
\qquad H_X=-\sum_i X_i,\quad H_{XX}=\sum_{ij} K_{ij}X_iX_j,$$

with $H_Z$ the embedded problem Hamiltonian and $H_{XX}$ an optional catalyst.
The **control** is the schedule $s(\tau)$, monotone with $s(0)=0$, $s(1)=1$,
executed over a runtime $T$. Feasibility is a bound on the traversal rate,
$\mathrm{d}s/\mathrm{d}\tau \le 4$ throughout; every schedule reported here
satisfies it by construction, and the bound is checked rather than assumed.

Given a schedule we propagate the closed system exactly, decode each chain by
majority vote, and define

$$\mathcal L \;=\; 1 - \Pr[\text{decoded state is a logical ground state}].$$

Lower is better. All propagation is adaptive with step-doubling diagnostics and
an independently tightened tolerance on a random subset; near-optimal
candidates are rescored at tighter tolerance before being ranked sharply.

## 2.3 The unit of independence

An embedding, a gauge, a chain strength and a runtime are all *variants of one
logical problem*. Treating them as independent observations would inflate every
interval in this paper by a large factor. We therefore keep **all variants of a
logical instance in the same split**, and take the **logical parent** as the
unit of independence: every mean is a mean of parent means, and every interval
is a bootstrap over parents.

Where a number is reported per record rather than per parent we say so. One
such case is corrected in §7: a published rank correlation had been computed
over 864 records where there were 48 independent units.

## 2.4 What a control rule is allowed to know

We distinguish four **cost classes** and never rank across them.

| class | what it may use at deployment |
|---|---|
| `fixed` | nothing instance-specific — e.g. a linear ramp |
| `amortised` | a trained model's forward pass; training and data are offline and charged separately |
| `privileged_spectrum` | eigendecompositions of the instance at deployment |
| `online_adaptation` | simulator or device calls on the test instance itself |

This is not bookkeeping pedantry. A spectral rule is `privileged_spectrum`; a
search is `online_adaptation`; the method of this paper is `amortised`. Placing
them in one column would be the central methodological error of the area, and
§5 shows how much the reported effect moves when the reference is changed
without saying so.
