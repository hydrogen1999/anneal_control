# 9. Related work

> **Verification status.** Groupings below are drawn from titles in the
> project bibliography and have **not** been checked against the papers'
> contents. Before submission every characterisation here must be verified
> against the source, and `scripts/verify_bib.py` must be run on the `.bib`
> (`MISMATCH` is the verdict that matters; `NOT_FOUND` alone is not proof of
> fabrication). Treat this file as a structure with placeholders, not as
> finished prose.

## 9.1 Schedules derived from the spectrum

The dominant approach chooses a schedule from adiabatic theory: bound the
diabatic error by the instantaneous gap and allocate time accordingly
[Jansen2007; RolandCerf2002], with the general setting reviewed in
[AlbashLidar2018]. Refinements replace the bare gap with a geometric or
response-based quantity [Kolodrubetz2017], and optimal-protocol analyses
characterise the shapes these rules produce [Brady2021; Shevchenko2010].

**Where we differ.** These rules are computed from a spectrum, and therefore
require one at deployment — `privileged_spectrum` in the accounting of §2.4.
Our method uses none. We also find (§7.1) that a raw first-gap rule is worse
than a linear ramp on our populations, while a transition-element-weighted
refinement is clearly better, which is consistent with the refinement
literature and sharpens it: the resolution matters *as a control rule*.

## 9.2 Schedule shape, empirically

Pausing and related shape interventions have a substantial empirical
literature [ChenLidar2020; Marshall2019], and catalyst terms alter the spectra
being traversed [Feinstein2024]. Our candidate library deliberately contains
pauses and slow windows, and §6.4 shows that eight such designed waveforms
carry two thirds of the full library's advantage.

## 9.3 Searching for a schedule per instance

Bayesian optimisation [Finzgar2024] and tree search [Chen2022] tune schedules
per instance, in the tradition of gradient-based quantum optimal control
[Khaneja2005].

**Where we differ.** These are `online_adaptation`: they spend simulator or
device calls on the test instance. We use such a search only as a
**denominator**, and §5.2 prices one forward pass against it in that search's
own currency — roughly 24 instance-specific calls on Pegasus, at least 15 under
the least favourable of three search strategies.

## 9.4 Learning schedules

Closest to us, deep models have been trained to output annealing schedules for
random Ising models [Hegde2023], to learn parameter curves in feedback-based
optimisation [PenaPerez2026], and to reconstruct spectral information with
sequence models [Lu2026]; susceptibility-based scheduling goes beyond local
adiabaticity [Singh2026].

**Where we differ.** These operate on the logical problem. The object a device
executes is the *embedded* one, and our central result (§5.1) is that an
encoder restricted to the logical graph is separated from every
embedding-aware encoder. We also find that the architecture consuming the
information does not measurably matter, which suggests the gap is not closed by
a better model of the logical instance.

## 9.5 Embedding

Minor-embedding parameter setting — chain strengths and the attendant
trade-offs — is a developed topic [Choi2008], and device documentation fixes
what a schedule may legally be [DWaveParameters; DWaveValidator]. This
literature treats the embedding as something to configure well; we ask what a
*controller* needs to know about it.

## 9.6 Graph networks and decision-focused learning

Our encoder ablation touches expressive-power questions [Xu2019] and
over-squashing [Alon2021]; graph networks have been used to initialise
variational quantum algorithms [Jain2022], and gauge-equivariant architectures
exist for lattice gauge theory [Rayat2026] — relevant to the covariant arm we
did **not** implement (§8.5). The training objective is decision-focused in the
established sense [Wilder2019]: optimise the quality of the chosen action, not
prediction accuracy on an intermediate quantity.

**A caution about our own result.** §5.1 finds that a statistics-only encoder is
not separated from a full message-passing token bank. That is a negative about
architecture on *this* task and sample size, not a general claim about graph
networks.

## 9.7 Adjacent

QAOA shares the variational-schedule framing [Farhi2014] but optimises
per-instance parameters, which is again `online_adaptation`.
