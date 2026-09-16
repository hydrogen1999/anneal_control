# Paper protocol: embedding-conditioned control from compositional Hamiltonians

Status: **v0.2 software contract and research protocol, not an experimental-results report**. The local closed-system workflow now includes generation, numerical teachers, selectable learning ablations, resumable training, evaluation, validation-only tuning, and report generation. Unit tests and smoke runs establish software behavior on small cases; they do not demonstrate an A* conference contribution. The scope below follows the supplied `main(2).tex` and the current source. Literature positions mentioned in that document remain claims to verify against original papers before submission; this protocol does not establish novelty priority.

## 1. A question worth answering

The main question is not whether a GNN can predict a schedule. It is:

> Which information in the programmed physical Hamiltonian is necessary to choose a good feasible annealing control, and when does learning that information improve new-instance decision quality or reduce search cost?

Three contingent claims can organize the paper:

1. **Representation:** particular embedding interventions alter good-action sets even when the logical objective is fixed; physical information improves decisions in those regimes.
2. **Compression:** low-complexity controls suffice on identifiable subsets, but a richer control family is needed on others. Demonstrate the boundary rather than assume one positive difficulty profile or one slow window is always adequate.
3. **Amortization:** an offline-trained controller improves the quality/cost frontier relative to strong global schedules, simple predictors, physics rules, and equal-budget classical search on independent held-out parents.

Each claim may fail. A globally tuned schedule, an outcome-only model, or classical search must be allowed to win. A useful negative result can delimit a representation's validity without manufacturing novelty from architectural terminology. None of these claims implies quantum speedup, universal spectral inference, or guaranteed publication acceptance.

## 2. Current implementation versus required evidence

| Component | Present scope | Not yet established |
|---|---|---|
| Logical generation | Four coefficient families; configurable support and logical sizes; exact labelled-coefficient duplicate audit | Representative hard-instance distribution; measured coverage; isomorphism/gauge-equivalent deduplication |
| Embedding generation | Integrated synthetic-lift and fixed-hardware growth/quotient routes; fixed/uniform/truncated-power-law chain targets | Commercial-topology results; exact achieved quota law under growth; embedding optimizer superiority |
| Compilation | Distributed logical fields/couplers plus chain penalties, explicit common scaling, exhaustive small-system audits | Vendor-specific autoscaling and calibration fidelity |
| Dynamics | Matrix-free X/ZZ/XX operator; single and batched CPU/CuPy propagation; memory guards; independent CPU checks | Measured GPU throughput or large-scale capacity; hardware fidelity |
| Spectral labels | Capped fixed/adaptive full-spectrum teacher; independent random-query audits; cached diagnostics and masks; explicit no-teacher regime | Uniform continuous-path certification; complete scalable sparse-response teacher |
| Sparse eigenpairs | CPU/CuPy low-energy eigensolver with diagnostics and explicit incompleteness flags | Proof all relevant ground/bright states and inverse moments are resolved |
| Learning | Hierarchical, flat physical, logical-only, and summary encoders with common auxiliary/policy/critic heads | Measured ablation advantage; response-bin prediction, device adaptation, uncertainty calibration |
| Training | Outcome/ranking and multimodal bank distillation; graph accumulation; best/latest checkpoints; content-verified epoch-boundary resume | Differentiable physical proposal optimization; full adjoint training; DDP/AMP performance |
| Evaluation | Fixed-bank and optional direct true-simulator scoring; exact-waveform family benchmarks; parent CIs; cost/TTS utilities | Paper-scale direct-policy superiority and measured control-complexity frontier |
| Search | Shared bank; charged local refinement; equal-budget Sobol/local family search; privileged gap/D2 comparators; finite-difference helper | Bayesian optimization and faithful external-method reproductions; continuous-optimum certificates |
| Orchestration | Frozen config/source manifest; stage/seed/method runner; validation-only grid tuning; audit and report exports | Completed paper-scale sweep, scientific conclusions, or an acceptance guarantee |
| Offline adapters | Supplied calibrated path tables; independent capped Lindblad solver; validated program export and sample ingestion | Authenticated device data, measured calibration fidelity, live vendor submission, CUDA-Q/QuTiP adapters |

Important implementation boundaries:

- `configs/pilot.json` proposes 120 logical parents, two embedding variants, two chain strengths, and three runtimes: at most 480 physical paths and 1,440 control tasks. With 64 candidates per task, that is 92,160 candidate outcomes before extra audits/refinement. The config is not evidence that this workload has been completed.
- That pilot currently has **three logical variables and eight physical qubits**, with a synthetic complete logical support. It is an integration dataset, not sufficient evidence of graph-size generalization or realistic hardware difficulty.
- `pipeline.py` now supports multiple logical sizes, configurable support, chain distributions, and declared held-out family/size splits. The fixed legacy pilot config above is only one choice, not the full supported distribution.
- Full-spectrum teachers remain capped at ten physical qubits. Explicit outcome-only generation supports caps up to twenty with endpoint/state memory guards and chunked exact endpoint enumeration. These are permitted limits, not measured GPU capacity or runtime guarantees; state vectors and endpoint enumeration remain exponential.
- Dataset generation executes nine-knot resampled waveforms. A named one-window or pause candidate before resampling is not necessarily the exact same family member afterward. `benchmark_record_controls` is the separate implemented exact-switching-knot path for family frontiers; store and plot its actual waveform and charge its search budget.
- The CLI's common bank does not use the test spectrum. Physics-derived candidates are a separate solver-assisted comparator whose inference cost must be charged.
- `evaluate_records` measures **finite-bank critic selection**. The CLI's optional `--direct` mode selects a proposal using the critic, then independently propagates that executed waveform for offline scoring. Never substitute the nearest bank label or an interpolated outcome for such a measurement. This implemented scoring path is not evidence of paper-scale direct-policy superiority.
- `candidate_batch_size` enables row-wise batched candidate propagation; `workers` parallelizes independent parents on CPU. CuPy generation requires one worker, performs explicit device/memory checks, and does not fall back silently. Per-row batch cost is an allocation of measured shared time, not isolated row latency. Graph-gradient accumulation in training is separate from vectorized simulator batching.

## 3. Data generation is the central experimental instrument

### 3.1 Separate three data regimes

| Regime | Purpose | Valid conclusion | Invalid shortcut |
|---|---|---|---|
| Exact small-system diagnostic/pilot | Audit physical construction, selection rules, labels, and control families | Mechanism under a declared small closed-system model | Treating thousands of related schedules as thousands of independent instances |
| Controlled hard-target generation | Create known structural/coherent stress regimes and broad randomized families | Behavior on declared generators, including failures | Calling connected growth or planted endpoint optima “quantum-hard” without measurement |
| Real hardware | Test executed feasible waveforms and sim-to-device transfer | Performance under measured device settings and uncertainty | Treating toy caps, generic Lindblad noise, or synthetic lifts as a D-Wave experiment |

Generate Hamiltonians from named parts, retaining their provenance: logical objective, logical-to-physical membership, distributed fields and inter-chain couplers, intra-chain penalties, programmed coefficient scaling, driver, and optional catalyst. The authoritative operator must be reconstructed from these parts, not stored as an unexplained dense random matrix.

Record both raw and programmed coefficients. Verify for every aligned logical assignment that

\[
E_{\rm phys}(z\circ\pi)=\alpha E_{\rm logical}(z)+C_{\rm chain}.
\]

This identity and preservation of an aligned physical ground state are different checks. Weak penalties can satisfy the identity but produce lower broken-chain states. Keep those failures as a declared stress stratum or exclude them by a predeclared physical-validity rule; do not silently discard them after seeing model results.

### 3.2 Two embedding-generation routes answer different questions

**Fixed logical parent, varying embedding:** synthetic lifts isolate controlled geometry/port/load effects in a toy physical graph. A subsequent commercial-topology experiment needs multiple feasible embeddings of the same logical support on one verified hardware graph. This route supports paired embedding interventions.

**Fixed hardware, growing chains then taking a quotient:** connected multi-source growth creates disjoint chains and a realizable quotient support. It guarantees that constructed support has a feasible embedding, not annealing hardness or exact sampled chain quotas. Report target and achieved lengths, jamming rate, unused sites, and the resulting support distribution. Two different quotients are generally different logical problems; they cannot be called “the same logical instance re-embedded.”

A truncated power law is one configurable target-length distribution, not a universally justified choice. Compare it with uniform and fixed-length controls at matched total physical size where possible. Growth-induced censoring can change the achieved distribution substantially. Publish achieved-length histograms and rejection/jamming logs; do not infer the exponent from requested quotas.

The implemented sampler also conditions requested lengths on the physical budget and records rejected draws. The hardware-growth route fixes the grown embedding for a parent; variant entries can change coefficient allocation, but do not silently supply alternative embeddings of the same objective. Declare those roles in the generator card.

### 3.3 Hardness is a measured attribute

Keep a broad randomized benchmark and a separate mechanism/stress benchmark. Candidate stress axes include zero/weak fields, bright versus dark low gaps, frustration at fixed absolute couplings, multiple transition regions, chain-boundary load concentration, and runtime-dependent interference. The existing weak-field family alone is not the full suite.

Define screening quantities and thresholds using training/validation parents only. Appropriate diagnostics include loss spread across controls, linear-to-best-found headroom, disagreement between weaker and richer representations, and family-restriction loss. Record the unscreened population, selection rule, selected fraction, and cost. Headroom-selected test instances are a conditional stress benchmark, not an unbiased claim about general deployment frequency. Do not filter on the proposed model's advantage.

Classically planted ground states certify endpoint energies only. Noncommuting component Hamiltonians do not have spectra obtained by adding independently diagonalized spectra. GPU acceleration changes cost, not either scientific fact.

## 4. Experimental units, splits, and seeds

Split **logical parents before** embeddings, coefficient-distribution variants, chain strengths, gauges, permutations, runtime variants, candidate outcomes, or reads. All descendants inherit their parent's primary split. Parent IDs require provenance checks: different IDs for duplicate/gauge-equivalent generated parents can still leak.

The generator now rejects exact duplicate labelled logical coefficients and supports family/size holdouts before descendants. Its duplicate check is not a graph-isomorphism or gauge-equivalence certificate. In fixed-hardware growth, the quotient and its defining embedding establish the logical parent before any subsequent coefficient/runtime variants are split.

The 120-parent pilot can use 72/24/24 train/validation/test parents, stratified over four families. That is a debugging allocation, not a power calculation. Determine the paper-scale parent count from diagnostic paired-variance estimates, desired confidence precision, and compute cost; freeze it before inspecting final test outcomes.

Predeclare separate evaluations:

- **New parents, familiar distribution:** main generalization test.
- **Held-out family:** leave-one-family-out training and validation design; do not call ordinary stratification unseen-family transfer.
- **Held-out physical sizes:** trained and tested physical/logical sizes explicitly listed. A config with one fixed size cannot support this claim.
- **Unseen embeddings of seen logical parents:** a distinct, weaker deployment-relevant task; never combine its scores with new-parent results.
- **Unseen runtime values:** distinguish interpolation from extrapolation and keep dimensionless runtime separate from microseconds.
- **Device or device-day transfer:** require held-out parent and calibration-batch protocols; otherwise report transductive adaptation honestly.

Suggested initial budget specification, to freeze after profiling: one public dataset master seed, five training seeds `[0,1,2,3,4]`, and three classical-reference search seeds `[11,23,37]` per diagnostic task. These are design choices, not a guarantee of statistical sufficiency. If compute permits fewer repetitions, record the change before final test inspection and report the reduced uncertainty coverage. Use independent random streams for parent generation, embeddings, allocation, bank construction, training, and local search. Do not retain only successful seeds.

A primary uncertainty unit is the independent logical parent. Average matched variants within a parent or use a declared hierarchical analysis. Parent bootstrap intervals do not include training-seed uncertainty automatically: show seed-level summaries or implement a predeclared hierarchical design. Hardware reads are not independent problem instances; programming cycles/device drift can also invalidate a simple binomial model.

## 5. Teacher and control-family protocol

### 5.1 Teacher correctness before teacher learning

For the exact diagnostic subset, compare operator application against independent dense construction; check eigenpair residuals and orthogonality; test the response variance/sum rule; and audit final propagation with an independent numerical method and tightened tolerances. Handle endpoints, degeneracies, and changes of ground-band rank explicitly. Projector-band response is not the instantaneous population of an evolved system.

Fixed-grid mode is not an interpolation guarantee. Adaptive mode now refines midpoint discrepancies in gaps, moments, bin mass, rank changes, and unresolved response, then performs independent held-out random-point audits. Audit queries never feed back into refinement, and both query sets consume the stated budget. Record budget exhaustion, unchecked intervals, minimum-width stopping, numerical failures, and failed audits. Passing sampled checks does not certify the entire continuous path (`uniform_certificate=false`). If regularization is used, label the target as regularized and vary its scale in sensitivity checks.

Adaptive coordinates are privileged auxiliary-training query locations. They do not enter the policy/critic structural encoding; fixed public queries suffice for deployment. Do not use a newly computed test spectrum or its adaptive locations as uncharged decision inputs. Explicit `teacher.mode=none` masks all absent response labels rather than treating zeros as observed targets.

For sparse scale-up, increase retained eigenpairs and audit response mass and moments. Small residuals only validate computed pairs; they do not prove that omitted bright modes are irrelevant or that the full low-energy band was found. The current `sparse_low_energy` return contract deliberately does not supply a complete D2 teacher. Do not feed it into production supervision until an audited completeness/truncation policy exists.

### 5.2 Separate a spectrum's accuracy from its decision value

On the same physical instances, runtime, feasible family, and schedule rule, compare exact logical-spectrum and exact physical-spectrum controls. Then compare raw first-gap, resolved accessible-gap, multilevel response, and finite-time search controls. This isolates logical/physical mismatch and the role of matrix elements before introducing learning errors.

Perturb spectral teachers in controlled ways: gap-only perturbation, transition-strength perturbation, omitted bright levels, coarser path grids, and degraded numerical tolerance. Plot resulting schedule outcomes against actual teacher-generation cost. This measures the label precision the decision needs, not merely an arbitrary eigenvalue-MSE target.

### 5.3 Control-complexity frontier

Evaluate linear, one-window, two-window, 8-bin, and richer monotone feasible controls with one objective and matched runtime/limits. Ordered bang controls, nonmonotone controls, and independently optimized QAOA angles are separate action spaces; do not place them on a supposedly nested monotone frontier.

For each family and declared evaluation budget, retain every evaluated control and the best found. Proposed diagnostic budgets are 64 shared candidates plus up to 64 local evaluations; a reference subset can use repeated larger budgets after cost profiling. A budget is a cap and accounting rule, not a certification of global optimality.

The implemented family benchmark alternates scrambled Sobol exploration and local incumbent perturbations. Every tunable family receives the declared total objective-call budget, including its initial linear incumbent and rejected trials; parameter-free linear receives one evaluation. Switching knots are integrated exactly as segment boundaries, with separate norm/convergence gates. Optional privileged gap-inverse-square and D2 baselines report spectrum costs outside the equal-budget zero-shot comparison. Test-instance family search requires explicit online-adaptation opt-in.

Only call families nested if the simpler executed waveform can be represented exactly in the richer family. In that case include the simpler incumbent explicitly, with consistent cost accounting, so increasing expressivity does not discard a previously feasible solution. At equal compute a richer optimizer can otherwise return a worse found solution. Report both expressivity and optimization effort; do not attribute all differences to control complexity.

Use distinct quantities:

\[
R_{\rm bank}(\hat\theta)=L(\hat\theta)-\min_{\theta\in\mathcal B}L(\theta),
\qquad
R_{\rm ref}(\hat\theta)=L(\hat\theta)-L(\theta_{\rm best\ found}).
\]

The first is nonnegative when the selected action belongs to that bank. The second may be negative when a model beats an imperfect search reference; retain that sign. Neither is global-control regret without a certificate. State the candidate family, seed, budget, and label error beside each reference.

## 6. Controlled learning ablations and credible competitors

| Factor | Matched comparison | Evidence required |
|---|---|---|
| Physical information | Logical-only versus physical-input model; matched heads/losses/control family/tuning budget | Paired held-out outcome improvement and embedding-intervention sensitivity |
| Hierarchy | Flat signed physical encoder versus physical-chain-logical tokens | Improvement beyond merely adding physical information or parameters |
| Spectral auxiliary | Same model, policy loss and data, response weight zero versus nonzero | Added decision value; include teacher-generation cost |
| Auxiliary target | First gap, resolved gaps, moments, response bins | More useful controls, not just lower supervised loss |
| Direct bypass | Mandatory scalar-profile route versus direct policy/critic route | Benefit on selected failure mechanisms and unbiased test population |
| Outcome supervision | Imitation-only versus outcome/ranking; policy distillation on/off separately | A single comparison must not simultaneously change several loss terms |
| Control family | Identical predictors with audited feasible output families | Learning error separated from family restriction |
| Online assistance | Zero-shot versus budgeted verified refinement | Cost/quality improvement at equal new-instance evaluator budgets |

The implemented encoder variants are `hierarchical`, `physical`, `logical`, and `summary`; loss switches permit a response-on/off test and a critic-only outcome/ranking test. Removing policy and response terms together does not isolate the spectral auxiliary. The logical variant excludes every physical/chain feature and programmed scale; the flat physical variant retains local same-chain channels and is not embedding-blind. Summary features are declared aggregate moments/counts, not a topology-complete representation. Parameter counts and tuning budgets must be reported. Response-bin prediction, a mandatory scalar bottleneck, and faithful external related-work implementations remain optional extensions beyond this completed baseline workflow.

Minimum comparator set for a persuasive paper:

1. Linear schedule at the **same total duration**, and a validation-tuned global window/curve.
2. Summary-statistics predictor and signed outcome-only graph predictor at comparable tuning budgets.
3. Gap-only and transition-aware local-adiabatic rules, with exact diagnostic teachers and properly charged solver-assisted deployment variants.
4. Shared-bank/Sobol search and stronger equal-budget classical optimization, including Bayesian optimization where appropriate. The current random local search is not a Bayesian-optimization implementation.
5. A faithful closest-method reproduction where feasible. If simplified, use “inspired by,” disclose deviations, and do not quote published numbers as same-benchmark results.

The attached document identifies relevant spectrum-learning, amortized scheduling, and susceptibility-based approaches. Verify their original papers and executable control conventions before finalizing the related-work table; do not infer implementation equivalence from method names. A cheaper classical surrogate or global schedule matching the proposed method is an important result.

## 7. Paired embedding interventions

For a fixed logical parent, hold logical coefficients, objective/decoder, driver family, runtime, candidate bank, numerical accuracy, and comparison budget fixed. Intervene on one declared physical feature: embedding geometry, number/location of boundary ports, field allocation, or chain strength. Start with matched physical size; separately test effects that intentionally change it.

Coefficient scaling is often a mediator of the embedding/chain-strength intervention. Report:

- **Total compiled effect:** use each embedding's declared programmed scaling.
- **Geometry/allocation control:** where feasible, use one common conservative scale across the pair and hold the driver and runtime unchanged. State that this is a distinct controlled experiment, not vendor autoscaling.

Evaluate a cross-control matrix on each pair: the actual loss of controls selected for embedding A and B, each executed on both physical Hamiltonians. Better evidence than merely changing predicted parameters is that the preferred control swaps and transferring the wrong control incurs measurable loss. Compare broad near-optimal sets when several schedules tie; a single unstable argmin is not enough.

For fixed-family exact/best-found diagnostic controls, record within-embedding advantage, cross-embedding transfer penalty, and the paired loss surfaces. For learned models, compare logical-only and physical-aware responses to the same intervention. These are causal interventions **inside the declared simulator**, not proof of a causal effect on unmeasured real hardware. The fixed-hardware quotient route cannot provide this pair unless the logical objective is genuinely preserved.

## 8. Primary metrics and uncertainty

Primary loss is `1 - decoded_logical_success` for the exact pilot. The accepted physical outputs are **all** strings whose declared decoding is logically optimal, including broken chains that decode correctly. Report physical chain-break rates separately. Fix tie handling and any gauge/un-gauging convention before evaluation.

Report absolute paired success difference, declared-reference loss difference, failure-tail quantiles, and decision cost. Include mean decoded energy as a separate objective/diagnostic. Schedule-parameter MSE, spectral MSE, feasibility rate, and raw chain-break reduction are secondary; none alone establishes better control.

Avoid normalized improvement ratios when the reference-to-linear headroom is near zero. Report low-headroom tasks and their prevalence; do not delete them to inflate ratios. Predeclare one primary comparison and a small set of mechanism comparisons. Distinguish exploratory subgroup patterns and account for multiple comparisons if making family-wide significance claims.

For measured success counts, use intervals and report zero-success censoring. A pseudocount must not manufacture precise finite TTS. TTS assumes an explicit repeated-trial model, and exact simulation probabilities are not hardware sample counts. For large systems without a certified optimum, use a frozen energy target and call the quantity time-to-target, not certified time-to-solution.

## 9. Paper figures and tables: claim-to-evidence map

| Item | Concrete content | Claim it can support |
|---|---|---|
| **Figure 1: data provenance and learning pipeline** | Named logical/physical/driver parts; compilation and validation; separate spectral and action-response labels; parent split before variants; training-only arrows distinguished from deployment inputs | The formulation and reproducible experimental instrument, not performance |
| **Figure 2: teacher correctness and selection rules** | A dark-gap example plus weak symmetry breaking; raw/accessible gaps and transition mass; residual/sum-rule checks; resulting loss surfaces for gap-only versus multilevel prescriptions | First gap can discard decision-relevant information under specified conditions |
| **Figure 3: control complexity versus best-found loss** | One-/two-window and bin families across runtime; nested-incumbent protocol; search-budget axis/insets; reference quality and unresolved cases shown | Distribution-dependent control compression and failure boundary |
| **Figure 4: same parent, different embeddings** | Matched physical structures and coefficient scales; 2x2 cross-control loss matrices; paired transfer penalties; logical-only versus physical-aware decisions | Embedding information changes actions and improves decisions in measured regimes |
| **Main result table** | Independent held-out parents, family, runtime, physical/logical size, selection mode, success, paired differences, reference regret and cost | Generalization on the declared benchmark, with clearly bounded inference claim |
| **Transfer/cost figure or table** | New parents, unseen sizes/runtimes, seen-parent re-embedding, and optional device transfer as separate blocks; quality versus online budget and deployment count | Where amortization pays off and where it does not |
| **Appendix data audits** | Generator coverage, achieved chain lengths, jam/exclusion rates, duplicates/splits, numerical convergence, teacher censoring, runtime/scale distributions, all seeds and costs | Reproducibility and limitations of the evidence |

For the main result table use separate panels or companion columns instead of one overcrowded table:

| Required row key | Required reported fields |
|---|---|
| Method, inference mode, test family, size stratum, runtime stratum | Independent-parent count; training seeds; absolute success; paired success difference with CI; declared-reference regret; online evaluator calls; inference/search time |
| Optional hardware block | Device and programming batches; reads; actual waveform restrictions; success counts/intervals; adaptation budget |

Mark missing experiments as **not run**, not zero, and do not fill templates with illustrative performance numbers. Unit-test pass counts and smoke losses belong in software validation, not the main-results table. The figure captions must identify exact diagnostic, synthetic pilot, hard-target, and hardware panels visibly.

## 10. Transfer, hardware, and full cost accounting

Report offline data generation and training costs, plus new-instance graph processing, inference, embedding when method-dependent, online candidate evaluations, programming, execution, and readout. Separate reusable offline teacher cost from any spectrum recomputed on a test instance.

\[
C(M)=\frac{C_{\rm data}+C_{\rm train}}{M}
+C_{\rm inference}+C_{\rm online\ search}+C_{\rm execution}
+C_{\rm embedding}+C_{\rm programming}.
\]

Show both unamortized cost and a curve over deployment count M; do not choose only a large M favorable to learning. Count failed labels, reference search and tuning separately, and avoid counting search executions twice. If the model evaluates K candidates with its critic, report K and its measured latency even when no physical oracle is called.

Hardware is a later, separately authorized execution stage. Before any claims, verify the selected solver's actual schedule limits and calibration; use executed waveform points, matched total duration, fixed compilation/readout settings, and interleaved method programming. Distinguish zero-shot transfer from same-instance refinement and adaptation on other labeled parents. Freeze adaptation budgets and held-out device/parent assignments before acquiring final results.

The offline adapter layer now accepts supplied calibration tables with explicit component factors and time/energy units, runs an independent capped dense Lindblad solver, validates device-constrained program exports, and ingests explicitly supplied sample counts after hash/energy/gauge checks. The ordinary training pipeline remains closed-system unless the caller deliberately creates a different declared dataset through those APIs. Computational-basis relaxation is not an interacting thermal bath; a generic Lindblad model does not certify D-Wave fidelity. Density-matrix storage is exponential (`O(4**N)`), with default six and absolute eight-qubit caps.

Program export makes no vendor API call and performs no implicit rounding or autoscaling. Ingestion establishes internal sample/program consistency, not authentic QPU provenance. Device authorization, current constraints, calibration dates, drift-aware uncertainty, and actual quantum runs remain external work. XX simulation is not evidence of native programmable XX hardware couplers. CUDA-Q and QuTiP are not implemented adapters. Gate digitization/QAOA, learned device-likelihood adaptation, and a full hardware-transfer study are separate optional extensions, not hidden capabilities of this prototype.

## 11. Reproducible execution, tuning, and report generation

`experiments.py` executes an explicit method × seed plan through `generate`, `train`, `evaluate`, and `report` stages. The resolved dataset/model/training/evaluation configuration, source hash, environment, and stage artifacts are written under an output lock. Resume refuses changed configuration/source and does not silently steal stale locks; partial failures remain recorded. Dry planning reports requested workload, not completed labels or anticipated performance.

`fit_records` supports configurable sequential graph accumulation, clipping once per optimizer update, validation-only early stopping, and atomic best/latest checkpoints. Resume verifies actual ordered train/validation contents, settings, architecture, and fingerprints and restores optimizer, RNG, history, and stopping state. An epoch target can be increased without relabeling the run as a fresh training attempt; an already-triggered early stop stays stopped. Best-validation weights are used for held-out evaluation. Cross-device bitwise equality, fused graph batching, DDP, and mixed precision are not promised.

The separate tuning runner explores a capped Cartesian grid over model/training keys, selecting the equal-seed mean validation bank regret. It never invokes held-out evaluation during selection. Training's validation metric weights records; balance descendant counts across parents or predeclare a different selection metric before making a parent-weighted claim. The runner requires every declared method/seed evaluation before producing final reports, preserving negative results and failed-run visibility.

`reporting.py` verifies matching record/reference sets across methods/seeds, averages seeds per record then variants per parent, and emits JSON/Markdown summaries, a LaTeX table, and optional plots. Parent-bootstrap CIs are conditional on the observed training seeds; seed-level standard deviation is a separate descriptor, not a joint uncertainty interval. These generated artifacts populate results from actual runs; they do not manufacture the mechanism panels, commercial-hardware evidence, or faithful external-method comparisons needed by a particular paper claim.

## 12. Sequential gates and stop/reframe decisions

| Gate | Work and deliverable | Decision rule |
|---|---|---|
| **G0 — Reproducible software** | Clean environment, unit tests, tiny train/checkpoint/evaluate cycle, manifest and resume audit | Fix numerical/schema/split failures before drawing research conclusions |
| **G1 — Generator validity and coverage** | Exhaustive compilation/endpoint checks on diagnostics; broad and stress generator cards; achieved-size and exclusion reports | If a route does not preserve the intended logical comparison or misses useful diversity, repair the generator before scaling |
| **G2 — Teacher and control-family validity** | Independent dynamics checks, dark-mode tests, teacher perturbations, search-budget and family frontier | If a target/family fails the declared application loss tolerance, revise it or define a fallback; do not train around a known invalid target |
| **G3 — Representation decision signal** | Paired embedding loss surfaces; summary, logical, flat physical, hierarchical and outcome-only controls | If physical/auxiliary information lacks a repeatable decision advantage at equal cost, narrow or reject that claim |
| **G4 — Independent generalization** | Frozen split/tuning/search specification; multiple training seeds; held-out family/size/runtime evaluation | If gains depend on leakage, one seed, or a conditional cherry-picked subset, report that limit and expand evidence before claiming generality |
| **G5 — New-instance cost and transfer** | Direct proposals truly propagated; stronger equal-budget search; amortization curves; optional separately audited hardware | If online cost or model mismatch erases gains, state the useful regime or reframe the contribution |
| **G6 — Paper readiness** | Every headline claim has a matched comparator, figure/table, uncertainty analysis, limitations and reproducibility artifact | Submit claims supported by completed experiments only; no A* acceptance guarantee |

Gate tolerances are project decisions. Before each relevant final experiment, write down numerical label precision, application loss tolerance, search/tuning budgets, and uncertainty targets using diagnostic/validation information. Do not choose a success-percentage threshold after seeing test results and present it as a theorem. If no operational tolerance is justified, report the entire regret/cost frontier instead of asserting an arbitrary pass/fail boundary.

The immediate sequence is G0–G2: make the compositional data generator and teacher defensible, measure control-family headroom, then expand the learner. A larger network or dataset cannot by itself repair an invalid Hamiltonian, a leaked split, an incomplete teacher, or an uncompetitive inference budget.
