# Kế hoạch triển khai: học biểu diễn phục vụ điều khiển embedded Ising annealing

Ngày: 16-09-2026. Đầu vào: ghi âm buổi họp 11-09 và `main(2).tex` đính kèm.

## Cập nhật triển khai v0.2

**Bắt đầu vận hành từ `RUNBOOK_VI.md` và README hiện tại.** Bản v0.2 đã hoàn thiện
runner cấu hình, dữ liệu nhiều sizes/supports, hardware-growth records, adaptive
teacher có independent audits, batch propagation, bốn encoder ablations, training
resume đầy đủ optimizer/RNG, validation-only tuning, exact-waveform control
baselines, báo cáo nhiều seeds, inference và offline hardware export/ingestion.
Lindblad/calibrated-path utilities và accepted-label profiling cũng đã có code.

Các mục 1–8 dưới đây lưu **lý do thiết kế và roadmap nghiên cứu của v0.1**; câu
“chưa triển khai” trong phần lưu lịch sử không mô tả trạng thái code v0.2. Đối
chiếu `docs/data.md`, `teacher.md`, `training.md`, `benchmarking.md`, `adapters.md`
và `paper_protocol.md` cho contract hiện tại. `reports/V02_VERIFICATION.md` ghi
đúng các kiểm chứng đã chạy; không thay lịch sử bằng một kết quả paper giả.

Các gates nghiên cứu vẫn còn giá trị: distribution/control headroom, lợi ích
representation, robustness/OOD, chi phí, nhiều seeds và hardware transfer.
Chưa có kết quả GPU/QPU thực, calibrated open-system training benchmark,
tensor-network teacher quy mô lớn, CUDA-Q/QuTiP adapter hay multi-GPU DDP.

---

## 1. Quyết định kỹ thuật

Đặt data generation và tính đúng của label làm nền móng; chưa lấy một architecture lớn làm trung tâm. Đối tượng sinh dữ liệu là **Hamiltonian vật lý theo thời gian, target đo/giải mã, và response với từng control**, không chỉ là graph cùng một vector spectral gap.

Đích nghiên cứu: tìm một biểu diễn của programmed physical system đủ để chọn control có chất lượng tốt trên instance mới, với ít chi phí tìm kiếm hơn. Các giả thuyết về compression, lợi ích của embedding information và transfer đều cần thực nghiệm. Source này là nền tảng có thể chạy, không phải bằng chứng đã đạt chất lượng paper A*.

Không đồng nhất ba việc:

1. Sinh instance có minor embedding hợp lệ.
2. Chứng nhận nghiệm tối ưu của classical endpoint.
3. Sinh instance đủ khó và đủ đa dạng cho **quantum control**.

Một generator có thể đảm bảo hai việc đầu nhưng chưa làm được việc thứ ba.

## 2. Công cụ GPU thầy nhắc là gì?

Chưa xác định chắc từ audio. Đoạn khoảng 05:51 nằm trong thảo luận closed-system/formal model và adaptation sang hardware; ASR có thể nhận “open … QPU” thành “Open GPU”. Không nên suy đoán thành một tên package đã được chốt.

**Q-GPU là một công trình thật ở HPCA 2022**, nhưng là framework tối ưu quantum circuit simulation, không đồng nghĩa với CUDA-Q. Nguồn: [trang tác giả Q-GPU](https://zhaoyilun.org/publication/zhao-2022-qgpu/), [DOI](https://doi.org/10.1109/HPCA53966.2022.00059).

Quyết định triển khai độc lập với tên chưa rõ đó:

| Công việc | Backend hiện tại | Backend cần đo tiếp |
|---|---|---|
| Sinh graph, partition, h/J, chain penalties, provenance | NumPy/Python | CPU workers theo parent/path |
| Exact spectral teacher hệ nhỏ | NumPy/SciPy dense Hermitian | GPU dense chỉ khi đo có lợi |
| Low-energy eigenpairs | SciPy LinearOperator/eigsh prototype | CuPy LinearOperator/eigsh prototype |
| Closed-system propagation | Matrix-free NumPy, Strang splitting | Cùng thuật toán với CuPy |
| Reference độc lập | Dense DOP853 nhỏ | CUDA-Q Dynamics hoặc QuTiP adapter tương lai |
| Open-system/device adaptation | Chưa triển khai | QuTiP/CUDA-Q Dynamics với bath model ghi rõ |

[CUDA-Q Dynamics](https://nvidia.github.io/cuda-quantum/latest/using/dynamics.html) hỗ trợ dynamics và time-dependent operators. [CuPy eigsh](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html) nhận LinearOperator và tìm low-energy eigenpairs. Cả hai không tự quyết định distribution, không tự chứng minh convergence của label, và không tự biến simulator thành một model D-Wave đã calibration.

Không cài CUDA stack khi chưa biết GPU/CUDA của máy triển khai. Chọn một CuPy build phù hợp theo [hướng dẫn cài đặt chính thức](https://docs.cupy.dev/en/stable/install.html), rồi chạy parity tests; không cài đồng thời nhiều biến thể CuPy. Chi tiết nguồn và giới hạn: `docs/gpu_backend_evidence.md`.

## 3. Data contract: sinh Hamiltonian từ các phần

Với physical qubit i:

\[
H_X=-\sum_iX_i,\quad H_Z=\sum_i h_i^{prog}Z_i+\sum_{ij}J_{ij}^{prog}Z_iZ_j,
\quad H_{XX}=\sum_{ij}K_{ij}X_iX_j.
\]

\[
\bar H(s)=a(s)H_X+b(s)H_Z+c(s)H_{XX}.
\]

Đường mặc định: a=1-s, b=s, c=0. API vật lý hỗ trợ catalyst c=4λs(1-s), nhưng model/CLI pilot hiện tại chủ động từ chối catalyst khác 0 vì chưa có driver-graph features tương ứng. Không dùng khả năng simulator để ngầm hứa model đã support.

Mỗi logical parent được tạo một lần, có split trước mọi augmentation. Từ parent, dựng embedding variants và chain-strength variants; mỗi physical path được đánh giá tại nhiều runtime/control. Mọi biến thể của cùng parent phải ở cùng split.

### 3.1 Compiler đúng hệ số

- Phân phối h_v lên các member của chain, tổng bằng h_v.
- Phân phối J_vw lên các physical boundary couplers, tổng bằng J_vw.
- Thêm -κ lên internal connected chain edges.
- Cộng các đóng góp rồi mới áp dụng coefficient scaling α.
- Lưu riêng problem_J, chain_J và total programmed J; không cộng penalty hai lần.
- Kiểm tra tất cả aligned logical assignments trên pilot:

\[
E_{phys}(z\circ\pi)=\alpha E_{logical}(z)+C_{chain}.
\]

Compiler dùng symmetric toy caps có khai báo, **không phải bản tái hiện autoscale của một solver D-Wave**. α chỉ scale H_Z, không tự scale driver hoặc runtime. Khi chuyển sang thiết bị thật phải thay bằng compiler/device calibration đã kiểm tra.

### 3.2 Hai generator không thay thế nhau

**A. Synthetic lift có kiểm soát.** Giữ logical parent; dựng chains path/star/random-tree, thay ports/field distribution. Tốt cho cơ chế và counterfactual vì biết chính xác cái gì thay đổi. Graph vật lý được dựng là toy graph, không được ghi nhãn Pegasus/Zephyr nếu chưa kiểm chứng trên topology đó.

**B. Fixed-hardware connected growth.** Đầu vào là hardware graph đã biết. Chọn seeds không trùng; mở rộng từng chain vào free frontier; quotient graph tạo logical support. Khi growth bị kẹt, trả achieved lengths và target_met=false, không giả vờ distribution thực tế giống distribution đã yêu cầu. Chỉ giữ active sites trong toán tử; lưu original hardware IDs và số qubit không dùng.

Có hàm sample truncated uniform/power-law chain lengths. Power-law là **một distribution để thử**, không phải tên distribution đã xác nhận được chốt trong audio ngày 11-09. Cần báo histogram target/achieved, occupancy, degree, boundary load, bottleneck và rejection rate.

Fixed-hardware growth tạo ra nhiều logical parents khác nhau; không tự cung cấp “nhiều embedding của cùng logical graph”. Experiment paired re-embedding cần một bước riêng giữ đúng logical graph hoặc bộ embedding maps đầu vào đã biết.

### 3.3 Logical coefficients và planting

Có weighted MaxCut, spin glass có fields, weak-field ferromagnetic stress, và weighted singly-frustrated-loop planting. Với loop dài L và trọng số w>0, mỗi loop đạt bound w(2-L) ở planted assignment. Tổng các loop chồng lấn vẫn có chứng chỉ vì cùng một assignment đạt mọi lower bound. Toàn bộ ground states vẫn được xác định bằng energy, không chỉ nhận một planted bitstring.

Không đặt quá nhiều kỳ vọng vào cycle-basis generator đơn giản: nó là reference correctness, không phải generator hard-instance hoàn chỉnh. Cần thêm loop length/density, frustration, near-degeneracy, multiple avoided-crossing stress và kiểm tra control surfaces trên subset trước khi mở rộng.

### 3.4 Ba tầng dữ liệu

| Tầng | Nội dung | Có được dùng làm inference features không? |
|---|---|---|
| Instance specification | Graphs, programmed h/J, chain membership, driver/path, runtime, scale | Có, nếu đã biết trước execution |
| Privileged physics labels | Spectrum, operator response, eigenpair residuals, censoring | Không; chỉ training/audited baselines |
| Action-response labels | Actual waveform, final success, energy, chain breaks, cost, uncertainty | Losses chỉ training/scoring; waveform có thể là candidate input |

Ground-state bitstrings, parent ID, split ID, winning-control ID, teacher residual pattern và runtime-dependent outcomes không được lén đưa vào encoder.

## 4. Những thuật toán đã có trong source

| Module | Thuật toán và ranh giới |
|---|---|
| `generation.py` | Certified loop planting; connected growth; synthetic lift; compiler/scaling; exhaustive energy/decoder validation; parent splits |
| `physics.py` | Little-endian XOR matvec; NumPy/CuPy backend; exact commuting X/XX rotations; midpoint Strang; step doubling; dense DOP853 cross-check |
| `spectral.py` | Ground-band/projector response; μ0, g_ss, D2; log-frequency bins; dark states; residual/orthogonality/unresolved metadata; incomplete sparse eigensolver |
| `physics_baselines.py` | Opt-in gap-only/D2 privileged controls; bounded allocation; explicit regularization and random interpolation checks |
| `schedules.py` | Strict monotone validation; residual-softmax durations; single/multiple windows; actual pauses; inverse-density schedule |
| `search.py` | Shared deterministic bank; Sobol duration logits; budgeted best-found search; feasible finite-difference interventions |
| `models.py` | Signed physical→chain→logical encoder; token bank; query attention; response head; multimodal policy; waveform-conditioned critic |
| `learning.py` | Train-only normalization; outcome/ranking/multimodal policy/physics losses; ambiguity masks; checkpoint with split provenance |
| `evaluation.py` | Parent bootstrap; negative regret retained; honest zero-success censoring; amortization accounting |
| `pipeline.py` / `cli.py` | Versioned NPZ+JSON; acceptance gates; resume; train; test-bank evaluation; independent true scoring of selected direct proposal |

Các thuật toán học không cần PyG. PyTorch là dependency tùy chọn; physics/tests cơ bản chạy bằng CPU. Imports không truy cập QPU, không đọc credentials, không tạo cloud jobs.

### 4.1 Matrix-free thay vì dense Hamiltonian

Lưu E_Z(x) và phép bit flip: H_X tác động lên ψ bằng ψ[x xor 2^i], H_XX bằng ψ[x xor 2^i xor 2^j]. Như vậy lưu trữ chính tỷ lệ 2^N thay vì 4^N; không có dense Kronecker product trong propagation production path.

Tuy nhiên state vẫn exponential. Một complex128 state N=28 đã là 4 GiB; 8 eigenvectors cần ít nhất 32 GiB trước Krylov workspace. GPU không làm full spectrum của instance hàng nghìn qubit trở nên khả thi. Dense teacher trong source được cap 10 physical qubits; pilot README không claim vượt cap đó.

### 4.2 Spectral labels đúng đối tượng

Tính V=∂_sH. Spectral response sử dụng |<m|V|0>|², không chỉ Δ. Với ground degeneracy, tính tổng cross-band strength theo projector và ghi rank. Không chọn eigenvector tùy tiện của degenerate ground space làm chân lý. Sparse k-eigenpairs trả truncated=true, full_response_resolved=false; không tự đổi thành full teacher.

Fixed-grid labels hiện tại không phải adaptive-grid certificate. Giai đoạn tiếp theo phải refine những khoảng có biến đổi projector/response nhanh và audit các điểm độc lập. Phase-sensitive reduced dynamics cũng chưa được triển khai.

### 4.3 Schedule và thực thi

Residual duration decoder:

\[
\Delta t_j=\frac{\Delta s_j}{v_j}+\left(T-\sum_k\frac{\Delta s_k}{v_k}\right)\operatorname{softmax}(\ell)_j.
\]

Reject T không khả thi. Pause là một đoạn có s lặp lại nhưng time tăng; không dùng “density rất cao” rồi gọi đó là pause chính xác. Phân biệt fsf slow windows với ordered BAB.

Để pilot critic đơn giản và nhãn đồng nhất, candidate bank được resample thành 9 knots trên τ-grid cố định **trước simulation**. Do đó không được dùng kết quả pilot này để tuyên bố đã đo đúng regret của nguyên bản one-window/two-window/BAB family. Frontier paper phải chạy đúng waveform/family và lưu đầy đủ knots.

### 4.4 Model đầu tiên và training

Default: width64, signed physical message passing, chain pooling, logical messages, token attention. Ba heads: auxiliary response; nhiều schedule proposals; schedule-conditioned loss critic. Direct proposal có ràng buộc slope normalized ds/dτ≤4; convert sang ds/dt theo runtime khi kiểm tra.

Train outcome/ranking trên toàn candidate bank, distill soft near-optimal set để không lấy trung bình hai mode khác nhau. Physics moments được mask khi unresolved. Có outcome-only ablation bằng response_weight=0, nhưng policy/ranking vẫn tồn tại; “outcome-only” ở đây nghĩa không dùng spectral labels, không phải chỉ một loss term.

Critic-selection trong shared bank và critic-selection trong direct proposals được báo riêng. Direct proposal được simulator score sau khi đã chọn bằng model; tuyệt đối không lấy loss của nearest neighbor trong training bank để giả là loss của proposal mới.

## 5. Lộ trình triển khai có gates

Các mốc là estimate tổ chức công việc, không phải deadline bảo đảm. Đạt gate mới tăng quy mô; không bù failure của data bằng model phức tạp hơn.

### G0 — Correctness/reproducibility (đã có nền, cần rerun trên server)

Chạy unit tests, dark-mode, two-level/Pauli checks, random compilation, disconnected embedding rejection, spectrum versus DOP853, parent leakage và resume. Ghi versions, bit convention, seed lineage, tolerances. Rerun GPU parity trước khi đổi backend.

### G1 — Data distribution và teacher validity (ưu tiên tuần 1–2)

Từ 12-parent smoke, mở rộng diagnostic subset có 6–10 physical qubits, nhiều logical sizes và chain layouts. Chạy controlled families, check endpoint equivalence, all-ground acceptance, rank changes, eigenpair convergence, accepted labels/sec. Vẽ distribution target/achieved chain lengths, coefficient scaling, bright gaps, response mass, failure/rejection reasons.

Không chỉ chạy random complete logical3 graph. `configs/pilot.json` là budgeted integration scaffold có 120 parent ×2 embedding variants ×2 κ ×3 runtimes =1,440 tasks, 64 schedules/task =92,160 outcomes; nó vẫn chỉ một logical size và không phải final benchmark. Có 480 physical paths×33 spectral points=15,840 diagonalizations. Đo 12 parents trước khi thực sự dùng config này.

### G2 — Control signal và compression (tuần 2–3)

Không cần GNN để hỏi: globally tuned schedule có thua instance-specific search không? Same-duration linear so với best-found 1-window/2-window/8-bin/richer controls ra sao? Có đủ instance mà best action thay đổi? Nếu gain nhỏ hơn numerical/statistical resolution, học policy trên distribution đó khó tạo contribution.

Chọn tolerance regret có ý nghĩa với application trước test; dùng percentile/tail chứ không chỉ mean. Nested control families phải warm-start bằng nghiệm từ family con và giữ incumbent; nếu optimizer family lớn trả tệ hơn, đó không chứng minh family lớn tệ hơn.

### G3 — Representation và causal embedding (tuần 3–5)

Matched comparisons: summary statistics → logical signed GNN → physical signed GNN → hierarchy, rồi thêm physics aux. Không đổi input, capacity, loss, teacher, control family, budget cùng một lúc.

Giữ parent, runtime, target và allowed control cố định; thay một factor mỗi lượt: geometry, physical ports, field distribution, κ, hoặc scaling. κ làm đổi α nên phải có raw-fixed và scale-controlled interventions riêng; không gọi mọi hiệu ứng đó là “chain geometry”. Test xem model dự đoán được **change of preferred control** không, không chỉ spectral MSE.

### G4 — Scale/generalization/cost (tuần 5–6)

Larger physical sizes cần validated sparse/tensor-network/trajectory teacher, không bỏ convergence checks. Hold out parent families, sizes, hardware topology, chain-length ranges. Tách cross-embedding transfer của seen parent khỏi generalization new parents. GPU benchmark đo accepted labels/s và peak memory, không chỉ matvec/s.

### G5 — Hardware/sim-to-real (phụ thuộc quyền dùng và ngân sách)

Dùng standard feasible waveform trước; kiểm tra point limit, timing quantization, coefficient scale, gauge, readout và calibration. Đo same-duration, same-read, same-search-budget; interleave methods across device batches. Freeze core rồi test residual adapter với một budget được predeclare; zero-shot và adapted báo riêng. Không dùng generic Lindblad labels để tuyên bố mô phỏng chính xác QPU.

### G6 — Paper claim audit

Nếu physical inputs không thắng logical-only, không claim embedding essential. Nếu aux không thắng outcome-only, bỏ claim necessity of spectrum. Nếu global schedule thắng learned policy, sửa distribution/claim theo evidence. A* cần insight và đối chứng này, không phải số layer hay số dòng code.

## 6. Cách trình bày kết quả cho paper

| Vị trí | Câu hỏi cần trả lời | Nội dung |
|---|---|---|
| Figure 1 | Đang học cái gì, dữ liệu đến từ đâu? | Compositional Hamiltonian + privileged train labels + no-oracle deployment |
| Figure 2 | Tại sao gap-only thiếu? | Dark/bright modes, controlled matching, loss surfaces; exact toy checks không trộn với main benchmark |
| Figure 3 | Control thấp chiều có đủ? | Control-complexity/regret/cost frontier, equal search budgets và optimization variability |
| Figure 4 | Embedding information có giá trị quyết định? | Paired counterfactual changes in embedding and preferred schedule; scale-controlled comparisons |
| Table 1 | Method có tốt trên unseen instances? | Linear, global window/bank, summary, logical, physical, outcome-only, full; paired parent CIs + failure tails |
| Table 2 | Gain có còn khi tính chi phí/transfer? | Zero-shot vs online refinement vs adapted; actual inference/search/execution/data costs |
| Appendix | Nhãn và distribution đáng tin không? | Provenance, all tests, censoring, achieved chain lengths, hardware validity, leakage, convergence, failures |

Teacher-assisted spectral baselines tính spectrum của test instance phải được ghi rõ và tính chi phí. Không dùng cùng tên với published method nếu chỉ inspired reimplementation. Tên/claim của các closest papers trong `main(2).tex` cần verified implementation/literature audit riêng trước submission; chưa có reproduction tương đương Tx-NQDT.

Main metric là terminal-loss regret, absolute success difference và cost; spectral MSE chỉ là diagnostic. Không clip negative teacher-relative regret. Bootstrap independent logical parents, không bootstrap hàng nghìn correlated reads như independent instances. Predeclare ít nhất nhiều training seeds cho full benchmark, báo mean/std qua seeds và parent CI rõ hai nguồn variation. Không hứa oral/acceptance.

Chi tiết tiếng Anh cho manuscript: `docs/paper_protocol.md`. Công thức/pseudocode: `docs/algorithms.tex`.

## 7. Việc code tiếp theo có giá trị cao nhất

1. Nạp topology/calibration thật từ file/API đã được cấp quyền; kiểm chứng embedding trên graph đó.
2. Adaptive spectral refinement + random audit + small-system dense/sparse cross-check, đủ metadata để censor labels.
3. Expand family/sizes và hardness qualification dựa trên control gain đo được, không chỉ gap quantile.
4. Tạo experiment runner factorial và seed matrix, giữ nguyên teacher/candidate budgets.
5. Batch vector propagation và Hψ trên GPU; cache theo physical path; log memory và accepted-label throughput.
6. Kiểm thử simulator/QPU adapter và train-only calibration khi có ngân sách thực.

Không đổi sang diffusion/Transformer trước khi xác định signal nào bị representation hiện tại bỏ mất. Diffusion là một hướng để kiểm chứng sau, chưa phải architecture bắt buộc từ buổi họp.

## 8. Trạng thái bàn giao

Source có các thuật toán chạy thật, tests, configs và CLI. Có end-to-end CPU smoke nhỏ để kiểm tra đường đi dữ liệu/gradients/evaluation. Các con số smoke nằm trong `reports/SMOKE_REPORT.md`; chúng không phải bảng kết quả paper và không chứng minh superiority.

Chưa có: kết quả GPU/QPU, calibrated open-system model, adaptive teacher hoàn chỉnh, benchmark scale lớn, head-to-head reproduction toàn bộ literature, full multi-seed conference experiment. Đây là những experiment quyết định claim, không phải việc chỉ cần sửa tên project hoặc tăng số epoch.
