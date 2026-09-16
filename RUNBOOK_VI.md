# Vận hành annealctrl v0.2 — chỉnh cấu hình rồi chạy

## 1. Bản này làm được gì?

Luồng nghiên cứu closed-system đã nối hoàn chỉnh:

**Config → sinh Hamiltonian → kiểm tra labels → train nhiều model/seeds → chọn checkpoint trên validation → đánh giá test → xuất bảng/hình.**

Bạn không cần tự nối các module. Phần phải quyết định vẫn là distribution, budget,
hyperparameters và giả thuyết thực nghiệm; phần GPU cần môi trường CUDA thật.
QPU cần topology/calibration/samples được cấp quyền, không có job trả phí tự động.

## 2. Chạy thử đầu tiên

Giải nén source, vào thư mục `anneal_control`, tạo môi trường riêng:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plots]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m annealctrl doctor
python -m pytest -q
python -m annealctrl run --config configs/experiment_smoke.json --output runs/my_smoke
```

Lệnh cuối sinh 12 logical parents, 24 tasks và 192 action outcomes; sau đó chạy
5 methods × 2 seeds. Đây là smoke test ba epochs, không phải cấu hình chứng minh
ưu thế học máy. Dữ liệu hai sizes và chain lengths khác nhau thực sự được sinh.

Mở `runs/my_smoke/paper/RESULTS.md` để xem kết quả; `table_results.tex` có thể đưa
vào manuscript. Hình có SVG chỉnh sửa được và PNG. JSON giữ số gốc.

## 3. Bạn cần sửa file nào?

| Việc muốn đổi | File/key |
|---|---|
| Logical sizes, families, support graph | `data_research.json`: `logical_sizes`, `families`, `logical_support` |
| Chain-length distribution | `chain_distribution`: `low`, `high`, `exponent`; hoặc dùng `chain_lengths` |
| Chain geometry, ports, field allocation | `variants` |
| Chain strength/runtime | `chain_strengths`, `runtimes` |
| Số control labels/instance | `candidates`, `candidate_batch_size` |
| Teacher và sai số | `teacher`, `label_state_tolerance`, `max_steps` |
| Model/loss/optimizer | `experiment_research.json`: `methods`, `training` |
| Seeds/device/workers | `seeds`, `execution` |
| Direct-policy scoring và bootstrap | `evaluation`, `report` |

Các ví dụ hoàn chỉnh trong `configs/`; không cần sửa Python cho các lựa chọn này.
`field_distribution`: `uniform` hoặc `concentrated`; `coupling_distribution`:
`uniform` hoặc `random`. Chain-length distribution được điều kiện hóa bởi
physical-qubit budget; xem target/achieved/rejected draws trong metadata.

Research template yêu cầu **240 parents, 1,440 physical paths, 4,320 tasks,
276,480 candidate outcomes**, chưa gồm direct scoring và audit phụ. Đây là ngân
sách yêu cầu, không phải kết quả đã chạy. Hãy chạy `--dry-run` và profile trước.

## 4. Training trên server GPU

Cài PyTorch CUDA và đúng một bản CuPy phù hợp với CUDA/driver của server theo
[tài liệu CuPy](https://docs.cupy.dev/en/stable/install.html). Không cần sudo.
Nếu server dùng CUDA 11.x, không mặc định lấy lệnh cài CUDA 12.x từ máy khác.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m annealctrl doctor --require-gpu
python -m pytest -q tests/test_physics.py tests/test_adaptive_teacher.py
python -m annealctrl profile-generation --config configs/data_smoke_v2.json \
  --backends numpy cupy --output runs/first_gpu_profile
python -m annealctrl run --config configs/experiment_research.json \
  --output runs/research_v1 --dry-run
```

Sau khi kiểm tra parity/memory/budget, bỏ `--dry-run` để chạy. Nếu chỉ CPU, đổi
`execution.device` thành `cpu`, `execution.backend` thành `numpy`. GPU mode không
tự fallback. Cross-device determinism và tốc độ cần kiểm tra thực tế; xem
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

`candidate_batch_size` batch nhiều waveform trên cùng Hamiltonian.
`training.batch_size × accumulation_steps` là số graph trong một optimizer step,
thực thi graphwise; chưa phải fused batching/DDP. Worker CPU chạy các parent độc
lập; CuPy hiện yêu cầu `workers=1`. Template SLURM ở `scripts/train.slurm`;
bạn tự điền partition/account phù hợp rồi mới submit.

## 5. Resume và đổi hyperparameters

```bash
python -m annealctrl run --config configs/experiment_research.json \
  --output runs/research_v1 --resume
```

Runner kiểm tra config/source. Không sửa source trong khi run đang chạy; source
drift sẽ bị từ chối. Không trộn một thí nghiệm với cấu hình đã thay đổi.

Mỗi run có `best.pt` và `latest.pt`: best dùng inference; latest giữ optimizer,
RNG, early stopping/history. Resume lặp lại tối đa phần epoch chưa checkpoint.
Sau hard kill có thể còn lock; kiểm tra process đang chạy trước khi tự xóa
**đúng file lock đó**, không xóa dataset.

Muốn điều chỉnh model trên dataset đã sinh:

```bash
python -m annealctrl train-config --config configs/training.json \
  --data runs/research_v1/data --output runs/custom/best.pt --seed 0
```

Resume bằng `--resume-from runs/custom/best.latest.pt`. Có thể tăng tổng `epochs`,
nhưng không đổi các thiết lập khác khi exact resume. Đổi loss/lr/architecture thì
tạo run mới. Chi phí resumed run được đánh dấu có thể thiếu phần trước hard kill.

## 6. Tuning chỉ dùng validation

```bash
python -m annealctrl tune --config configs/tuning_smoke.json --output runs/tune_v1
```

Grid có `model.KEY` hoặc `training.KEY`, giới hạn bởi `max_trials`. Dữ liệu sinh
một lần rồi tái dùng. Chọn trial bằng mean validation bank-regret qua seeds,
không gọi test evaluator. `selected_experiment.json` là cấu hình được chọn.

Sau khi chốt tuning, dùng cấu hình được chọn chạy `--stage evaluate --resume`
trên output trial tương ứng, rồi `--stage report --resume`. Không tiếp tục hiệu
chỉnh dựa trên test. Audit trước training chỉ đọc và xuất training outcomes.

## 7. Baseline và paper outputs

Các ablation đã khai báo:

- `summary`: statistics, không message passing.
- `logical`: logical coefficients/graph và runtime; không chain/physical scale.
- `physical`: signed physical graph.
- `hierarchy_outcome`: hierarchy nhưng không auxiliary spectral loss.
- `hierarchy_physics`: hierarchy và physics auxiliary loss.

Tất cả có critic và policy heads. “Outcome-only” nghĩa không spectral labels;
vẫn có outcome/ranking/policy losses. Equal width không đồng nghĩa equal parameter
count; checkpoint lưu count để thiết kế capacity-matched comparisons.

Linear và global schedule chọn trên validation là baselines. Best-bank là
privileged reference, không phải deployment method. Direct proposal được chọn
bằng critic trước, sau đó mới simulator score; không nội suy outcome từ bank.

Muốn đo control-family frontier đúng waveform:

```bash
python -m annealctrl control-benchmark --data runs/my_smoke/data \
  --split validation --record-id RECORD_ID --budget 32 --output runs/frontier.json
```

Lấy `RECORD_ID` từ manifest. Search giữ knots thật của window/pause, tính budget
và incumbent. Test-time search cần `--allow-test-adaptation` và phải được báo là
online adaptation. Không gọi best-found là global control optimum.

Report dùng mean theo logical parent, tách parent CI với seed SD; thiếu bất kỳ
method/seed đã khai báo thì không xuất bảng chính. Hình phản ánh số đo, không tự
biến result thành kết luận có ý nghĩa thống kê hoặc ưu thế paper A*.

## 8. Inference cho instance mới

```bash
python -m annealctrl infer --record INSTANCE.npz \
  --checkpoint runs/custom/best.pt --output runs/predicted_schedule.json
```

NPZ cần instance specification đúng schema, không cần candidate outcomes.
`graph_from_record` mô tả inputs. Output có waveform, critic predictions và
runtime; **chưa quan sát true loss**, không tự gửi QPU.

## 9. Hardware/open-system: đã có đường code riêng

- `data_hardware_toy.json`: connected growth tích hợp vào training records.
  Thay graph bằng topology đã xác minh, không đổi tên toy graph thành hardware thật.
- `simulate-open`: Lindblad density matrix độc lập, local dephasing/relaxation,
  kiểm tra trace/Hermiticity/positivity. Không phải bath model đã fit với QPU.
- `export-hardware`: topology/coefficient/time/slope/point constraints,
  gauge/decoder và payload offline có hash. Không submit job.
- `ingest-hardware`: samples bạn cung cấp; kiểm tra hash/energy/order,
  tính decoded success, chain breaks và binomial interval.

JSON/API đầy đủ ở `docs/adapters.md`. `hardware_export_toy.json` chỉ dành cho
instance ba physical qubits và thiết bị giả; sửa mapping/constraints theo thiết
bị thật. Schedule dimensionless không tự được coi là microseconds; calibration
dùng `us` cần runtime vật lý riêng trong config.

## 10. Giới hạn khi diễn giải

Full spectral teacher cap 10 physical qubits. `teacher.mode=none` cho outcome-only
cap 20 với explicit memory/endpoint budgets trên10. State/acceptance vẫn
exponential, không phải khả năng xử lý hàng nghìn qubits.

Adaptive teacher có independent audits/censoring; hết budget hoặc audit fail
được ghi rõ, không phải uniform certificate. Individually exact labels không
chứng minh curve giữa chúng đã resolved.

ML input hiện canonical X/Z, không catalyst khác0/global energy_scale khác1.
Physics APIs rộng hơn không đồng nghĩa model đã hỗ trợ. Lindblad/calibration
utilities chưa thay closed-system training contract. Không có CUDA-Q/QuTiP adapter,
tensor-network large-scale teacher, DDP hay tự động QPU submission.

Trong scope hiện tại, các bước nối pipeline đã có code. Công việc còn lại là
cấu hình, chạy thực nghiệm, đo trên server và kiểm tra giả thuyết bằng evidence thật.
