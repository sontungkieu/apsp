# Lite2 collector — B200 qualification, 23/10/2026

Code đã có trong `apsp.lite2`; node dùng [RUNBOOK_LITE2.md](RUNBOOK_LITE2.md).
Core SHA256 `0fe3d02d408041ef446cac53e17ee5257346a07ebf6ccd052eb32c4ea7962179`. Runtime lock giữ nguyên, không rebuild hoặc tải
model về máy cá nhân. Source/runtime/model/data tách riêng.

## Kết quả thật

Modal profile billionaireproject8, một B200 sm_100, CPU6/RAM64GiB, runtime pinned.
Source cuối:40 calibration task =10 UID ×2 bank ×2 profile,320 candidate;
2,560 preview rows và320 delivered rows. Main train/dev/test chưa thu trong lượt này.

Từng task: MiniLM384 masked-mean/L2/truncation, raw IR/PickScore, feature/masks,
native dtype/value hash trước/sau serialization, final pixel re-decode và checksum
backup PASS. Export full/Core384, tar parts và restore/replay ở process khác PASS.
Replay còn chạy trên một bản copy chỉ có core/, không có PNG hoặc latent.
Backup cùng Modal Volume chỉ chứng minh checksum/round-trip, không chứng minh
independent disaster recovery hay retention của shared công ty.

20 task (mọi bank0) so native canonical với optimized trên đủ9 frames ×8 seeds:
1440 ảnh, max pixel diff=0, max IR abs delta=5.4195523262023926e-05, max PickScore abs
delta=7.62939453125e-06; validity và IR winner giữ nguyên. Không nới gate (pixel≤2,
mean≤0.01, score abs≤0.001); không so với v1 cast-fp16 để ép V2 giống đường cũ.

## Full-task timing

Mỗi profile có2 cặp crossover bổ sung: canonical→fast và fast→canonical,
warmup ngoài timer; source/seed/model giống nhau, tensors native bit-identical.
Timer gồm generation, decode, hai scorer, serialize/checksum và backup.

| Profile | Canonical mean s | Fast mean s | Mean paired speedup |
|---|---:|---:|---:|
| hyper_sd15_cfg8_512 | 5.499048 | 3.388555 | 1.622916× |
| sdxl_lightning8_1024 | 14.179701 | 9.430972 | 1.503514× |

Đây là2 paired ratios/profile, không phải fleet benchmark8 GPU hoặc 40 prompt
độc lập. Primary qualification cũng có2 paired tasks/profile, thứ tự fast trước;
chi tiết trong từng receipt. Crossover có thứ tự cân bằng là số đo ở bảng trên.
Đã chạy CLI worker riêng cho cả2 profile; resume Lightning ở process mới giữ
nguyên mọi COMPLETE hash và không tạo task mới. K=2/4/8 cost samples là timing
native batch thật; chi phí active IR tách khỏi PickScore diagnostic. Không phải
nghiệm thu online pruning/native-batch trajectory equivalence.

## Native dtype và storage

Native dtype đo thật: `{"hyper_sd15_cfg8_512": {"final": "float16", "initial": "float16", "x0_01": "float16", "x0_02": "float16", "x0_03": "float16", "x0_04": "float16", "x0_05": "float16", "x0_06": "float16", "x0_07": "float16", "x0_08": "float16"}, "sdxl_lightning8_1024": {"final": "float16", "initial": "float16", "x0_01": "float32", "x0_02": "float32", "x0_03": "float32", "x0_04": "float32", "x0_05": "float32", "x0_06": "float32", "x0_07": "float32", "x0_08": "float32"}}`.
Lightning preview fp32 không bị ép fp16; final/initial vẫn là dtype thực của
pipeline. Hyper native fp16 được giữ fp16. Torch CPU synthetic regression riêng
đã dùng sentinel fp32 sẽ mất bit nếu bị cast fp16; không coi fixture là CUDA receipt.

Forecast từ max bytes/task trên20 calibration task/profile, không dựa compression:
native main+calibration=49.745945GB,
Core/provenance=0.401713GB,
QA PNG=1.641148GB; cộng5GB dự phòng thành
**56.788805GB**, mode `core_latent_qa`.
Đây là forecast cho full4,800 main +120 calibration/QA, không phải full data đã thu.
Final export/parts kiểm tổng byte thực≤80GB; fallback Core+QA ghi latent chưa nhận.
Working storage cần raw/backup/export/parts/cache riêng, lớn hơn80GB.

Core package có planned/status/attempts/durable-ACK index, raw feature/scoring
records, embedding index, per-task model/runtime/source/feature provenance,
coverage, retention, dataset card, collection report và checksums. Node exporter
đưa thêm qualification/cost receipts. Missing/pending không thành completed;
low reward không bị bỏ. Native-latent availability tách Core availability.

## Reuse và giới hạn rollout

Reuse prompt pool v1: train2000, chọn dev200 theo quota/UID từ dev300 trước reward,
calibration50/test500 giữ disjoint groups và test locked. Native task importer
chỉ nhận artifact Lite2 tương thích đã verify; data v1 bị cast không được upcast
để giả native hoặc đổi nhãn FLUX/Base50 thành Hyper-SD15. Số839 cũ không phải
auto-deduction. Giữ toàn bộ v1 làm lineage/reserve, không sửa output đang dùng.

Launcher kiểm UUID/allocation rảnh, qualification tại node, storage forecast/disk,
canary rồi scale2→4→8 worker mới, partition freeze và reservation exclusions.
Đã unit-test từ chối busy/duplicate GPU và manifest drift; **chưa launch8 GPU đồng
thời tại công ty**. Node phải tự qualify đúng runtime/source qua launcher. Không
restart/cancel/mutate job running/queued/reserved, scheduler SQL hay global env.

## Bằng chứng và ngân sách

Local135 passed/3 skipped bằng toolchain sẵn có; Torch CPU regression remote PASS.
R1 đã pass hai profile nhưng chưa có đủ metadata planned/retry/ACK trong export;
đã bổ sung rồi R2 kiểm lại đúng source cuối, thêm crossover/worker CLI/resume.
Giữ log/source snapshot cả hai lượt, không xoá evidence.

Chi phí V2 ghi nhận **2.45938788USD**, conservative duration upper
2.7043410800000002USD, dưới khoản bổ sung5USD.
Mọi app V2 đã stopped; billing có thể trễ, không phải final invoice/remaining credit.
Volume/model/results retained. Chỉ tải receipt/metadata/source zip nhỏ về D.
Raw GPU receipts, forecast, budget và source audit: `reports/lite2-v2/`.
`env/lite2/verify_release.py` kiểm content binding offline, không cấp node GPU PASS.
