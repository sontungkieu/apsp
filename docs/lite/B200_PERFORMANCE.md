# B200 collector throughput — 22/10/2026

Đã tối ưu theo policy opt-in; không đổi default legacy, model/scorer weights, seeds,
step count, runtime dependency lock hoặc bank v1 đang chạy. Runtime không rebuild.
Core source SHA256: `75c1109bbbb8ef95179d8d039b526723accca3284ddf590b113b1b3e737b2019`.

Bàn giao cập nhật HF, kiểm B200 và rollout cho job mới:
[COMPANY_B200_OPTIMIZATION_HANDOFF_20261022.md](COMPANY_B200_OPTIMIZATION_HANDOFF_20261022.md).

## Kết quả đo thật

Modal `billionaireproject8`, một NVIDIA B200 sm_100,6 CPU cores,64GiB RAM.
Thời gian task gồm generation, decode, hai scorer, ghi file/checksum và backup copy.
Model load/warmup và validator riêng nằm ngoài timer task. Mỗi profile có20 task
calibration thật:10 prompt từ manifest calibration cũ ×2 bank ×8 seed. Không sinh
train/dev/test500 và không train controller trong lượt tối ưu.

| Profile | Legacy task, mean s | Fast task, mean s | Mean paired full-task speedup | Median decode+score speedup | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| hyper_sd15_cfg8_512 | 5.146850 | 2.738815 | 1.879254× | 1.938815× | 16.84 | 31.52 |
| sdxl_lightning8 | 17.765904 | 9.984933 | 1.779254× | 1.775801× | 18.26 | 56.19 |

Full-task speedup là mean của2 paired ratios/profile trên2 calibration prompt;
hai cột giây là mean latency của các pair đó. Decode+score median tính trên các
task có legacy reference; mọi bank0 đã được chọn để so trước khi xem reward.
Bank1 được kiểm schema/finite/re-decode; vài bank1 có reference từ phiên trước
được giữ nguyên. Xem `reference_checked_tasks` trong raw receipt, không coi tất cả
40 task là40 prompt độc lập. Speedup chưa phải phép đo trên fleet công ty.

So baseline trên 1584 preview/final images: max pixel diff=0,
max IR abs delta=4.780292510986328e-05, max raw PickScore abs delta=7.62939453125e-06.
Mọi20 task/profile có8 candidate và64 preview rows, valid và final re-decode PASS.
Các full-task pairs có stored trajectories giống nhau. Đây không phải QA thẩm mỹ,
chứng minh controller thắng hoặc online pruning/native-batching equivalence.

## Thay đổi được chọn

- Scorer batch8 với row mapping đầy đủ; lỗi latent/decode/scorer vẫn giữ đúng seed.
- VAE giữ shape batch1; decode được chạy trên CUDA streams độc lập, tái dùng stream
  pool giữa các task để allocator không tích lũy pool mới theo mỗi lần gọi.
- Gom checkpoint để che bớt khoảng nghỉ giữa decode và scorer, giữ thứ tự j/seed.
- Giữ clean predictions trên GPU trong denoising rồi serialize một lần; không
  thêm denoiser forward, không đổi seed và không cast tensor theo tối ưu mới.
- PNG compress_level1 vẫn lossless; file có thể lớn hơn. Tỷ lệ size so PNG legacy
  trong các pair: {"hyper_sd15_cfg8_512": 1.060744, "sdxl_lightning8": 1.081082}.
- Cache common prompt embedding theo encoder repo/revision/text, tối đa4096 entry;
  không đổi encoder768 của collector v1 hoặc giả MiniLM384 đã tích hợp V2.

## Cấu hình đã loại

ROUND1 thử VAE batch2/4/8: nhanh hơn nhưng thay pixel và làm score vượt gate.
ROUND2 thử scorer batch32: pixel giữ nguyên khi VAE batch1/streams, nhưng score
vẫn vượt gate. Không nới gate sau khi thấy kết quả, không chọn các cấu hình này.
Gate cố định: pixel max≤2, pixel mean≤0.01, IR/PickScore max abs≤0.001 và IR
winner không đổi trên reference subset. Kết quả được chọn thực tế có pixel diff0.

## Chạy trên node

Source mới ở GitHub main và HF source bundle; cập nhật checkout idle theo
`SOURCE_DISTRIBUTION.md`. Giữ job v1 và dữ liệu niêm phong nguyên trạng. Tạo release
root/backup mới; source hoặc execution policy đổi sẽ làm qualification/resume hash
đổi, không ghi nối vào `release-v1`.

Kiểm source/receipt offline sau khi đặt PYTHONPATH về checkout mới:

```bash
python3 env/lite/verify_performance_release.py
```

Opt-in của CLI collector hiện có:

```text
--execution env/lite/performance/sdxl_lightning8.json
--qualification reports/performance/sdxl_lightning8-qualification.json
```

Hai flag này ghép với task manifest/profile/output/backup đã được operator phê
duyệt, dùng GPU được giao và root mới. Không có launcher tự chọn node/GPU hoặc
restart job. Trước bulk kiểm runtime fingerprint và smoke/re-decode trên node
thực: Modal receipt không thay company-node qualification. Thiếu khớp phải dừng.

Hyper-SD15 có execution JSON và20-task receipt riêng cho adapter, nhưng chưa được
thêm vào TRAIN_PROFILES v1. Tích hợp nó vào collector/schema namespace Lite2 vẫn
là bước riêng; không đổi tên `hyper_flux8` hoặc coi native-dtype/Core384 đã DONE.

`reports/performance/ROUND1.json`, `ROUND2.json` là exploratory tuning receipts;
core hash có thể khác final. `CONFIRMATION.json`, từng profile qualification và
`RELEASE.json` là binding cho source hiện tại. Legacy `docs/lite/RELEASE_MANIFEST.json`
vẫn mô tả source/runtime v1 gốc, không bị viết lại để giả nó qualify code mới.

## Verification và ngân sách

Local targeted suite121 passed/3 skipped; Torch CPU batch/mask/row-count regression
và snapshot preflight PASS trên runtime remote. Không cài Torch/dependency local.
Hai round exploratory và final calibration chạy B200 thật, dữ liệu trên Modal
Volume, chỉ tải metadata/receipt/source snapshot nhỏ về D. Backup ở benchmark là
copy/checksum cùng Volume; không gọi nó là independent disaster-recovery backup.

Một attempt dừng trước model vì ZIP timestamp image<1980; đã fix và đưa check về
CPU preflight. Hai confirmation attempt được dừng để tái dùng stream pool và bỏ
fast decode lặp trong harness; task đã verified được resume với cùng source hash.
Không xóa bằng chứng hoặc đổi numerical gate để làm xanh test.
Chi phí snapshot và trạng thái app nằm trong `BUDGET.json`, có thể trễ billing.
Trần toàn task7USD giữ nguyên. Giá dùng để bound thời gian: https://modal.com/pricing.
