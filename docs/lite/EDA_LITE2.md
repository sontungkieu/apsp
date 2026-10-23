# EDA Lite2 trên dữ liệu công ty

Notebook: [`APSP_LITE2_EDA_COMPANY.ipynb`](../../notebooks/APSP_LITE2_EDA_COMPANY.ipynb).
Helper CPU: [`env/lite2/eda_v2.py`](../../env/lite2/eda_v2.py).
Không sửa `apsp/`, runtime lock hay receipt collector. Không dùng GPU, tải model,
chạy worker, sửa queue hoặc update checkout đang có job dùng.

## Input đã xác nhận từ operator

```text
RUN=/workspace/storage-shared/nlp/tungks/apsp_all/apsp-lite2-fast2/lite2-r1
EXPORT=$RUN/release/exports/1792706875990190337
COLLECTION=$RUN/release
MANIFESTS=$RUN/release/manifests
BACKUP=$RUN/backup
```

Theo `delivery.json` operator gửi: main4800, calibration120, mode
`core_latent_qa`, `coverage=complete`, `status=RELEASE_VERIFIED`. Notebook phải
đọc lại và kiểm tại máy công ty; đây chưa phải số đo EDA. Các export trong
`release/qualification/` là tập nhỏ, không chọn chúng làm export chính.

## Chạy notebook

Tải source EDA vào **checkout mới** theo pins của lần bàn giao, hoặc chỉ tải notebook
và helper vào một thư mục riêng giữ layout `notebooks/`, `env/lite2/`. Không cần
update source collector, không rebuild runtime. Source bundle mới vẫn giữ nguyên
`apsp/` đã qualify, nhưng EDA không cần import module collector.

Chọn kernel có sẵn của `/opt/apsp-runtime/bin/python` trong Jupyter/VS Code công
ty nếu đã có kernel đó. Không tự cài Jupyter, ipykernel, matplotlib vào runtime.
Thiếu giao diện/kernel thì dùng lệnh CPU headless dưới đây: cùng helper và kết quả
phân tích. Notebook không bắt buộc GPU hay một trong8B200 rảnh.

```bash
export APSP_EDA_SOURCE=/duong/dan/checkout-eda-moi
# Notebook đã điền đúng các input trên; chỉ đặt source và mở notebook rồi Run All.
```

Mỗi run ghi vào thư mục mới `.../apsp_all/eda-v2/<UTC>/` ngoài collection/export/
backup, không ghi đè. Muốn đổi input/output dùng biến `APSP_EDA_EXPORT`,
`APSP_EDA_COLLECTION`, `APSP_EDA_MANIFESTS`, `APSP_EDA_BACKUP`, `APSP_EDA_OUT`.

## Chạy cùng phân tích không cần Jupyter

Lệnh đã điền input, default full export hash và native4task/profile:

```bash
bash "$APSP_EDA_SOURCE/env/lite2/run-eda.sh"
```

Hoặc gọi Python trực tiếp để thấy toàn bộ tham số:

```bash
(
set -euo pipefail
: "${APSP_EDA_SOURCE:?checkout EDA moi}"
RUN=/workspace/storage-shared/nlp/tungks/apsp_all/apsp-lite2-fast2/lite2-r1
OUT=/workspace/storage-shared/nlp/tungks/apsp_all/eda-v2/$(date -u +%Y%m%dT%H%M%SZ)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p /workspace/storage-shared/nlp/tungks/apsp_all/eda-v2
/opt/apsp-runtime/bin/python "$APSP_EDA_SOURCE/env/lite2/eda_v2.py" \
  --export "$RUN/release/exports/1792706875990190337" \
  --collection "$RUN/release" --manifests "$RUN/release/manifests" \
  --backup "$RUN/backup" --out "$OUT" --full-hash --native-values 4 \
  >"${OUT}.log" 2>&1
echo "EDA_REPORT=$OUT/report.html"
)
```

Chạy foreground trong tmux/job CPU được công ty giao nếu cần giữ phiên lâu;
không lấy GPU hoặc thay đổi job manager. Exit2 nghĩa audit đã xong nhưng còn FAIL;
exit1 thường là lỗi input/code/integrity, xem log và `checks.csv`.

Để giảm I/O, bỏ `--full-hash` (notebook: `FULL_HASH=False`), đặt
`--native-values 0`. Báo cáo ghi rõ Core/metadata hash, native chỉ header,
extension chưa hash. Muốn kiểm toàn bộ backup bytes thêm `--backup-full-hash`;
không tự bật vì sẽ đọc thêm toàn bộ payload backup. Hash stream không cần đưa
toàn bộ50–80GB vào RAM. Mọi native header đều đọc; native value chỉ mẫu4task/profile.
Mọi PNG đã giữ đều CPU decode/so pixelhash; gallery chỉ thumbnail tối đa48ảnh.

## 22 phần phân tích

1. Config, identity, dependency check và phạm vi I/O.
2. File-set/checksum export, manifest và SHA256SUMS.
3. Core cardinality, exact seed/checkpoint keys, foreign keys, embedding384.
4. Coverage profile/split/bank/source; canonical4800 + calibration120.
5. Frozen manifests, full task equality, raw COMPLETE, ACK và backup marker.
6. UID/text/group split leakage; external test-locked manifest-only audit.
7. Validity/finiteness/failure/missingness, giữ mẫu số.
8. Reward distribution/tails theo profile/split/source.
9. Seed diversity, duplicate pixel, seed-index bias.
10. Exact expected best-of-k trên mọi tổ hợp seed, không prefix sampling.
11. Preview trajectory, within-task ranks, increments/plateau.
12. Pruning oracle survival/regret ở j và k; terminal chỉ evaluator.
13. Paired bank/profile contrasts, CI bootstrap theo source group.
14. IR/PickScore winner conflict và feature/final associations.
15. Legal masked offline baseline schedules, logical operation counts.
16. Token truncation, Core384 PCA và exact cross-split train neighbors.
17. Native header/value/provenance/RNG recipe audit.
18. Timings theo profile, qualification costs K2/4/8.
19. Bytes/component/cap/retention, ACK limitations.
20. Biểu đồ Matplotlib nếu sẵn có; bảng/HTML vẫn chạy khi thiếu.
21. Full saved-QA pixel audit, low/mid/high gallery và manual-review CSV.
22. Report, Material Passport, phạm vi thống kê, artifact checksums.

Notebook đánh số0..21; các phần trên gom cùng nội dung để dễ tìm.

## Đọc kết quả

Mở `report.html` và `qa_gallery.html`, bắt đầu từ `summary.json`, `checks.csv`,
`coverage.csv`, `canonical_plan_coverage.csv`, `live_source_backup.csv`.
Hash/cardinality lỗi sẽ dừng trước thống kê. Các lỗi coverage/feature/provenance
được giữ trong checks, không che bằng thống kê còn lại. Native dtype recipe khác
receipt đo ban đầu là FAIL để điều tra, không tự ép dtype hay sửa data.

40 calibration task tái dùng qualification có thể nằm ở
`release/qualification/<profile>/fast/tasks/<id>`, backup riêng
`release/qualification/<profile>/backup/<id>`, không có worker ACK. Audit chỉ theo
đúng layout này trong collection được chỉ định, ghi riêng loại source/backup và
`NOT_EMITTED_QUALIFICATION_REUSE`; không giả ACK và không báo thiếu task nhầm.

Các CSV chính cho phân tích controller: `seed_budget_exact`, `trajectory_ranking`,
`pruning_diagnostics`, `paired_cluster_intervals`, `scorer_agreement`,
`feature_rank_associations`, `offline_policy_diagnostics`, `qualification_cost_samples`.
Giữ split/source khi so sánh. Nhiều seed/bank cùng prompt không độc lập; CI dùng
group-level paired differences, không seed bootstrap. Best-of-k/pruning dùng
outcome privileged chỉ làm headroom; không biến thành claim controller thắng.

`valid=true` không phải QA ảnh đẹp. Pixel khác không phải semantic diversity.
Gallery chỉ saved calibration subset và chọn low/mid/high nên không estimate
population quality. CI exploratory không có p-value hoặc tuyên bố significance.
Logical forward/verify units không phải wall-time saving hay USD. Hash/ACK
không chứng minh backup độc lập. Notebook không đọc score/generate test.

Gửi lại thư mục báo cáo (hoặc trước mắt các CSV/JSON chính và log), chưa cần
chuyển latent/model/PNG gốc. Report có prompt text và đường dẫn nội bộ: chỉ chuyển
trong phạm vi công ty/người dùng cho phép; code không tự upload.

## Kiểm chứng trước bàn giao

`tests/lite2/test_eda.py` chạy mọi code cell notebook trên export V2 tổng hợp và
kiểm lỗi hash, checkpoint duplicate, output isolation, native-byte hash/nonfinite,
backup payload tamper, exact best-of-k và workload counters. Đây là kiểm notebook,
không phải kết quả chạy data4920task trên công ty. Regenerate notebook bằng
`python env/lite2/build_eda_notebook.py` (stdlib); source notebook không chứa output.
