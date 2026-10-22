# Lite2: lệnh chạy thật trên 8 B200

Namespace `apsp.lite2`, dataset `apsp-lite2-fast2`, Core schema `apsp-core-lite2-v1`.
Spec: [plan nguyên bản](PLAN_DATA_CONTROLLER_LITE_V2_80GB.md). Collector, MiniLM384,
native storage, manifests, replay accessor, export/parts/restore và launcher đã có
trong source; agent công ty **không phải viết collector V2 nữa**.

## 1. Source và môi trường

Lấy đúng Git/HF pins trong tin nhắn bàn giao mới nhất, bootstrap vào checkout mới
theo §4 [handoff](COMPANY_LITE2_80GB_HANDOFF_20261023.md). Không update checkout,
symlink, env, source, queue hoặc claim của job cũ, kể cả job queued/reserved.
`env/lite/uv.lock` giữ nguyên; không pip install/rebuild runtime. Dùng runtime và
companion đã restore, thêm model add-on theo [LITE2_MODEL_ADDON](LITE2_MODEL_ADDON.md)
nếu chưa có. Cache view riêng dùng chung cho 8 worker, không tải 8 bản model.

Activate trong shell mới theo thứ tự: companion → add-on → đặt lại source:

```bash
(
set -euo pipefail
: "${APSP_SOURCE:?dat checkout moi da verify}"
SOURCE_NEW="$APSP_SOURCE"
ROOT=/workspace/storage-shared/nlp/tungks/apsp_all
source "$ROOT/apsp-lite-b200-cu128/companion-v1/activate_companion.sh"
source "$ROOT/lite2-model-addon-v1/activate_lite2_models.sh"
export APSP_SOURCE="$SOURCE_NEW" PYTHONPATH="$SOURCE_NEW"
export APSP_PYTHON=/opt/apsp-runtime/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$APSP_PYTHON" "$APSP_SOURCE/env/lite2/verify_release.py"
)
```

Expected `LITE2_RELEASE_BINDING_PASS`: source/lock/receipt khớp, **chưa phải node
GPU PASS**. Các biến trong subshell không giữ ngoài block: launcher riêng phải
activate và export lại các giá trị đúng, không dựa vào state shell cũ.

V1 `verify_performance_release.py` là binding source v1 lịch sử; không dùng nó
để nghiệm thu source V2 vì fingerprint bao gồm mọi `apsp/*.py`. Không viết lại
receipt v1 để giả đã qualify source mới. Node mới qualify bằng `apsp.lite2.qualify`.

## 2. Chuẩn bị inventory và task reservations

Agent re-query hostname, GPU UUID/name/compute apps và scheduler allocation.
Chỉ dùng đúng 8 B200 được giao và rảnh. Tên node/UUID trong handoff là snapshot,
không phải trạng thái live. Launcher từ chối GPU đang bận; không tự kill/restart.

Từ inventory scheduler hiện tại, xuất **frozen JSONL** các task running/queued/
reserved cũ. Mỗi dòng có `profile` (tên v1 cũng được), `prompt_uid` hoặc
`prompt.prompt_uid`, `bank_id`. Nếu không có task liên quan, tạo file rỗng sau
khi kiểm live, không bỏ flag để vô tình sinh trùng. Không sửa SQL/claims của queue.
Snapshot có thể cập nhật giữa các attempt đã dừng; không đổi partition của epoch
đang chạy. V1 Lightning alias chỉ dùng để loại reservation, không tự chứng minh
native compatibility. Các task pending do reservation vẫn hiện trong coverage.

Input text dùng lại prepared manifests v1 trên shared (chỉ text/metadata):
`.../companion-v1/data/lite-v1/prepared/manifests/`, có train2000/dev300/calibration50/
test_id300/test_shift200. CLI chọn dev200 theo quotas/UID trước reward, kiểm group
disjoint, giữ train rank và nguồn provenance. Không dùng tên file task manifest
v1 thay input prompt directory này. Thiếu quota/group/source metadata phải báo,
không tự bổ sung prompt giả. Test500 chỉ lưu manifest, không sinh/chấm.

## 3. Một entry point để qualify, canary, collect và export

Các biến dưới đây do agent/operator chốt từ inventory; launcher từ chối biến
thiếu. `OUTPUT_NEW` và `BACKUP_NEW` là root mới, tách cây và ngoài v1. Xác nhận
storage retention/failure domain thực tế; disk check không tự chứng minh backup
độc lập. Log/manager/job đều thuộc lượt Lite2 mới, không sửa manager toàn cục.

Chạy qua scheduler/job manager hiện hữu hoặc trong tmux/nohup của job mới để không
phụ thuộc chat. Payload dưới chạy foreground khi được manager/tmux giữ; không
chạy lặp để tạo hai manager. Lock riêng bảo vệ một manager/root.

```bash
(
set -euo pipefail
: "${APSP_SOURCE:?}" "${APSP_PYTHON:?}" "${HF_HOME:?}"
: "${V1_PROMPT_MANIFESTS:?}" "${OUTPUT_NEW:?}" "${BACKUP_NEW:?}"
: "${GPU_UUIDS:?8 UUID cach nhau boi dau phay}" "${OLD_RESERVATIONS:?file JSONL da audit}"
: "${OLD_RELEASE_ROOT:?release v1 chi doc}"
bash "$APSP_SOURCE/env/lite2/run-node.sh" \
  --v1-manifests "$V1_PROMPT_MANIFESTS" \
  --out "$OUTPUT_NEW" --backup-root "$BACKUP_NEW" \
  --gpu-uuids "$GPU_UUIDS" --old-reservations "$OLD_RESERVATIONS" \
  --old-root "$OLD_RELEASE_ROOT" --hyper-workers 2
)
```

Trình tự thực tế:

1. Kiểm source receipt offline; inventory 8 B200 và reservation snapshot.
2. Freeze manifests: main4,800; calibration/QA union120; qualification40;
   test500 locked. Nếu input thay đổi, từ chối resume namespace cũ.
3. Qualification tại node trên một GPU: 20 task/profile × 8 seed, native dtype,
   MiniLM384, preview/final, scorer masks, parity trên 10 bank0 task/profile,
   full-task timings, native-batch cost K=2/4/8, export/parts/restore process khác
   và replay từ bản Core tách khỏi PNG/latent. Receipt node gắn source/runtime/
   registry/policy; không mượn receipt Modal thay node gate.
4. Forecast từ **max bytes thực từng loại file** của calibration, cộng main2,400
   task/profile và calibration60/profile, 800/480 QA PNG, reserve5GB. Nếu native
   không vừa cap, chọn mode `core_qa`, giữ raw latent ở source và ghi chưa nhận.
   Working-disk gate tính raw + backup + export + parts; không giả node chỉ cần80GB.
5. Audit v1 COMPLETE/hash để tái dùng record thực sự tương thích. Dùng lại text,
   UID/split/seed pool hợp lệ. V1 fp16 đã mất native fp32 không được upcast để nhận
   full V2; giữ nguyên như reserve. Reuse report chỉ nhận `accepted_full` từ dữ
   liệu Lite2 native đã validate. Vì vậy số839 cũ không mặc định trừ khỏi2,000.
6. Canary tối đa10 task/profile; kiểm valid/native trước mở rộng. Chạy thêm wave4
   nhỏ rồi tới 8 worker (2 Hyper +6 Lightning), fixed partitions trong mỗi wave.
   Mỗi wave tự drain trước đổi partition. Model/scorers giữ resident mỗi worker.
7. Phase1 train1k+dev200 rồi phase2 train2k, chỉ task thiếu và không bị old
   reservation giữ. Không cần duyệt lại từng phase. Cuối cùng thu phần QA120 còn
   thiếu, reuse40 task qualification nên không sinh trùng calibration.
8. Export Core/latent/QA có checksum, provenance/cost receipts, dataset card,
   collection report và bảng planned/collected/failure/backup ACK. Pack thành tar
   parts1GB mặc định. `--no-pack` chỉ bỏ đóng parts, vẫn export/verify.

Các phép GPU/restore fail hoặc mean paired speedup dưới1.10×: không mở optimized
bulk, giữ logs và báo blocker; không nới gate hoặc đổi recipe. Handoff bổ sung
[B200_LITE2_QUALIFICATION.md](B200_LITE2_QUALIFICATION.md) ghi số đo source phát hành.
Đây không phải benchmark đồng thời 8 GPU trên node công ty.

## 4. Lệnh tách bước đã có

Để debug/tiếp tục có kiểm soát, các CLI sau tồn tại thật. Luôn đặt PYTHONPATH,
HF_HOME/offline và CUDA_VISIBLE_DEVICES theo GPU được giao trong shell riêng.

```text
python -m apsp.lite2.manifests --v1-manifests INPUT_DIR --out NEW_MANIFEST_DIR
python -m apsp.lite2.qualify --tasks NEW_MANIFEST_DIR/qualification.jsonl --out NEW_CHECK_DIR
python -m apsp.lite2.reuse --old-root OLD_RELEASE --tasks NEW_MANIFEST_DIR/main.jsonl --out NEW_REUSE.json
python -m apsp.lite2.worker --init --tasks TASKS.jsonl --out NEW_RELEASE
python -m apsp.lite2.worker --tasks TASKS.jsonl --out NEW_RELEASE --backup-root NEW_BACKUP --profile hyper_sd15_cfg8_512 --qualification NODE_CHECK/hyper_sd15_cfg8_512.json --worker-index 0 --workers 2 --blocked-tasks OLD_RESERVATIONS.jsonl --reuse-audit NEW_REUSE.json
python -m apsp.lite2.worker --tasks TASKS.jsonl --out NEW_RELEASE --backup-root NEW_BACKUP --profile sdxl_lightning8_1024 --qualification NODE_CHECK/sdxl_lightning8_1024.json --worker-index 0 --workers 6 --blocked-tasks OLD_RESERVATIONS.jsonl --reuse-audit NEW_REUSE.json
```

Text block là cú pháp CLI, không copy nguyên placeholder để launch. Default
`--phase 2` xử lý phase1 trước phase2 theo manifest; node manager thêm barrier QA
giữa phase. `--max-tasks` dùng cho canary. Một process/UUID; không chạy cả hai
profile trên cùng GPU. Worker0..N-1 và N phải cố định trong manifest epoch.

## 5. Export/restore riêng, không cần model hay GPU

```text
python -m apsp.lite2.export --root MAIN_ROOT --root CALIBRATION_ROOT --reuse-audit MAIN_REUSE.json --reuse-audit CALIBRATION_REUSE.json --planned-tasks MANIFESTS/main.jsonl --planned-tasks MANIFESTS/calibration.jsonl --failures-root MAIN_ROOT --failures-root CALIBRATION_ROOT --qualification-root NODE_CHECK --include-calibration --expected-tasks 4800 --mode core_latent_qa --out NEW_EXPORT
python -m apsp.lite2.replay --core NEW_EXPORT/core
python -m apsp.lite2.pack pack --release NEW_EXPORT --out NEW_PARTS --part-bytes 1000000000
python -m apsp.lite2.pack restore --parts NEW_PARTS --out NEW_RESTORED
python -m apsp.lite2.replay --core NEW_RESTORED/core
```

Mode40GB: `--mode core_qa`. Metadata ghi latent chưa nhận; Core vẫn đủ chạy ba
baseline approximate fixed/threshold/random-masked. Accessor không đưa score
chưa VERIFY, inactive scorer, future/terminal hay pruned candidate vào actor view.
Không biến replay PASS thành online equivalence hoặc controller thắng.

Core exported có `task_plan.parquet` cho planned/status/attempts/ACK và
`tasks.parquet` cho completed records. `coverage.json` phân profile/split/status;
pending/failed_infra không được đếm vào collected. `retention.json` ghi source URI,
native export status và expiry chưa xác nhận. Default snapshot/backup cùng volume
trên Modal chỉ kiểm hash/restore, không phải bằng chứng independent durability.

Không có PNG/latent là điều kiện bắt buộc để load Core; GPU validation tùy chọn
mới mở native tensor/re-decode. `RELEASE_MANIFEST.json` và `SHA256SUMS` bao toàn
gói; parts restore chặn path traversal/symlink/duplicate/tamper, không ghép thêm
một tar lớn trên đĩa. Tổng bytes thực cả overhead bị chặn ở80GB. Nếu vượt cap,
giữ staging và chọn mode rõ ràng, không cast hoặc xóa bằng chứng.

## 6. Báo kết quả và resume

`OUTPUT_NEW/delivery.json`: actual main/calibration counts, coverage, package mode
và trạng thái `RELEASE_VERIFIED`, `CORE_READY_LATENT_PENDING` hoặc partial. Source
node manager/jobs/logs và per-task ACK/failures còn trong manager-runs/, workers/,
acks/, failures/; báo path và timestamp, không chỉ báo exit0/running.

Nếu old reservations chưa hết, release vẫn partial. Agent audit snapshot mới
sau khi các job Lite2 hiện tại đã dừng, rồi chạy lại cùng manifest/source/roots:
COMPLETE đúng contract được skip, backup được verify, không đổi seed hoặc sinh
lại task đã xong. Không steal stale claim tự động; kiểm process/queue thật trước
khi xử lý claim của **job Lite2 mới thuộc đợt này**. Không sửa claim/job cũ.
Input manifest, registry, source hoặc execution policy đổi phải namespace/gate
mới; không reset output để vượt mismatch. Export lần resume có thư mục mới,
không ghi đè gói đã bàn giao.
