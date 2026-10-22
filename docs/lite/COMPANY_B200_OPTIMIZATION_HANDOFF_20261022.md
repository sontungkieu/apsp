# Handoff công ty: HF source → B200 gate → tối ưu job mới

Ngày 22/10/2026, giờ Việt Nam. Người nhận: agent đang vận hành APSP ở công ty.
Yêu cầu của Tung: cập nhật source từ HF, kiểm tra trên B200 rồi bật tối ưu cho
job mới; giữ nguyên mọi job đang chạy. Đây là hướng dẫn thực thi cho agent công
ty, chưa phải receipt đã triển khai trên node.

## 1. Phạm vi thực hiện

1. Đọc trạng thái queue/process và allocation hiện tại; ghi node, GPU UUID, job
   ID, source/output root của các job đang chạy. Chỉ dùng B200 được giao và đang
   rảnh; nếu chưa có thì chuẩn bị source/CPU checks rồi chờ slot. Không giành GPU.
2. Clone HF bundle vào **checkout mới dành riêng cho release này**, kể cả khi
   checkout chung đang clean. Không pull/update checkout đang được process dùng;
   không sửa symlink source chung, env chung, runtime, cache model hoặc launcher
   đang được job chạy/queue cũ tham chiếu. Không restart/cancel/migrate các job đó.
3. Chỉ bật policy đã kiểm cho **job Lightning8 mới tạo sau gate PASS**. Job đã
   chạy hoặc đã queue với source/config cũ giữ nguyên; tạo job ID/spec mới riêng.
4. Không ghi nối vào `release-v1`, `backup-v1`, task staging/claims cũ. Root output,
   backup, qualification, logs của đợt mới phải riêng; giữ task ID/seed/profile và
   liên kết provenance. Root mới không làm task ID thành dữ liệu khoa học mới.
5. Không rebuild runtime, pip install, nâng Torch hay đổi model/scorer weights.
   Nếu lock/ABI/model không khớp thì dừng phần GPU, báo đúng phần thiếu.

## 2. Identity đã chốt

| Thành phần | Giá trị |
|---|---|
| GitHub canonical | `https://github.com/sontungkieu/apsp`, branch `main` |
| Code commit đã đo | `d0436c3d5c24e5bbd29d1156bd5862faa80e4126` |
| HF dataset | `codemaivanngu/apsp-lite-b200-cu128` |
| HF source revision ghim | `1de26708f53b5307db73b69e6ca696c2c3dffac5` |
| Bundle | `source/apsp.bundle`, 1,009,196 bytes |
| Bundle SHA256 | `7b8e033d391c002cfff7c9a3eae8e7a624581c42f10d0a84c257e0d35ef7fb90` |
| Core source SHA256 | `75c1109bbbb8ef95179d8d039b526723accca3284ddf590b113b1b3e737b2019` |
| `env/lite/uv.lock` SHA256 | `e404558d31cc6befaa79aa693b99650eee3d269db3d8aec00517af1845aff793` |

Revision này cố định code triển khai; commit chỉ thêm handoff về sau không phải
lý do resolve HF `main` rồi tự đổi source giữa lượt kiểm/launch.
Runtime đã đo: Python 3.11.6, Torch 2.9.1+cu128, Diffusers 0.39.0,
Transformers 4.57.6, PEFT 0.17.1; B200 sm_100. Dùng cùng model/scorer revisions
trong registry và runtime/companion đã restore, không dùng revision `main` của model.

Policy chọn: VAE batch 1, 4 CUDA streams có pool tái dùng, scorer batch 8,
capture checkpoint trên GPU, fuse checkpoint, PNG compression 1 và cache prompt
embedding. PNG vẫn lossless nhưng byte size đo được lớn hơn khoảng 6–8%.
Không tăng VAE batch hoặc scorer batch 32: các cấu hình đó đã fail numerical gate.

Modal đo Lightning khoảng 1.78×, Hyper-SD15 khoảng 1.88× full-task trên 2 cặp/profile;
đó không phải cam kết throughput tại công ty. 40 task calibration trên Modal pass,
1,584 ảnh được so trực tiếp có pixel diff 0. Receipt chi tiết ở
`reports/performance/`; đọc `docs/lite/B200_PERFORMANCE.md` trong checkout mới.

**Giới hạn quan trọng:** `hyper_sd15_cfg8_512` mới được đo qua adapter benchmark;
chưa có trong `TRAIN_PROFILES`/CLI collector v1. Không đổi tên `hyper_flux8` thành
Hyper-SD15, không bật policy này cho `hyper_flux8` hoặc `sdxl_base50`. Không coi
Lite2 Core384/native-dtype/schema V2 là đã hoàn tất. Nếu công ty có collector V2
riêng, báo commit/contract và gate riêng trước khi áp dụng; không patch nóng.

## 3. Bootstrap source qua HF/proxy

Đọc note proxy/download DSH đang dùng ở công ty. Chỉ lấy nguồn từ HF qua proxy đã
cấu hình; không GitHub clone, không in giá trị proxy/token, không bật `set -x`.
Bundle source chỉ khoảng 1 MB nên dùng curl thường; không tải lại model/runtime
hoặc hai part companion vì source đổi.

Root dưới đây lấy từ EDA bàn giao, **agent phải xác minh mount thực tế trước khi
ghi**. Nếu khác thì thay ROOT; không suy ra server/GPU từ đường dẫn shared.
Chọn đường dẫn SOURCE chưa được job nào tham chiếu. Nếu target đã có và chưa xác
minh idle thì chọn checkout mới khác; updater clean-check không phát hiện job active.

```bash
(
set -euo pipefail
set +x
ROOT=/workspace/storage-shared/nlp/tungks/apsp_all
test -d "$ROOT"
PROXY="${PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"
test -n "$PROXY" || { echo 'STOP: thieu proxy'; exit 2; }
export PROXY
REV=1de26708f53b5307db73b69e6ca696c2c3dffac5
BASE="https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/resolve/$REV/source"
BOOT="$ROOT/source-bootstrap-d0436c3"
SOURCE="$ROOT/source-releases/apsp-d0436c3"
mkdir -p "$BOOT" "$(dirname "$SOURCE")"
for F in update_source.py update-apsp.sh; do
  curl --silent --fail --location --retry 3 --connect-timeout 20 \
    --max-time 120 --proxy "$PROXY" --noproxy '' "$BASE/$F" -o "$BOOT/$F"
done
printf '%s  %s\n' \
  c7626386d1df707d64a26db03b3ea2e924f6c800254c3efe1b69020763e857a3 "$BOOT/update_source.py" \
  5ba9da1bae64b5410af3708f951ca78e47c8f5b3ab87fa2afa8f07a6712299e7 "$BOOT/update-apsp.sh" | sha256sum -c -
bash "$BOOT/update-apsp.sh" --target "$SOURCE" --revision "$REV"
test "$(git -C "$SOURCE" rev-parse HEAD)" = d0436c3d5c24e5bbd29d1156bd5862faa80e4126
test -z "$(git -C "$SOURCE" status --porcelain)"
)
```

Chỉ `UPDATE_PASS` + hash/ref đúng mới tiếp tục. Updater giữ staging/receipt cạnh
checkout; bảo toàn bằng chứng. Không dùng reset/stash/force để vượt clean/ancestry.

## 4. Chuẩn bị shell riêng và kiểm CPU

Mỗi block GPU/worker chạy trong shell/subprocess riêng. Agent xác định các biến
sau từ inventory thực tế, không copy giá trị ví dụ thành một allocation:

- `APSP_SOURCE`: checkout mới từ §3.
- `APSP_PYTHON`: interpreter của runtime đã restore, thường là
  `/opt/apsp-runtime/bin/python`; kiểm file executable và receipt runtime.
- `HF_HOME`: cache model đúng companion, đã restore/check; activate companion chỉ
  trong shell mới, sau đó đặt lại `APSP_SOURCE`/`PYTHONPATH` vì activate có thể trỏ
  về snapshot source cũ. Không sửa cache đang dùng để chữa thiếu model.
- `CHECK_ROOT`: thư mục kiểm mới bên ngoài source, ví dụ
  `$ROOT/qualification-throughput-d0436c3/<node>-<UTC-attempt>`.
- `ASSETS_RECEIPT`: receipt assets đã verify đúng registry/model revisions/cache
  hiện tại. `apsp.lite.qualify` đọc `$CHECK_ROOT/assets.json` nếu `--out` là
  `$CHECK_ROOT/qualification`; chỉ copy receipt thực đã kiểm, không tự điền PASS.
- `GPU_UUID`: B200 rảnh thuộc allocation đã giao cho lượt kiểm này.

Đặt biến trong shell mới; block dưới từ chối biến trống:

```bash
(
set -euo pipefail
: "${APSP_SOURCE:?}" "${APSP_PYTHON:?}" "${CHECK_ROOT:?}" "${ASSETS_RECEIPT:?}" "${HF_HOME:?}"
test -x "$APSP_PYTHON"
test -f "$ASSETS_RECEIPT"
test ! -e "$CHECK_ROOT" || { echo 'STOP: can CHECK_ROOT moi'; exit 2; }
mkdir -p "$CHECK_ROOT"
export PYTHONPATH="$APSP_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cp -- "$ASSETS_RECEIPT" "$CHECK_ROOT/assets.json"
cd "$APSP_SOURCE"
"$APSP_PYTHON" env/lite/verify_performance_release.py
"$APSP_PYTHON" env/lite/check_performance_cpu.py
"$APSP_PYTHON" -c 'import apsp,sys; print("python=",sys.executable); print("apsp=",apsp.__file__)'
)
```

Expected: `PERFORMANCE_RELEASE_PASS` đúng core hash, `PERFORMANCE_CPU_PASS`, import
từ checkout mới. Không gọi `apsp.lite.assets` online để tải cả registry khi chỉ
cần Lightning; thiếu asset thì inventory trước và dùng add-on/runbook đã ghim.
Không coi CPU PASS là GPU/node PASS.

## 5. B200 qualification tại node, rồi gate parity/throughput

Ghi baseline job inventory trước test; dùng GPU UUID rảnh đã được giao, một process
trên GPU này. Không đổi env hay CUDA_VISIBLE_DEVICES của process đang chạy khác.

```bash
(
set -euo pipefail
: "${APSP_SOURCE:?}" "${APSP_PYTHON:?}" "${CHECK_ROOT:?}" "${GPU_UUID:?}"
export PYTHONPATH="$APSP_SOURCE" CUDA_VISIBLE_DEVICES="$GPU_UUID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$APSP_SOURCE"
"$APSP_PYTHON" -m apsp.lite.qualify \
  --profiles sdxl_lightning8 \
  --execution "$APSP_SOURCE/env/lite/performance/sdxl_lightning8.json" \
  --out "$CHECK_ROOT/qualification" > "$CHECK_ROOT/qualify.log" 2>&1
"$APSP_PYTHON" - "$CHECK_ROOT/qualification/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s.get('pass') and s['profiles']['sdxl_lightning8']['pass'], s
print('NODE_SMOKE_PASS')
PY
)
```

**Không chỉ dựa vào exit code**: qualifier có thể ghi FAIL nhưng process vẫn exit
0. Kiểm summary và receipt. Dùng receipt của node này:
`$CHECK_ROOT/qualification/sdxl_lightning8.json`; không thay bằng receipt Modal.
Node khác fingerprint phải qualify riêng; receipt không được sửa tay.

Sau smoke, agent thực hiện gate nhỏ trên cùng B200 trước khi mở queue production:

1. Chọn trước **3 prompt calibration** theo prompt UID tăng dần từ manifest
   calibration đã phê duyệt, × bank 0/1 × 8 seed = **6 task / 48 candidate**.
   Không chọn theo reward; không dùng train/dev/test để tuning. Ghi hash manifest,
   3 UID và task IDs trước khi chạy. Không chạy trực tiếp `qualify_throughput.py`:
   harness đó có đường dẫn `/persist/...` của Modal, không phải launcher công ty.
2. Tạo harness nhỏ ngoài checkout core trong CHECK_ROOT. Tái dùng APIs thật:
   `LiteRuntime`, `ExecutionPolicy.load`, `fingerprint`, `collect_task`,
   `backup_task`, `validate_task`; helper `score_frames`/`compare` trong
   `env/lite/benchmark_performance.py` có thể import qua sys.path riêng. Không sửa
   `apsp/*.py`, registry hoặc contract để làm pass. Ghi hash harness vào receipt.
3. Với 6 task, collect policy đã chọn vào root kiểm mới. Kiểm COMPLETE/hash,
   8 seed rows, 64 preview rows/task, dtype/layout v1, CandidateId/seed ordering,
   scorer provenance, finite masks, final re-decode và backup checksum/restore.
   `validate_task(path, runtime)` mới có GPU re-decode; CLI validator chỉ kiểm
   schema/checksum. V2 native-dtype/Core384 vẫn nằm ngoài gate này.
4. Trên **cùng stored trajectories** (x0 từng checkpoint + final), chạy decode/
   score với `runtime.execution=None` rồi policy mới; giữ mọi frame/seed, không
   chỉ so final PNG. Gate cố định: max pixel ≤2, mean pixel ≤0.01, max abs delta
   IR/PickScore ≤0.001, validity masks và IR argmax không đổi. Modal đạt pixel 0;
   báo con số thực tại node, không nới gate hoặc chỉ giữ task pass.
5. Đo ít nhất **2 cặp full-task** trên 2 prompt đã chọn, cùng bank/seed/profile,
   hai root legacy/fast riêng. Warmup model/VAE/scorers/common encoder trước timer;
   đổi thứ tự legacy→fast / fast→legacy giữa hai cặp. Timer gồm generation,
   decode, hai scorer, serialize/checksum và backup; loại model load và validator.
   Kiểm tensors initial/x0/final giống nhau giữa hai nhánh. Không dùng task đã
   COMPLETE/resume skip để đo latency. Tính mean các paired speedup ratios.
6. Quan sát ít nhất 6 task liên tiếp: GPU utilization/VRAM, allocated/reserved,
   wall time p50/p90, free disk và growth PNG/latent. Stream pool phải tái dùng;
   không để reserved VRAM tăng không giới hạn. Không tự tăng workers/batch/stream.

**Cho phép bật** khi smoke + cả 6 task + parity + backup/restore đều PASS, không
OOM/corruption, mean paired full-task speedup ≥1.10× và không ảnh hưởng job cũ.
Ngưỡng 1.10× là tiêu chí rollout tại công ty, không phải kết quả đã đo ở node.
Nếu nhiễu shared I/O hoặc performance không đạt thì giữ tắt policy, báo số đo và
nguyên nhân; không chạy lặp vô hạn/tuning vào test hoặc đổi policy đã qualify.
Lưu `node-gate.json` chứa các phép kiểm/số đo, runtime identity, config hash,
source/HF/model/runtime/data identity và danh sách task PASS/FAIL.

## 6. Bật cho job mới sau gate PASS

Re-audit queue trước submit để tránh duplicate và allocation đã đổi. Tạo launcher/
job spec mới, dùng đường dẫn tuyệt đối tới checkout d0436c3 và policy JSON. Không
đổi default source/PYTHONPATH/policy của manager hoặc queue cũ. Source pin theo
job spec; không chạy updater mỗi lần worker khởi động.

Lập manifest **chỉ task còn thiếu và chưa được job running/queued/reserved nào
nhận**; giữ nguyên original task definition/ID/seed. Đọc COMPLETE/hash từ release
cũ để loại task đã có, không copy completion marker cũ vào root mới. Ghi lineage
và bản đồ release theo task ID để downstream đọc cả hai root mà không đếm trùng.
Không dùng nguyên manifest full để vô tình sinh lại 5,162 task từ snapshot EDA;
con số EDA đã cũ, phải audit live. Nếu hệ thống chưa hỗ trợ nhiều output roots,
giữ rollout chặn ở bước này và bổ sung mapping trong job spec/downstream mới,
không buộc collector ghi nối release-v1.

Template worker sau khi operator/agent đã chốt allocation và manifest:

```bash
(
set -euo pipefail
: "${APSP_SOURCE:?}" "${APSP_PYTHON:?}" "${GPU_UUID:?}" "${TASKS_NEW:?}"
: "${OUTPUT_NEW:?}" "${BACKUP_NEW:?}" "${CHECK_ROOT:?}"
: "${WORKER_INDEX:?}" "${WORKERS:?}" "${MAX_WAVE:?}"
export PYTHONPATH="$APSP_SOURCE" CUDA_VISIBLE_DEVICES="$GPU_UUID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$APSP_SOURCE"
"$APSP_PYTHON" -m apsp.lite.collector \
  --tasks "$TASKS_NEW" --profile sdxl_lightning8 \
  --out "$OUTPUT_NEW" --backup-root "$BACKUP_NEW" \
  --worker-index "$WORKER_INDEX" --workers "$WORKERS" --max-wave "$MAX_WAVE" \
  --execution "$APSP_SOURCE/env/lite/performance/sdxl_lightning8.json" \
  --qualification "$CHECK_ROOT/qualification/sdxl_lightning8.json"
)
```

Đây là payload để đưa vào manager đang dùng, không phải yêu cầu chạy foreground
hoặc dựng manager thứ hai. WORKERS/INDEX phải cố định cho cùng manifest; không
đổi modulo partition giữa lượt. Mỗi GPU một worker lúc đầu. Root output và backup
phải tách cây; backup ở storage độc lập theo policy công ty, không chỉ hai thư
mục trên cùng Volume rồi gọi là disaster-recovery backup.

Cho **một job mới nhỏ tối đa 10 task** trước, kiểm hash/schema/throughput và backup,
rồi mới mở các job mới còn lại trong allocation hiện có. Thu receipt từng node.
Job Base50/FLUX/Hyper-SD15 chưa đủ gate/CLI giữ nguyên cấu hình đã được phép trước
đó; không tự chuyển chúng sang collector v1 mới để tận dụng policy Lightning.

## 7. Dừng/rollback và báo lại

- Gate fail: không submit production mới; giữ logs, task IDs và exception đầu tiên.
- Canary mới fail: ngừng mở rộng, chỉ dừng đúng canary/job mới của đợt này nếu cần;
  không đụng job cũ. Không xoá artifacts/claims của job khác.
- Fallback tạo **job spec mới** dùng source/qualification cũ đã biết tốt và root
  phù hợp; không bỏ `--execution` rồi resume root optimized vì contract hash đổi.
- Không khẳng định thiếu model từ một traceback wrapper: kiểm lỗi load đầu tiên,
  cache snapshot/revision và add-on receipt; gửi metadata, không gửi secrets.

Gửi báo cáo JSON/text ngắn kèm đường dẫn shared:

```text
deployment_attempt, timestamp, node, assigned_gpu_uuid
old_jobs_before_after: job_id/state/source_root/output_root (không thay đổi)
git_commit, hf_revision, core_source_hash, lock_hash, runtime_archive_hash
model/scorer revisions, python/torch/cuda/gpu, execution_policy/hash
source_update_receipt, cpu_check, node_smoke_receipt, node_gate.json
calibration_manifest_hash, selected_uids, 6 tasks expected/actual/pass/fail
max/mean_pixel_diff, max_IR_delta, max_PickScore_delta, validity/winner_equal
paired_full_latencies/speedups, p50/p90, allocated/reserved/free_disk
primary/backup restore result và storage failure domains
new_manifest_hash + exclusion snapshot + lineage roots
new_canary_job_id/state/task_count, new_job_ids, log/receipt paths
blocked_profiles/reasons và bước tiếp theo
```

Trạng thái kết thúc: `SOURCE_READY_GPU_PENDING`, `NODE_GATE_FAILED`,
`NODE_GATE_PASS_CANARY_PENDING`, hoặc `ENABLED_FOR_NEW_LIGHTNING_JOBS`.
Chỉ trạng thái cuối khi canary và job spec mới đã được kiểm chứng; source tải được
hoặc process running riêng lẻ không đủ để tuyên bố rollout thành công.
