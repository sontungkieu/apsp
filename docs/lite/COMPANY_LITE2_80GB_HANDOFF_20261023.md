# Handoff chính: Lite2 80GB trên 8 B200, reuse và bảo toàn job cũ

Ngày 23/10/2026, giờ Việt Nam. Người nhận: agent công ty đang giữ queue và shared
storage của Tung. Đọc tài liệu này **trước** handoff tối ưu v1 ngày 22/10.

**Bổ sung triển khai:** collector V2 và các CLI đã được viết trong `apsp.lite2`.
Agent công ty chạy [RUNBOOK_LITE2.md](RUNBOOK_LITE2.md) và
`env/lite2/run-node.sh`; không còn phải tự implement collector/schema.
Các mục bên dưới giữ vai trò checklist hợp đồng và vận hành;
trạng thái code/kiểm chứng hiện tại theo
[B200_LITE2_QUALIFICATION.md](B200_LITE2_QUALIFICATION.md). Luôn lấy Git/HF pins
trong tin nhắn phát hành mới nhất, không quay lại bundle chỉ có handoff f83ba32.

## 1. Mục tiêu và thứ tự tài liệu

Yêu cầu đã được Tung giao: triển khai plan mới trên 8 B200 thuộc node
`duytx-snipe-dgx06-attempt-4-0-0`, tận dụng data cũ đã verify, giữ nguyên job đang
chạy. Đây là quyền triển khai trong allocation đó, không phải quyền thuê thêm máy,
chi thêm Modal hoặc can thiệp các workload khác.

1. [PLAN_DATA_CONTROLLER_LITE_V2_80GB.md](PLAN_DATA_CONTROLLER_LITE_V2_80GB.md):
   **toàn văn spec khoa học/workload/storage**, không phải CLI đã có.
2. Handoff này: current implementation status, migration/reuse, allocation,
   qualification và thứ tự thực thi. Chỗ xung đột về chuyển job, yêu cầu mới của
   Tung ưu tiên: **không đụng job running/queued/reserved cũ**, không chuyển queue
   đó sang source mới. Chỉ tạo job/manifest Lite2 mới cho GPU được giao và rảnh.
3. [LITE2_MODEL_ADDON.md](LITE2_MODEL_ADDON.md): weights/cache view đã đóng gói,
   HF revision và hash download/restore.
4. [B200_PERFORMANCE.md](B200_PERFORMANCE.md) và
   [handoff tối ưu v1](COMPANY_B200_OPTIMIZATION_HANDOFF_20261022.md): tái dùng
   execution-policy patterns và evidence v1, **không dùng như plan Lite2**.

Plan được chép nguyên byte từ file Tung đã gửi, không sửa nội dung:
`PLAN_DATA_CONTROLLER_LITE_V2_80GB.md`, 29,976 bytes,
SHA256 `b85748af36edd40f6000a5a00e1ab41b25317c9220248b85643fb748b361dd32`.
Đã đưa vào source Git/HF lần này; không yêu cầu Tung gửi lại file trong Download.

## 2. Contract dữ liệu mới phải đạt

Namespace `apsp-lite2-fast2`; schema Core `apsp-core-lite2-v1`.

| Profile đích | Recipe ban đầu cần khóa/gate | Train tasks | Dev tasks |
|---|---|---:|---:|
| `hyper_sd15_cfg8_512` | SD1.5 + Hyper CFG-preserved8 LoRA; 512², DDIM trailing, CFG 7.5, eta 0, scale 1 | 2,000 | 400 |
| `sdxl_lightning8_1024` | SDXL + Lightning8 UNet; 1024², Euler trailing, guidance 0, s_churn 0 | 2,000 | 400 |
| Tổng | 8 seed/task, đủ 8 checkpoint, không prune khi thu | 4,000 | 800 |

- **2,000 train + 200 dev UID dùng chung giữa hai profile**; train bank 0,
  dev bank 0/1. **4,800 task / 38,400 candidate**, 307,200 preview rows,
  38,400 delivered rows, 691,200 score values dự kiến (2 scorer × preview/final).
  Failure/null có lý do, không giả mọi score đều valid.
- Train source quotas: DiffusionDB 1,000, COCO 400, structured 600; dev 100/40/60.
  V1 dev quota trong code hiện tại là 150/60/90, **không gọi make_tasks/prompts v1
  rồi gán nó thành dev200**. Freeze V2 manifest/UID/source/group trước reward audit.
- Train nested 250 ⊂ 1,000 ⊂ 2,000, cùng UID trên hai profile; normalize NFC/space,
  split theo dedup/group, COCO image_id và template/paraphrase family. Kiểm leakage
  với test/v1 manifests, báo coverage detector thực tế.
- 50 calibration UID riêng. Test500 gồm 300 ID + PartiPrompts-200: chỉ chuẩn bị
  manifest/khóa quyền đọc, **không sinh/chấm test** trước method freeze.
- Core dùng MiniLM-L6-v2 384d, attention-mask mean pooling, L2, log truncation;
  không dùng CLIP768 v1 giả thành embedding mới. Final scores riêng preview@8;
  Lightning cfg_gap không có thì availability=false, không bịa feature bằng 0.
- Lưu latent **dtype thực trước serialize**, từng tensor có metadata native dtype,
  shape, scaling/shift/layout/decode recipe. Không cast để đạt dung lượng; không
  suy ngược tensor fp32 gốc bằng cách upcast file fp16.
- Core chạy fixed/threshold/random-masked replay trên dữ liệu thật mà không mở
  PNG/latent, có no-leak accessor/sentinel tests. Không train controller/RL trong
  collection job, không dùng quality để bỏ prompt/seed khó.

Dung lượng tính theo tổng byte **mọi file bàn giao**, GB = 10^9 byte:
Core ≤5GB, latent ≤65GB, QA ≤5GB, dự phòng 5GB, tổng ≤80GB.
QA: 800 final PNG + 480 preview PNG chọn trước theo spec, ngoài train/dev.
Không kèm model/cache/runtime, không xuất toàn bộ preview/train-dev PNG. Nếu vượt
forecast, giữ Core và báo latent optional/retention; không cast/xóa raw. Mode 40GB
là Core+QA, không tự giảm 2,000 train. Working disk có dự toán riêng, lớn hơn 80GB.

## 3. Trạng thái implementation hiện tại

| Thành phần | Source hiện đã có | Việc agent công ty phải làm trước bulk V2 |
|---|---|---|
| Runtime | Python 3.11.6, Torch 2.9.1+cu128; lock hiện có | Restore/verify node nếu thiếu; reuse dependency, không pip upgrade |
| Lightning V2 | `apsp.lite2.contracts/collector` có profile riêng, native storage, features | Node qualification bằng CLI V2, không dùng receipt v1 |
| Hyper-SD15 V2 | Production profile/adapter và collector chung, không alias hyper_flux8 | Restore add-on nếu thiếu, node gate 20 task |
| MiniLM384 | `apsp.lite2.encoder` có masked mean/L2/truncation/index | Activate cache view đúng, không đổi common encoder |
| Performance | VAE batch1 + 4 streams reuse, scorer8, GPU capture/fusion | Recheck native path tại node; giữ policy đã khóa |
| Core/manifests/export | `apsp.lite2.manifests/validate/replay/export/pack` | Chạy CLI có sẵn; Core replay/no-leak/restore và forecast trước bulk |
| Source phát hành | Receipt/binding tại `reports/lite2-v2/`, verifier `env/lite2/verify_release.py` | Source commit/HF pins mới nhất, node qualification riêng |

**Agent công ty chỉ tải source/runtime/model đúng hash và chạy các CLI thật theo
RUNBOOK_LITE2.** Không còn nhiệm vụ viết lại collector/schema. Nếu gặp bug/missing
asset thì giữ log, báo exception đầu tiên và làm tiếp phần độc lập; không chạy
v1 thay V2 để lấp số task. Không đổi scientific recipe, schema hoặc policy ngầm
giữa bulk. Xem receipt phát hành để phân biệt kiểm chứng Modal với node công ty.

## 4. Nhận source/handoff đầy đủ từ HF, vào checkout riêng

HF dataset `codemaivanngu/apsp-lite-b200-cu128`, canonical GitHub
`https://github.com/sontungkieu/apsp`, branch `main`. **Dùng HF revision + Git commit
trong tin nhắn bàn giao lần này**, không resolve `main` giữa lượt. Handoff không
ghi chính revision chứa nó để tránh vòng lặp self-reference khi publish.

Source baseline đã đo trước đây là Git `d0436c3d5c24e5bbd29d1156bd5862faa80e4126`,
HF `1de26708f53b5307db73b69e6ca696c2c3dffac5`. **Revision cũ đó KHÔNG chứa full
plan 80GB/handoff mới**. Không dùng lại nó để nạp context của lần này.

Theo DSH proxy note đang có; không in proxy/token, không fallback IP proxy viết
cứng. Xác minh ROOT mount thực tế. Block dùng target mới, từ chối overwrite:

```bash
(
set -euo pipefail
set +x
: "${HF_SOURCE_REV:?lay tu tin nhan ban giao}" "${EXPECTED_SOURCE_COMMIT:?lay tu tin nhan ban giao}"
ROOT=/workspace/storage-shared/nlp/tungks/apsp_all
test -d "$ROOT"
PROXY="${PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"
test -n "$PROXY" || { echo 'STOP: thieu proxy'; exit 2; }
export PROXY
BOOT="$ROOT/source-bootstrap-lite2-80gb-$(date -u +%Y%m%dT%H%M%SZ)-$$"
SOURCE="$ROOT/source-releases/apsp-lite2-80gb-${EXPECTED_SOURCE_COMMIT:0:12}"
test ! -e "$SOURCE" || { echo 'STOP: target da co, kiem active va chon checkout moi'; exit 2; }
mkdir -p "$BOOT" "$(dirname "$SOURCE")"
BASE="https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/resolve/$HF_SOURCE_REV/source"
for F in update_source.py update-apsp.sh; do
  curl --silent --fail --location --retry 3 --connect-timeout 20 --max-time 120 \
    --proxy "$PROXY" --noproxy '' "$BASE/$F" -o "$BOOT/$F"
done
printf '%s  %s\n' \
  c7626386d1df707d64a26db03b3ea2e924f6c800254c3efe1b69020763e857a3 "$BOOT/update_source.py" \
  5ba9da1bae64b5410af3708f951ca78e47c8f5b3ab87fa2afa8f07a6712299e7 "$BOOT/update-apsp.sh" | sha256sum -c -
bash "$BOOT/update-apsp.sh" --target "$SOURCE" --revision "$HF_SOURCE_REV"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
printf '%s  %s\n' \
  b85748af36edd40f6000a5a00e1ab41b25317c9220248b85643fb748b361dd32 \
  "$SOURCE/docs/lite/PLAN_DATA_CONTROLLER_LITE_V2_80GB.md" | sha256sum -c -
printf 'LITE2_CONTEXT_READY source=%s\n' "$SOURCE"
)
```

Chỉ `UPDATE_PASS` + `LITE2_CONTEXT_READY` mới coi là nhận đủ context. Source nhỏ
không cần tải 12 luồng; không tải lại hai part runtime/companion vì docs đổi.
Không update symlink/env/source chung hoặc checkout d0436c3 đang được job dùng.
Khi viết code V2, commit vào checkout riêng, ghi baseline HF revision + local V2
commit/core hash; mọi qualification/bulk phải dùng đúng source V2 đã freeze đó.
Không tự push/force/rewrite canonical main từ công ty để giải quyết divergence.

## 5. Inventory và reuse data cũ: không sinh lại phần hợp lệ

Snapshot người dùng đã cung cấp `20261021T192438Z`, **không phải live count**:
`VERIFY_OK 5162/5162, BAD 0`, 1,000 prompt UID, tất cả split=train.

| Data cũ | Bank 0 train | Bank 1 train | Cách dùng trong Lite2 |
|---|---:|---:|---|
| Lightning8 | 839 | 823 | Bank0 là ứng viên chính; bank1 reserve, không đổi thành dev/test |
| Hyper-FLUX8 | 750 | 750 | Dữ liệu bổ sung v1, không phải Hyper-SD15 |
| SDXL Base50 | 1,000 | 1,000 | Dữ liệu bổ sung v1, không nằm target Lite2 |

Nguồn audit: `eda-v1/reuse-counts-20261021T192438Z.json`,
`COMPANION_MANIFEST.json` và task provenance trên shared. Collector cũ source
`c7d51304f4d91e75d839cdade1a13584ab88dd02`; lock SHA
`e404558d31cc6befaa79aa693b99650eee3d269db3d8aec00517af1845aff793`.
Không lấy số count từ EDA khác thời điểm để cộng trừ coverage.

### 5.1 Audit live trước khi đóng manifest

Đọc nguyên trạng `release-v1`, backup đã có và mọi release cũ thực tế; verify
COMPLETE/checksum **task đã atomic complete**, không đọc staging như completed.
Snapshot chung timestamp gồm profile × split × bank × wave, unique UID, task IDs,
current source/model/lock, dtype/shape/decode/scorer contract. Ghi riêng running,
queued, reserved IDs và output roots. Không sửa DB scheduler, steal claim hay
đụng staging/process cũ. Bảo toàn inventory và old-job snapshot trước/sau.

Freeze V2 prompt pool/splits theo quotas/dedup **trước khi nhìn reward**, tận dụng
text/source/group/UID cũ nếu hợp lệ. Dev không được lấy UID đã dùng train v1 để
gọi là unseen-dev. Wave250/1000 cũ không tự chứng minh đúng V2 stratified subset;
join theo frozen UID và subset rank mới. Test/calibration disjoint như spec.

### 5.2 Compatibility matrix, không chỉ đổi nhãn model

Mỗi task cũ có một record audit với lineage và trạng thái:

- `accepted_full`: prompt/split/seed/recipe/runtime/numeric path/scorers/features
  đều tương thích, latent native dtype/shape đủ evidence, được tính vào V2 target.
- `accepted_core_only`: chỉ khi Core observation/scores đúng contract V2 và replay
  đạt, nhưng thiếu latent native. Ghi latent extension thiếu rõ, không tính là
  native-latent complete. Không chấp nhận scalar từ decode khác chỉ vì finite.
- `needs_derivation`: không cần sinh diffusion lại nhưng phải bổ sung MiniLM384,
  index/masks/export từ artifact gốc hợp lệ. Chỉ tính sau derivation+validation.
- `blocked_native_dtype`, `blocked_recipe`, `blocked_split`, `invalid_checksum`,
  hoặc `pending_old_job`: giữ nguyên v1, báo lý do, không tính thành V2 complete.

Đặc biệt `collector.py` v1 có `.to(runtime.dtype)` khi ghi x0/final/initial;
sample Lightning cũ x0/final/initial đều fp16. P1 dtype probe trong log công ty
chưa có kết quả trong handoff này. Phải kiểm native dtype ở điểm tap trước cast
và đường decode thực trên source/profile cũ, đối chiếu V2. File fp16 không chứng
minh x0 gốc fp16. Nếu đã mất fp32, không khôi phục được bằng cast ngược; không
nhận full native provenance bằng cách sửa metadata. Reuse-only Core cũng cần
kiểm parity feature/score với contract V2, không mặc định được nhận.

Tên `sdxl_lightning8` v1 → `sdxl_lightning8_1024` V2 cần bảng mapping recipe/hash
có kiểm chứng, không đổi task ID/file cũ tại chỗ. Core V2 giữ cả old ID/URI/hash,
new CandidateId và conversion receipt; dedup theo khoa học (profile, UID, bank,
seed_idx), không chỉ theo task ID vốn chưa chứa source/dtype contract.

### 5.3 Task còn thiếu và job cũ chưa xong

Ví dụ **chỉ khi cả 839 bank0 đều đạt full/core contract cần thiết**: Lightning
train còn 1,161; dev còn 400; Hyper-SD15 còn 2,400; tổng 3,961 task main cần thu.
Đây là upper-bound reuse từ snapshot cũ, **không phải launch count đã chốt**.
Nếu không task nào đạt thì main cần thu 4,800, trừ live reuse được nghiệm thu.
Luôn báo coverage Core và native-latent riêng, kèm record không tương thích.

Không cấp lại task đã được job cũ nhận. Với task `pending_old_job` chỉ hoãn tới
khi job đó complete và audit được; nếu kết quả cũ không thể đạt V2, đánh dấu cần
V2 generation riêng rồi xếp sau khi reservation cũ kết thúc. Các UID/task độc
lập khác vẫn tiến hành. Không cancel/chuyển profile job cũ để tăng số reuse.
Không append/mirror completion marker vào `release-v1`/`backup-v1` hoặc chép cả
raw bank sang release mới chỉ để tăng count. Reference artifact cũ chỉ khi có
durability/retention + hash; nếu cần snapshot byte copy thì copy artifact đã
complete sang root mới, ghi lineage, không thay file/metadata nguồn.

## 6. Node, runtime và model: setup một lần, không tám bản cache

Log người dùng ngày 23/10 cho host `duytx-snipe-dgx06-attempt-4-0-0` thấy 8 B200,
shared mount có source d0436c3, khoảng 2TB RAM. UUID snapshot:

| Index | GPU UUID |
|---:|---|
| 0 | GPU-79be5e9f-178e-5de4-4714-5d50b2554b0c |
| 1 | GPU-d4a9f380-253b-11fb-3310-dc386ed41115 |
| 2 | GPU-48ed4aea-a65f-7d52-2f97-d518940729ea |
| 3 | GPU-12c44a3f-a35c-237e-7396-9754233bf993 |
| 4 | GPU-654c372c-01fb-9f78-efa2-c3ae4753b451 |
| 5 | GPU-8cb6790e-e514-d9df-bf7d-da5001fa90ed |
| 6 | GPU-0df4fc7d-8df8-cc9c-627a-031032fc140d |
| 7 | GPU-47300b26-33c3-62c8-4ffc-5029e8c869a3 |

Đây là snapshot cũ, **re-query `hostname`, `nvidia-smi -L`, compute apps và queue
allocation ngay trước test/launch**. UUID đang bận/reserved thì không dùng. Không
chạy trên node khác chỉ vì tên gần giống. Agent không có remote shell thì cung
cấp block đúng từng node cho operator, không bịa receipt đã chạy.

RY1 trong log đã báo restore `/opt/apsp-runtime` + `/opt/apsp-python` PASS,
Torch 2.9.1+cu128; **không mặc định phải restore lại**. Recheck interpreter/runtime
manifest/lock. Chỉ nếu thiếu thật mới dùng bootstrap runtime đã verify sẵn trên
shared. Activate companion trong shell mới rồi đặt lại APSP_SOURCE/PYTHONPATH;
activate cũ trỏ về `companion-v1/source`, không phải checkout Lite2.

Add-on đã public: HF revision `246ca32ce4dcb287296a740ed7cbd17998d19024`,
archive `apsp-lite2-model-addon-v1.tar`, 4,133,734,400 bytes, SHA256
`3ba6fc1a2cd47ee9e42b7857146c6f967ed56c4c9dbb6d28ae9cbffcdc306533`.
Xem LITE2_MODEL_ADDON để dùng template 12 luồng/proxy/restore đúng hash. Tải/restore
một bản trên shared và tạo cache view riêng, không sửa companion cũ đang dùng.
SD1.5 base rev `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, Hyper LoRA rev
`bc08d970a87c74c71209491d64e3525845698863`; MiniLM384 theo manifest đã ghim.
Lightning/scorers reuse pinned assets hiện có, không tải lại mỗi worker/GPU.

Trong worker mới: companion activate → add-on activate → đặt source V2/PYTHONPATH
→ offline flags → chọn đúng GPU UUID. Không sửa shell/config global hoặc process
env của job cũ. Tính working disk cho weights + staging + raw + export + backup;
80GB là cap bàn giao, không phải cap node storage hay VRAM.

Lỗi RY2 trong đoạn chat được gửi có thiếu `-` ở Python heredoc. Cú pháp đúng:

```bash
"$APSP_PYTHON" - "$CHECK_ROOT/qualification/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s.get('pass'), s
print('NODE_SMOKE_PASS')
PY
```

Không gọi qualifier v1 ở đây như thể nó nghiệm thu native dtype/Core V2; receipt
v1 chỉ có giá trị đúng source/policy/schema đã đo. Exit code 0 không thay JSON PASS.

## 7. Thứ tự triển khai và gate bắt buộc

### A. Inventory và kiểm source trước khi mở 8 GPU

Ghi old-job snapshot, mount/runtime/weights, prompt/data inventory; clone source
mới qua §4. Đọc full spec và chạy entry point trong RUNBOOK_LITE2, không yêu cầu
operator tự viết collector nữa. Không sửa `apsp/*.py` trong checkout v1 đang dùng.
Không đổi dependency khi không có blocker đã chứng minh; báo constraint thay vì
pip install đè runtime.
Không dùng script Modal `/persist/...` làm node launcher, không tạo CLI tưởng tượng.

Release đã khóa adapter/model/scorer/tokenizer/scheduler/VAE revisions,
precision/attention backend, execution policy và native tensor semantics.
Agent giữ các giá trị này, dùng CLI thật trong `RUNBOOK_LITE2.md`, kiểm binding
và import từ source mới; launcher freeze prompt manifests tại node. Một thay đổi
`apsp/*.py` làm core source hash đổi: receipt phát hành/node không còn qualify
cho source sửa đó; không patch nóng giữa bulk.

### B. Calibration V2 và forecast trước bulk

Trên tối đa 2 GPU rảnh đã giao, mỗi profile một process resident. Chọn trước 10/50
calibration prompt × 2 bank = **20 task/profile**, hai profile 40 task/320
candidate; đủ 8 checkpoint, hai scorer, native latents và Core. Phase receipts
riêng INSTALL/IMPORT/GPU/GEN/PREVIEW/DECODE/IR/PICK/SAVE/RESTORE/CORE_REPLAY.
Một adapter fail thì giữ blocked adapter đó; phần độc lập đã PASS tiếp tục trong
phạm vi được phép. Lỗi common scorer chặn completion cả hai.

- Shape/dtype metadata khớp tensor **trước** serialize, không chỉ header file;
  kiểm masks/null/row map, no extra denoiser forward, initial/final hashes và
  delivered pixels. Restore ở process khác. Không dùng full v1 casts để làm pass.
- So optimized với canonical V2 trên cùng trajectory; các ngưỡng performance-v1
  giữ: max pixel ≤2, mean ≤0.01, max IR/PickScore abs delta ≤0.001, validity và
  IR winner giữ nguyên. Không dùng reference v1 cast-fp16 để ép native V2 giống nó.
- Ít nhất 2 cặp full-task/profile, warmup ngoài timer, đảo thứ tự legacy/fast,
  cùng prompt/seed; timer gồm generation/decode/scorers/serialize/backup. Chỉ bật
  policy khi parity/restore PASS và mean paired speedup ≥1.10×. Nếu không đạt,
  giữ canonical V2 đã qualify cho profile đó, báo đo thật; không nới gate.
- Đo bytes/task theo Core/latent/PNG/metadata và dtype; forecast 4,800 main task
  cộng hợp calibration/QA, không double count phần giao nhau. QA50 + qualification
  có hợp 120 calibration task / 960 candidate; ngoài main 4,800.
- Nếu không vừa 80GB, ưu tiên Core+QA, latent optional với byte/URI/expiry report;
  không lặng lẽ đổi yêu cầu raw hay giảm prompt. Không gọi optional chưa nhận là
  đã bàn giao full latent. Log capacity/retention của storage trước bulk.

### C. Canary rồi tăng 2 → 4 → 8 worker

Hai canary mới, mỗi profile tối đa 10 task từ main manifest (được tính vào main,
không sinh lại lúc bulk), output/backup riêng. Gate hash/schema/row count/native
dtype/scores/heartbeat/restore và không ảnh hưởng job cũ. Sau canary PASS mở thêm
worker tới 4, kiểm throughput/I/O/VRAM rồi tới 8 nếu allocation vẫn rảnh. Tất cả
đều là process/job **mới**, không restart/repurpose job cũ.

Không chờ người dùng duyệt từng phase đã được giao. Nếu vướng missing critical
contract/capability thì báo blocker cụ thể, tiếp tục audit/Core/adapter độc lập;
không bật job sai contract cho đủ 8 GPU. Không tạo fake workload để né idle policy.

## 8. Phân việc trên 8 B200 và queue không trùng

Khởi đầu đề xuất **2 Hyper-SD15 + 6 Lightning**, một process/GPU, model/scorers
resident; UUID chọn từ live inventory, không buộc GPU0 phải Hyper. Lightning đắt
hơn theo đo Modal cũ nhưng V2 native path có thể khác. Đây là scheduling heuristic,
không phải scientific recipe hoặc cấu hình distributed training/DDP.

Sau calibration/reuse audit, tính `W_profile = remaining_tasks × measured_p50_s`,
chia worker gần tỷ lệ W, ít nhất 1/profile còn việc, tối đa 8 GPU được giao.
Ví dụ 2/6 chỉ là khởi điểm; dùng throughput end-to-end của node, không lấy GPU
utilization tức thời hay 1.78× Modal làm ETA. Nếu shared I/O quá tải, hạ số worker
mới hoặc điều tiết backup/export mới, không đụng job cũ.

Tạo task manifest chỉ phần thiếu, loại accepted reuse/canary COMPLETE và running/
queued/reserved IDs cũ. Ghi manifest hash, audit snapshot, exclusions và trạng
thái blocked/pending. Không lấy toàn bộ manifest v1/wave-2000 làm workload Lite2.
Giữ profile ID/UID/bank/seed theo V2 manifest, không alias FLUX thành SD15.

Dùng manager hiện hữu bằng job group/spec mới hoặc scheduler allocation hiện tại;
không sửa config chung/SQL trực tiếp, không dựng manager cạnh tranh job cũ.
Nếu partition theo modulo: freeze `worker_index/workers` cho từng profile và
manifest/epoch; không thay 2/6 thành 3/5 khi worker cùng epoch còn chạy. Chỉ
repartition epoch mới sau khi các worker của **epoch Lite2 mới** đã tự drain,
rescan COMPLETE/leases và freeze remaining manifest mới. Task claim là atomic,
không steal active claim. Heartbeat/lease timeout phải theo trạng thái process
thực, không dùng elapsed time để tự cướp job lâu.

Đợt 1: train1k một bank × hai profile (2,000 tasks), dev200 hai bank × hai profile
(800 tasks), trừ reuse/canary; ưu tiên lấp dev còn thiếu, không chờ train controller.
Đợt 2: thêm train1k còn lại (2,000 tasks), tự tiếp tục sau QA. Tổng target vẫn
4,800 logical tasks, không cộng task calibration/QA thành train. Shard exports
incremental, không gom toàn bộ rồi mới backup.

Mỗi worker spec ghi source V2 commit/hash, baseline HF revision, profile lock,
policy/hash, runtime/model/scorer/data revisions, assigned UUID, manifest hash/
partition/epoch, output/backup, qualification receipt và log path. Retry infra
cùng task/seed tối đa 2 lần sau attempt đầu theo spec; không áp retry 3 lần bổ
sung hoặc thay seed khi reward thấp. Status model failure/infra/pending tách rõ.

## 9. Roots, retention và nghiệm thu cuối

Đề xuất roots dưới shared đã xác minh (tên cụ thể ghi vào run manifest):

```text
apsp_all/source-releases/apsp-lite2-80gb-<handoff-commit>/  # mới, không sửa source job cũ
apsp_all/apsp-lite2-fast2/<release_id>/
  manifests/ audit/ qualification/ logs/ core/ latent/ qa/ export/
<backup-storage-da-xac-minh>/apsp-lite2-fast2/<release_id>/
```

Output, staging/claims và backup mới tách cây v1; không sửa/quét/xóa v1 để giải
phóng dung lượng. Xác nhận storage failure domain/retention thật; hai thư mục
trên cùng mount không tự thành independent backup. Cross-process restore là
technical gate, chưa chứng minh durability. Nếu retention chưa xác nhận, ghi
blocked durable handover, ưu tiên chuyển Core verified trước khi mất máy.

DONE chỉ khi có đủ Core record/status và planned-vs-complete counts theo cả
profile × split × bank, 2,200 UID main đúng split, lineage reuse, checksums,
Core replay/no-leak trên data thật, native latent status, QA, cost/throughput/
storage forecast vs actual và restore ở process/máy khác. Dataset card/license,
collection report, release manifest, SHA256SUMS, runtime/model/scorer locks,
cost raw samples và runbook lệnh thật theo §10 spec. Không claim đã sinh đủ
ảnh valid vì đủ records, không claim controller thắng/test hoàn tất.

Xuất Core trước, latent/QA riêng cùng release_id. Part 1–2GB nếu cần; tổng bytes
nhận ≤80GB hoặc mode 40GB Core+QA với latent pending nêu rõ. Tách test artifacts
ngoài budget. Không re-upload model/runtime, không provider Stop/Destroy.

## 10. Báo lại và trạng thái trung gian

Agent báo ngắn gọn sau inventory, reuse/native probe, qualification, canary/bulk
và export; tiếp tục tự chủ các bước đã được giao. Mỗi báo cáo có timestamp và
path receipt, không chỉ nói đang chạy hoặc setup xong:

```text
host + live allocated UUIDs + old-job before/after snapshot (unchanged)
handoff Git/HF pins + plan SHA + V2 implementation commit/core/lock hash
runtime/model/scorer/encoder/precision/backend/policy lock
V2 manifest hash, quotas/dedup/leak checks, train/dev/calib/test UID counts
live reuse: accepted_full/core_only/needs_derivation/blocked/pending theo profile/split/bank
native dtype probe trước/sau serialize + decode parity của v1 reuse
expected/new/reused/complete/failed/pending counts, không double count
compatibility matrix và 20 calibration tasks/profile receipts
Core replay/no-leak/restore, forecast và actual bytes từng package
assigned UUID→profile/job/partition/epoch, canary IDs và bulk heartbeat/throughput
primary/backup/retention/failure domains; còn thiếu gì và bước tiếp theo
```

Các trạng thái hợp lệ: `CONTEXT_READY`, `V2_IMPLEMENTATION_PENDING`,
`REUSE_AUDITED`, `V2_CALIBRATION_PARTIAL`, `V2_QUALIFIED`, `CANARY_PASS`,
`BULK_RUNNING`, `CORE_READY_LATENT_PENDING`, `RELEASE_VERIFIED`, hoặc blocker
cụ thể. Không gọi `V2_QUALIFIED` từ smoke/receipt Lightning v1. Nếu lỗi, dừng mở
rộng và chỉ xử lý job Lite2 mới thuộc đợt này; giữ nguyên toàn bộ job cũ.
