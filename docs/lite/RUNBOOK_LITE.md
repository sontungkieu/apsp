# APSP Lite v1 — runbook

Namespace riêng từ bundle `396b3132eeb35bf7c98e0e4b1a851bf88268cc1b`.
Giữ nguyên `env/b200`, seed/grammar v1 và mọi dữ liệu Full. Plan nguồn:
[PLAN_CODEX_CONTROLLER_DATA_LITE_V1.md](PLAN_CODEX_CONTROLLER_DATA_LITE_V1.md).

Model/scorer/tokenizer/text/source được gom riêng ngoài dependency runtime theo
[COMPANION_BUNDLE.md](COMPANION_BUNDLE.md), cùng private Modal Volume. Chưa thu full
bank thì archive companion chỉ chứa text/task/raw metadata hiện có.

## Runtime

Dựng runtime mới từ lock B200 của bundle, thêm PEFT 0.17.1. Không dùng TTS làm
dependency base: TTS là Python 3.10 / Transformers 4.38.2 / Diffusers git cũ,
trong khi bundle dùng Python 3.11 / Transformers 4.57.6 / Diffusers 0.39.0.
Tái dùng quy trình path-preserving archive, SHA256, gate và asset cache khi hash khớp.
Không cài đè `/opt/venvs/tts-b200`.

`env/lite/uv.lock` là nguồn dependency; mọi worker dùng cùng lock. Modal image có
dependency layer riêng, source overlay riêng, weights tại Volume riêng. Mọi source
thay đổi sau qualification sẽ làm receipt không khớp collector và buộc requalification.

ImageReward pinned còn import ba helper từ `transformers.modeling_utils`; bản Lite
alias chúng tới đúng hàm upstream trong `transformers.pytorch_utils` trước khi import
ImageReward. Không sửa checkpoint, third-party files hoặc công thức score. Bridge nằm
trong source fingerprint. Import pass chưa chứng minh scorer/model pass.

Build/qualify (chỉ trên remote đã được cấp quyền, chạy billing guard trước):

```bash
uvx --offline --from modal modal run --profile PROFILE env/lite/modal_runtime.py::cpu
uvx --offline --from modal modal run --profile PROFILE env/lite/modal_runtime.py::prepare_assets
uvx --offline --from modal modal run --profile PROFILE env/lite/modal_runtime.py::qualify
uvx --offline --from modal modal run --profile PROFILE env/lite/modal_runtime.py::package
uvx --offline --from modal modal run --profile PROFILE env/lite/modal_runtime.py::restore
```

`--offline` ở đây chỉ cấm uvx tải CLI trên máy điều khiển; asset download chạy trên
CPU remote. Không chạy `uv sync` trên laptop. Trên node có sẵn uv và được phép cài:
`uv sync --locked --default-index https://pypi.org/simple` trong `env/lite`.

Volume: `apsp-lite-runtime-v1`, mount `/persist`. Model/scorer snapshots ở `hf/`;
asset checksums ở `assets.json`; receipts ở `receipts/`, `qualification/`; portable
archive nếu mọi gate đạt ở `export/`. Volume thuộc profile đã chọn; không public.
Không gọi package thành qualified nếu summary false. Package không chứa source,
weights hoặc credential. Gói Python gồm cả prefix thật và alias symlink của uv.

Restore trên node đích đã được cấp quyền, lấy archive/manifest trực tiếp tới node:

```bash
python3 env/lite/ensure_runtime.py --archive /shared/apsp-lite/runtime.tar.gz \
  --manifest /shared/apsp-lite/manifest.json
```

Helper kiểm size/hash, traversal/link target, lock restore và không ghi đè prefix
không do runtime này quản lý. Cần ghi `/opt`, Python >=3.10,
Linux x86_64, driver tương thích cu128 và B200. Library ABI của node vẫn cần gate thực.
Không coi restore trên Modal là xác nhận node công ty.

Archive đã được thử trên Ubuntu22.04/glibc2.35 riêng. System libraries trong
restore image: `libgl1`, `libglib2.0-0`, `libgomp1`; node đích phải có các library
tương ứng. Không copy đè libc/libstdc++ của host. Có thể cài/import trước khi có GPU
bằng `--cpu-only`; kết quả đó được ghi rõ là CPU restore, không phải GPU qualification.

Lấy archive trực tiếp tới node đã được cấp quyền, bằng Modal CLI/profile có sẵn:

```bash
mkdir -p /shared/apsp-lite/payload
modal volume get apsp-lite-runtime-v1 /export/runtime.tar.gz \
  /shared/apsp-lite/payload/runtime.tar.gz --profile billionaireproject8
modal volume get apsp-lite-runtime-v1 /export/manifest.json \
  /shared/apsp-lite/payload/manifest.json --profile billionaireproject8
```

Chỉ là lệnh bàn giao, chưa chạy trên node công ty. Nếu dùng directory download,
tạo destination directory trước rồi kiểm hash từng file; không chỉ tin thông báo CLI.

Qualification cuối đã chạy một prompt riêng × tám seed cho mỗi profile, cùng hai
scorer; restore archive trên container Ubuntu22.04/B200 cũng đạt. Saved final latent
re-decode trong từng run có max pixel diff=0. So sánh **giữa** build container và
restore container: Lightning/FLUX có 8/8 PNG giống SHA, SDXL50 có 0/8. SDXL50 có
max mean channel difference=0.2115631103515625 trên thang 0–255, max IR delta=
0.04556381702423096 và max raw PickScore delta=0.04733085632324219. Chưa xác định
nguyên nhân; không coi pass functional restore là bitwise parity. Cần freeze
host/runtime thực tế và nghiệm thu drift/native batching trước mass collection.
Xem `reports/lite/CROSS_CONTAINER_PIXEL_DRIFT.json` và `LITE_DATA_REPORT.md`.

## Prompt input và manifests

Đã có text preparation candidate tại Volume:
`/text/lite-v1-20261019/manifests/` và local `data/lite-v1/prepared/manifests/`.
Xem dataset card/hash audit; manual/language QA còn pending.

CLI cũng nhận JSONL text-only có provenance. Một row:

```json
{"text_raw":"A ceramic cup on a wooden table.","source":"diffusiondb","source_row_id":"123","source_revision":"PINNED_REVISION_OR_SOURCE_HASH","parent_group":"source-parent-123"}
```

`source`: `diffusiondb`, `coco_captions`, `structured`, `parti`.
COCO thêm `image_id`; structured thêm `template_family_id`, `scene_spec`.
Giữ thông tin URL, licence và attribution trong provenance của nguồn. Không thay
corpus thật bằng fixture. Không lấy prompt theo reward. Corpus phải có đủ group
riêng cho train/dev/test/calibration; thiếu quota thì CLI fail, không bù giả.

```bash
python -m apsp.lite.prompts --input /shared/text/candidates.jsonl \
  --exclusion /shared/text/geneval.jsonl --exclusion /shared/text/v1.jsonl \
  --out /shared/apsp-lite/prompts-r1
python -m apsp.lite.tasks --prompts /shared/apsp-lite/prompts-r1/train.jsonl \
  /shared/apsp-lite/prompts-r1/dev.jsonl --out /shared/apsp-lite/tasks-r1.jsonl
```

Parti input toàn bộ được dùng làm exclusion trước split ID/train/dev; chỉ selected
200 Parti rows vào test_shift. UID SHA256 trên NFC/whitespace-normalized text;
exact aliases giữ mọi parent/image/family group. Detector lexical Jaccard 3-gram
>=0.85; không có tuyên bố semantic decontamination. Thiếu v1 manifest phải ghi rõ.
Nested train250/1000/2000 có quota nguồn 50/20/30 phần trăm. Structured có nhãn
primary_category được chia đều năm nhóm; domain/độ dài/naturalness còn cần QA.

Manifests thực có train2000, dev300, calibration50, test_id300, test_shift200 và
13.800 train/dev tasks. 110.400 candidate records/1.177.600 preview rows là kỳ vọng,
chưa phải dữ liệu đã thu. Full chưa khóa nên Lite manifest độc lập.

## Worker và resume

Mỗi worker giữ resident một profile. Có thể chạy bằng scheduler hoặc tmux được cấp.
Đặt `APSP_GPU_UUID`, `APSP_SOURCE`, `APSP_PYTHON`, `APSP_PROFILE`, `APSP_TASKS`,
`APSP_OUTPUT`, `APSP_BACKUP`, `APSP_QUALIFICATION`, `HF_HOME`, rồi:

```bash
bash env/lite/launch_worker.sh
```

`APSP_WORKERS`/`APSP_WORKER_INDEX` chia task độc lập, không phải distributed training.
Không đổi partition trong lúc worker cũ còn chạy. Claim directory không tự bị cướp;
crash để lại claim cần operator xác minh PID/host đã chết rồi xử lý đúng task.
Không đoán quyền GPU từ số GPU nhìn thấy, không kill job khác.

Tasks theo wave250/1000/2000 trong cùng manifest. `APSP_MAX_WAVE=250` giới hạn pilot;
CLI hiện xử lý task theo wave nhưng chưa có controller tự nghiệm thu một wave rồi
phát hành wave kế. Chạy report/QA sau mỗi wave trước khi tăng limit cho collection thật.

Mỗi task là tám full trajectories, không prune. Checkpoint count lấy profile (8/8/16).
Tap lấy clean prediction từ scheduler forward hiện có; actual denoiser calls phải
bằng T. CPU FP32 noise hash và actual initial tensor/hash được lưu. FLUX unpack,
VAE scale/shift và dtype có provenance. Decode chunk=1 và hai scorer được dùng cùng
đường code cho preview/final; score thấp không bị loại. Nonfinite có failure/null.

Mỗi task ghi vào staging riêng, fsync, checksum toàn bộ, atomic rename và COMPLETE.
Resume chỉ skip khi contract/hash khớp. Infrastructure retry tối đa hai lần ngoài
attempt đầu, cùng task/seed. Contract/corruption không retry như lỗi infrastructure.
Source/runtime khác dùng namespace mới hoặc requalify. Không xóa staging lỗi tự động.

Backup root phải ngoài primary tree. Hash-copy verification không chứng minh hai
root ở hai storage failure domains: operator phải cấp đích backup độc lập thật.
Qualification có copy/restore drill trên cùng Volume; chưa là backup độc lập.

```bash
python -m apsp.lite.validate /shared/apsp-lite/release/tasks/TASK_ID
python -m apsp.lite.report --tasks /shared/apsp-lite/tasks-r1.jsonl \
  --root /shared/apsp-lite/release --backup-root /backup/apsp-lite/release \
  --out /shared/apsp-lite/LITE_DATA_REPORT.json
```

Collector timings tách generation/decode/IR/PickScore/artifact dump. Chúng **không**
thay deployment cost profile với một active verifier. Đo den(k), ver(k), fin(k),
prune/setup/dispatch trên B200 trước khi freeze budget. Không suy ETA từ số diffusion
steps hoặc đánh đồng đủ record với đủ valid.

## Accessor, grammar và giới hạn

`PaidAccessor` chỉ trả current/alive/active bought scores; FINAL bị khóa tới J.
Environment vẫn phải charge ledger trước khi gọi purchase API. Python object API
là contract chống leakage khi lập trình, không phải sandbox bảo mật chống code độc.
`Grammar` triển khai DP/mask riêng `G-842-optional-lite1`, 0/1/2 prune, fin(8/4/2),
hai prune khác checkpoint, CONTINUE theo đúng segment length. Chưa nối Lite bank
vào trainer v1. Không train PPO/REINFORCE hoặc mở test trong data job.

Chưa đủ cho DONE dữ liệu: corpus/source/QA thật, measured cost primitives, 50-prompt
replay-online drift study, full collection, independent backup, held-out Hyper-SD15
contract và node-specific verification cần hoàn tất riêng. Báo cáo preparation
hiện tại có tại `LITE_DATA_REPORT.md`; release manifest không đánh dấu dataset DONE.

Lượt build/qualification dùng profile `billionaireproject8`, usage snapshot ghi nhận
1.56326821 USD so với cap 7 USD. Mọi app của lượt này đã stopped; Volume được giữ
private. Snapshot billing có thể trễ, không phải số dư credit xác nhận hay invoice
cuối. Không tự chạy full bank: ngoại suy một prompt từ timings đã đo cho riêng
generation đã khoảng 162 USD cho full bank, chưa decode/scoring/CPU/RAM/storage.
Đây chỉ là budget screen, không thay forecast 20 task/profile hoặc cam kết chi phí.
