# APSP Lite v1 — runtime và data preparation

Ngày bàn giao: 20/10/2026 (Asia/Saigon). **Hoàn tất runtime riêng, collector contracts
và text/task preparation; chưa hoàn tất dataset L0–L6 của plan.** Chưa train
controller, chưa chạy ảnh/chấm điểm test500, chưa chạy trên node công ty.

## Quyết định runtime

Dựng environment riêng từ dependency lock B200 của bundle396b313; không nâng đè
runtime TTS. TTS phục vụ Static BFS/SD1.5 và stack Python3.10/Transformers4.38.2 cũ;
Lite dùng Python3.11.6/Torch2.9.1+cu128/Diffusers0.39.0/Transformers4.57.6 và thêm
PEFT0.17.1. Tái sử dụng cách đóng gói, checksum, restore và cache đúng hash.
Chi tiết ở [RUNTIME_DECISION.md](RUNTIME_DECISION.md).

ZIP gốc có SHA256 `21f148b678db3dda8df06c04123cd681e6765d59c466fbfe2b8cfb004273b123`;
30/30 SHA256SUMS entries khớp. Git base396b3132eeb35bf7c98e0e4b1a851bf88268cc1b chỉ
có lock check, không coi README cũ là bằng chứng clean install/B200.

## Đã kiểm chứng

- 102 local regression tests PASS: contracts, budget DP, replay leakage và Lite.
  Local không có Torch; không dùng kết quả này thay remote GPU qualification.
- Remote CPU import/BLIP-BERT forward và Euler/Flow clean-prediction checks PASS.
- B200 thật: Lightning8, Hyper-FLUX8, SDXL50 chạy một prompt riêng × tám seed/profile;
  24/24 final valid và 256 preview rows, hai scorer ImageReward/PickScore PASS.
  Denoiser calls bằng8/8/50 theo profile; final saved latent re-decode pixel diff=0.
- Portable dependency archive được restore/import trên CPU và chạy lại cả ba
  profile/scorers trên Ubuntu22.04/glibc2.35, B200 khác container: PASS functional.
  CPU restore được đánh dấu riêng, không gọi là GPU qualification.
- FLUX và Lightning có8/8 PNG giống SHA giữa build/restore; SDXL50 có0/8.
  SDXL50 max mean channel diff=0.2115631103515625 trên0–255, max channel diff=100;
  max IR delta=0.04556381702423096, max raw PickScore delta=0.04733085632324219.
  **Chưa xác định nguyên nhân, chưa có bitwise parity SDXL50.** Cần freeze môi
  trường collection và làm drift/native-batching qualification trước mass run.

Scope là một prompt calibration authored riêng, không phải study50 prompt,
bank khoa học hoặc kết quả controller. Xem [compatibility matrix](MODEL_SCORER_COMPATIBILITY.md)
và raw receipts trong `reports/lite/`.

## Text và task manifests

Đã tạo2850 UID: train2000, dev300, calibration50, testID300, Parti source-shift200.
Quota train/dev/calibration/testID đúng50% DiffusionDB,20% COCO,30% structured;
structured train có120 row/nhóm chính và100 template families. Group overlap=0 theo
UID/parent/image/template và lexical3gram Jaccard>=0.85 detector đã khóa. Nested
250⊂1000⊂2000 giữ quota; không khẳng định semantic decontamination hoàn toàn.

13.800 train/dev tasks gồm ba profile × hai bank ×2300 prompt, tám seed/task.
**Actual train/dev candidates=0.** Kỳ vọng110.400 candidate,1.177.600 preview,
2.576.000 score values không phải số record đã thu. 500 test chỉ có text manifest.

Đã kiểm checksum nguồn/manifests, UID, quota, split/group, nested subsets, task
reconstruction,36.800 distinct seed identities và36.800 unique seed values. Cùng
seed được cố ý chia sẻ giữa profile. Full chưa freeze nên Lite là manifest độc lập.

DiffusionDB dùng metadata thật, COCO captions chính thức, Parti/GenEval pin commit;
không tải ảnh nguồn. Manual language/content QA và tokenizer truncation còn pending.
DiffusionDB candidate lấy theo thứ tự metadata, có ASCII heuristic; chưa gọi là
random đại diện corpus hoặc language QA. Exclusion dùng553GenEval,1632Parti,
370 v1 text khả dụng và prompt qualification;220 v1 row thiếu text còn là gap.
Chi tiết nguồn/licence/provenance ở [dataset card](DATASET_CARD_LITE.md).

## Ngân sách và lý do chưa chạy full collection

Profile `billionaireproject8` vượt qua billing guard trước launch. Usage snapshot
2026-10-19T17:01:59Z ghi **1.56326821 USD** cho mọi app `apsp-lite-*`, cap task7USD.
Workspace usage gồm job cũ là1.77478443USD; không lấy nó làm riêng chi phí task.
Đây là usage API có thể trễ, không phải invoice cuối/số dư credit xác nhận.
Mọi app của task đã stopped; private Volume giữ artifact/weights/runtime.

Budget screen dùng timings có checksum của một task/profile ở hai container.
Lấy mẫu nhanh hơn/profile và nhân4600 task/profile cho full bank: generation
riêng khoảng162.22USD GPU; các stage collector được đo cộng lại khoảng536.97USD
GPU. Wave250 train prompt generation riêng khoảng17.63USD, đã vượt cap7USD.
Đây là **ngoại suy một prompt, không phải forecast đã nghiệm thu**, không gồm
load, CPU/RAM/storage, backup/retry hoặc biến thiên prompt. Giá B2000.001736USD/s
được kiểm tại [Modal pricing](https://modal.com/pricing). Không tự tăng ngân sách.
Raw timings và phép tính ở `reports/lite/COLLECTION_COST_SCREEN.json`.

## Trạng thái plan và phần còn thiếu

| Task | Trạng thái thực tế |
|---|---|
| L0 | Audit provenance, namespace, gap và model registry xong |
| L1 | Text/task manifests xong; manual language/content và truncation QA pending |
| L2 | Locked remote install và ba model adapters/scorers trên B200 PASS |
| L3 | Collector/validator variableJ, real production8seed task/profile, latent/scorer/restore checks PASS; drift limitation ở trên |
| L4 | Task manifest và raw collector timings có;20task/profile throughput/storage forecast và active-verifier deployment cost primitives chưa có |
| L5 | Resident launcher, deterministic task partition, retries, atomic completion/hash/resume/backup primitives có; full collection chưa chạy; auto QA-gated wave promotion chưa triển khai |
| L6 | Grammar version riêng và paid/no-leak API có tests; feature integration/trainer views đầy đủ, independent backup và cross-machine data restore còn pending |
| L7 | Controller training/online evaluation chưa chạy; không thuộc lượt dữ liệu này |

Hyper-SD15 held-out disabled tới khi freeze checkpoint/scheduler. Không tự tune
target-model quality. Copy/restore drill dùng hai root trên cùng Volume, chưa có
backup độc lập. Company node cần host/GPU/storage được chỉ định và gate thực.
Không chuyển kết quả one-prompt thành DONE checklist hoặc tuyên bố khoa học.

## Artifact và khôi phục

- Private Modal Volume `apsp-lite-runtime-v1`, profile `billionaireproject8`:
  `/export/runtime.tar.gz` và `/export/manifest.json`; model/scorer cache ở `/hf`.
- Archive4,571,656,042 bytes; SHA256
  `d944669e3211621416e5e017fdc010ea04d8726a89ec35cea2eb30d686bfda84`.
  Chứa Python+dependencies; không chứa source, model weights hoặc credential.
- Qualified core source SHA256
  `7346430ea5f5a1bcfdd65d517c537dd82dd553dabce9ac125276d178ccdf6023`.
  Modal overlay không có.git nên `source_commit=null`; exact hash đối chiếu local.
- Text/task local `data/lite-v1/prepared/manifests/`, remote
  `/text/lite-v1-20261019/manifests/`; handoff ZIP chứa text manifests và source.
- [RUNBOOK_LITE.md](RUNBOOK_LITE.md) có direct-download và idempotent restore commands;
  [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) khóa hashes, receipt scope và pending.

Branch `feat/apsp-lite-data-v1`; commits local đã ký. Origin là git bundle local,
chưa có GitHub backup/push cho triển khai này. Gói bàn giao nhẹ trong `outputs/`
chứa source archive, git bundle, docs/receipts và text/task manifests; runtime lớn
giữ tại Modal, tải trực tiếp tới node được cấp quyền theo runbook.
