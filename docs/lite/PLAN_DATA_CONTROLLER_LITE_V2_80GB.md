# Adaptive PSP — Plan dữ liệu Lite v2, bàn giao tối đa 80 GB

**Ngày:** 21/10/2026. **Đối tượng:** Codex/Claude Code chuẩn bị code; người phụ trách B200 chạy inference và bàn giao dữ liệu.

**Trạng thái:** đặc tả để triển khai. Chưa có tuyên bố adapter, env B200 hay collector mới đã đạt nghiệm thu. Các tên config/API trong tài liệu là yêu cầu cần ánh xạ vào repo thật, không phải CLI đã tồn tại.

## 1. Quyết định chính

- Hai profile: **Hyper-SD15 8 bước, 512²** và **SDXL-Lightning 8 bước, 1024²**.
- **2.000 prompt train, 200 prompt dev**, cùng danh sách prompt cho hai profile.
- Train có **1 bank/prompt/profile**; dev có **2 bank/prompt/profile**; mỗi bank có **8 seed**.
- Tổng lượt thu chính: **4.800 task, 38.400 candidate trajectory**; mỗi trajectory chạy đủ tám bước, không prune khi collection.
- Thu ImageReward và PickScore cho cả tám preview và ảnh cuối; giữ đặc trưng cần cho replay controller MLP.
- Xuất **Core trước**, latent và ảnh QA thành gói độc lập. Tổng bản bàn giao đích **≤80 GB**, chừa khoảng trống so với khả năng nhận 100 GB của người dùng. Đây là dung lượng dữ liệu xuất, không phải trần tài chính.
- Nếu chỉ nhận được 40 GB: nhận Core + QA trước; không cần giảm prompt để train MLP. Gói latent có thể nhận sau hoặc dùng trực tiếp từ storage được phép truy cập.
- Chuẩn bị manifest **500 test prompt**; chưa sinh/chấm test trong nhiệm vụ làm train/dev.
- Không tự Stop/Destroy máy, không xóa dữ liệu cũ, không restart workload đang chạy chỉ để đổi cách đóng gói.

Lite v2 thay phần workload/storage của Lite v1 trong lượt thu mới. Giữ dữ liệu, kết quả và config v1/Full nguyên trạng, dùng namespace mới `apsp-lite2-fast2`.

## 2. Model và cấu hình phải khóa

| Profile | Base và trọng số bổ sung | Resolution | Recipe khởi đầu cần qualification |
|---|---|---:|---|
| `hyper_sd15_cfg8_512` | SD1.5 mirror của dự án + `ByteDance/Hyper-SD/Hyper-SD15-8steps-CFG-lora.safetensors` | 512×512 | T=8; CFG=7,5; DDIM trailing; eta=0; LoRA scale=1,0 |
| `sdxl_lightning8_1024` | `stabilityai/stable-diffusion-xl-base-1.0` + `ByteDance/SDXL-Lightning/sdxl_lightning_8step_unet.safetensors` | 1024×1024 | T=8; guidance=0; EulerDiscrete trailing; s_churn=0 |

SDXL-Lightning yêu cầu checkpoint khớp số bước và sampler tương ứng [1]. Hyper-SD có bản SD15 8-step CFG-preserved hỗ trợ guidance 5–8 [2]. CFG=7,5 và recipe DDIM nêu trên là lựa chọn study cần xác nhận tích hợp; không nhầm checkpoint CFG-preserved với bản 8-step không CFG hoặc unified LoRA. Không dùng scale=0,125 của Hyper-FLUX cho SD15 chỉ vì cùng repo.

Codex pin chính xác revision/hash của base, LoRA/UNet, VAE, tokenizer, scheduler và hai scorer. Negative prompt rỗng, không prompt expansion. Giữ native VAE; nếu pipeline yêu cầu upcast thì thực hiện và ghi lại. Không tự đổi VAE để tăng tốc rồi trộn dữ liệu vào cùng profile.

Precision UNet, VAE, preview và attention backend phải được qualification rồi khóa. Lưu tensor theo **dtype thực tế**, không ép fp32/bf16 sang fp16 khi ghi file. Ví dụ Euler có phép tính upcast fp32 [3].

Env B200 tái sử dụng branch/bundle hiện có sau khi audit trạng thái thật. `uv lock --check` chỉ chứng minh lock hợp lệ; phải có `uv sync --locked`, import và chạy model/scorer thật trên B200. Thêm PEFT/adapter nếu thiếu, regenerate uv.lock khi dependency đổi. Người sinh data không phải tự thiết kế hay chọn phiên bản thư viện.

## 3. Số lượng chính xác và cách chia prompt

Một **task** = một prompt × một profile × một bank, gồm tám candidate chạy đủ quỹ đạo. Candidate không phải một rollout controller.

| Split | Prompt UID riêng biệt | Bank/profile | Task mỗi profile | Candidate mỗi profile | Candidate hai profile |
|---|---:|---:|---:|---:|---:|
| Train | 2.000 | 1 | 2.000 | 16.000 | 32.000 |
| Dev | 200 | 2 | 400 | 3.200 | 6.400 |
| **Tổng thu chính** | **2.200** | | **2.400** | **19.200** | **38.400** |

Tổng hai profile: **4.800 task**. Preview rows = 38.400×8 = **307.200**. Final rows = **38.400**. Hai scorer ở preview và final: 2×(307.200+38.400) = **691.200 score values dự kiến**. Lỗi phải có validity/null tương ứng; không ghi tất cả là score valid khi có failure.

### Nguồn prompt

| Nguồn | Train | Dev | Test ID, giữ kín |
|---|---:|---:|---:|
| DiffusionDB | 1.000 | 100 | 150 |
| COCO captions | 400 | 40 | 60 |
| Prompt cấu trúc tự tạo | 600 | 60 | 90 |
| **Tổng** | **2.000** | **200** | **300** |

Test có thêm **200 PartiPrompts**, thành 500 prompt. Báo tên `PartiPrompts-200`, không gọi là toàn benchmark [4]. Thêm 50 prompt calibration riêng; QA/profiling dùng split này và báo workload ngoài tổng 38.400.

Prompt cấu trúc gồm năm nhóm: thuộc tính; quan hệ nhiều vật thể; đếm/binding; vật liệu/ánh sáng/phong cách; nhiều ràng buộc. Giữ thêm nhãn domain, độ dài và độ khó để báo coverage. Không chọn prompt theo reward hay mức thuận lợi cho prune.

Gom exact/near duplicate và parent group trước split. Caption cùng COCO image_id cùng split; paraphrase/template family cùng split. Dedup với test/GenEval/PartiPrompts và manifest đánh giá v1 còn truy cập được. Ghi coverage thật của detector; không tuyên bố loại mọi tương đồng ngữ nghĩa.

Giữ UID và split giống nhau trên hai profile. UID lấy từ SHA256 của text đã NFC/whitespace normalize; lưu text_raw và quy tắc normalize. Seed scheme giữ `apsp-v2-sha256-cpu` đã định nghĩa nếu tương thích; nếu cần sửa phải version mới, không đổi ngầm. Seed không phụ thuộc rank, số worker hay thứ tự task.

Tạo subset train lồng nhau **250 ⊂ 1.000 ⊂ 2.000**, có phân tầng và cùng UID trên hai profile. Một bank train là đánh đổi để ưu tiên nhiều prompt; không khẳng định nó đã đủ cho mọi biến thiên seed.

## 4. Ba lớp dữ liệu và giới hạn xuất

### 4.1 Core — bắt buộc, đủ cho replay MLP theo observation đã chốt

Giữ đầy đủ cho mọi candidate:

- Prompt, source/split/group/nhãn; common prompt embedding; profile/environment context.
- CandidateId = (profile_id, prompt_uid, bank_id, seed_idx), seed_global và initial-latent hash.
- Từng checkpoint: forward_idx, timestep/sigma, preview IR, raw PickScore trước softmax, validity/failure code.
- Final: IR/PickScore riêng cho ảnh giao; không thay bằng preview cuối; delivered-pixel hash.
- Các feature thực sự dùng: progress, drift/cfg_gap nếu có, availability mask, metadata định nghĩa feature. Budget, alive set và lịch sử paid score do env/controller dựng khi rollout.
- Cost table, raw profile samples, model/env/scorer hashes, uv.lock hash, runtime fingerprint, sampler config và checksum.

Common prompt encoder là frozen `sentence-transformers/all-MiniLM-L6-v2`, output 384 chiều theo bản Full: pin revision, attention-mask mean pooling, L2 normalization và log truncation. Không ghép native CLIP embedding SD15/SDXL như thể cùng một feature space.

Core không cần đủ ảnh/latent để train MLP này. Nhưng **không đủ để học controller đọc latent/ảnh**, không hỗ trợ tùy ý thêm feature chưa lưu hay thay reward model mà không decode/chấm lại. Core có thể dùng trainer REINFORCE hiện có sau khi adapt schema; không tự nhận tương thích chỉ vì số cột gần giống.

### 4.2 Latent — phần mở rộng

Giữ preview latent tại cả tám checkpoint và final latent thực sự dùng cho decode, bằng safetensors, nguyên dtype và shape; có scaling/shift/packing/decode metadata [5]. Không cần nhân đôi final latent nếu cùng tensor đã tồn tại: dùng tham chiếu có kiểm chứng, không mặc định preview@T = delivered latent.

Không bắt buộc xuất toàn bộ initial latent: seed, RNG version và hash đủ làm danh tính; giữ initial tensors cho QA subset và kiểm tái sinh. Native kernel/runtime đổi có thể làm ảnh decode khác; latent không bảo đảm tái tạo byte-identical PNG trên mọi máy. Ghi pixel hashes và re-decode evidence theo runtime đã khóa.

### 4.3 QA images — chọn trước, không chọn theo điểm

Chọn 50 calibration prompt × hai profile × một bank × tám seed = **800 final PNG** làm bộ QA cơ sở, ngoài lượt train/dev. Chọn trước 10 trong 50 prompt đó để lưu preview PNG tại j={1,4,8}, tổng **480 preview PNG**. Giữ scalar/scoring evidence và sample IDs.

Không xuất PNG cho mọi preview train/dev. Ảnh cuối train/dev ngoài QA giữ ở shared nếu được phép và có độ bền đã xác nhận; chúng không bắt buộc nằm trong gói Core. Không gọi QA subset là đánh giá chính. Nếu shared không bền vững, bàn giao Core/latent/QA đã kiểm trước khi mất máy; không coi đường dẫn trên máy là bản sao lưu.

### 4.4 Budget dung lượng

Tính GB theo 10^9 byte, báo thêm GiB. Ngân sách phân bổ, không phải dung lượng đã đo:

| Gói | Budget xuất |
|---|---:|
| Core | ≤5 GB |
| Latent train/dev | ≤65 GB |
| QA images/evidence | ≤5 GB |
| Overhead đóng gói/dự phòng | 5 GB |
| **Tổng đích** | **≤80 GB** |

Ước tính raw latent train/dev cho hai profile, mỗi profile 19.200 candidate:

| Tensor | Tất cả 16-bit | Tất cả fp32 |
|---|---:|---:|
| SD15 preview: [19.200,8,4,64,64] | 5,03 GB | 10,07 GB |
| SDXL preview: [19.200,8,4,128,128] | 20,13 GB | 40,27 GB |
| Final latent hai profile, lưu riêng nếu cần | 3,15 GB | 6,29 GB |
| **Tổng raw** | **28,31 GB** | **56,62 GB** |

Nếu dtype pha trộn thì tính theo từng tensor. Chưa gồm QA, metadata và file overhead. Forecast bằng `numel × element_size` cùng size file thực; không dựa vào giả định archive sẽ nén mạnh tensor.

Không đưa model weights, HF cache, venv, staging cũ, bản backup trùng, optimizer/controller checkpoint vào dataset release. Không hứa storage làm việc trên B200 cũng chỉ cần 80 GB: working storage phải chứa weights/cache và staging riêng.

Trước bulk, đo **20 calibration task/profile** = 10 prompt × hai bank × tám seed, đã chứa một bank của QA subset. Báo bytes/task riêng từng loại file, max dtype/shape và dự báo full. Không áp dụng tỷ lệ 28 MB/task của plan ba model cũ cho hai model mới.

Nếu xuất đủ bộ QA50 và qualification nêu trên, hợp của chúng có 120 calibration task/960 candidate: 100 task cho QA50 một bank trên hai profile, cộng 20 task bank thứ hai của 10 prompt qualification. Reuse phần giao nhau, không chạy hai lần; profiling/parity có thể thêm lượt đo và báo riêng. Các lượt calibration này không nằm trong 4.800 task train/dev.

Nếu forecast vượt 80 GB: tiếp tục giữ Core và scalar collection, tách latent thành optional release/để ở storage durable; không cắt prompt theo reward, không cast tensor âm thầm. Nếu bắt buộc mọi raw tensor đều phải tải về mà không vừa, báo lựa chọn giảm dữ liệu cụ thể trước khi sửa manifest. Không xóa raw để tạo cảm giác vừa cap.

Mức nhận **40 GB** mặc định là `core + qa`; latent là phần nhận thêm. Giới hạn tính trên tổng file nhận, không chỉ dung lượng một archive đã nén.

## 5. Collector phải làm gì?

1. Nạp model/scorer đã pin, giữ resident; nhận task từ manifest, không tự đổi prompt/seed.
2. Sinh tám candidate k=8 đầy đủ; không có controller/prune trong job làm bank.
3. Sau mỗi forward j=1…8, lấy clean prediction từ kết quả vừa tính, không thêm denoiser forward chỉ để lấy x0. Ghi cả j và actual timestep/sigma.
4. Decode theo recipe đã khóa, kiểm finite trước clamp; chuẩn hóa pixel bằng cùng một đường cho collection và online.
5. Chấm IR và PickScore trên đúng pixel đó, ghi scalar/validity và hash. PickScore lưu raw pre-softmax.
6. Tính/lưu feature đã đăng ký. Lightning guidance=0 không có hai nhánh CFG; cfg_gap thiếu thì ghi availability=false, không tạo số giả. Hyper-SD15 có CFG phải ánh xạ đúng hàng uncond/text.
7. Ghi final latent/score riêng. Decoder/scorer có batch/chunk cố định đã qualification; log batch composition và runtime fingerprint đủ audit.
8. Atomic commit shard + checksum + completion marker sau validation; task đang ghi không được đánh complete. Worker lỗi infrastructure retry cùng ID/seed tối đa hai lần rồi báo. Model/scorer failure giữ nguyên record và nguyên nhân; không đổi seed để thay ảnh xấu/lỗi.

Một scorer lỗi không có nghĩa generator lỗi. Compatibility matrix tách generator, preview, decode, IR, PickScore, serialization và restore. Task không coi complete khi scorer bắt buộc còn thiếu, dù generator PASS.

ImageReward là verifier/reward training chính. PickScore là đánh giá phụ, không chọn prompt hoặc checkpoint policy chính theo PickScore. Một episode deployment chỉ trả phí scorer nó thực sự gọi; chi phí thu hai scorer không gán thành phí VERIFY bắt buộc dùng cả hai.

## 6. Schema tối thiểu và quyền đọc

Xuất version `apsp-core-lite2-v1`. Reuse schema đa model của repo khi tương thích, không sửa nghĩa bank v1. Những bảng sau là logical schema; Codex ánh xạ vào implementation và viết validator.

| Thành phần | Nội dung chính |
|---|---|
| release_manifest.json | schema/version, planned/complete/failed counts, profiles, hashes, retention flags, total bytes |
| prompts.parquet | uid, text/raw, source, split, group, tags, truncation |
| prompt_features.safetensors + index | common embedding 384 chiều và uid → row |
| candidates.parquet | CandidateId, seed, init hash, validity |
| previews.parquet | CandidateId, j=1…8, forward_idx, t/sigma, raw scores, feature values/masks, validity |
| delivered.parquet | CandidateId, final scores/flags, pixel hash, image_uri nullable, final_latent_uri nullable |
| tasks.parquet | planned/status/failure/retries/provenance và durable ack |
| provenance/ | env/model/scheduler/scorer lock, cost tables/raw samples, compatibility/evidence |
| latent extension | task-level safetensors + index/hash/decode recipe |

Validator kiểm cardinality, unique keys, liên tục checkpoint, foreign key, split consistency, finite flags, file hashes và URI nullable theo retention. `complete_valid`, `complete_with_model_failure`, `failed_infra`, `pending` phải phân biệt; đủ record không đồng nghĩa đủ ảnh valid.

Actor chỉ đọc paid score ở checkpoint đã VERIFY, feature hiện tại được phép dùng, prompt/environment context và trạng thái controller. Không đọc terminal score sớm, future checkpoint, candidate đã prune, scorer inactive hoặc hidden test labels. Có no-leak test bằng sentinel/proxy accessor. Không lộ split/source ID như shortcut cho quality.

Một Core release hợp lệ phải đủ chạy **fixed, threshold và random masked policy** trong replay mà không cần mở latent/PNG. Test trên dữ liệu thật, không chỉ fixture. Replay vẫn là mô phỏng xấp xỉ, không phải môi trường online chính xác.

## 7. Phương pháp downstream mà dữ liệu phải hỗ trợ

### Grammar và chi phí

Đề xuất giữ `G-842-optional-lite1`: 0/1/2 prune, 8→4→2; prune cần fresh paid score; không hai prune cùng checkpoint; chỉ prune ở j=1…7. CONTINUE đi một checkpoint, tức một forward. FINAL tại j=8 có phí decode/chọn/giao với k=8/4/2. Extra VERIFY-only mua thông tin nhưng không tự đổi latent.

Có 1+7+21 = **29 lịch prune cố định** để tune/oracle replay. Đúng-hai-prune có 21 lịch, không phải 105 của v1 16 checkpoint. Version/tests DP riêng; không gán nhãn grammar cũ cho behavior mới. Người làm dữ liệu không phải tự implement RL/DP nếu phần đó do Codex khác sở hữu.

Profile thật trên B200 cho den(k), ver(k), fin(k), prune và optional feature, với k={1,2,4,8}; tính cả setup/prompt encoding và controller dispatch khi đo deployment. Cost controller chưa có artifact được ghi provisional, không bịa số. I/O xuất dataset/model loading/warmup tách riêng khỏi latency request.

Budget recipe mới dùng reference ledger BoN8: beta={0,5;0,75;1,0}; mọi method so tại cùng cap theo profile. Report cap infeasible thay vì nâng riêng cho RL. Chi phí phụ thuộc measurement hardware; wall-clock trên B200 không suy ra latency 4090.

### Training/evaluation kế tiếp — không nằm trong job sinh data

- Train controller trên subset 250/1k/2k, cùng architecture và protocol update; báo learning curves và kiểm hội tụ. Normalizer/PCA chỉ fit đúng train subset.
- Tối thiểu ba policy training seeds; so shared policy với specialist nếu muốn hiểu tác động trộn hai model.
- Baselines: single, BoN2/4/8, lịch tham chiếu (2,4), best-fixed 29 lịch, threshold và one-query selector. Tune lại bằng dữ liệu được cấp công bằng; giữ masked random làm sanity check.
- Dev dùng chọn cấu hình; tuning cuối và mọi kết luận xác nhận dùng online sampler thật. Batch thay đổi sau prune có thể khác quỹ đạo bank; kiểm discrepancy trên calibration, không gọi oracle replay là cận trên online.
- Main train reward IR; báo PickScore của ảnh giao, quality difference với CI, mean/p50/p95 e2e latency, saving và failure rate. Không diễn giải IR thành phần trăm chất lượng ảnh.
- Test500 chỉ mở sau freeze policy/baseline/cap/quality margin; hai bank/profile cho online evaluation. 500 prompt là 500 cụm độc lập, không coi bank/method repeats là các prompt độc lập. Bootstrap theo prompt, giữ pairing và report training-seed variability riêng.
- Lưu đầy đủ ảnh giao của online test để audit/metric khác; storage test có dự toán riêng, **không nằm trong 80 GB train/dev release này**.

Phạm vi chính là prompt mới trên hai diffusion profile 8 bước đã thấy. Chưa chứng minh flow/many-step/unseen-model transfer. Nếu làm cross-model ablation A→B, policy/norm/hyperparameter không được dùng quality của B để tune; ghi rõ đó là ablation riêng. Không warm-start policy v1 SD1.5 rồi gọi SD1.5 chưa từng thấy.

## 8. Triển khai theo đợt, không mất công đã chạy

### Đợt 0 — kiểm code và 20 task/profile

Codex audit env/collector mới nhất thực tế, thêm hai adapter và export modes, rồi người chạy B200 chạy qualification trên calibration. Xuất INSTALL/IMPORT/GPU/GEN/DECODE/IR/PICK/SAVE/RESTORE PASS/FAIL riêng. Đo storage/throughput từ 20 task mỗi profile và kiểm restore ở process khác. Một adapter PASS có thể tiếp tục thu trong khi sửa adapter khác; lỗi chung scorer phải xử lý trước khi coi shard complete.

### Đợt 1 — train1k và dev200

Thu một bank cho 1.000 prompt train trên hai profile: **2.000 task/16.000 candidate**. Thu dev200 hai bank: **800 task/6.400 candidate**. Tổng đợt này **2.800 task/22.400 candidate**, ngoài calibration. Subset250 nằm bên trong train1k, không sinh lại.

### Đợt 2 — bổ sung lên train2k

Thêm 1.000 train prompt còn lại: **2.000 task/16.000 candidate**. Tổng lũy kế **4.800 task/38.400 candidate**. Sau QA dữ liệu thành công, tự tiếp tục đợt sau; không cần xin duyệt lại từng đợt và không chờ phải train controller trước. Các chẩn đoán reward/headroom không dùng để chọn bỏ prompt bất lợi.

### Nếu plan cũ đang chạy

Không kill/restart job giữa task, không xóa hay đổi nhãn output cũ. Xuất inventory gồm profile/hash/prompt/bank/checkpoint/status/bytes. Chỉ tái sử dụng record khớp hoàn toàn recipe và provenance mới; tên model giống nhau không đủ. Chọn subset UID/dev200 trước khi nhìn quality, giữ bank0 theo manifest. Bank1 train dư có thể lưu reserve, không biến thành unseen-prompt test. FLUX/Base50 không thuộc Lite2 nhưng là dữ liệu bổ sung hợp lệ nếu chạy đúng recipe; không ghi chúng như Hyper-SD15/Lightning mới.

Nếu không có record tương thích, thu task mới trong namespace riêng. Khi đổi plan, hoàn tất task đang ghi trước, rồi chuyển queue; báo rõ phần đã reuse và phần sinh mới. Không ép dừng toàn bộ chỉ vì export thay đổi.

## 9. Chạy độc lập, sao lưu và đóng gói

- Dùng scheduler hoặc worker độc lập phiên chat; tối thiểu một worker/profile nếu có hai GPU. Không bắt buộc dùng hết 10–20 B200.
- Worker giữ model resident, không load lại mỗi task. Thêm worker theo GPU-seconds còn lại đo được; không chia ETA theo marketing B200 hay giả định step ratio.
- Ghi tasks done/failed, tasks/phút/worker, bytes từng gói, heartbeat thật và raw cost. Tách thời gian denoise/decode/scoring/I/O để biết chậm ở đâu. Không tạo tải giả để né idle policy của mentor.
- Shared storage chỉ coi durable khi quyền lưu/retention đã xác nhận. Copy tăng dần những shard đã atomic complete; không chờ sinh xong toàn bộ mới backup.
- Core, latent và QA có checksum riêng và dependency cùng release_id. Người nhận kiểm hash và restore subset trước khi xác nhận nhận đủ. Không dùng số file làm bằng chứng duy nhất.
- Có thể chia part 1–2 GB, tránh một file hàng chục GB. Giữ một bản dữ liệu chuẩn; nếu archive làm tăng working disk gấp đôi phải forecast staging trước. Không coi nén chia part là giảm tổng dung lượng.
- Không đưa credential/token vào config/log/release; dùng tài khoản/đường dẫn cache theo quyền cụm.
- Không có đường tự gọi provider Stop/Destroy trong launcher thu data; không xóa dữ liệu cũ. Job lỗi phải báo và làm tiếp task độc lập an toàn, không báo đã xong toàn bộ.

## 10. Gói bàn giao và trách nhiệm

**Codex/Claude Code:** adapters, uv env/lock, prompt pool/splits, manifest/schema, collector, feature export, validator/accessor, profiler, storage forecast, resume/export/restore và docs. Có quyền giải quyết routine engineering trong scope; không tự sửa scientific recipe giữa bulk.

**Người chạy B200:** cài đúng env, chạy qualification, thu task theo manifest, theo dõi lỗi/throughput, export và chuyển gói verified. Không cần implement trainer, chọn thuật toán RL hoặc tự đoán những field khoa học còn thiếu.

Bàn giao tối thiểu:

1. Core release đủ record, raw scores/flags/features và checksum.
2. Latent extension nếu nằm trong mức nhận đã chọn; nếu không xuất phải có retention report/location/size/expiry, không giả là người nhận đã có.
3. QA package và compatibility/round-trip evidence.
4. `DATASET_CARD_LITE2.md`: source/license refs, splits, profiles, validity, limitation/retention và uses được phép theo license gốc.
5. `COLLECTION_REPORT_LITE2.md`: planned/complete/failed/missing theo profile/split, dtype/shape/bytes, số đo B200 thật và discrepancy findings.
6. `RELEASE_MANIFEST.json`, `SHA256SUMS`, env/model/scorer locks, raw cost samples.
7. `RUNBOOK_LITE2.md`: lệnh thật đã chạy, resume/export/restore và vị trí dữ liệu. Không ghi CLI tưởng tượng.

### Nghiệm thu DONE

- Đúng 2.000 train/200 dev UID, 4.800 task terminal status, 38.400 candidate record; failed/pending báo riêng. Không che thiếu data bằng tổng số planned.
- Preview/final/scorer counts khớp hoặc có failure flags/null giải thích được; score thấp không bị loại.
- Env và hai profile/hai scorer đã chạy thật B200; safetensors bảo toàn dtype và shape.
- Core chạy replay baseline trên data thật không cần ảnh/latent; no-leak tests đạt.
- Gói xuất đo thật ≤80 GB, hoặc chế độ 40 GB `core+qa` với latent extension ghi chưa nhận rõ ràng.
- Hash/restore giữa hai process hoặc máy đạt; người nhận biết phần nào đã có và phần nào còn trên shared.
- Không claim controller thắng/generalize hoặc test xong chỉ vì collection xong.

## 11. Config để Codex ánh xạ vào implementation

```yaml
dataset_name: apsp-lite2-fast2
schema_version: apsp-core-lite2-v1
seed_namespace: apsp-v2-sha256-cpu
train_prompts: 2000
dev_prompts: 200
calibration_prompts: 50
test_prompts: {id: 300, parti_subset: 200}
train_sources: {diffusiondb: 1000, coco_captions: 400, structured: 600}
dev_sources: {diffusiondb: 100, coco_captions: 40, structured: 60}
profiles:
  - {id: hyper_sd15_cfg8_512, steps: 8, checkpoints: 8, height: 512, width: 512}
  - {id: sdxl_lightning8_1024, steps: 8, checkpoints: 8, height: 1024, width: 1024}
banks: {train: 1, dev: 2, test: 2}
candidates_per_bank: 8
train_nested_subsets: [250, 1000, 2000]
collect_verifiers: [imagereward_v1, pickscore_v1]
main_training_verifier: imagereward_v1
active_verifiers_per_online_episode: 1
common_prompt_encoder: sentence-transformers/all-MiniLM-L6-v2
qualify_tasks_per_profile: 20
save_preview_checkpoints: [1, 2, 3, 4, 5, 6, 7, 8]
preserve_tensor_dtype: true
expected_train_tasks: 4000
expected_dev_tasks: 800
expected_train_candidates: 32000
expected_dev_candidates: 6400
expected_preview_rows: 307200
expected_score_values: 691200
export:
  mode: core_latent_qa
  target_max_total_bytes: 80000000000
  fallback_40gb_mode: core_qa
  core_budget_bytes: 5000000000
  latent_budget_bytes: 65000000000
  qa_budget_bytes: 5000000000
  reserve_bytes: 5000000000
  include_model_weights: false
  include_venv: false
  include_all_preview_png: false
  include_all_train_dev_final_png: false
  tensor_cast_for_storage: false
  qa_final_png_expected: 800
  qa_preview_png_expected: 480
  test_artifacts_in_this_budget: false
allow_test_sampling_before_method_freeze: false
allow_controller_training_in_collection_job: false
automatic_provider_stop: false
automatic_provider_destroy: false
```

Không chạy bulk với revision/hash còn placeholder. Các định nghĩa feature, failure handling và prompt normalization phải nằm trong lock có version, không để mỗi worker tự quyết.

## 12. Prompt bàn giao nguyên văn cho Codex

Đọc PLAN_DATA_CONTROLLER_LITE_V2_80GB.md và triển khai namespace apsp-lite2-fast2. Mục tiêu là dữ liệu train controller, không phải train controller trong lượt này. Audit repo/env thật trước; giữ nguyên v1/Full và dữ liệu đang có. Chuẩn bị Hyper-SD15 CFG-preserved8 ở 512² và SDXL-Lightning8 ở 1024², 2.000 train/200 dev dùng chung, train một bank, dev hai bank, tám seed/bank: 4.800 task/38.400 candidate. Thu đủ tám checkpoint, IR + raw PickScore, feature và provenance; Core phải chạy replay MLP không cần latent/ảnh. Export Core trước, latent nguyên dtype và QA riêng, tổng mục tiêu ≤80 GB; mức nhận 40 GB chỉ Core+QA. Đo 20 calibration task/profile để forecast dung lượng thật; không hứa tỷ lệ nén, không cast tensor, không xuất toàn bộ PNG preview. Lưu dữ liệu trên storage có retention rõ, atomic shard/checksum/resume và kiểm restore. Nếu plan cũ đang chạy, để task hiện tại hoàn tất rồi chuyển queue; reuse chỉ record khớp profile/hash/split/seed, không xóa hoặc gán lại nhãn. Hai đợt train1k rồi bổ sung2k tự tiếp tục khi QA đạt. Hoàn tất adapter, uv lock, manifests, collector, validator/accessor, profiler, export/restore và runbook thật. Tách trạng thái generator/decode/scorer. Không mở test, không claim generalization, không tự Stop/Destroy máy. Bàn giao compatibility matrix và collection report dựa trên CUDA thật, không dừng ở synthetic test hoặc chỉ tạo YAML.

## 13. Nguồn chính đã đối chiếu

1. SDXL-Lightning official model card: https://huggingface.co/ByteDance/SDXL-Lightning
2. Hyper-SD official model card và SD15 CFG8 checkpoint: https://huggingface.co/ByteDance/Hyper-SD ; https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SD15-8steps-CFG-lora.safetensors
3. Diffusers Euler scheduler source (tham khảo dtype behavior; implementation phải pin version thật): https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py
4. PartiPrompts: https://github.com/google-research/parti
5. Safetensors API/docs: https://huggingface.co/docs/safetensors/en/index
6. Common encoder: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Các kích thước dataset, ngân sách lưu trữ, grammar và protocol ở đây là đề xuất của dự án; nguồn model không bảo đảm controller sẽ tiết kiệm hoặc tổng quát hóa thành công.
