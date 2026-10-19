# Adaptive PSP — Plan dữ liệu Lite v1: few-step + nhiều bước

**Ngày:** 19/10/2026. **Phạm vi bàn giao:** Codex implement và chuẩn bị dữ liệu; người phụ trách B200 chạy inference/collector.
**Trạng thái:** thiết kế đề xuất để triển khai; chưa khẳng định adapter/model mới đã chạy thành công.
**Quan hệ với bản Full:** đây là amendment Lite riêng, không ghi đè bank, kết quả hay scientific config v1/Full.

## 0. Chốt phương án

Thu bank trên **ba execution profile**: SDXL-Lightning 8 bước, FLUX.1-dev + Hyper-FLUX LoRA 8 bước, SDXL Base 50 bước. Dùng cùng **2.000 prompt train + 300 prompt dev**, hai bank/prompt/model, tám seed/bank.

Tạo manifest test giữ kín gồm **500 prompt mới**, nhưng chưa chạy test trong lượt thu train/dev. Thêm Hyper-SD15 8 bước làm profile giữ riêng cho kiểm tra chuyển model sau khi controller đã khóa.

Thiết kế này giảm khối lượng nhưng vẫn kiểm tra được: policy có học hơn lịch cố định không; có làm việc trên cả diffusion và flow không; có dùng được ở cả few-step và nhiều bước không; và có chuyển sang prompt/profile chưa thấy không.

Không có kích thước Lite nào tự bảo đảm một kết luận thống kê hoặc kết quả dương. Plan này nhằm tránh so sánh sai và đo giới hạn thực tế của controller đã chọn; chưa đủ để tuyên bố controller dùng chung mọi model/dataset.

## 1. Các model và cấu hình

| ID | Cách nạp | T | J checkpoint | Resolution | Guidance / scheduler | Vai trò |
|---|---|---:|---:|---|---|---|
| sdxl_lightning8 | SDXL Base + `ByteDance/SDXL-Lightning`, file `sdxl_lightning_8step_unet.safetensors` | 8 | 8 | 1024² | guidance=0, EulerDiscrete, trailing | train/dev/test |
| hyper_flux8 | FLUX.1-dev + `ByteDance/Hyper-SD`, file `Hyper-FLUX.1-dev-8steps-lora.safetensors` | 8 | 8 | 1024² | guidance=3.5, fuse scale=0.125, native flow scheduler | train/dev/test |
| sdxl_base50 | `stabilityai/stable-diffusion-xl-base-1.0`, không refiner | 50 | 16 | 1024² | CFG=5.0, EulerDiscrete, native config | train/dev/test |
| hyper_sd15_8 | SD1.5 mirror + Hyper-SD bản SD15 8 bước, cấu hình checkpoint tương ứng | 8 | 8 | 512² | freeze từ hướng dẫn checkpoint khi tích hợp | chỉ test chuyển model |

Nguồn chính: SDXL-Lightning có checkpoint và sampler riêng cho số bước tương ứng [1]. Hyper-SD cung cấp adapter FLUX 8 bước và các bản SD1.5 [2]. SDXL Base là backbone nhiều bước giữ trong study [3].

Ba profile training đại diện **hai backbone chính**: SDXL và FLUX. SDXL Base/Lightning chia sẻ nguồn gốc; không gọi chúng là ba kiến trúc độc lập.

Hyper-SD15 test là một checkpoint/backbone thực thi chưa dùng để fit policy v2, không phải kiến trúc hoàn toàn mới. Khởi tạo controller v2 mới; không warm-start từ policy v1 đã học trên SD1.5 rồi gọi SD1.5 là hoàn toàn chưa thấy.

Pin base model, distilled weights/LoRA, tokenizer, scheduler và processor bằng revision/hash. Negative prompt rỗng, không prompt expansion. Giữ precision/runtime profile đã qualification, không đổi giữa shard.

## 2. Khối lượng thu ngay

Một task = một prompt × một model profile × một bank; task chứa tám candidate chạy đủ quỹ đạo, không prune khi collection.

| Profile | Prompt train | Prompt dev | Task train | Task dev | Candidate train | Candidate dev |
|---|---:|---:|---:|---:|---:|---:|
| SDXL-Lightning8 | 2.000 | 300 | 4.000 | 600 | 32.000 | 4.800 |
| Hyper-FLUX8 | 2.000 | 300 | 4.000 | 600 | 32.000 | 4.800 |
| SDXL Base50 | 2.000 | 300 | 4.000 | 600 | 32.000 | 4.800 |
| **Tổng** | **2.000 UID dùng chung** | **300 UID dùng chung** | **12.000** | **1.800** | **96.000** | **14.400** |

**Thu ngay: 13.800 task, 110.400 candidate trajectory/ảnh cuối dự kiến.** Thêm 50 prompt calibration riêng; collection/qualification trên chúng báo ngoài số lượng trên.

Preview rows train+dev:

- Lightning8: 36.800 × 8 = 294.400.
- Hyper-FLUX8: 36.800 × 8 = 294.400.
- SDXL Base50: 36.800 × 16 = 588.800.
- Tổng: 1.177.600 preview rows.
- Hai verifier chấm preview và final: 2 × (1.177.600 + 110.400) = **2.576.000 phép chấm ảnh–verifier**.

Các con số là workload yêu cầu, chưa phải số ảnh valid đã có. Số lượng hợp lệ/thất bại phải được bàn giao thật. Không nhân lại bank theo cap/policy.

## 3. Prompt đủ đa dạng mà vẫn nhỏ

### 3.1 Train và dev

| Nguồn | Train | Dev | Vai trò |
|---|---:|---:|---|
| DiffusionDB | 1.000 | 150 | Prompt người dùng thực tế |
| COCO captions | 400 | 60 | Mô tả cảnh thông dụng |
| Prompt cấu trúc tự tạo | 600 | 90 | Kiểm soát độ khó và compositionality |
| **Tổng** | **2.000** | **300** | |

600 prompt cấu trúc chia đều năm nhóm chính: thuộc tính chủ thể; nhiều vật thể/quan hệ; đếm/attribute binding; vật liệu/ánh sáng/phong cách; nhiều điều kiện. Từng câu có scene specification và template family; không sao chép template benchmark.

Theo dõi các nhãn chéo: chân dung/người, động vật, sản phẩm, phong cảnh/kiến trúc, minh họa; độ dài; số vật thể và mức độ ràng buộc. Quota nguồn là cứng; domain/độ dài là mục tiêu mềm, không tạo toàn bộ tích Descartes làm tăng kích thước.

Không chọn prompt dựa trên reward, mức dễ prune, ảnh đẹp hay lợi thế oracle. Không chỉ lấy prompt mà PSP/controller làm tốt.

### 3.2 Giữ tương thích bản Full

Chuẩn bị pool text Full trên CPU nếu chưa có, rồi chọn Lite thành subset có phân tầng. Việc chuẩn bị text không yêu cầu infer toàn bộ Full.

Giữ cùng prompt_uid, seed namespace, bank_id và model profile khi mở rộng. Xuất ba subset train lồng nhau **250 ⊂ 1.000 ⊂ 2.000**, đều có đủ ba profile. Dùng để đo hiệu ứng tăng dữ liệu với số môi trường không đổi.

Nếu Full pool chưa được khóa, ghi Lite manifest độc lập và giữ nguyên UID khi đưa vào Full sau; không nói đã là subset của một manifest chưa tồn tại.

### 3.3 Chống leakage

- UID từ SHA256 của text đã NFC/whitespace normalize; giữ text_raw.
- Gom exact/near duplicate và parent group trước khi split.
- COCO: caption cùng image_id cùng split; synthetic: paraphrase/scene/template family cùng split.
- Loại trùng train/dev với text GenEval, PartiPrompts và prompt đánh giá v1 mà dự án còn manifest.
- Một UID có cùng split ở tất cả model và bank.
- Normalizer/PCA chỉ fit trên train; experiment 250 prompt chỉ fit bằng subset 250 đó.
- Báo coverage của detector và các manifest v1 thiếu; không tuyên bố đã loại mọi tương đồng ngữ nghĩa.

## 4. Tập test giữ riêng, không dùng để tune

Test500 gồm:

- **300 prompt ID:** 150 DiffusionDB +60 COCO +90 cấu trúc, group chưa xuất hiện trong train/dev.
- **200 prompt source-shift:** lấy có phân tầng từ PartiPrompts, khóa UID trước khi train. Ghi kết quả là `PartiPrompts-200`, không gọi đó là toàn benchmark [4].

Test chạy cùng các prompt/seed bank trên ba profile đã thấy và Hyper-SD15 chưa thấy. Báo riêng hai nhóm prompt và từng profile; không để mean gộp che nhóm bị giảm chất lượng.

Trong nhiệm vụ hiện tại chỉ tạo manifest test và exclusion index. **Chưa sinh/chấm test**; bước đó thực hiện sau khi policy, baseline, cap, preprocess và tiêu chí so sánh đã freeze.

Khi đánh giá, controller tương tác với sampler thật; mỗi episode bắt đầu từ pool tám seed của bank, rồi được phép prune. Chỉ một ảnh winner được giao. Không quy mọi episode policy thành tám ảnh đầy đủ như khi thu bank.

Không cần thu một bank test đầy đủ chỉ để thay cho online evaluation. Bank test bổ sung chỉ dùng phân tích, có mục đích và chi phí riêng.

Hai bank của một prompt giúp giảm nhiễu seed; test500 vẫn có **500 cụm prompt**, không phải 1.000 mẫu độc lập.

## 5. Collector và thông tin phải lưu

Giữ common schema đa model của bản Full nhưng J phải lấy từ profile, không hard-code 16.

Mỗi task:

1. Tám initial latent, danh tính và hash; seed không phụ thuộc GPU rank/worker order.
2. Chạy đủ quỹ đạo k=8, không dùng controller để lọc dữ liệu khi collection.
3. Tại checkpoint lấy clean prediction từ forward vừa thực hiện; không gọi thêm denoiser.
4. Lưu latent preview bằng safetensors bảo toàn dtype của tensor dùng để decode; packing/scaling/shift có metadata.
5. Decode → finite check → pixel conversion chuẩn → ImageReward/PickScore; cùng một đường preprocessing cho bank và online.
6. Lưu PNG cuối và score final riêng với preview cuối.
7. Ghi validity/failure code, content hashes và durable completion marker.

Checkpoint:

- Model8: sau mỗi forward thực, j=1,…,8.
- SDXL50: step_end(j)=ceil(50j/16), j=1,…,16.
- Ghi forward_idx, actual timestep/sigma và preview origin. Không đồng nhất j=4 của hai profile là cùng noise level; controller nhận cả progress chuẩn hóa và metadata môi trường.

Ảnh có score thấp vẫn là dữ liệu hợp lệ. Không đổi seed để thay candidate lỗi. Failure phải được giữ trong manifest; worker lỗi infrastructure retry cùng task/seed, tối đa hai lần tự động trước khi báo.

## 6. Verifier và đặc trưng controller

Thu cả **ImageReward-v1.0 và PickScore-v1**. PickScore lưu raw score trước softmax. Hai scorer frozen; không train lại.

Để Lite dễ diễn giải, training chính dùng **ImageReward làm verifier active/reward**. PickScore là phép kiểm tra chất lượng phụ, không dùng để chọn prompt hay checkpoint policy chính. Có thể chạy ablation train với PickScore riêng, nhưng gắn tên riêng và không gọi PickScore là metric độc lập cho ablation đó.

Lưu embedding prompt bằng frozen common encoder theo bản Full; không dùng native SDXL/FLUX embedding với số chiều khác nhau như thể cùng một feature space. Encoder/norm version nằm trong manifest.

Actor chỉ được đọc:

- prompt feature;
- progress, số candidate, budget/cost context;
- lịch sử score đã mua bằng VERIFY;
- optional cheap trajectory features nếu variant có trả phí.

Không đọc score tương lai, scorer inactive, terminal score trước FINAL hay candidate đã bị prune. Lưu tất cả score trong bank không có nghĩa actor được nhìn tất cả.

Cheap feature thiếu CFG hai nhánh phải có availability mask. Lightning/Hyper-FLUX profile dùng guidance-distilled/native guidance nên không giả tạo uncond/text branch gap.

## 7. Grammar để đánh giá controller công bằng

Collection k=8 đầy quỹ đạo không phụ thuộc grammar training, nên cùng bank dùng được hai grammar sau.

**Anchor:** giữ grammar v1 với đúng hai prune 8→4→2 để đối chiếu lịch sử.

**Đề xuất chính cho Lite few-step:** thêm grammar có version `G-842-optional-lite1`: được thực hiện **0, 1 hoặc 2 prune**; k_final có thể là 8/4/2 nếu đủ budget. Mỗi prune vẫn cần score fresh tại checkpoint hiện tại; không prune hai lần tại cùng checkpoint.

Lý do: với model rất nhanh, mua thêm verifier trung gian có thể không đáng chi phí. Cho phép kết thúc không prune giúp kiểm tra controller có học được khi nào nên bỏ qua VERIFY; không ép nó phải mua hai lần rồi mới kết luận RL kém hiệu quả.

Đây là thay đổi đề xuất cần implement rõ, không khẳng định DP hiện tại đã hỗ trợ. DP phải có fin(8), fin(4), fin(2); mask bảo đảm trả đủ mọi action/setup/final cost. Không có rescue miễn phí.

Trong replay deterministic xấp xỉ và ranking cố định, số lịch prune không gồm extra VERIFY-only:

- J=8: 1 lịch không prune +7 lịch một prune +21 lịch hai prune =29.
- J=16: 1+15+105 =121.

Extra VERIFY-only vẫn có thể giúp policy chọn thời điểm sau này, nhưng tốn phí. Oracle từ các lịch này là oracle simulator, không phải cận trên online.

CONTINUE tiến tới checkpoint kế tiếp: một forward đối với profile 8 bước, hoặc đoạn step_end(j+1)−step_end(j) đối với SDXL50. Khi hết quỹ đạo, environment thực hiện FINAL và tính đủ phí. Actor không điều chỉnh số bước nội bộ của checkpoint trong study Lite này.

## 8. Chi phí và các baseline tối thiểu

Profile trên B200: den(k), ver(k), fin(k), prune, setup/prompt encoder, optional feature và controller dispatch. Có k=1/2/4/8; k16 chỉ cần nếu thực hiện baseline BoN16 riêng. Phiên bản scorer/precision khớp collector/online.

Dataset factory overhead ghi latent/PNG không thuộc latency deployment. Mỗi episode chỉ gọi một verifier active; chi phí thu hai scorer không được coi là chi phí của một lần VERIFY triển khai.

Chọn budget từ train/calibration rồi freeze trước test. Recipe Lite đề xuất: **Bβ=β·C_BoN8, β∈{0,5;0,75;1,0}**, với C_BoN8 là ledger đầy đủ của reference ở profile tương ứng. Tính setup, dispatch, decode và final selection vào ledger được so. Model loading/warm-up và artifact dump tách riêng.

Đây là phiên bản budget mới của Lite; không tái sử dụng bảng cap v1 bằng đổi tên. β thấp có thể không khả thi cho controller k0=8. Báo infeasible và lý do; không tự nâng budget cho riêng learned policy. Các method khả thi cùng một operating point phải dùng cùng cap.

Baselines bắt buộc:

1. Single và BoN-2/4/8 dùng cùng nested seed pool.
2. Lịch PSP-style cố định: fast profile ví dụ (2,4), SDXL50 dùng checkpoint (4,8); gọi đúng là lịch tham chiếu của study, không gán là mặc định paper cho mọi model.
3. Best-fixed: tune trên train/dev, gồm cả 0/1/2 prune trong grammar được đánh giá.
4. Threshold controller và one-query selector.
5. Random policy có budget mask, làm sanity check.
6. Learned controller chính, tối thiểu ba training seeds; giữ original MLP làm architecture baseline nếu đổi sang set/history controller.

Trên Hyper-SD15 giữ riêng, fixed/threshold tham số chuyển từ train/dev mà không tune bằng quality của Hyper-SD15. Được đo runtime/cost và kiểm API bằng prompt calibration riêng. Nếu tune bằng quality target-model thì báo thành few-shot adaptation, không gọi zero-shot.

## 9. Replay dùng để học, online dùng để kết luận

Training có thể sử dụng replay bank để tạo rollout mới từ policy; không cần action label khi collection. Future scores là target đặc quyền của environment, không phải actor input.

Batch size thay đổi sau prune có thể làm survivor trajectory khác bank ngay cả khi sampler eta=0. Vì vậy:

- Qualification so replay với online trên 50 calibration prompt, các lịch xác định trước; ghi discrepancy và keep/winner agreement.
- Tuning cuối trên dev chạy online với artifact và cost profile thật.
- **Mọi kết luận test dùng online sampler thật.** Không lấy replay score rồi gọi là latency/chất lượng triển khai.
- Online fine-tuning nếu thực hiện dùng prompt train, policy version rõ và namespace seed mới. Không dùng test làm môi trường update.

Lượt làm dữ liệu này chỉ xây bank/collector/accessor; training controller và online test là nhiệm vụ tiếp theo. Người sinh data không phải implement PPO hay optimizer.

## 10. Thế nào là kết quả phản ánh đúng khả năng controller?

Primary report trên mỗi profile và budget:

- Difference về delivered ImageReward giữa controller và baseline mạnh đã chọn trên dev, cùng paired CI.
- Mean/p50/p95 wall-clock e2e, saving ratio, failure rate.
- PickScore của ảnh giao để kiểm tra chuyển giao chất lượng ngoài training reward chính.
- Tỷ lệ VERIFY, số prune, thời điểm prune, số action bị mask và số episode policy không có lựa chọn.

Hai cách đọc: chất lượng tại cùng cap và chi phí tại chất lượng tương đương. Chốt quality margin theo metric và contrast trước test; không suy ra non-inferiority chỉ từ hai mean gần nhau. Mức tiết kiệm thực tế ≥10% có thể dùng làm mục tiêu thực dụng, không phải kết quả bảo đảm.

Bootstrap theo prompt, giữ bank và method ghép cặp; báo biến động giữa ba policy training seeds riêng. Trên 300 ID/200 Parti và held-out profile, CI có thể rộng. Dùng paired SD trên dev để dự toán MDE trước khi mở test; nếu CI chưa phân biệt được lợi ích nhỏ thì báo chưa kết luận, không tự đổi margin sau khi nhìn test.

Learning curves dùng cùng architecture và tổng training rollout gần bằng nhau trên 250/1k/2k prompt. Chỉ thay dataset size; fit norm đúng subset. Kiểm riêng shared-policy so specialist-policy trên seen profiles nếu cần xác định tác hại của việc dùng chung model.

Lite hỗ trợ kết luận về các profile, nguồn prompt và budget đã thử. Thất bại của một controller không bác bỏ mọi kiến trúc/thuật toán; thành công trên ba profile cũng không chứng minh universal controller.

## 11. Storage và mở rộng

Giữ mọi preview latent và final PNG train/dev. Với native latent SDXL 4×128×128 và FLUX tương đương 16×128×128, lưu 16-bit:

| Profile | Raw preview latent train+dev ước tính |
|---|---:|
| SDXL-Lightning8 | 38,59 GB |
| Hyper-FLUX8 | 154,35 GB |
| SDXL Base50 | 77,18 GB |
| **Tổng** | **270,11 GB** |

Chưa tính ảnh cuối, final latent, weights, metadata, QA, staging và backup. FP32 preview tăng gấp đôi phần tương ứng. Đo shape/dtype/bytes thật trước khi forecast.

Đề xuất **1–2 TB working storage bền vững và backup độc lập** cho đợt Lite; kiểm dung lượng từ 20 task/profile và chọn mức đáp ứng forecast có headroom. Không cần cấp 40 TB như bản Full để bắt đầu Lite.

Checkpoint dtype bảo toàn bằng safetensors; validate round-trip và re-decode. Shard commit atomic, checksum và durable ack. Không tự xóa dữ liệu hoặc Stop/Destroy máy. Không tạo tải giả để tránh idle policy của cụm.

## 12. Công việc giao Codex và người chạy B200

| Task | Codex chuẩn bị | Người chạy B200 thực hiện | Nghiệm thu |
|---|---|---|---|
| L0 | Audit bundle 396b313 và repo, registry Lite | cung cấp môi trường/storage thực tế | gap report, không coi bundle cũ đã có Lightning/Hyper |
| L1 | Pool prompt, dedup, split và nested subsets | có thể chạy script CPU nếu cần | train2k/dev300/test500/calib50, manifest hash |
| L2 | Adapter Lightning UNet và Hyper-FLUX LoRA; thêm/pin PEFT nếu đường LoRA cần | cài bằng uv lock đã cập nhật | install/import + revision report |
| L3 | QA/collector/validator J variable | một shard production8seed/profile, decode +hai scorer +restore | PASS từng generator/decode/scorer, output schema đúng |
| L4 | Profiling và distributed task manifest | đo cost và throughput trên B200 | raw samples, fingerprint, forecast từ số đo |
| L5 | Launcher resume +backup | thu 12k train task và1.8k dev task | 110.4k candidate record, flags đầy đủ, durable verified |
| L6 | Accessor/feature views và dataset card | restore subset khác process/máy | no-leak tests, restore và final data report |
| L7 — sau training | freeze evaluator/policy/baseline | online test500, held-out Hyper-SD15 | paired results và cost thật; không nằm trong L5 |

L1 và L2 có thể làm độc lập. Khi một adapter đạt qualification, nó có thể tiếp tục collection trong khi sửa adapter khác. Hai scorer là dependency chung: lỗi scorer phải được báo ở đúng tầng, không ghi cả ba generator FAIL chỉ vì scorer chưa dùng được.

Bundle 396b313 hiện được báo là candidate; chưa có bằng chứng env/scorer chạy B200. Không giả rằng thêm tên model vào YAML đã hoàn tất adapter. Mọi sửa dependency phải regenerate uv.lock, clean install check và ghi compatibility thật.

Sử dụng scheduler/worker độc lập với phiên chat, model resident, tối thiểu một worker mỗi profile nếu có đủ GPU. Không cần dành hết 10–20 B200 cho dataset này. Phân bổ thêm worker theo remaining measured GPU-seconds; không chia đều chỉ theo số task.

Không đặt ETA bằng cách chia số bước của SDXL/FLUX hoặc dùng hệ số marketing B200. Đo giây/task gồm decode/scoring/I/O trên profile thật rồi dự báo.

Triển khai L5 theo ba đợt nằm trong cùng manifest: 250 → 1.000 → 2.000 prompt train, mỗi đợt bổ sung phần còn thiếu. Đợt đầu chứa 12.000 candidate trajectory trên ba profile. Sau QA dữ liệu thành công, tự tiếp tục đợt kế tiếp; không cần xin duyệt lại từng đợt. Các thống kê headroom/ranking chỉ là chẩn đoán, không dùng để loại prompt bất lợi hoặc gọi oracle replay là bằng chứng online.

## 13. Config đề xuất

Các khóa dưới đây là contract cần ánh xạ vào code thực tế, chưa phải cam kết CLI đã tồn tại.

```yaml
dataset_name: apsp-lite-mixed-v1
train_prompts: 2000
dev_prompts: 300
calibration_prompts: 50
test_prompts:
  id: 300
  parti_subset: 200
train_sources:
  diffusiondb: 1000
  coco_captions: 400
  structured: 600
train_profiles:
  - {id: sdxl_lightning8, steps: 8, checkpoints: 8}
  - {id: hyper_flux8, steps: 8, checkpoints: 8}
  - {id: sdxl_base50, steps: 50, checkpoints: 16}
heldout_profiles: [hyper_sd15_8]
banks_per_prompt_profile: 2
candidates_per_bank: 8
train_nested_subsets: [250, 1000, 2000]
seed_namespace: apsp-v2-sha256-cpu
collect_verifiers: [imagereward_v1, pickscore_v1]
main_training_verifier: imagereward_v1
optional_grammar: G-842-optional-lite1
budget_beta_vs_bon8: [0.5, 0.75, 1.0]
expected_train_tasks: 12000
expected_dev_tasks: 1800
expected_train_candidates: 96000
expected_dev_candidates: 14400
expected_preview_rows: 1177600
expected_score_values: 2576000
save_all_preview_latents: true
save_all_final_png: true
allow_test_sampling_before_method_freeze: false
allow_controller_training_in_data_job: false
automatic_provider_stop: false
```

## 14. Definition of DONE cho lượt dữ liệu Lite

- [ ] Source/prompt/profile/seed/schema lock và actual runtime fingerprint đầy đủ.
- [ ] Có 2.000 train UID, 300 dev UID, 50 calibration UID; 500 test UID giữ riêng; group overlap bằng 0 theo detector đã khóa.
- [ ] Có 13.800 task terminal status, 110.400 candidate record; không thiếu task bị gọi là complete.
- [ ] Preview/scorer counts theo J từng profile; failure flags/null đúng; không bỏ dữ liệu reward thấp.
- [ ] Latent/PNG/scoring round-trip và row map theo CandidateId đạt.
- [ ] Kiểm chứng B200 thật từng profile và cả hai scorer; native batching/preview behavior có báo cáo.
- [ ] Profiler có raw costs, inference timer không trộn artifact dump; phần controller dispatch chưa có artifact được ghi provisional.
- [ ] Accessor mua thông tin/no-leak tests đạt; subset views dùng được cho trainer.
- [ ] Dữ liệu có bản durable verified, backup và restore drill; có checksum và resume instructions.
- [ ] Bàn giao DATASET_CARD_LITE.md, LITE_DATA_REPORT.md, RELEASE_MANIFEST.json, RUNBOOK_LITE.md và model/scorer compatibility matrix.
- [ ] Chưa có tuyên bố controller thắng, generalize hoặc test hoàn tất khi mới thu bank.

## 15. Prompt giao việc

Đọc PLAN_CODEX_CONTROLLER_DATA_LITE_V1.md và chuyển lượt thu dữ liệu hiện tại sang plan Lite này trong namespace riêng. Giữ file/data v1 và bản Full. Kiểm bundle b200/inference-env thực tế; thêm adapter SDXL-Lightning8, Hyper-FLUX8, J variable, prompt manifests và collector theo task L0–L6. Tái sử dụng env uv đang chuẩn bị, chỉ sửa dependency khi cần và ghi lock mới. Không coi model/scorer đã kiểm chứng chỉ vì lock check pass. Hoàn tất 2.000 prompt train + 300 dev dùng chung ba profile, hai bank × tám seed, đúng 13.800 task / 110.400 candidate record. Chưa train controller, chưa mở test; không tự Stop/Destroy máy. Làm tiếp task độc lập khi adapter khác lỗi; bàn giao số đo/validity/backup thật, không dừng ở plan hoặc synthetic tests. Thay đổi grammar/budget Lite phải có version và tests riêng, không sửa nghĩa v1. Người chạy B200 chỉ cần gói env+collector, manifests, launcher, output specification và các bước nghiệm thu.

## 16. Nguồn chính

1. [SDXL-Lightning: checkpoint 8-step, UNet loading và sampler](https://huggingface.co/ByteDance/SDXL-Lightning).
2. [Hyper-SD: FLUX LoRA8, scale/guidance và SD1.5 variants](https://huggingface.co/ByteDance/Hyper-SD).
3. [SDXL Base1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
4. [PartiPrompts, nguồn benchmark ngoài](https://github.com/google-research/parti).
5. [Procgen, bằng chứng trong RL về tầm quan trọng của đa dạng môi trường và test tách biệt](https://proceedings.mlr.press/v119/cobbe20a.html). Đây là động cơ thiết kế, không phải chứng minh T2I controller sẽ generalize.
6. [Plan Full của dự án](PLAN_CODEX_CONTROLLER_DATA_V2.md): giữ các quy tắc schema, provenance, paid information và fail/resume tương thích; các kích thước/model/grammar/budget mới dùng bản Lite này.
