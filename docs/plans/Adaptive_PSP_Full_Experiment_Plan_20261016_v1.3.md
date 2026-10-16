# Adaptive PSP — kế hoạch hoàn thành toàn bộ thí nghiệm trên RTX 4090

**Phiên bản 1.3 · 16/10/2026, 23:07 Asia/Saigon.** Người thực hiện: Claude Code/Codex. Research: người dùng và nhóm nghiên cứu.

**Văn bản này thay thế kế hoạch v1.0/v1.1/v1.2 về thứ tự thực hiện, máy hiện tại và vận hành.** HANDOFF/CONTRACT vẫn là nguồn chuẩn cho schema và method, trừ các amendment được duyệt. Đây là kế hoạch thực thi và gói quyết định đề xuất; không phải chữ ký PREREG hay báo cáo kết quả.

## 1. Mục tiêu và phạm vi hoàn thành

Mục tiêu: giảm chi phí inference đầu-cuối, giữ chất lượng gần tham chiếu theo margin đã khóa. So sánh với single thông thường, single ít bước, BoN, PSP và baseline thích nghi mạnh; không chỉ so với một đối chứng thuận lợi.

Objective:
`min E[C_method], subject to E[Q_method - Q_reference] >= -epsilon`.

Tham chiếu xác nhận chính đề xuất là PSP-default. Báo đầy đủ các tham chiếu single64 và BoN4; một method đắt hơn single64 không được gọi là tiết kiệm so với single64. Grammar 8→4→2 có chi phí tối thiểu riêng, có thể không đạt mức chi phí của single; phải báo giới hạn này từ cost table.

**Phạm vi chính:** hoàn tất SD1.5: bank, toàn bộ baseline nội bộ, controller/RL, đo chi phí thật, xác nhận online, held-out test, ablation và báo cáo. **Phạm vi khái quát:** SDXL với ma trận thu gọn đã khóa và đối chứng ngoài phù hợp. Flow backbone nằm ngoài phạm vi mặc định của phiên bản này; không claim diffusion + flow khi chỉ chạy SD1.5/SDXL.

Một RTX 4090 24 GB là máy thực hiện. Các job GPU chạy tuần tự, evaluator và generator không tranh VRAM. Điều kiện bộ nhớ của từng cấu hình phải được kiểm trong lượt chạy đầu; không mặc định cấu hình ở H200 có cùng mức sử dụng trên 4090. Không cần thuê thêm máy để bắt đầu hoặc hoàn thành suite SD1.5.

## 2. Quyết định vận hành đã rõ

- Instance hiện tại: **56267276, RTX 4090 24 GB**, giá được Claude báo **$0.626/giờ**; xác minh fingerprint và giá thực lúc chạy. Máy mới ở Tây Ban Nha, driver được báo 615.71.09, quota CPU khoảng 31 và disk 200 GB trống. Dùng runtime thực để fingerprint; đặt thread phù hợp quota. Không dùng ID 56261825/56224386/56070930/56044352 trong lệnh điều khiển.
- Repo: `NguyenNgocMinh30012005/DMRL`, branch `bootstrap/p0`. Commit gần nhất được báo là `790d7cb`; resolve SHA đầy đủ từ repo, không reset đè sửa đổi mới. PSP read-only ở `590f59f58384169c719e431dc01d14006fb0fc0c`.
- Người dùng đã bỏ trần tiền thuê. Không hỏi lại approved_total_usd, giá tối đa hay số phút tối đa để được chạy. Vẫn ghi chi phí tích lũy.
- **Không thử Stop thật thêm.** Ghi nhận bài thử hiện có chỉ kiểm được một phần; tiếp tục dự án, không mở vòng debug hạ tầng mới.
- **Chỉ thị mới nhất: CẤM Stop instance trước khi hoàn tất toàn bộ thí nghiệm.** Không Stop khi xong P1, chờ PREREG, không còn job ready tạm thời, job lỗi/treo, phiên SSH mất hoặc laptop tắt. Các trạng thái đó không phải project_completed.
- Máy giữ chạy giữa mọi phase. Tắt tất cả nhánh tự gọi Stop instance khi dự án còn unfinished, kể cả nhánh unrecoverable_failure. Khi lỗi, chỉ dừng/retry job bằng PID/PGID, lưu checkpoint và báo lỗi. Không Stop/Destroy/release instance; không thay bằng thao tác khác làm mất GPU. Chỉ xem xét Stop sau khi toàn bộ checklist cuối đã hoàn tất, hoặc khi người dùng ra chỉ thị mới rõ ràng.
- Watchdog và workload tách process group. Trong giai đoạn chưa hoàn tất, watchdog chỉ giám sát/cảnh báo và xử lý job; tuyệt đối không gọi API Stop instance. Watchdog không có deadline tài chính; tiến độ thật của job là căn cứ phát hiện treo. Mất log ngắn, GPU rảnh lúc CPU training hoặc đang chạy evaluator CPU không tự chứng minh job treo.
- Workload chạy detached trên máy thuê, có log, checkpoint và resume. Không phụ thuộc một tiến trình chờ SSH trên laptop hay phiên Claude đang mở.
- Giữ nguyên DP cap theo episode, quy tắc trả phí thông tin, no-leak, provenance và PREREG. Đây là ràng buộc khoa học, khác với tiền thuê máy.
- Nếu công cụ yêu cầu quyền, trình đúng thao tác để người dùng xét duyệt. Không coi prompt này là thay đổi cấu hình quyền của công cụ.
- Chỉ sửa những lỗi chặn thực thi hoặc làm sai kết quả. Không thay engine, backend, precision hay sampler chỉ để đạt bitwise invariance giữa batch.

## 3. Trạng thái xuất phát và ưu tiên

Theo báo cáo người dùng: có code orchestrator, baseline, collector, profiler, evaluator adapter; nhiều test CPU pass. Smoke/diag CUDA cũ đã chạy trên 3090 và phát hiện batch drift. **Chưa có calibration hoàn chỉnh, bank nghiên cứu hay bảng kết quả xác nhận trên 4090.**

Đầu ra tiếp theo phải là số liệu thật: calibration + bảng fixed baselines. Sau đó tự động chuyển sang bank và thí nghiệm còn lại khi prerequisite đã đủ. Không bàn giao một lượt “sẵn sàng chạy” rồi dừng nếu vẫn còn việc được phép.

Các phần độc lập có thể làm xen kẽ: chuẩn bị evaluator, sửa report, train controller nhỏ và tính thống kê trên CPU trong lúc collector chạy. Benchmark latency dùng GPU độc quyền, không chạy kèm workload gây nhiễu.

## 4. Ma trận thí nghiệm

| Method | Train/tune | Dev | Val/Test | Ghi chú |
|---|---|---|---|---|
| Single64 | Không | Đo thật | Bắt buộc | RM đánh giá sau delivery không phải phí selection |
| Single32 | Không | Đo thật | Bắt buộc | Scheduler T=32 thật |
| BoN2/4/8 | Không | Đo thật | Bắt buộc | Đúng batch và final verifier |
| BoN16 | Không | Khi cần frontier | Nếu được khóa trước val | Không suy từ bank k8 |
| PSP-default 8→4→2 | Không | Đo thật | Bắt buộc | Đối chứng chính; cuts j=4,8 |
| B3 best-fixed 105 lịch | Replay train | Chọn + xác nhận online | Lịch đã khóa/cap | Không quét 105 lịch trên val/test |
| B4b grid PSP rộng | Replay có coverage | Chọn + online | Cấu hình đã khóa/cap | Có thể k0≤16, keep không chia đôi; không làm tròn checkpoint |
| B5 screen-and-resume | Train | Chọn cutoff/keep | Cấu hình đã khóa | Một paid query; hỗ trợ final k=1 |
| B6 threshold | Train | Chọn tham số | Bắt buộc | Đối chứng đơn giản mạnh |
| B7 prompt-only selector | Train | Chọn model | Bắt buộc | PCA/norm fit đúng fold |
| B8 one-query selector | Train | Chọn model | Bắt buộc | Trả đủ query; không VERIFY lặp miễn phí |
| B9 random feasible | Không | Sanity check | Báo phụ | Dùng mask thật |
| B10 predictor rẻ | Train | Chọn model | Bắt buộc | Không paid VERIFY trung gian; tính phí feature |
| RL-M1 REINFORCE | Train, ít nhất 3 training seeds cuối | Chọn config/checkpoint | Bắt buộc | Không chọn training seed thắng trên val |
| FK-Steering chính thức | Tune trên train/dev | Xác nhận adapter | Đối chứng ngoài SD1.5 | Cấu hình stochastic hợp lệ; tính mọi phí |
| Flash-BoN | Audit compatibility | Nếu backbone phù hợp | Chỉ khi chạy method thật | B5 không được đổi tên thành Flash-BoN |

Các caps khảo sát: alpha ∈ {0.25, 0.5, 1.0, 1.5} so với cùng C_ref đã khóa; ưu tiên 0.5 và 1.0. Loại điểm không khả thi bằng DP và ghi rõ, không cấp thêm compute để đủ một hàng bảng. So sánh theo cùng cap số tuyệt đối và theo actual wall-clock; cùng alpha không tự bảo đảm cùng chi phí.

Đối chiếu ngoài: PSP là baseline nền bắt buộc; FK là baseline external có kế hoạch chạy trên SD1.5. Flash-BoN chính thức dùng cấu hình Wan/FLUX và verifier server riêng [R2]; khả năng chạy đúng method trên một 4090 không được mặc định. Audit khả năng tương thích trong lúc chạy main suite, ghi bằng chứng nếu không hỗ trợ backbone/không đủ bộ nhớ. Không để việc port Flash-BoN chặn toàn bộ SD1.5; cũng không tuyên bố đã thắng nó nếu chưa chạy so sánh hợp lệ.

## 5. Chuỗi công việc từ hiện tại đến kết quả cuối

### E0 — Đưa bản hiện tại vào chạy, không thêm vòng thử Stop

1. Dùng máy mới 56267276 mà người dùng vừa giao và đã kết nối; xác minh đang Running rồi giữ nguyên. Đây là máy đo thay thế 56261825. Không chờ Start lại máy cũ, không cần đọc state.json cũ để tiếp tục.
2. Kiểm đúng ID, HEAD/submodule, fingerprint, disk và không có job trùng. Chỉ chạy kiểm tra cần thiết cho thay đổi sau commit đã test; không lặp lại toàn bộ smoke/diag cũ.
3. Cập nhật orchestrator/watchdog để phân biệt completed và blocked; vô hiệu hóa mọi đường gọi Stop instance trong khi dự án unfinished. Deploy code/prompts từ bản local đã kiểm lên disk mới, tải model theo pin; không cần truy cập ổ máy cũ. Triển khai commit rõ ràng, bật detached runner.
4. Tạo job manifest toàn suite với dependency và trạng thái. Giữ nguyên outputs cũ, không stage clone lồng, token, weights, bank hoặc ảnh.

**Đầu ra:** runner thực sự chạy job P1; log có hoạt động UNet/scoring, không chỉ “process started”.

### E1 — Calibration + bảng fixed baseline đầu tiên

- Dùng 50 prompt calibration riêng, M9 subset 20 đã định trước; kiểm input near-duplicate trước khi sinh ảnh.
- Full-pool language/dedup: theo §9. Lọc và chia theo input, không theo kết quả ảnh. Không dựng ma trận similarity NxN trong RAM.
- Nạp model/reward/tokenizer đã pin, kiểm hash và cấu hình DDIM eta=0, T64, CFG7.5, 512², delta4.
- M3: đo validity trước clamp, NaN, sai lệch decode và tốc độ fp16/fp32. Chọn preview dtype/chunk theo quy tắc validity/bộ nhớ/tốc độ đã có, không chọn vì IR cao.
- M2/M9: đo den/ver/fin/prune/feat ở k=1,2,4,8,16 và các k khác mà B4b dùng. Chi phí không có số đo thì chưa được dùng làm cost thật.
- Đo single32 thực; equivalence cùng định nghĩa scoring. Không yêu cầu mọi batch bitwise giống nhau.
- Chạy **online thật** single64, single32, BoN2/4/8, PSP-default trên 50 prompt ×2 bank. Pair prompt/seed đúng; không lấy điểm replay k8 làm kết quả BoN2/k1.
- Xuất bảng tối thiểu ngay khi đủ IR-delivered, actual e2e latency, lỗi và chi phí. HPS bổ sung khi adapter đạt; việc chuẩn bị HPS không chặn bảng tối thiểu.
- Bảng calibration là mô tả: không dùng để tune method, chọn epsilon hay đặt ngưỡng thắng.

**Đầu ra:** calibration_report, cost table có coverage/provisional rõ, calibration_quality_cost.csv, fingerprint, prompt manifest và forecast E2–E8 từ throughput thực.

**Đi tiếp:** không Stop sau E1; chạy phần đã đủ điều kiện, gom các quyết định PREREG còn thiếu thành một gói.

### E2 — Khóa protocol và thu bank train/dev

- Gom một gói nghiên cứu đầy đủ theo §9, thay vì hỏi rời từng Q. Chữ ký/ủy quyền duyệt phải đến từ người dùng/research thật, có nguồn và hash; coding agent không tự ký.
- Sau PREREG-A: thu train_fit 190 + train_dev 50 prompt, mỗi prompt 2 bank ×8 seed. Tổng mặc định 3,840 candidate trajectories; mỗi trajectory có 16 checkpoint preview.
- Đây là kích thước train pilot mặc định, không phải bằng chứng đủ power cho test paper. Nếu protocol hiện hành đã khóa kích thước khác, đối chiếu trước khi thay.
- Thu train/dev trước, chưa cần val bank khi confirmatory evaluation sẽ chạy online.
- validate_bank, checksum, CandidateId, đủ dòng, seed provenance, scoring path và config hash. Resume shard nguyên tử; lỗi được ghi chứ không rerun âm thầm đến khi thành công.
- Lưu latent train theo contract nếu phục vụ B10/ablation, ước lượng dung lượng trước; không tự thêm latent held-out.
- Song song chuẩn bị HPS/GenEval và FK trong môi trường riêng.

**Đầu ra:** bank train/dev hợp lệ, collector report và bộ artifact đủ train/tune.

### E3 — Baseline, anchor, replay discrepancy và headroom

- Fit preprocessing/PCA/norm trên train_fit; CV theo prompt/group và fit lại preprocessing trong fold.
- Tune B3/B4b/B5/B6/B7/B8/B10 trên train, chọn trên dev. Search space và số trial hữu hạn, log đầy đủ trước khi chọn; không mở rộng vô hạn vì bỏ trần tiền thuê.
- Anchor PSP−BoN4 phải dùng dữ liệu train đúng protocol; nếu lấy replay, ghi phạm vi xấp xỉ. Đề xuất dùng online train cho anchor cuối vì batch drift đã biết.
- Tính variance/power, epsilon, reward scaling theo công thức §9. Khóa PREREG-B trước script/cổng bị guard.
- G1 dùng CI của **sai lệch hiệu ứng online−replay so với PSP-default**, không dùng tỷ lệ winner agreement làm tiêu chuẩn thay thế.
- O1 trên bank là O1-replay; không gọi là cận trên online. Hindsight schedule chạy online là witness dùng tương lai, không phải policy triển khai.
- Replay không có headroom nhưng online chưa kiểm đủ → inconclusive; không tự kết luận method vô dụng.
- Native batching được giữ. Không quay lại debug kernel hoặc padding toàn bộ UNet.

**Đầu ra:** baseline_dev.csv, anchor/power, replay_online_discrepancy, headroom và các operating points cần tiếp tục.

### E4 — Train controller và chọn operating points

- Train B7/B8/B10; RL dùng method REINFORCE/MLP đã có, không thay PPO/GRPO mới trước khi có kết quả bản tham chiếu.
- Reward terminal lấy delivered IR, trừ baseline B3 cùng cap, chia sigma_R train; final reward chỉ vào trainer, không thành feature trước FINAL.
- Cấu hình RL cuối chạy ít nhất 3 training seeds; dùng checkpoint selection cố định theo dev. Lưu mọi seed và kết quả, không chọn seed dựa trên val.
- Observation chỉ chứa thông tin đã mua/được phép. Trajectory feature có phí riêng. Mọi cấu hình dùng đúng DP mask với chi phí method tương ứng.
- Đo cost controller thực: build_obs, mask, act, truyền dữ liệu, once/disp. Cập nhật CPH/CCH đúng, không làm mất hiệu lực bank khi generator/scoring không đổi.
- Chọn điểm có actual mean e2e cost thấp nhất trong các điểm đạt quality target trên dev. Chuẩn bị 3 nhóm claim: adaptive allocation, paid verification, lợi ích riêng của RL.

**Đầu ra:** policies/model checkpoints, norm/PCA, controller costs, online_dev.csv, frozen_methods.json và contrast_manifest.json.

### E5 — Xác nhận online trên held-out validation

- Trước lần xem val đầu: khóa methods, caps/operating points, endpoint, margin, primary contrasts, multiplicity, sample size, failure rules và phần mở rộng mẫu.
- Val pilot 80 + val_ext tối đa220 chỉ dùng theo protocol hai-look đã duyệt. Không mặc định tối đa300 prompt đủ cho margin nhỏ.
- Tính power bằng variance ghép cặp train/dev trước khi xem val. Nếu mục tiêu paper đòi N>300, đưa **một amendment kích thước mẫu/tập xác nhận độc lập trước lần nhìn đầu**, không nới mẫu sau khi thấy kết quả chưa đẹp.
- Chạy online đúng từng method/batch. Không quét lại 105 lịch/grid trên val.
- Báo quality difference/CI, savings/CI, p50/p95, failure, VRAM, số VERIFY, denoise forwards, ledger và actual cap overrun.
- Với RL, báo đủ 3 training seeds; không lấy seed tốt nhất làm claim. Đơn vị thống kê chính vẫn là prompt; bank và training seed không biến thành số prompt độc lập.
- Nếu CI rộng, giữ inconclusive. Kết quả âm/thiếu power không bị sửa thành GO.

**Đầu ra:** validation_quality_cost.csv và kết luận có/không tiết kiệm ở quality target đã khóa.

### E6 — Test paper trên GenEval và metric độc lập

- PREREG-C thật và method artifacts đã khóa trước khi sinh/chấm ảnh test.
- Dùng official GenEval prompt manifest/aggregation, pin detector, code, weights và HPSv2.1 artifacts [R3,R4].
- Đề xuất protocol test: **4 episode sinh ảnh độc lập/prompt/method**, mỗi episode giao đúng1 ảnh; cùng seed protocol giữa các method. Đây là đề xuất cần khóa ở PREREG-C, không tự coi là protocol PSP đã tái lập.
- Chạy các baseline bắt buộc và cấu hình đã chọn; không đem toàn bộ trial train lên test. Test không được dùng để chọn checkpoint/epsilon.
- GenEval overall/per-category theo official aggregation; CI bảo toàn cấu trúc category/prompt. IR/HPS là metric khác, báo bất đồng giữa metrics.
- Evaluator chạy riêng sau delivery. Không cộng chi phí chấm GenEval vào inference method; mọi verifier cần để method chọn ảnh vẫn nằm trong inference.
- Không tự thực hiện human study hay gửi ảnh cho người khác; human evaluation cần protocol riêng.

**Đầu ra:** test_metrics.csv, test protocol, per-category table, CI, failure cases và hình minh họa chọn theo quy tắc đã khóa.

### E7 — Ablation và giải thích cơ chế

Chọn/configure trên train/dev, khóa trước tập xác nhận ablation; không dùng val/test đã xem để phát triển lại.

| Ablation | Câu hỏi |
|---|---|
| VERIFY bắt buộc kèm PRUNE (A1) | Tách mua thông tin khỏi prune có ích không? |
| Threshold / one-query / RL | Có cần controller RL phức tạp? |
| B10 không query giữa chừng | Feature rẻ có thay verifier được không? |
| Bỏ paid-score features | Policy có thực sự sử dụng thông tin đã mua? |
| Thêm trajectory features, tính phí | Lợi ích có lớn hơn overhead? |
| Policy điều kiện hóa budget vs riêng cap | Có chuyển giữa operating points? |
| Replay so với online | Sai lệch simulator tác động kết luận ra sao? |
| Reward-preview vs delivered, nếu protocol cho phép | Terminal convention ảnh hưởng gì? |
| Verifier cost ×{0,0.5,2} trong replay | Độ nhạy accounting, gắn nhãn counterfactual |

Chi phí/mask phải cập nhật cho từng ablation. Không gọi kết quả mô phỏng đổi giá verifier là wall-clock đã đo. Xuất action/query distributions, prune timing, regret và breakdown compute.

### E8 — SDXL và đối chứng ngoài, sau đó tổng hợp

- SDXL là kiểm tra khái quát targeted: single, BoN4/8, PSP, best-fixed, threshold, one-query, B10 và method được khóa từ SD1.5; ít nhất hai operating points khả thi đã chọn trước theo cost table SDXL.
- Tạo ModelAdapter, embedding/schema tương thích, pin weights/scheduler, calibration và cost table riêng. Không dùng cost/norm SD1.5 cho SDXL.
- Đề xuất giữ 512² như SD1.5 sẽ làm setup SDXL kém đại diện; cấu hình SDXL phải được khóa riêng trước chạy. Ưu tiên cấu hình native của backbone nếu bộ nhớ thực cho phép.
- Nếu không fit, ghi chính xác thành phần/batch gây OOM. Có thể chunk decode; đổi denoise batch/offload/quantize phải áp công bằng các method và ghi thành cấu hình mới. Không âm thầm đổi để tuyên bố cùng setup.
- PSP báo peak SDXL 13.27GiB ở setup H200 của họ [R1]; đây là cơ sở để thử trên 24 GB, không phải bảo đảm.
- Hoàn tất FK baseline trên backbone phù hợp. Với Flash-BoN, chỉ benchmark khi giữ được method và backbone chung trong bộ nhớ máy; nếu không, ghi unsupported rõ và thu hẹp claim.
- Không kéo một flow model lớn vào phiên này chỉ để đủ tên loại model. Flow là mở rộng riêng nếu nhóm quyết định.
- Tổng hợp kết quả dương/âm, độ bất định, overhead và khả năng hoàn vốn chi phí training.

**Đầu ra:** sd15_final/, sdxl_final/, external_baselines.md, ablation/, REPRODUCE.md và gói artifacts kiểm checksum.

## 6. Accounting và thống kê

1. Inference e2e gồm prompt encoding/init, denoise, decode, selection verifier, features/controller, prune và đưa ảnh thắng về CPU. C_fixed ngoài ledger cap vẫn phải vào e2e.
2. Scoring để đánh giá ảnh sau delivery báo riêng; single không bị tính thêm selection RM mà nó không dùng.
3. Load model/setup/warmup báo riêng, kèm chi phí nghiên cứu tổng: collection, tuning, training, experiment bị loại và evaluation.
4. Ledger dùng mean primitive không bảo đảm mọi request wall-clock dưới cap. Báo p95 và actual overrun; không gọi modeled budget là deadline latency chắc chắn.
5. Savings estimator: `1 - mean(C_method)/mean(C_reference)`. CI bootstrap theo prompt cluster và ghép cặp; không coi các seed của cùng prompt là mẫu độc lập.
6. NI: lower bound của mean(Q_method−Q_ref) ≥ −epsilon. “Không có khác biệt có ý nghĩa” không đủ để kết luận NI.
7. Mục tiêu chính đạt khi quality NI đồng thời savings đạt mức thực tế đã khóa; không chỉ IR tăng.
8. Frontier dùng các operating points đã chạy. Đường nối chỉ để minh họa, không claim điểm chưa đo.
9. Break-even `C_setup / savings_per_request` khi cùng đơn vị và savings>0. Không có savings thì không claim hoàn vốn.

## 7. Scheduling và dự toán

Không hứa một ETA full suite khi chưa có throughput calibration thật. Sau20 prompt M9, tạo forecast theo:
`n_prompts × episodes/prompt × methods × operating_points × training_seeds × repeats`.
Liệt kê mỗi job GPU/CPU riêng; sau job đầu của từng phase cập nhật ETA bằng tốc độ thực. Dự toán không phải cổng xin phép chi tiền.

Trình tự chính:
**E0 → E1 → E2 → E3 → E4 → E5 → E6 → E7 → E8 → final package.**
Chuẩn bị evaluator/external adapter và báo cáo CPU xen kẽ, không cản benchmark timing.

SD1.5 train/dev bank mặc định 3,840 candidate trajectories. Test GenEval theo đề xuất 4 episode/prompt/method có thể lớn hơn rất nhiều phần calibration; không được báo ETA vài giờ của smoke như ETA full paper.

Giữ một GPU worker. Tune RL/replay chủ yếu CPU; không cần một model generator resident để train controller nhỏ. Chỉ tăng concurrency khi không làm sai latency/bộ nhớ; không lấy số CPU host thay quota cgroup.

## 8. Runner và báo cáo tiến độ

Job states: pending, ready, running, completed, failed, blocked_prereg, blocked_dependency, cancelled.

Project states: running, waiting_research, completed_positive, completed_negative, incomplete_blocked, unrecoverable_failure.

**Bắt buộc:** “không có job ready” không được tự chuyển thành project_completed. Hoàn thành phase không ghi global completion marker. PREREG chưa ký → waiting_research và báo gói quyết định, máy giữ chạy theo chỉ đạo hiện tại.

Mỗi job có command/config/code/data hashes, prerequisites, timestamps, exit code, outputs/checksums, actual n và costs. Resume chỉ skip khi output hợp lệ, không dựa trên tên file rỗng.

Báo cáo theo mốc: calibration table → bank complete → baseline/dev → controller/dev → val → test → final. Trong mỗi báo cáo, nêu kết quả thực, nhánh đang chạy và ETA cập nhật; tránh chỉ báo số test CPU.

Tệp cuối:
- calibration_quality_cost.csv, calibration_report.md, cost_tables và fingerprints.
- bank_validation.json, baseline_dev.csv, headroom.md, power_report.md.
- controller artifacts và frozen method/contrast manifests.
- validation_quality_cost.csv, test_metrics.csv, sdxl_quality_cost.csv.
- frontier, cost breakdown, ablation, failure cases, amortization.
- RUN_MANIFEST.jsonl, DECISIONS_LOG, EXPERIMENT_STATUS, REPRODUCE.md.
- Tarball với checksums; tải bản sao về laptop. Không đưa ảnh/weights/bank/credential vào Git.

## 9. Gói quyết định khoa học — gom duyệt một lần

Đây là các đề xuất để research duyệt trước khi thu bank/khóa đánh giá. Không đánh dấu signed chỉ vì file này tồn tại. Khi người dùng duyệt cả gói, agent ghi rõ nguồn ủy quyền và ánh xạ sang A/B/C; các trường phụ thuộc measurement được điền theo quy tắc đã duyệt, không tự đổi công thức theo kết quả.

| Mục | Đề xuất cụ thể / quy tắc |
|---|---|
| Machine | 56267276, 4090 24GB; generator/eval không benchmark đồng thời |
| Main generator | SD1.5 frozen,512²,T64,DDIM eta0,CFG7.5,delta4; giữ pin model/RM/tokenizer đã kiểm |
| Native batching | Replay xấp xỉ; confirmatory results chạy online; không yêu cầu batch invariance |
| Prompt source | DiffusionDB metadata revision/SHA thực, ghi giấy phép; không tải ảnh nguồn nếu không cần |
| Filter | fastText English top1≥0.90; nsfw<0.1;4–60 CLIP tokens, giữ quy tắc chuẩn hóa đã có |
| Dedup | Full-pool connected components cosine>0.95; group split; GenEval text/template exclusion theo spec hiện hành; không nhìn test outputs |
| Default split | Calibration50;train_fit190;train_dev50;val80;val_ext220 niêm phong. Kích thước xác nhận full quyết định từ power trước nhìn val |
| Preview/chunk | Chọn theo validity/NaN/memory/speed trong M3; nếu spec chưa có tolerance cụ thể phải đưa giá trị trong gói duyệt, không ngụy tạo rằng đã duyệt |
| Reference | PSP-default chính; single64 và BoN4 đối chiếu đầy đủ |
| Caps | {0.25,0.5,1,1.5}×C_ref chung; cap không khả thi ghi NA; primary0.5/1 |
| Anchor | Mean online train IR_PSP−IR_BoN4, ghép cặp theo prompt/bank |
| Margins IR | tau=max(0.2×anchor,0.02IR);epsilon_IR=tau/2;theta_RL=tau/4 |
| RL scaling | sigma_R đề xuất = population SD của IR_deliv hợp lệ trên toàn candidate train_fit, floor1e−6; freeze trước train RL |
| RL failure | r_fail đề xuất = Q0.001(IR_deliv train_fit hợp lệ)−SD cùng tập; khai báo population quantile/SD implementation |
| RL selection | Ít nhất3 training seeds cuối; checkpoint/config chọn trên dev; không chọn seed trên val |
| Endpoint | Delivered IR là primary quality; actual e2e cost là primary efficiency; HPS/GenEval độc lập theo vai trò đã khóa |
| NI+savings | GO khi IR NI và LCB(savings)≥10%; STOP theo evidence quality-inferior hoặc UCB(savings)<5%; còn lại INC |
| Multiple looks | Nếu dùng val80 rồi mở220: alpha0.025 mỗi look, M khóa trước lần nhìn đầu; correction theo scalar contrasts thật |
| Precision CI | Bootstrap≥10,000 và đủ tail resolution theo alpha/M; tăng số bootstrap khi cần, không thay alpha |
| Full confirmatory N | Power80% từ paired train variance/effect mục tiêu; khóa N và split trước val; nếu vượt300 cần amendment trước xem outcomes |
| GenEval | Official prompt manifest/aggregation; đề xuất4 independent delivered images/prompt/method; freeze C trước generate/evaluate |
| HPS/GenEval margins | Mặc định report external/generalization; không claim NI các metric này nếu chưa khóa margin/power riêng |
| Search/ablation | Khóa search grids trên train/dev; ablation không phát triển trên val/test; log mọi trial |
| External | FK thật trên cùng backbone; Flash-BoN compatibility audit, không dùng B5 giả danh |
| SDXL | Targeted replication, adapter/costs/protocol riêng; cấu hình độ phân giải/schema khóa trước chạy |

Coding agent tạo một bản checklist điền đủ hash/giá trị còn phụ thuộc đo đạc, chỉ ra chính xác mục nào chưa có quyết định. Không biến mọi lựa chọn engineering thành câu hỏi research. Không xóa guard vì muốn chạy nhanh; đồng thời không để guard thiếu thông tin nằm im mà không đưa proposal có thể duyệt.

## 10. Khi nào dự án được gọi là hoàn tất?

**SD1.5 suite complete:** tất cả baseline bắt buộc, RL/control, online confirmation, held-out test, ablation, statistics và cost breakdown đã chạy thật; kết quả dương hoặc âm đều được bàn giao.

**Paper suite theo kế hoạch này complete:** thêm SDXL targeted và external comparison có phạm vi trung thực. Nhánh không chạy được phải ghi incomplete/unsupported; synthetic tests không thay số liệu CUDA.

Một research STOP hợp lệ có thể kết thúc bằng completed_negative khi bằng chứng đủ và nhóm chấp nhận dừng claim tương ứng. Thiếu PREREG, thiếu power hoặc chưa chạy test không phải completed_negative.

Hoàn thành thí nghiệm không bảo đảm method sẽ thắng hoặc paper được nhận. Mục tiêu thực thi là tạo bằng chứng đầy đủ, không ép RL thắng.

## 11. Nguồn kiểm chứng

- [R1] PSP tác giả: https://www.vision.caltech.edu/psp/ ; code https://github.com/rogerioagjr/PSP . Nguồn cho lịch pruning, overhead, SD1.5/SDXL và peak VRAM H200. Không dùng số H200 làm latency4090.
- [R2] Flash-BoN chính thức: https://github.com/flash-bon/flash-bon và https://flash-bon.github.io/ . Method có draft acceleration, selection/refine và verifier riêng; README có cấu hình Wan/FLUX.
- [R3] GenEval chính thức: https://github.com/djghosh13/geneval .
- [R4] HPSv2 chính thức: https://github.com/tgxs002/HPSv2 .
- [R5] FK-Steering chính thức: https://github.com/zacharyhorvitz/Fk-Diffusion-Steering .
- Tiến độ và machine/commit hiện tại lấy từ báo cáo Claude người dùng gửi ngày 16/10/2026; người soạn chưa truy cập repo private hay SSH để audit code. Agent thực thi xác minh hiện trạng trước chạy.

**Chỉ đạo thực thi:** chạy toàn chuỗi đến kết quả, không dừng ở “code đã sẵn sàng”; không lặp Stop test; gom quyết định khoa học; giữ 56267276 chạy đến khi hoàn tất mọi thí nghiệm. Job lỗi chỉ xử lý job; không Stop instance. Báo số liệu thật sớm và bàn giao toàn bộ bằng chứng.

