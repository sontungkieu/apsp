# Adaptive PSP — kế hoạch hoàn thiện và chạy bộ thí nghiệm

**Phiên bản:** 1.0 · **Ngày:** 16/10/2026 · **Người thực hiện:** Claude Code/Codex · **Research:** người dùng + ChatGPT/Claude web.

**Mục tiêu bàn giao:** bộ kết quả SD1.5 đầy đủ, có thể kiểm tra và tái lập; kế hoạch mở rộng SDXL/flow được định lượng riêng. Đây là kế hoạch thực thi, không phải bằng chứng phương pháp đã tốt hơn baseline và không thay thế chữ ký PREREG.

## 1. Quyết định và phạm vi

Mục tiêu chính là **giảm expected end-to-end inference cost trong khi chất lượng trung bình không thấp hơn tham chiếu quá epsilon**. Câu hỏi bổ sung: ở cùng chi phí, chất lượng có tốt hơn không; có cần paid verification và RL không?

\[
\min_\pi \mathbb{E}[C_\pi]\quad\text{s.t.}\quad
\mathbb{E}[Q_\pi-Q_{ref}]\geq-\varepsilon.
\]

- Primary reference đề xuất: PSP-default. Báo đầy đủ single, single ít bước, BoN và các baseline mạnh; chọn một reference chính để khóa giả thuyết và power, không loại các baseline khác khỏi đánh giá.
- Main SD1.5: frozen generator + ImageReward, 512², DDIM eta=0, T=64, CFG=7.5, delta=4, grammar 8→4→2. Giữ cấu hình đã thống nhất trong repo; thay đổi khoa học phải có amendment và version.
- Giữ **native batching**. Replay là simulator xấp xỉ để train/tune/sàng lọc; kết luận chính chạy online. Không tiếp tục debug kernel nhằm ép batch invariance.
- Không fine-tune generator. Method chính có thể là threshold/selector nếu RL không đem lại lợi ích; không sửa cách báo kết quả để ép RL thắng.
- Calibration, thu bank, train, val và test là các công việc có quan hệ phụ thuộc. Orchestrator chạy nối tiếp khi điều kiện đã đủ; không hỏi lại những quyết định triển khai thường lệ.
- Hoàn tất SD1.5 trước. Mở rộng backbone là phase trong kế hoạch paper, không được ghi là đã hoàn thành chỉ vì SD1.5 pass.

## 2. Máy hiện tại và giới hạn vận hành

Ảnh người dùng cung cấp: instance **56224386**, nhãn `NgocMinh_robotos_data`, hiển thị 1×RTX 4090, VRAM 48 GB, RAM 258 GB, CPU 64/256, SSD 200 GB, giá **$0.778/giờ**, trạng thái Running. Đây là thông tin UI, phải xác minh qua máy/API. RTX 4090 tiêu chuẩn có 24 GB [R6]; chưa được coi 48 GB hiển thị là bộ nhớ thật của một GPU.

1. Xác minh ID, GPU name/UUID/total memory bằng NVIDIA và PyTorch, CPU/RAM cgroup, disk còn trống, driver và runtime fingerprint. Nếu nhận dạng khác ảnh, ghi đúng thực tế; không đổi hardware fingerprint để giữ hash cũ.
2. Xác nhận workspace dành cho DMRL, không ảnh hưởng job/dữ liệu của dự án khác. Không tự xóa dữ liệu hay Destroy instance. Dùng thư mục và process group riêng.
3. Một GPU worker. Dùng CPU cho implement, thống kê và train controller nhỏ. Sinh ảnh và evaluator nặng chạy theo lịch để tránh tranh VRAM; benchmark latency phải đo khi không có workload gây nhiễu.
4. Giữ môi trường generator đã pin; không nâng Torch/diffusers theo CUDA banner. Môi trường HPS/GenEval riêng nếu dependency xung đột.
5. Bật và kiểm watchdog thật trước workload dài. Watchdog độc lập phiên SSH/Claude, bảo toàn artifacts, Stop đúng **56224386**; không dùng ID cũ. Phân biệt stop_requested, api_accepted và confirmed_stopped.
6. Giá mới khác kế hoạch RTX 3090 ≤$0.40/h. Cập nhật cấu hình giá/deadline sau khi có ngân sách hợp lệ; không âm thầm tắt kiểm tra giá. 5 giờ × $0.778 = **$3.89**, chưa gồm phần phí chưa được bao gồm. Ngân sách $2.50 cũ không cấp quyền chạy 5 giờ trên máy này.
7. Chưa có tổng ngân sách được duyệt cho cả dự án. Hoàn thiện code/kế hoạch là công việc đã được giao; không suy diễn thành quyền thuê máy khác, chạy không giới hạn, hoặc tự tăng cap chi phí.

**Nếu cần nhiều giờ lập trình trước khi dùng GPU:** hoàn tất trên laptop và Stop máy nhàn rỗi trong phạm vi đã được người dùng cho phép. Storage vẫn tính phí khi Stop [R5]; không tự Destroy để xử lý khoản này.

## 3. Nguồn chuẩn và hiện trạng cần audit

- Repo: `NguyenNgocMinh30012005/DMRL`, branch `bootstrap/p0`.
- Commit được báo gần nhất: `10690e44d43f1e77db14239694188ca4346799c9`; trước đó `ff82cd9f6023b0945316ef9ef7766f5f6edba175`. Đọc HEAD thực tế, không reset/checkout ghi đè công việc mới.
- PSP submodule read-only: `590f59f58384169c719e431dc01d14006fb0fc0c`.
- Đọc các bản hiện hành của HANDOFF, CONTRACT, PREREG, DECISIONS_LOG, QUESTIONS, CALIB1_SESSION_PLAN, PILOT_RUN_PLAN và configs. Kế hoạch này cập nhật ưu tiên cost-to-quality, native batching và máy mới; agent phải ghi rõ mọi xung đột với tài liệu cũ.
- Báo cáo hiện có cho biết phần baseline và runner mới đã test CPU/fake backend. Không gọi chúng là đã được nghiệm thu trên CUDA thật. Smoke/diag cũ là bằng chứng kỹ thuật của phiên trước, không phải kết quả mới trên 4090.
- Không sửa `third_party/psp`; không stage thư mục clone lồng `DMRL/`, weights, ảnh, bank, cache hoặc credentials.

## 4. Ma trận đối chứng phải hoàn thiện

| ID | Method | Công việc/kiểm chứng bắt buộc |
|---|---|---|
| B0 | Single, 64 bước | Không gọi RM để chọn ảnh; IR đánh giá sau delivery có chi phí riêng |
| B0-32 | Single, 32 bước | Scheduler T=32 thật, provenance riêng; không lấy checkpoint của quỹ đạo T=64 làm ảnh kết thúc |
| B1 | BoN-2/4/8; BoN-16 khi có điểm vận hành cần thiết | Chạy online đúng batch; không coi delivered score bank k=8 là ground truth k=1/2/4/16 |
| B2 | PSP-default | Chung sampler/scoring với method; ghi khác biệt với setup paper nếu có |
| B3 | Best-fixed trong grammar 105 lịch | Tune train-fit, chọn train-dev, đóng băng riêng từng cap |
| B4b | Grid PSP rộng | Hoàn thiện chọn theo cap; k_init≤16, keep không nhất thiết chia đôi, tối đa 4 cut theo grid gốc; tính đủ verifier |
| B5 | Screen-and-resume | M=8/16, một VERIFY, giữ q=1/2; ledger phải có fin(1) đúng |
| B6 | Threshold | Tune trong ngân sách tìm kiếm đã log; không tự tăng số tham số khi thấy val |
| B7 | Prompt-only selector | Fit train-fit; prompt embedding/PCA không fit trên dev/val/test |
| B8 | One-query rồi commit lịch | Giữ preflight, dùng lại query đúng checkpoint; CV theo prompt và fold preprocessing đúng |
| B9 | Random feasible | Sanity check; seed và budget mask được log |
| B10-min | Predictor rẻ, không VERIFY trung gian | Dùng feature được phép, fixed cuts; phí feature/predictor/prune và terminal selection phải đủ |
| RL-M1 | REINFORCE controller | Registry replay/online, checkpoint, normalization và ctrl costs đầy đủ; không nhầm tên với bước profile M1 |
| External | Phương pháp gần nhất phù hợp | Kiểm khả năng chạy cùng backbone; Flash-BoN chỉ được gọi đúng tên khi chạy method thật |

**Giới hạn B4b:** checkpoint không nằm trên bank grid phải có dữ liệu phù hợp hoặc đánh giá online; không tự làm tròn checkpoint. Pool 16 ghép hai bank 8 chỉ cung cấp replay xấp xỉ trong chế độ native batching. Một pool 16 tạo một episode mỗi prompt, không nhân đôi thành hai lần lặp độc lập.

**External baseline:** chuẩn bị compatibility matrix cho Flash-BoN/FK-Steering và các phương pháp gần nhất đã có code. Chọn đối chứng chạy được trên cùng backbone, nguồn chính thức, cấu hình và ngân sách công bằng. Nếu Flash-BoN chưa hỗ trợ SD1.5, giữ trạng thái blocked/deferred và đối chiếu ở backbone chung trong phase mở rộng; B5 không được đổi tên thành Flash-BoN [R2].

## 5. Lộ trình thực thi

### P0 — Audit và hoàn thiện khả năng chạy nối tiếp

**Làm ngay trên CPU:**

- Tạo `docs/FULL_EXPERIMENT_PLAN.md` từ kế hoạch này, bảng gap thực tế và checklist baseline. Không viết lại module đã hoạt động.
- Hoàn thiện B4b/B7/B10-min, tích hợp RL-M1 vào registry/benchmark, artifact đóng băng và báo cáo nếu còn thiếu. Tách nhiệm vụ cần bank khỏi skeleton có thể kiểm bằng synthetic.
- Hoàn thiện HPSv2.1 và GenEval adapters, pin dependency/code/weights/tokenizer/preprocessing. Audit package HPS có tự tải OpenCLIP weights khác không; pin mọi artifact thực sự dùng, không chỉ checkpoint HPS.
- Tạo một job manifest có trạng thái, dependencies, command/config hashes, đầu ra, predicted cost, resume rule và prerequisites PREREG.
- Implement một orchestrator tuần tự có resume, preflight ngân sách, chống chạy trùng, ghi lỗi và watchdog. Ví dụ interface cần tạo nếu chưa có: `python -m apsp.run_project --plan configs/full_experiments.yaml --resume`. Đây là API đề xuất, chưa phải lệnh đã tồn tại.
- Hoàn thiện đường xuất thống kê, bảng/figures và provenance. Không tự tạo kết quả khoa học từ fixture.

**Nghiệm thu:** chạy checks hiện hành và targeted tests có ý nghĩa: ngân sách/once/disp, no-leak, CandidateId, single32, pool16, B8 query reuse, train/dev/val guards, resume và evaluator isolation. Báo chính xác code path nào chưa chạy CUDA thật.

### P1 — Calibration trên đúng máy measurement

**Trình tự:** watchdog live check → xác nhận Stop bên ngoài → Start lại khi được phép → environment/fingerprint/model → full-pool check → M3/M9 → chọn preview/chunk → equivalence → M2/M2b.

- Giữ English top-1, p≥0.90 cho pilot; ghi hạn chế về phong cách prompt. Không đổi ngưỡng theo quality outcome.
- Near-duplicate: cosine >0.95 trên pool quy định, blocked computation, connected components; không tạo ma trận NxN trong RAM. Nhóm không được vắt qua calibration/train-fit/train-dev/val/val_ext. Xử lý xung đột trước khi freeze. Nhóm được xác định từ input, không từ score sinh ảnh.
- Khử trùng với văn bản GenEval theo quy tắc hiện hành (cosine >0.9/template trùng trong đặc tả cũ), pin encoder/manifest. Trước PREREG-C chỉ dùng văn bản cho bước này, không sinh/chấm ảnh test.
- Giữ calibration 50 prompt riêng; M9 subset 20 được xác định trước. Pilot size được chọn bằng throughput và ngân sách, không bằng kết quả method trên calibration.
- M3 so fp16/fp32 trên cùng x0, ghi validity trước clamp, NaN/failure, drift và thời gian. Dùng quy tắc đã có; nếu vượt ngưỡng lỗi phải giải quyết trước bank, không chọn precision vì IR cao hơn.
- Chunk: log actual CHUNK_N thống nhất giữa command và report; chọn bằng tính hợp lệ, bộ nhớ và tốc độ, không thay đổi kết quả chọn method theo ý muốn.
- M2 trước hết đo các k=1,2,4,8,16. Đo thêm k mà B4b thực sự cần trước khi dùng chúng. Coverage phải ghi rõ; chi phí controller mới chưa đo vẫn provisional.
- Đo single32 bằng sampler 32 bước riêng; không lấy nửa thời gian single64 làm số đo cuối.
- Equivalence nằm trong phiên này để đủ đầu vào PREREG-A; không yêu cầu tái hiện bitwise giữa batch size khác nhau.

**Nghiệm thu:** cấu hình sinh ảnh/scoring đã pin; pool không leakage; validity đạt quy tắc; bảng chi phí có coverage; M9 thực đo; equivalence có bằng chứng; job forecast mới theo GPU thực. Không tái dùng latency 3090 làm latency 4090.

**Đầu ra:** `calibration_report.md`, runtime fingerprint, generator cost table, machine config, pool manifest, proposal PREREG-A, dự toán toàn bộ P2–P7.

### P2 — Thu bank nghiên cứu một lần, kiểm và resume

- Chỉ bắt đầu sau PREREG-A thật và ngân sách đã đủ. Coding agent không tự ký.
- Kế hoạch nhỏ hiện hành: train-fit 190 prompt, train-dev 50, val 80, val_ext 220 niêm phong; calibration 50 riêng. Hai bank ×8 seed cho mỗi prompt nghiên cứu. Nếu repo có split đã được research duyệt khác, đối chiếu trước khi thay.
- Thu **train-fit/train-dev trước**. Chưa cần sinh full bank val/test để có dữ liệu train; đánh giá xác nhận sẽ chạy policy online. Thu val bank chỉ khi protocol đã duyệt thực sự yêu cầu.
- Không đổi EPH/BMH hoặc scoring giữa shard. validate_bank, checksum ảnh, đủ dòng, CandidateId, seed/hash và manifest trước khi dùng.
- Bật lưu latent train nếu đã có trong hợp đồng và phục vụ baseline/ablation đã chốt; không âm thầm thêm dữ liệu val.
- Resume shard nguyên tử, không tự ghi đè run cũ. Không rerun miễn phí episode lỗi rồi chỉ giữ lần thành công.

**Nghiệm thu:** bank đầy đủ/valid, không có prompt niêm phong bị truy cập, log collector throughput/RAM/VRAM và tổng chi phí. Lỗi dữ liệu thật dừng nhánh phụ thuộc; phần CPU độc lập tiếp tục.

### P3 — Baseline, ngưỡng và đo dư địa trên train/dev

- Fit preprocessing/PCA/normalization trên train-fit. Các fold của ridge/selector phải fit preprocessing riêng nếu dùng CV.
- Tune B3/B4b/B5/B6/B7/B8/B10 trên train-fit, chọn bằng train-dev. Online xác nhận một tập lịch/operating points được chỉ định trong protocol, vì replay selection có thể đổi thứ hạng khi batch thay đổi.
- Tính anchor, variance/power, epsilon và ngưỡng theo công thức đã duyệt; chuẩn bị PREREG-B. Không tự điền approved=true cho configs còn null.
- Chỉ chạy các script G1/headroom bị guard sau khi PREREG-B có chữ ký thật. Việc chuẩn bị anchor/ngưỡng trên train-fit dùng đường proposal được phép, không lách guard.
- G1 dùng **paired effect discrepancy** so PSP-default; không thay bằng winner agreement. Dù G1 tốt, final confirmation vẫn online.
- O1-replay/O1-free/O2/O3 là chẩn đoán đúng phạm vi. O1-replay không phải upper bound của online. Đánh giá online các lịch hindsight có triển vọng để có bằng chứng dư địa; chúng chưa phải controller triển khai được.
- Nếu replay dự đoán không có dư địa nhưng online chưa xác nhận, kết quả là INC. Replay mismatch tự nó không chứng minh dự án thất bại.

**Nghiệm thu:** bảng train-dev chất lượng–chi phí đầu tiên; baseline mạnh nhất/operating points được chọn; cổng headroom và các hạn chế rõ ràng. Không gọi thiếu power là không có hiệu ứng.

### P4 — Train và chọn controller

- Hoàn thiện/train B7, B8 và B10-min trước khi đánh giá vai trò RL. B10-min dùng đúng feature rẻ được phép, không lén đọc reward trung gian chưa trả phí.
- RL-M1 dùng REINFORCE/MLP trong handoff. Không đổi thuật toán lớn trước khi có kết quả bản tham chiếu. Đầu tiên kiểm training thật hoạt động trên bank; sau đó chạy tuning budget hữu hạn, log toàn bộ config và training seeds.
- Reward dùng terminal delivered IR và baseline B3 cùng episode/cap, sigma_R fit train-fit rồi freeze. Định nghĩa chính xác sigma_R phải được research duyệt trong PREREG-B; không thêm một giá trị tùy tiện để bỏ blocker.
- Chạy ít nhất 3 training seeds cho cấu hình cuối để đánh giá ổn định; không chọn seed thắng trên val. Quy tắc aggregate/checkpoint selection phải freeze trên dev.
- Chỉ thông tin đã mua/được phép vào observation. Final reward của trainer không được biến thành feature policy trước FINAL.
- Profile build_obs/mask/policy/feature, đo once và disp; cập nhật đúng controller hashes/cost coverage và tính lại feasibility. Đổi controller không làm mất bank nếu generator/scoring provenance vẫn phù hợp.
- Chọn operating point bằng **actual mean e2e cost thấp nhất trong các điểm đạt quality target trên train-dev**, không đơn thuần chọn alpha nhỏ nhất.
- Implementation hiện tại học dưới hard cap rồi chọn trên các operating points đã đo. Đây là cách tìm nghiệm trong family đang xét, không phải chứng minh đã giải tối ưu toàn cục bài toán cost-constrained-quality ở mục 1.

**Nghiệm thu:** model artifacts + norm/PCA + config + hash + ctrl costs; kết quả online dev; selected methods và comparator đóng băng. RL thua baseline đơn giản được ghi đúng; baseline đó vẫn có thể là candidate chính của Adaptive PSP.

### P5 — Đánh giá xác nhận trên val

- PREREG-B và artifacts đóng băng phải hợp lệ. Chạy tất cả baseline cần thiết tại các operating points đã chọn; không quét lại 105 lịch, toàn bộ grid và hyperparameter trên val.
- Cùng prompt pool, seed scheme, model revision, evaluator và điều kiện timing. Mọi method dùng đúng batch thật; CandidateId ghép cặp không có nghĩa pixel phải giống nhau giữa batch size.
- Báo mean delivered IR, HPSv2.1 theo vai trò đã khai báo, paired difference, latency mean/p50/p95, peak VRAM, failure rate, số VERIFY/candidate-forward, ledger và actual cap overruns.
- Chất lượng và savings phải được xác nhận đồng thời cho claim cost-saving. Nếu HPS mới chỉ được khai báo diagnostic trên val, không đổi policy dựa vào HPS rồi dùng chính val làm xác nhận.
- Nếu có lần nhìn val_ext thứ hai, chỉ mở theo protocol đã duyệt, cùng method/config, tổng val không vượt giới hạn. Không tự mở vì CI chưa đẹp hoặc tự tăng mẫu theo kết quả.

**Nghiệm thu:** bảng xác nhận với GO/STOP/INC theo đúng CI; kết luận riêng cho adaptive compute, paid verification và RL. Không nâng verdict chỉ vì một metric tốt.

### P6 — Test độc lập và kết quả paper

- Đóng băng toàn bộ method/operating points/evaluator/primary contrasts trước PREREG-C. Chỉ sau chữ ký thật mới sinh/chấm ảnh test GenEval.
- Dùng official GenEval prompt manifest và protocol [R3], pin detector/code/weights; không tự đổi protocol vài ảnh/seed rồi gọi cùng score với paper. HPSv2.1 pin đầy đủ [R4]. Eval có thể chạy venv riêng trên ảnh đã giao.
- Có single64, single32, BoN, PSP, best-fixed và các adaptive/learned baselines đã chốt trong bảng test; tách method không tương thích thành đúng phạm vi backbone.
- Giữ test tách khỏi train/dev/val theo quy tắc khử trùng đã duyệt. Test chỉ đánh giá, không sửa reward, epsilon hoặc chọn checkpoint.
- Báo GenEval overall/per-category, HPS và IR; thất bại sinh ảnh được tính theo quy tắc đã khai báo. Không giấu tác động lên nhóm prompt khó hoặc chỉ chọn ảnh đẹp.
- Human evaluation là hạng mục tùy chọn có protocol và ngân sách riêng; không tự thuê annotator, liên hệ người khác hoặc phát sinh khoản chi này.

**Nghiệm thu:** test report có CI, protocol/version, số ảnh/số prompt thực, failure, trường hợp bất đồng giữa metrics; không chỉ một con số IR trung bình.

### P7 — Ablation và cơ chế

Chốt danh sách trước tập xác nhận dành cho ablation. Có thể khảo sát trên dev, sau đó freeze các ablation cần xác nhận; không dùng val đã xem làm tập phát triển.

| Ablation | Câu hỏi |
|---|---|
| VERIFY bắt buộc kèm PRUNE, grammar A1 | Mua thông tin riêng có thực sự cần không? |
| Bỏ paid-score features | Controller có sử dụng thông tin verifier không? |
| Thêm trajectory features, tính đủ phí | Feature rẻ có giúp tiết kiệm hơn không? |
| B10-min không query giữa chừng | Có cần trả phí verifier thay vì dự đoán rẻ? |
| Threshold/one-query so RL, 3 training seeds | RL tạo thêm lợi ích đủ ổn định không? |
| Một policy điều kiện hóa budget so policy riêng cap | Có chuyển được giữa các mức ngân sách không? |
| Đổi reward preview@T/delivered nếu đã được duyệt | Khác biệt ảnh preview và ảnh giao ảnh hưởng lựa chọn ra sao? |
| Verifier cost ×{0,0.5,2} trong replay | Phân tích độ nhạy chi phí; gắn nhãn counterfactual, không phải latency thực đã đo |

Mỗi ablation cần ghi chi phí thay đổi và tính lại mask/caps phù hợp. Hai method chỉ được gọi là so cùng cap nếu số cap thực tế và thành phần phí so sánh được. Alpha giống nhau không tự bảo đảm cùng số microsecond khi method có overhead khác.

**Nghiệm thu:** giải thích được khi nào query/prune sớm có ích, khi nào mất quality hoặc tốn tiền; có action/query/cost breakdown và failure cases. Ablation không tạo cải thiện vẫn phải báo.

### P8 — Mở rộng SDXL/flow và đối chứng ngoài

- Bắt đầu khi SD1.5 có kết quả đáng tiếp tục và ngân sách mở rộng đã chốt. Ưu tiên SDXL, rồi một flow backbone được research chọn theo khả năng triển khai và bộ nhớ thực.
- Thực hiện ModelAdapter, hook x0, sampler, text feature và schema/hash phù hợp. Không áp công thức epsilon/DDIM của SD1.5 máy móc cho flow model. Không ép prompt embedding của backbone khác vào schema 768 chiều nếu nó không tương thích; version contract khi cần.
- Mỗi backbone có calibration, bank/costs, normalization, operating points và đánh giá online riêng. Không chuyển cost table hay chấm điểm thuộc pipeline cũ sang pipeline mới mà không kiểm chứng.
- Kiểm external baselines trên backbone chung. Flash-BoN thật có draft acceleration và multi-stage verification; B5 không thay thế đầy đủ method đó [R2]. Với phương pháp cần stochastic sampler, dùng cấu hình hợp lệ của tác giả và ghi khác biệt thay vì làm baseline suy biến bằng eta=0.
- Full matrix SD1.5 không cần lặp máy móc trên mọi backbone: chọn trước các đối chứng then chốt đủ kiểm tra claim, cùng các điểm ngân sách đã chốt theo protocol backbone đó.
- Với 4090 chuẩn 24 GB, không bảo đảm mọi backbone/batch lớn đều fit. PSP báo SDXL 13.27 GiB và SD3.5 34.85 GiB trên H200 trong setup của họ [R1]. Kiểm VRAM thực, rồi mới đề xuất GPU khác; không tự thuê hay offload/quantize và coi setup không đổi.

**Nghiệm thu:** ghi rõ backbone nào thực sự đã chạy, chất lượng–chi phí và khả năng chuyển giao. Không claim diffusion + flow nếu mới có SD1.5/SDXL.

## 6. Quy tắc thống kê và diễn giải phải giữ

### 6.1 Endpoint và đơn vị thống kê

- Prompt là cluster; bank nằm trong prompt. Pair các method trên cùng prompt và seed protocol. Pool 16 tạo một episode/prompt; không nhân đôi n bằng cách coi cùng pool là hai bank độc lập.
- Primary cost saving: `1 - mean(C_method)/mean(C_reference)`. Bootstrap ratio trên cluster, không lấy trung bình phần trăm tiết kiệm từng episode làm estimator thay thế mà không khai báo.
- NI dùng lower confidence bound của quality difference so với -epsilon. P-value không có ý nghĩa hoặc CI chứa 0 chưa chứng minh NI.
- Cost-to-target chọn trên dev, xác nhận trên val/test. Đường nối giữa các điểm frontier chỉ minh họa; không claim tại điểm nội suy chưa đánh giá policy/mix thực.
- Cùng ngân sách nhưng method không khả thi phải được đánh dấu; không cấp rescue miễn phí hoặc tự tăng budget sau khi hết tiền.

### 6.2 Ngưỡng và chữ ký

Giữ công thức trong handoff nếu bản hiện hành vẫn dùng chúng:

`Delta_anchor = mean(IR_PSP-default - IR_BoN4)` trên train.

`tau_prac = max(0.2 * Delta_anchor, 0.02 IR)`; `epsilon_IR = tau_prac/2`; ngưỡng lợi ích RL/paid-query đề xuất `tau_prac/4`.

Chi phí: GO khi quality NI đạt và LCB(savings)≥10%; STOP theo quy tắc đã duyệt nếu UCB(savings)<5% hoặc có bằng chứng quality kém quá margin; phần còn lại INC. Không bỏ khoảng INC để làm kết luận có vẻ dứt khoát.

- R_fail đề xuất theo handoff: quantile 0.001 của IR_deliv hợp lệ train-fit trừ SD cùng tập. Sigma_R, đơn vị lấy mẫu, norm/PCA phải được định nghĩa chính xác và đóng băng trong B. Không tái dùng penalty IR cho HPS vì khác thang đo.
- Epsilon GenEval/HPS và failure handling của từng metric phải được duyệt ở đúng vai trò endpoint. Mức GenEval 0.01 trong tài liệu cũ là proposal, không phải ngưỡng đã tự được ký hoặc đảm bảo đủ power.
- Trước val: sinh `contrast_manifest.json`, đếm M từ primary scalar contrasts thật. Giữ correction/multiple-look protocol đã duyệt, không tùy ý dùng CI95% cho mọi kết luận xác nhận.
- Nếu giữ hai look theo handoff: alpha_1=alpha_2=0.025, mỗi contrast/look dùng mức CI 1-0.025/M; M không giảm khi bỏ nhánh. Bootstrap tối thiểu 10,000 và tăng đủ phân giải tail; CI95% mô tả phải ghi descriptive.
- PREREG-A trước bank; PREREG-B trước G1/headroom/val; PREREG-C trước test. Agent tạo proposal và kiểm artifact signatures thật; không tự ký hay chỉnh guard. Gom các quyết định cần duyệt thành một gói rõ ràng, không hỏi lại quyền cho từng task thường lệ.

### 6.3 Replay, oracle và quyết định tiếp tục

- Amendment phải sửa giả thiết replay exact: eta=0 không đủ; cần numerical execution phù hợp và candidate independence. Với native batching, đổi k có thể đổi trajectory; 105 lịch không đồng nghĩa online chỉ có 8 ảnh cuối.
- G1 CI rộng → INC; native replay mismatch → ưu tiên xác nhận online và báo limitation. Lỗi implementation thật phải sửa; không tự xem mọi numerical drift là lỗi chặn dự án.
- Online chạy lịch hindsight s* tìm từ replay là một witness khả thi có dùng tương lai, không phải deployable controller và không phải upper bound của tối ưu online.
- O2-avg phải tách dual lower bound chi phí và primal witness. Chỉ có dual hoặc oracle replay tốt chưa chứng minh policy có thể học được lợi ích.
- Nếu RL không hơn B6/B8, hạ claim RL và giữ kết quả adaptive nếu nó đứng được. Nếu paid-query không hơn B10, hạ claim cần verifier giữa chừng. Nếu không còn lợi ích đủ đáng kể trên các cap chính với bằng chứng online đủ mạnh, báo negative/STOP thay vì mở rộng thêm compute vô hạn.

## 7. Đo chi phí đúng và công bằng

1. **Per-request inference:** prompt encoding/init, denoising, decode, score dùng để lựa chọn, feature/controller, prune, chuyển winner về CPU. C_fixed có thể ở ngoài ledger cap theo hợp đồng, nhưng vẫn nằm trong e2e đầy đủ.
2. **Đánh giá sau delivery:** IR/HPS/GenEval chỉ dùng để chấm kết quả được ghi thành chi phí evaluation riêng. Không cộng evaluator vào method single nếu nó không cần evaluator để ra ảnh; không bỏ verifier mà BoN/PSP/policy cần để chọn ảnh.
3. **Setup:** nạp model, compilation/warmup, tải weights. Báo riêng cold-start và warm inference. Không lấy model-load vài phút làm latency một request, cũng không giấu nó khỏi bill nghiên cứu.
4. **Nghiên cứu/training:** thu bank, tune, train controller, các thử nghiệm không được chọn, calibration và việc chạy lại. Báo tổng GPU/CPU time và bill; không chỉ cost của model cuối.
5. Ledger là accounting model; dùng ceil(mean_us) không bảo đảm mọi request wall-clock nằm dưới cap. Báo cap violation thực tế/p95. Cùng hardware/timing boundary và điều kiện đo cho các method.
6. Nếu setup/training có chi phí C_setup và mỗi request tiết kiệm Delta_C>0 theo cùng đơn vị, báo `N_break_even = C_setup / Delta_C`. Nếu chưa đủ dữ liệu hoặc Delta_C≤0 thì không claim đã hoàn vốn.

## 8. Ngân sách và lịch chạy toàn đợt

**Phần cần người dùng chốt một lần:** tổng tiền/tổng giờ được phép, bandwidth/storage reserve, các phase được phép chạy nối tiếp và quy tắc dừng. Đây là bước duyệt một kế hoạch cụ thể, không phải yêu cầu hỏi quyền lại sau mọi job.

Sau M9 trên máy thật, agent xuất bảng mỗi phase gồm:

`số prompt × bank × methods × operating points × training/evaluation seeds × repeats`, throughput đo được, GPU/CPU hours, transfer bytes, chi phí ước tính trung tâm/cận dự phòng và dependency.

- Không coi 33 GPU-giờ trên báo cáo cũ là full paper hoặc ETA trên 4090.
- Không nhân 0.778 với giờ của 3090 rồi coi là forecast cuối; đo throughput mới. Các con số sau chỉ là phép tính chi phí nếu chạy đúng số giờ đó theo giá UI hiện tại:

| Runtime giả định | Runtime × $0.778/h, trước phần phí chưa bao gồm |
|---|---:|
| 5 giờ | $3.89 |
| 12 giờ | $9.34 |
| 24 giờ | $18.67 |
| 48 giờ | $37.34 |

- Tính đủ compute, storage, upload/download theo hợp đồng thực [R5]; không cộng đôi storage đã nằm trong hourly quote. Ngân sách phiên và storage còn tính sau Stop phải ghi riêng.
- Giữ ngân sách tổng xuyên các lần restart. Không reset cumulative bill/deadline chỉ bằng đổi RUN_ID.
- Trước mỗi job, kiểm phần tiền/thời gian còn lại đủ chạy job và đóng gói/Stop. Khi chưa có ngân sách hợp lệ, không tự bật workload dài. Khi thiếu budget cho phase sau, hoàn tất báo cáo và dừng trong cap.
- Nếu phiên agent/SSH mất, watchdog vẫn chạy; download bị treo không được giữ máy chạy vô hạn. Artifacts cần có checkpoint trên disk và bản tải về có checksum.
- Implementation dài và controller training CPU nên được sắp lịch để GPU không ngồi chờ nhiều giờ. Chỉ tự Start lại khi đã có quyền/cơ chế điều khiển hợp lệ; không giả định container key hoạt động khi máy đã tắt.

**Thứ tự ưu tiên để có kết quả nhanh:** P0 CPU và P1 có lịch rõ → P2 bank → P3 bảng baseline/dev → P4 controller → P5 xác nhận → P6 test → P7 ablation cuối → P8 mở rộng. Những việc CPU độc lập/evaluator preparation được làm song song khi không ảnh hưởng timing benchmark. Không bỏ các điều kiện khoa học chỉ để chạy nhiều job hơn.

## 9. Bộ điều phối và báo cáo trạng thái

Đề xuất `configs/full_experiments.yaml` và `data/runs/<project_run>/job_manifest.jsonl`. Ví dụ field, phải ánh xạ sang contract thực tế trước khi dùng:

```yaml
project_run: apsp_full_sd15_v1
instance_id: 56224386
measurement_gpu: null          # điền bằng fingerprint thực, không suy từ UI
approved_total_usd: null       # người dùng chốt, không tự đặt
approved_max_wall_hours: null
actual_hourly_price: null
bandwidth_storage_reserve_usd: null
scope: [sd15]
prereg: {A: null, B: null, C: null}
resume: true
gpu_workers: 1
stop_instance_on_deadline: true
destroy_instance: false
```

Mỗi job cần có ID duy nhất, prerequisites, method/config/split hashes, status `pending/running/done/failed/blocked`, timestamps, exit code, output checksums và cost thực. Resume chỉ skip khi artifacts và provenance đúng; không skip vì tồn tại một file rỗng. Phân biệt bug engineering, dữ liệu lỗi, PREREG thiếu, budget hết và kết quả nghiên cứu âm.

Runner đi tiếp tự động khi prerequisites đã đủ. Khi bị chặn một nhánh, agent tiếp tục code/report nhánh độc lập trong phạm vi đã được giao. Nếu không còn GPU work hợp lệ thì xử lý Stop, không chờ người dùng nhiều giờ trong khi GPU chạy rỗng.

Không sửa science config để làm test pass hoặc qua guard. Contract/schema thay đổi có version và decision log đúng quy định. Cập nhật code/commit giữa run phải tạo run/provenance mới; không ghi kết quả commit mới thành kết quả commit cũ.

## 10. Đầu ra cuối cùng và tiêu chí hoàn thành

### 10.1 Tệp đề xuất trong repo

- `docs/FULL_EXPERIMENT_PLAN.md`, `docs/EXPERIMENT_STATUS.md`, `docs/DECISIONS_LOG.md` và prereg artifacts.
- `configs/full_experiments.yaml`, method grids, frozen method/operating-point manifests, evaluator locks.
- Báo cáo calibration, bank validation, anchor/power, replay-online discrepancy, headroom, baseline tuning, controller training và confirmatory evaluation.
- `reports/final/sd15_quality_cost.csv`: mean quality, effect/CI, NI, e2e savings/CI, p50/p95, failure, VRAM, query/compute count, actual n, verdict và provenance.
- `reports/final/quality_cost_frontier.*`, `cost_breakdown.*`, `ablation.*`, `test_metrics.*`, `failure_cases.*`, `amortization.*` và `REPRODUCE.md`.
- Model artifacts/data ngoài Git; manifest/checksums và đường dẫn rõ ràng. Gói bàn giao tải về được và kiểm checksum.

### 10.2 Khi nào được gọi là hoàn tất?

**SD1.5 complete:** tất cả đối chứng trong phạm vi đã duyệt đã chạy thật; RL và controls có artifacts; operating points freeze; xác nhận online và test độc lập đúng protocol; ablation cốt lõi, uncertainty, failure và cost đầy đủ. Nếu cổng nghiên cứu kết luận STOP đủ bằng chứng, báo completed-negative thay vì giả kết quả dương.

**Paper experiment suite complete:** thêm kiểm tra khái quát backbone/dataset và đối chứng ngoài theo claim cuối. Bất kỳ phần chưa chạy nào phải gắn blocked/deferred; synthetic pass không được thay thế số liệu CUDA/test.

**Một kết quả có thể viết thành claim chính:** chất lượng không kém quá margin đã chốt và chi phí thấp hơn đủ mức thực tế, trên dữ liệu xác nhận độc lập. Nếu method đắt hơn single nhưng chất lượng tốt hơn, báo đúng tradeoff; không gọi là tiết kiệm so với single. Không đòi hỏi một method thắng mọi baseline ở mọi operating point.

**Không bảo đảm:** hoàn tất suite không bảo đảm được nhận top-tier. Cần đóng góp và bằng chứng tương xứng; formulation POMDP, DP mask và số test pass không tự là bằng chứng chất lượng–chi phí tốt hơn.

## 11. Checklist Claude thực hiện ngay sau khi đọc

1. Trả lời ngắn cấu hình/giá/VRAM thực và HEAD đang dùng; kiểm ngân sách/watchdog của instance mới.
2. Tạo gap list và job manifest toàn đợt. Hoàn thiện các phần CPU/code thiếu ngay; không chỉ gửi một plan rồi dừng.
3. Đưa gói quyết định còn thiếu gồm ngân sách tổng, các trường science còn null và PREREG proposals, có giá trị/công thức đề xuất rõ. Phân biệt điều kiện cần ký thật với lựa chọn engineering tự xử lý được.
4. Chuẩn bị lệnh calibration đã cập nhật cho 4090; sau throughput thật, cập nhật dự toán các phase và chạy nối tiếp phần đã đủ điều kiện/ngân sách.
5. Báo tiến độ bằng kết quả và số tiền/giờ còn lại. Commit/push code theo quy trình repo; giữ nguyên artifacts cũ. Không chờ có kết quả đẹp mới bàn giao bảng đầu tiên.

## 12. Nguồn và căn cứ

- **[R1] PSP, nguồn tác giả:** https://www.vision.caltech.edu/psp/ — phương pháp, lịch mặc định, nhiều backbone, overhead và VRAM. Repo: https://github.com/rogerioagjr/PSP . Các con số H200 không phải benchmark máy hiện tại.
- **[R2] Flash-BoN, nguồn tác giả:** https://flash-bon.github.io/ — đánh giá wall-clock có verifier, draft-and-refine và multi-stage verification. Repo: https://github.com/flash-bon/flash-bon . B5 của ta không phải implementation paper này.
- **[R3] GenEval chính thức:** https://github.com/djghosh13/geneval — pin đúng bản code/protocol dùng trong thực nghiệm.
- **[R4] HPSv2 chính thức:** https://github.com/tgxs002/HPSv2 — kiểm checkpoint v2.1 và mọi pretrained artifact thực sự nạp.
- **[R5] Vast pricing:** https://docs.vast.ai/guides/instances/pricing — compute, storage và bandwidth là các khoản chi phí cần kiểm.
- **[R6] NVIDIA RTX 4090:** https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/ — bản tiêu chuẩn 24 GB; ảnh UI 48 GB phải xác minh bằng runtime.
- **[R7] fastText language ID:** https://fasttext.cc/docs/en/language-identification.html — bộ nhận dạng ngôn ngữ; threshold là lựa chọn lọc dữ liệu của dự án.
- **Nguồn tiến độ:** báo cáo người dùng cung cấp `Pasted text(8).txt`, `Pasted text(9).txt` và ảnh instance ngày 16/10/2026. Không có quyền truy cập repo/SSH thực trong lần soạn kế hoạch này; các trạng thái code dựa trên báo cáo, Claude phải kiểm lại trên repo thật.

**Quy tắc bàn giao:** báo rõ “đã chạy thật / chỉ synthetic / chưa chạy / blocked”; giữ mọi kết quả âm và giới hạn. Mục tiêu là một bộ bằng chứng có thể kiểm chứng về tiết kiệm chi phí và giữ chất lượng.
