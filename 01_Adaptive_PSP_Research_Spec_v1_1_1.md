# Adaptive PSP: nghiên cứu về mua thông tin verifier dưới ngân sách suy luận

**Research specification v1.1.1 — 09/10/2026**  
**Dành cho:** research team, người hướng dẫn, người viết paper.  
**Tài liệu đi kèm:** `02_Adaptive_PSP_Implementation_Guide_v1_1_1.md`.  
**Trạng thái:** đặc tả nghiên cứu và kế hoạch paper; chưa có kết quả GPU hoặc kết quả ảnh thật được xác nhận trong tài liệu này. Đây không phải bản manuscript đã chứng minh hiệu quả. Các PREREG-A/B/C vẫn cần chữ ký thật của research.

## 1. Quyết định nghiên cứu

Nghiên cứu một controller nhỏ điều phối **CONTINUE, VERIFY và PRUNE** khi nhiều seed của một diffusion model đang được denoise. Generator và verifier được giữ frozen. Controller quyết định khi nào nên trả chi phí để xem điểm verifier và khi nào đủ thông tin để loại candidate.

Pilot dùng **PSP** làm base, SD1.5 và DDIM tất định. Giữ grammar 8→4→2 để có oracle tính được bằng vét cạn. RL là một ứng viên; chỉ trở thành method chính nếu vượt các đối thủ đơn giản mạnh dưới cùng protocol chi phí.

Tên paper tạm thời, tùy kết quả:

- Nếu closed-loop thắng: *Adaptive Verification and Seed Pruning for Budgeted Diffusion Inference*.
- Nếu RL tạo lợi ích riêng: *Learning When to Verify: Budgeted Control of Diffusion Seed Pruning*.
- Nếu headroom nhỏ: *When Is Intermediate Verification Worth Its Cost in Diffusion Sampling?*

Không dùng “significantly surpasses recent methods” trong title/abstract trước khi có kết quả và baseline tương ứng.

## 2. Vấn đề và động cơ

Progressive Seed Pruning duy trì nhiều candidate ban đầu rồi giảm dần số seed dựa trên điểm của ảnh sạch dự đoán x̂₀. PSP là nền phù hợp vì cơ chế đơn giản, có code công khai, và không cần chỉnh trọng số generator [R1–R2].

Một lịch cố định trả phí verify tại cùng thời điểm cho mọi prompt. Nhưng ảnh trung gian có thể đủ thông tin rất sớm ở một prompt và vẫn chưa đáng tin ở prompt khác. Câu hỏi của dự án là **thông tin nào đáng mua, vào lúc nào, khi quyết định prune không thể đảo ngược và ngân sách phải đủ để giao ảnh**.

Budget thấp ở đây là **ngân sách inference cho từng request**. Tổng chi phí thu bank và huấn luyện controller là một đại lượng riêng, cần báo cáo để tính điểm hòa vốn.

Trong pilot, “chất lượng” được đo chính bằng ImageReward. IR không đại diện đầy đủ cho chất lượng thị giác, sự đa dạng hay sở thích con người. GenEval và HPS là kiểm chứng chuyển giao, không được dùng để tune policy.

## 3. Câu hỏi và giả thuyết có thể bác bỏ

| ID | Câu hỏi/giả thuyết | Đối chiếu |
|---|---|---|
| RQ1 / H1 | Trong grammar 8→4→2, còn headroom chất lượng hoặc chi phí đủ đáng nghiên cứu không? | O1, primal hindsight khả thi, B3 |
| H2 | Seed pruning có cạnh tranh được với dùng budget cho bề ngang không? | B4b và B5 theo từng cap |
| RQ2 / H3 | Một method triển khai được cải thiện chất lượng tại cùng cap hoặc giảm chi phí tại chất lượng tham chiếu không? | B6/B7/B8/RL đối với B3 và B2 |
| H4 | Quan sát tuần tự tốt hơn mua thông tin một lần rồi chọn lịch không? | Closed-loop đối với B8 |
| RQ3 / H5 | RL thêm lợi ích so với threshold và one-query selector không? | G4 |
| H6 | Lợi ích trên IR chuyển giao sang đánh giá độc lập không? | GenEval, HPS, có thể human study |
| H7 | Verifier trung gian trả phí có lợi hơn predictor từ feature rẻ không? | G3b, B10-min; kết luận giới hạn trong baseline đã thử |

Một kết quả G4 âm không bác bỏ toàn bộ ý tưởng adaptive PSP. Khi baseline đơn giản thắng, baseline đó có thể trở thành method chính.

## 4. Related work và novelty audit

| Công trình | Liên quan | Khác biệt cần kiểm chứng |
|---|---|---|
| PSP [R1–R2] | Base cho candidate pruning theo điểm trung gian | Dự án học khi mua thông tin, tách VERIFY khỏi PRUNE |
| Flash-BoN [R3] | Draft rẻ, lựa chọn candidate, nhấn mạnh wall-clock và verifier overhead | B5 là đối chiếu cùng bank; không được gọi là tái lập nguyên bản Flash-BoN |
| ELASTIC [R4] | RL điều phối compute cho diffusion/flow control policy, có candidate pruning | Miền robotics; đóng góp dự kiến của ta là quyết định mua verifier trung gian trong T2I, với baseline kiểm chứng trực tiếp |
| DDPO [R5] | RL fine-tune diffusion model theo reward | Ta train controller; không fine-tune denoiser |
| Feynman–Kac steering [R6] | Steering bằng particle reweighting/resampling | Pilot không nhân bản hay làm nhiễu seed; pruning loại candidate vĩnh viễn |

PSP và Flash-BoN được nguồn tác giả ghi nhận là ECCV 2026. ELASTIC được trích dẫn ở trạng thái preprint, không gán venue chưa xác minh. Appendix 7.4 của ELASTIC mô tả latency triển khai π0.5 có tính meta-policy, generator và verifier; không mở rộng kết luận đó cho mọi bảng của bài [R4].

**Không claim novelty cho:** replay bank, successive halving, budget DP, controller RL nhỏ, pruning từ trạng thái trung gian, hay dùng verifier để chọn seed. Đó là những thành phần hoặc hướng đã có.

**Đóng góp dự kiến:**

1. Formulation về thu nhận thông tin trả phí và pruning dưới hard ledger cap.
2. Một so sánh kiểm soát giữa lịch tune, prompt-only, one-query, threshold, RL và learned pruning không dùng verifier trung gian.
3. Một method cải thiện frontier chất lượng–chi phí, nếu thí nghiệm xác nhận.

Các mệnh đề lý thuyết dưới đây làm rõ tính đúng đắn; bản thân chúng không đủ tạo novelty cho paper top tier. Trước submission phải audit lại công trình song song và trạng thái venue từ nguồn chính.

## 5. Thiết lập cố định của pilot

| Thành phần | Giá trị |
|---|---|
| Base repo | `https://github.com/rogerioagjr/PSP` |
| Base commit | `590f59f58384169c719e431dc01d14006fb0fc0c` |
| Generator | `stable-diffusion-v1-5/stable-diffusion-v1-5`; revision và trọng số SHA phải pin khi triển khai |
| Sampling | DDIM, η=0, T=64, CFG=7.5, negative prompt rỗng |
| Resolution | 512×512 |
| Checkpoint | Δ=4 forward; j=1,…,16 |
| Grammar | `G-842-v1`: k₀=8, đúng hai prune 8→4→2 tại 1≤j₁<j₂≤15 |
| Reference | PSP-default j₁=4, j₂=8, tương ứng forward 16 và 32 |
| Verifier | ImageReward v1.0, frozen, pin trọng số |
| Scoring | Chung một đường tensor decode → PIL → RM cho bank, preview, terminal và online |
| Main controller | Observation 38 chiều, không trajectory feature |
| Trajectory ablation | Observation 42 chiều, có phí feature |

Các timesteps thực tế, precision decode, chunk VAE và runtime backend được chốt sau P0. Không thay thế số đo bằng cấu hình giả định. Ảnh preview tại j=16 và ảnh được giao cuối cùng phải lưu và chấm riêng.

## 6. Formulation: partially observed control với thông tin trả phí

### 6.1 Candidate và quỹ đạo

Cho prompt p và seed ξᵢ, generator frozen tạo quỹ đạo:

\[
x_i^{\ell+1}=F_\theta(x_i^\ell,t_\ell,p),\qquad \ell=0,\ldots,63.
\]

Preview tại checkpoint j là x̂₀ của forward thứ 4j, trước khi đồng nhất nó với latent sau scheduler step. Ảnh giao của seed i là \(I_i=D(x_i^{64})\). Đặt \(Y_i=IR(p,I_i)\), hoặc giá trị phạt hữu hạn nếu episode thất bại.

CandidateId là `(bank_id, seed_idx)`. Một episode chính dùng 8 seed của một bank. Baseline pool 16 dùng hai bank cùng prompt. Các hàng tensor luôn có cùng thứ tự với CandidateId.

### 6.2 State đầy đủ và information state

State đầy đủ chứa prompt, các latent còn sống, checkpoint, trạng thái sampler, thông tin đã mua và budget. Với η=0, dynamics là tất định trong mô hình toán học, điều kiện theo prompt và seed.

Controller chỉ thấy information state:

\[
h_t=(p, j_t, A_t, q_t, r_t, \mathcal H^{paid}_t, \mathcal H^{cheap}_t),
\]

trong đó A là candidate còn sống; r là budget còn lại; lịch sử paid chứa score đã mua; lịch sử cheap chỉ chứa feature đã được phép lấy và tính phí.

Main policy dùng \(o_t=\phi(h_t)\), một summary 38 chiều. Đây là **POMDP với history compression**; không khẳng định vector 38 chiều là Markov sufficient. MLP là lựa chọn thực nghiệm nhỏ, không phải hệ quả của một định lý sufficiency.

### 6.3 Actions

Trạng thái grammar rút gọn \(q=(j,k,f,p_r,v)\): f=score tại checkpoint hiện tại còn fresh; pᵣ=đã prune tại checkpoint này; v=có preview.

| Action | Điều kiện cấu trúc | Hiệu ứng |
|---|---|---|
| CONTINUE | j<16 và còn đường hoàn tất | Chạy thêm Δ forward cho candidate còn sống; j←j+1; fresh/pruned←False |
| VERIFY | 1≤j≤15, v, không fresh, k>2 | Decode và chấm candidate hiện tại; không đổi latent hay tập seed |
| PRUNE | 1≤j≤15, fresh, chưa prune, k>2 | Giữ top k/2 theo score vừa mua; giữ thứ tự hàng cũ |
| FINAL | j=16 và k=2 | Decode/chấm ảnh giao; chọn winner; kết thúc |

FINAL tự động, không phải action của policy. VERIFY ở terminal không được dùng để cứu một trace chưa hoàn thành hai prune. Hai prune không được diễn ra tại cùng checkpoint.

Tại PRUNE, score không hợp lệ được xếp cuối. Hòa điểm dùng CandidateId tăng dần. Policy quyết định **thời điểm**, còn phép chọn top-k là quy tắc cố định.

### 6.4 Objective

Với distribution prompt/seed và cap B:

\[
\max_\pi\ \mathbb E[Y(w_\pi)]
\quad\text{s.t.}\quad C^\pi(\tau)\le B\ \text{cho mọi trace hợp lệ}.
\]

Đây là ràng buộc ledger theo từng episode, không chỉ ràng buộc expected compute. Không thay hard cap bằng penalty λ trong method chính.

RQ2 riêng: giảm expected end-to-end cost khi mean quality không thấp hơn PSP-default quá δ_IR. Mục tiêu quality-constrained này được đánh giá bằng operating point đã chọn trên train-dev, không bằng chọn λ trên val.

## 7. Ledger và mask khả thi

### 7.1 Primitive không chồng lấn

Mỗi chi phí là số nguyên microsecond: \(c=\lceil mean\_us\rceil\), đo trên measurement machine.

- den(k): một UNet forward CFG và scheduler step.
- feat(k): feature trajectory cho một checkpoint, khi method sử dụng.
- ver(k): validity checks, preview decode, PIL và RM.
- prune(k→k′): ranking, deterministic tie-break, gather tensor và hook state.
- fin(k): terminal decode, lựa chọn và giao ảnh; RM chỉ cần khi k>1.
- lfeat(k), pred(k): feature và predictor của B10 tại cut.
- dispᵐ: build observation, mask, policy action và synchronization cho một decision.
- onceᵐ: setup riêng của method, gồm prompt feature/PCA hoặc selector work; trừ trước feasibility check.

**Công thức CONTINUE đúng:**

\[
c^m(q,CONT)=\Delta\,den(k)+u^m feat(k)+disp^m.
\]

Không cộng feat vào den rồi nhân Δ. Scheduled methods có disp=0 theo định nghĩa, nhưng ranking/gather vẫn bị tính ở prune. Các phần Python orchestration chưa được kế toán phải báo riêng.

### 7.2 End-to-end cost

\[
C^m=once^m+\sum_t c^m(q_t,a_t)+fin(k_{final}),
\qquad C^m_{e2e}=C^m_{fixed}(k_0)+C^m.
\]

C_fixed gồm text encoding, chuẩn bị latent và scheduler setup. Nó nằm ngoài cap nhưng trong endpoint e2e, và phải đo theo cấu hình startup thực tế; không mặc định k₀=1,8,16 có cùng C_fixed.

Nạp model, warm-up, artifact I/O không thuộc steady-state inference endpoint. Báo wall-clock riêng, cùng phần dư chưa kế toán. Hard ledger cap không bảo đảm hard wall-clock latency vì primitive dùng timing trung bình.

### 7.3 DP

Sau khi trả once, \(r_0=B-once^m\). Đặt Lᵐ(q) là chi phí hoàn tất nhỏ nhất:

\[
L^m(q)=\min_{a\in\mathcal A(q)}\{c^m(q,a)+L^m(\tau(q,a))\},
\qquad L^m(16,2,\cdot)=fin(2).
\]

State không có completion hợp lệ nhận +∞. Mask:

\[
\mathcal M^m(q,r)=\{a\in\mathcal A(q):c^m(q,a)+L^m(\tau(q,a))\le r\}.
\]

Không chạy episode nếu r₀<Lᵐ(q₀). Không dùng action cứu hộ miễn phí. Controller vẫn được tính disp khi mask chỉ cho một action. Các policy có phí hoặc grammar khác phải tính L của chính chúng.

### 7.4 Cap chung

Từ primitive của method cố định, không controller/trajectory:

\[
B_\alpha=L_0+\lfloor\alpha(C_{ref}-L_0)\rfloor,
\quad \alpha\in\{0.25,0.5,1,1.5\}.
\]

L₀ là completion rẻ nhất của G-842; C_ref là chi phí PSP-default. Hai cap chính là α=0.5 và 1. Các method cùng một cap dùng cùng B, nhưng feasibility khác nhau do phí riêng.

Chi phí fixed schedule hai prune:

\[
C(j_1,j_2)=\Delta[j_1den(8)+(j_2-j_1)den(4)+(J-j_2)den(2)]
+ver(8)+prune(8\to4)+ver(4)+prune(4\to2)+fin(2).
\]

## 8. Cơ sở lý thuyết và giới hạn của từng mệnh đề

### Mệnh đề 1: mask bảo toàn khả năng hoàn tất

Giả sử chi phí ledger không âm, là chi phí đúng của grammar, và state ban đầu thỏa r₀≥L(q₀). Nếu mọi action thuộc mask thì episode hoàn tất với C≤B.

**Chứng minh.** Bất biến tại state hiện tại là r≥L(q). Khi chọn a, định nghĩa mask cho r−c(q,a)≥L(τ(q,a)); bất biến được bảo toàn. Grammar có thứ tự tiến hữu hạn: CONT tăng j, VERIFY làm fresh=True, PRUNE giảm k và khóa prune tại j. Không có cycle cho phép hành động vô hạn. Tại terminal, budget còn lại ≥fin(2), nên hoàn tất không vượt cap. □

Mệnh đề áp dụng cho ledger đã định nghĩa. Không suy ra bound latency từ mean timing, và không áp dụng khi online tự thực hiện extra work ngoài ledger.

### Mệnh đề 2: replay tương đương trong sampler tất định độc lập theo candidate

Giả sử update và scoring của mỗi candidate không phụ thuộc vào candidate khác, cùng initial latent, timestep, model và numeric execution. Loại candidate không đổi quỹ đạo của survivor. Vì vậy bank của toàn bộ seed có thể replay mọi trace pruning mà không chạy lại UNet.

**Chứng minh.** Quy nạp theo forward: seed còn sống ở bước ℓ có cùng latent với bank; áp dụng cùng F cho cùng latent tạo cùng bước kế tiếp. PRUNE chỉ chọn subset, không sửa tensor survivor. Score và terminal decode giống nhau theo giả thiết. □

Trong code CUDA, thay batch size có thể đổi rounding/kernel. **η=0 không đủ bảo đảm bitwise equality**. M6 và G1 kiểm chứng mức sai lệch thực tế. Khi G1 chỉ cho phép replay để train, mọi cổng kết luận phải dùng online.

### Hệ quả 2a: 105 lịch, tối đa 8 ảnh giao trong một bank

Grammar chọn hai checkpoint từ 1,…,15, nên có \(\binom{15}{2}=105\) cặp (j₁,j₂). Với ranking và terminal selection đã cố định, mỗi cặp xác định survivor và winner. VERIFY không đi kèm PRUNE không thay quỹ đạo hay winner của một cặp đã cố định.

Có 105 lịch, **không phải 105 ảnh khác nhau**: winner thuộc tối đa 8 ảnh giao đã có trong bank. Extra VERIFY chỉ có thể giúp controller chọn cặp checkpoint tốt hơn, đồng thời tiêu budget.

### Mệnh đề 3: O1 là cận trên trong cùng pruning grammar

\[
O1_i(B)=\max_{s\in\mathcal S:C_0(s)\le B}Y_i(W_i(s)).
\]

C₀ là phí lịch cố định. Mọi trace main policy hợp lệ hiện thực hóa một cặp s; phí controller và extra VERIFY không âm, nên C₀(s)≤C_policy≤B. Do đó Y_policy≤O1_i(B).

Cận này dùng giả thiết replay và cùng paid ranking. **Không phải cận trên của B10 đổi ranking, pool 16, resampling hay model khác.**

### Mệnh đề 4: cận Lagrange cho mean-quality constrained hindsight cost

Với instance i=(prompt,bank), tập lịch khả thi Sᵢ và reference Yᵢᵣᵉᶠ, định nghĩa primal:

\[
P^*=\min_{s_i\in S_i}\sum_i C_i(s_i),
\quad\text{s.t.}\quad\sum_i[Y_i(s_i)-Y_i^{ref}+\delta]\ge0.
\]

Đặt Dᵢ(s)=Yᵢ(s)−Yᵢʳᵉᶠ+δ. Với λ≥0:

\[
LB(\lambda)=\sum_i\min_{s\in S_i}[C_i(s)-\lambda D_i(s)]\le P^*.
\]

Vì với mọi nghiệm primal khả thi, \(\sum_iD_i\ge0\), Lagrangian không vượt chi phí của nghiệm đó. Lấy max theo λ vẫn giữ bound. □

Tính chất này **không** khẳng định strong duality cho bài toán chọn lịch discrete. Giá trị LB lớn nhất tìm được bằng số là một bound, chưa chắc tight. Chia LB cho số instance trước khi so với chi phí trung bình.

Vì LB là cận dưới chi phí, savings suy từ nó là cận trên tiềm năng. G2-cost GO cần một **primal assignment khả thi** đạt target và mức tiết kiệm; không GO chỉ từ dual. Dual không đủ loại hướng nếu target δ khác target trong công thức.

O2-inst áp target theo từng instance nên chỉ dùng chẩn đoán. Mean-quality constraint cho phép bù trừ giữa prompt, nên O2-inst không phải cận dưới chi phí cho RQ2.

### Vai trò của value of information

Ở history h, giá trị của VERIFY là chênh lệch giữa continuation được quyết định sau khi nhìn score và continuation khi không nhìn, trong các action còn đủ budget. Không mặc định nó dương: trả verifier có thể làm mất một lịch tốt đang khả thi.

Không đưa định lý multinomial/missing mass từ trao đổi cũ vào core theory: hiện chưa có mapping xác lập giữa định lý đó và control problem này.

## 9. Method candidates

### 9.1 Observation

| Nhóm | Chiều | Nội dung |
|---|---:|---|
| Điều khiển | 11 | j/J; one-hot k; fresh/pruned; r/C_ref; L/C_ref; slack/C_ref; B/C_ref; n_verify/J |
| Paid scores | 8 | has_score; tuổi; mean/std/max score chuẩn hóa; cut gap; top gap; fraction invalid |
| Ổn định | 3 | has_two; Kendall τ; thay đổi cut gap |
| Prompt | 16 | PCA-16 của pooler_output |
| +traj | 4 | Mean/std drift và CFG gap hiện tại, có phí |

PCA và normalization chỉ fit train-fit. Observation luôn hữu hạn. No-score và invalid dùng mask/flag và số 0 hữu hạn; không truyền −∞ vào mạng.

Quy ước tính feature cụ thể được ghi trong implementation guide và đóng băng tại PREREG-A/B. Feature chỉ lấy từ candidate còn sống và thông tin đã mua hoặc được phép lấy.

### 9.2 Baseline chính

| ID | Method | Điều cần kiểm soát |
|---|---|---|
| B0 | Standard một seed | F(1) không gọi RM trong inference |
| B1 | Best-of-N, seed lồng nhau, N≤16 | Chọn N khả thi; cuối kỳ chấm để chọn |
| B2 | PSP-default (4,8) | Reference chất lượng/chi phí |
| B3 | Best-fixed trong 105 lịch | Tune train, chọn dev, riêng từng cap |
| B4b | Grid PSP rộng: k_init≤16, keep không nhất thiết halving, ≤4 cut | Đọc đúng grid tại commit; tính thêm verifier |
| B5 | Screen-and-resume với M∈{8,16}, một screen rồi giữ q∈{1,2} | Đối chiếu bề ngang cùng bank; không gọi là nguyên bản Flash-BoN |
| B6 | Threshold sáu tham số | Hai ngưỡng tuyến tính theo j, nhịp verify và j_start |
| B7 | Prompt-only selector chọn lịch | Nhãn đặc quyền chỉ train; phí setup/selector |
| B8 | Một query trả phí rồi cam kết lịch | Preflight completion trước query; tính toàn bộ trace |
| B9 | Random action khả thi | Sanity baseline |
| B10-min | Predictor từ feature latent rẻ, fixed cuts, không verifier giữa chừng | Feature/predictor/prune có phí; terminal RM có phí |
| M1 | REINFORCE, main 38 chiều | RL có phí controller; evaluation greedy |
| M1+ | M1 với trajectory | Tính thêm feature; không nhầm benefit với free information |

B10-CNN và closed-loop learned pruning mạnh hơn để Phase 5, chỉ bật sau quyết định research trên train-dev. G3b trong pilot chỉ hỗ trợ claim đối với B10-min.

### 9.3 RL

Policy MLP 2×128 → ba masked logits; value MLP riêng. Một policy điều kiện theo cap. Sample action trong train, greedy có tie-break cố định trong eval. Chỉ sample cap mà method khả thi.

Reward terminal:

\[
R=\frac{Y(w_\pi)-Y(w_{B3,\alpha})}{\sigma_R}.
\]

Reference B3 không phụ thuộc action của policy. Việc trừ nó không đổi expected-quality objective trong cùng sampling distribution. σ_R fit train-fit rồi cố định.

\[
\mathcal L=\frac1N\sum_{episodes}\sum_t
\big[-(R-\operatorname{sg}V(o_t))\log\pi(a_t\mid o_t,\mathcal M_t)
+c_v(V(o_t)-R)^2-\beta H(\pi_t)\big].
\]

Giữ mask trong normalization của distribution; action không khả thi có xác suất 0. Forced action có log-prob và entropy bằng 0. Adam 3e−4, β=0.01, c_v=0.5, clip gradient 1.0 là mặc định.

Tune tối đa 24 config ×3 initialization: lr∈{1e−4,3e−4,1e−3}, β∈{0.003,0.01,0.03}, hidden∈{64,128}, batch∈{64,256}, wd∈{0,1e−4}. Chọn config/early stopping chỉ trên train-dev; ghi mọi thử nghiệm.

## 10. Thí nghiệm và dữ liệu

### 10.1 Split

DiffusionDB được dataset card ghi CC0 [R7]. Dùng metadata/prompt, không cần tải toàn bộ ảnh nguồn. Lọc English, prompt_nsfw<0.1, 4–60 CLIP tokens. Near-duplicate cosine>0.95 được nhóm; split theo group. Loại cosine>0.9 với văn bản GenEval hoặc template trùng. Pin encoder dùng cho lọc.

| Pilot | train-fit | train-dev | val lần 1 | val_ext tối đa | Tổng val tối đa |
|---|---:|---:|---:|---:|---:|
| Nhỏ | 190 | 50 | 80 | 220 | 300 |
| Lớn | 320 | 80 | 200 | 100 | 300 |

Mỗi prompt có hai bank ×8 seed. Calibration pool riêng **50 prompt**; M9 dùng subset 20 đã định trước. Không dùng score calibration để quyết định kích thước pilot. Chọn pilot theo M9 và budget GPU-hours do user/team cung cấp, trước khi nhìn outcome của phương pháp.

GenEval gồm 553 prompt theo protocol dự kiến; triển khai kiểm lại official manifest [R8]. Trước PREREG-C chỉ đọc văn bản để khử trùng, không sinh/chấm ảnh GenEval. HPS trên val chỉ chẩn đoán, không dùng để chọn policy.

### 10.2 Replay và online

Thu bank một lần, train/screen bằng replay, kiểm G1 trên train-dev, rồi xác nhận cuối bằng online. Không đưa chi phí collector vào per-request inference; báo thành một bảng setup/training riêng.

G1 xét sai lệch **hiệu ứng ghép cặp**, không chỉ winner agreement:

\[
\Delta_{disc}=\overline{(Y_m-Y_{ref})_{online}}-
\overline{(Y_m-Y_{ref})_{replay}}.
\]

CI nằm trong ±0.25τ_prac: replay dùng cho cổng. Nằm trong ±0.5τ_prac nhưng không đạt mức trên: chỉ dùng replay train, cổng chạy online. Nếu chưa xác lập được tương đương do CI rộng, tăng kiểm tra train-dev trong budget; không gọi đó là bằng chứng parity. Sai lệch quá lớn: STOP-ENG và sửa engine.

### 10.3 Endpoint và operating point

RQ1: mean delivered Y theo cap; paired differences theo prompt/bank. Báo failure rate, số query/prune, ledger actual spend và wall-clock median/p95.

RQ2: trên train-dev, chọn **mean actual e2e cost thấp nhất** trong các điểm khả thi đạt quality target. Tie-break cố định trước val. Không chọn đơn thuần α nhỏ nhất. Confirm trên val bằng NI và tiết kiệm:

\[
LCB(\bar Y_m-\bar Y_{B2})>-\delta_{IR},\qquad
S_{e2e}=1-\bar C^m_{e2e}/\bar C^{B2}_{e2e}.
\]

Đường nối frontier chỉ để vẽ. Muốn claim tại điểm nội suy phải định nghĩa và đánh giá một randomized mixture policy thực sự.

### 10.4 Oracle

- O1: hindsight best schedule trong grammar và cap.
- O2-inst: cost-to-target theo từng instance, chẩn đoán.
- O2-avg: dual lower bound và **primal feasible witness tách riêng**, cho mean-quality target đã chọn.
- O3: chọn tốt nhất trong 8 ảnh giao, outer diagnostic; không mặc định đủ budget.
- O1-free: verifier miễn phí, optimistic diagnostic, không phải baseline thực thi.

Primal witness có thể giải bằng MILP trên bảng lịch hoặc solver khác trả được assignment kiểm lại. Chưa cần chứng minh solver optimal để có witness; chỉ cần feasibility và chi phí đúng. Không nhầm primal với deployable policy.

### 10.5 Ablation

Pilot: A1 gộp VERIFY+PRUNE; A2 bỏ paid-score feature; A3 +traj có phí; A4 nhân phí verifier {0,0.5,2}, tính lại L; A5 policy riêng từng cap; A6 reward preview@T đối với delivered reward.

Phase 5 nếu có nguồn lực: Δ∈{2,8}, k₀=16, verifier khác, objective λ, B10-CNN/closed-loop no-query. A4 là sensitivity của ledger, không thay cho đo lại wall-clock của một verifier khác.

## 11. Thống kê và preregistration

### 11.1 Đơn vị và nhiều lần nhìn

Prompt là cluster; hai bank lồng trong prompt, các method ghép cặp. Với pool 16, một episode dùng cả hai bank: vẫn bootstrap theo prompt, không coi nó là hai episode độc lập.

Trước lần nhìn val đầu tiên, PREREG-B khóa danh sách đầy đủ các **primary scalar contrasts**, tổng số M, gồm các cap và endpoint chất lượng/NI/chi phí. Candidate và đối thủ được chọn trên train-dev.

Hai lần nhìn tối đa: α₁=α₂=0.025. Mỗi contrast mỗi look dùng CI mức \(1-0.025/M\). M giữ nguyên khi loại cap hoặc không chạy một nhánh. Union bound kiểm soát tổng error allowance qua hai look; không dựa vào “thứ tự cổng” chưa chứng minh. Việc bootstrap xấp xỉ coverage phải được đánh giá riêng.

Cluster bootstrap percentile là mặc định; số lần resampling phải đủ phân giải tail ở mức CI đã điều chỉnh. Dùng tối thiểu 10.000 và tăng để mỗi tail có ít nhất khoảng 100 draw dự kiến; báo Monte Carlo sensitivity. CI 95% mô tả được ghi rõ là descriptive.

Lần nhìn 2 chỉ mở phần val_ext cần thiết, tổng val≤300; dùng lại config đã freeze, không tune. Kết quả vẫn INC sau look 2 thì ghi “inconclusive”, không chuyển thành bằng chứng không có hiệu ứng.

### 11.2 Power

Cho paired contrast với P prompt, n_b=2 bank:

\[
SE\approx\sqrt{\sigma_p^2/P+\sigma_b^2/(Pn_b)},\quad
MDE\approx(z_{1-\alpha_{eff}/2}+z_{0.8})SE,
\quad \alpha_{eff}=0.025/M.
\]

Ước lượng phương sai từ train, không từ val để tùy ý đổi mục tiêu. Đây là xấp xỉ planning; có thể kiểm Monte Carlo theo estimator thực tế. Không dùng hệ số 2.80 cho tất cả phép thử sau multiplicity adjustment.

### 11.3 Ngưỡng

Đặt Δ_anchor=mean Y(B2)−mean Y(BoN-4) trên train và:

\[
\tau_{prac}=\max(0.2\Delta_{anchor},0.02),\quad
\theta_H=\tau_{prac},\quad
\theta_q=\delta_{IR}=\tau_{prac}/2,\quad
\theta_{RL}=\theta_{pay}=\tau_{prac}/4.
\]

Chất lượng: GO nếu LCB≥θ; STOP cho hiệu ứng thực tế đó nếu UCB<θ; còn lại INC. “STOP” không có nghĩa hiệu ứng toán học bằng 0.

Chi phí: GO nếu NI đạt **và** LCB(savings)≥10%; STOP nếu UCB(savings)<5% hoặc UCB(quality difference)<−δ_IR; còn lại INC. Khoảng tiết kiệm 5–10% không bị ép thành GO hoặc STOP.

GenEval NI dự kiến δ_GE=0.01, đóng băng trong PREREG-C cùng family correction riêng cho test.

### 11.4 Các cổng

| Gate | Quy tắc |
|---|---|
| G0 | Các yêu cầu kỹ thuật bắt buộc trước bank đạt; không yêu cầu controller final đã train |
| G1 | Replay-online effect discrepancy trên train-dev đủ nhỏ |
| G2-quality | O1−B3 có headroom thực tế ở cap chính |
| G2-cost | Có primal hindsight witness đạt mean target và tiết kiệm; dual chỉ chẩn đoán/loại hướng khi đủ chặt |
| Breadth | B4b/B5 vượt O1 trong cap nào thì cap đó bị chi phối; không loại cap khác |
| G3 | Candidate chọn dev từ {B6,B7,B8,RL*} so B3; hoặc operating point qua NI và tiết kiệm |
| G3b | Best-paid chọn dev từ {B6,B8,RL*} so B10-min; chỉ quyết định claim paid verifier |
| G4 | RL* so đối thủ mạnh hơn trong {B6,B8}, chọn dev riêng mỗi cap; quyết định vai trò RL |

Project STOP nếu G2 thực sự STOP, mọi cap chính bị breadth chi phối, hoặc mọi nhánh G3 thực sự STOP. GO nếu không STOP và G3 GO. Trường hợp còn lại INC. RL được train bank thật sau trạng thái G2-evaluated và không có STOP nghiên cứu; G2 INC không được viết thành đã chứng minh headroom.

Không dùng max theo từng prompt để tạo “best baseline” triển khai được. Với G2-cost thiếu primal witness, đường cost là INC trừ khi optimistic upper bound đã đủ loại mục tiêu.

### 11.5 PREREG

- **A, trước bank:** machine/budget, prompt split, sampler/seed, precision, scoring, feature definitions, EPH, generator costs/caps, collector equivalence, P0 acceptance.
- **B, trước G1/headroom/val:** anchor, ngưỡng, failure penalty, reward scale, normalization/PCA; candidate selection protocol, primary contrasts và multiplicity; chỉ train/dev được dùng để tạo các giá trị này.
- **C, trước test:** method artifacts, config và operating point đóng băng; evaluator/seed protocol, primary endpoints/correction; chữ ký research thật.

Coding agents chỉ tạo proposal và kiểm guard; không tự ký.

## 12. Validity, lỗi và tính công bằng

Ghi riêng latent_ok, img_ok trước clamp/PIL, score_ok. Không có valid delivered candidate thì episode thất bại. Rank invalid bằng −∞ nội bộ; reward/observation luôn hữu hạn.

\[
r_{fail}=Q_{0.001}(IR_{deliv}\mid train\_fit,valid)
-SD(IR_{deliv}\mid train\_fit,valid).
\]

Đóng băng từ train-fit. Đây là heuristic, không guarantee thấp hơn mọi ảnh hợp lệ. Nếu không đủ dữ liệu hợp lệ để tính, STOP-ENG; không tự thêm giá trị mặc định. Report failure rates và sensitivity; không loại failure khỏi metric chính. GenEval failure tính không thành công. Không retry miễn phí.

M3 dùng 50 prompt×8 seed×16 checkpoint cho từng precision đang đánh giá. Bất kỳ mức lỗi >10⁻³ hoặc failure delivered >10⁻³ phải được research xử lý trước bank. Đổi VAE FP32 không sửa latent NaN.

Train/eval fairness: seed lồng nhau, scoring chung, ledger theo method, tune cùng split, log số config. B10 nhận feature rẻ có phí; policy main không được lén nhìn feature bank chưa mua.

## 13. Kế hoạch paper và bằng chứng cần có

### 13.1 Dàn ý manuscript

1. Introduction: vấn đề trả phí verifier và irreversible pruning; câu hỏi thay cho claim RL mặc định.
2. Related work: TTS, seed pruning, controller RL, verification cost; phân biệt post-training generator.
3. Problem formulation: information state, actions, budget, quality target.
4. Method: controller được chọn, DP mask, observation, training.
5. Experimental protocol: cost calibration, replay validation, baselines, statistical controls.
6. Results: headroom, fixed-budget quality, NI cost savings, necessity of RL/verification, independent evaluation.
7. Limitations: restricted grammar, reward dependence, batch effects, amortization.
8. Appendix: proofs, schemas, costs, tuning space, seeds, manifests, full per-cap/failure results.

### 13.2 Bảng/figure bắt buộc

| Artifact | Nội dung |
|---|---|
| T1 | Backbone, sampling, precision, verifier, machine và provenance |
| T2 | Mọi method: delivered IR, paired CI, actual cost, failure rate theo cap |
| T3 | Operating point: NI, e2e savings, measured wall-clock |
| T4 | G4 và G3b, đối thủ freeze trên dev |
| T5 | GenEval official aggregate/per-tag, HPS và NI trên test |
| F1 | Frontier actual e2e cost–quality, CI; không chỉ NFE |
| F2 | Oracle headroom, distribution lịch prune và query count |
| F3 | Replay-online discrepancy; không chỉ winner agreement |
| F4 | Benefit theo prompt/score uncertainty; phân tích đã khai báo là exploratory |
| Appendix | Primitive timing, all ablations, dual/primal gap, config counts, failures |

Một pilot dương trên SD1.5 chưa đủ chứng minh một paper top tier. Trước submission cần ít nhất kiểm chứng online/test đủ power; thêm backbone hoặc sampler hiện đại nếu muốn claim rộng về diffusion/flow. Phần mở rộng phải qua research decision riêng, không làm chậm P0.

### 13.3 Chi phí train và điểm hòa vốn

Report GPU-hours collector, profiling, parity và controller tuning riêng. Nếu controller tiết kiệm t_save giây/request và setup t_setup giây tương ứng thì rough break-even requests = t_setup/t_save. Không claim tiết kiệm tổng chi phí khi chưa vượt điểm hòa vốn.

## 14. Giao việc và quyết định tiếp theo

Claude Code sở hữu GPU engine/collector/online/evaluator. Codex sở hữu contracts/replay/DP/policies/stats/reports. Ownership không quyết định agent có GPU hay không.

P0 bắt đầu ngay từ tài liệu này. Trước thu bank phải hoàn tất measurement machine, prompt pool, generator profile, lỗi M3, batch/parity checks và PREREG-A. Không cần controller final để thu bank. Sau bank: train anchor → PREREG-B → G1 → G2 → train/chọn controller → đóng băng trước G3/G3b/G4 → PREREG-C → online test.

Các routine choices như CLI layout, vectorization và caching nội bộ được coding agents quyết định trong phạm vi đã chốt. Đổi scientific configuration, cap, split, reward, paid information, gate hoặc ngưỡng phải ghi QUESTIONS/DECISIONS và được research duyệt.

## 15. Nguồn tham khảo và nguồn quyết định

**Nguồn chính tra cứu ngày 09/10/2026:**

- **R1.** Guimaraes & Perona, *Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning*: https://arxiv.org/abs/2607.21591 ; author page: https://www.vision.caltech.edu/psp/
- **R2.** Official PSP repository: https://github.com/rogerioagjr/PSP ; base commit `590f59f58384169c719e431dc01d14006fb0fc0c`. Những ghi chú code chi tiết kế thừa audit của nhóm; coding agent kiểm lại tại commit này trong P0.
- **R3.** *Flash-BoN: Instant Drafts for Inference-Time Scaling in Diffusion Models*: https://arxiv.org/abs/2607.04461 ; https://flash-bon.github.io/ ; https://github.com/flash-bon/flash-bon
- **R4.** *ELASTIC: Efficiently Learning to Adaptively Scale Test-Time Compute for Generative Control Policies*: https://arxiv.org/html/2606.31132v1 ; đặc biệt §4 và Appendix §7.4.
- **R5.** *Training Diffusion Models with Reinforcement Learning*: https://rl-diffusion.github.io/
- **R6.** *A General Framework for Inference-time Scaling and Steering of Diffusion Models*: https://arxiv.org/abs/2501.06848 ; https://github.com/zacharyhorvitz/Fk-Diffusion-Steering
- **R7.** DiffusionDB official dataset card: https://huggingface.co/datasets/poloclub/diffusiondb
- **R8.** GenEval official repository: https://github.com/djghosh13/geneval

**Nguồn quyết định nội bộ:** bốn tài liệu handoff/contract/brief v1.1 người dùng cung cấp và amendment research v1.1.1 trong trao đổi ngày 09/10/2026. Tài liệu này cùng implementation guide đã tích hợp amendment; ưu tiên chúng khi một đoạn v1.1 mâu thuẫn.

## 16. Changelog khoa học v1.1 → v1.1.1

1. Phí feature một lần/block, không nhân Δ.
2. Ranking/tie-break nằm trong prune primitive.
3. O2 dual không được dùng như witness để GO; target NI thống nhất primal/dual.
4. G3 xét RL, tránh false STOP.
5. Family correction bao phủ toàn bộ primary contrasts qua hai look.
6. Cost GO≥10%, negligible STOP<5%; khoảng giữa INC.
7. Tổng val≤300, không cộng thêm 300.
8. Operating point chọn actual mean cost thấp nhất, startup cost theo method.
9. B8 preflight trước query; F(1) không gọi inference RM.
10. Code provenance và accessor phân biệt paid/cheap/prompt information.
11. Calibration pool 50; P0 không chờ controller final.
12. ELASTIC latency cập nhật theo Appendix 7.4; B10-CNN chưa bật pilot.
