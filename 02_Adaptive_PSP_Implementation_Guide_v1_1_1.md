Ưu tiên triển khai và chuẩn bị repo trên local trước khi thuê GPU.

Đọc hai tài liệu v1.1.1, làm trong phạm vi ownership và phối hợp
với Codex cho contracts/DP/replay/oracle. Implement engine,
collector và các lệnh P0 cần chạy trên GPU.

Chạy các kiểm tra CPU và synthetic có ý nghĩa. Kiểm cài đặt,
import và CLI từ một bản clone mới. Không báo CPU/synthetic
tests thành đã kiểm đúng trên CUDA hoặc ảnh thật.

Đẩy code lên repo GitHub private được chỉ định. Commit code,
config mẫu, tài liệu và báo cáo nhỏ; giữ weights, banks, ảnh,
cache và credentials ngoài Git.

Khi repo đủ để chạy GPU smoke test, bàn giao commit SHA,
hướng dẫn cài, các lệnh theo thứ tự và danh sách kiểm tra
chưa thực hiện. Không chờ hoàn thiện RL hoặc mọi baseline
mới bàn giao. Không tự thuê máy.

# Adaptive PSP: hướng dẫn triển khai và chạy thí nghiệm cho Claude Code / Codex


**Implementation guide v1.1.1 — 09/10/2026**  
**Dành cho:** Claude Code, Codex và engineering team.  
**Nguồn khoa học:** `01_Adaptive_PSP_Research_Spec_v1_1_1.md` (gọi tắt Research Spec).  
**Trạng thái:** đặc tả giao việc, chưa phải repo đã implement. API, CLI và test dưới đây là các deliverable agents phải tạo. Không có khẳng định test đã pass hoặc số đo GPU đã tồn tại.

## 0. Prompt khởi động cho Claude Code

Copy phần sau cùng hai file này và các file v1.1 hiện có vào workspace của coding team:

> Bạn là implementation lead cho Adaptive PSP. Đọc Research Spec và Implementation Guide v1.1.1 đầy đủ trước khi sửa code. Hai file đã tích hợp amendment research mới nhất và được ưu tiên khi v1.1 mâu thuẫn. Hãy kiểm repo thực tế, lập plan theo phase, rồi triển khai P0 trong phạm vi ownership. Bạn sở hữu GPU engine, model/hook/decode/scoring, profiling, collector, online runner và evaluator; Codex sở hữu contracts/replay/DP/policies/stats/reports. Dùng module dùng chung, không tự viết lại ledger, mask hay observation. Không sửa third_party/psp. Không tự ký PREREG, đổi scientific config hay dùng val/test để tune. Routine engineering choices được tự quyết và ghi RUNBOOK. Dùng synthetic fixtures để tích hợp khi chưa có contracts, nhưng fixture phải cùng schema và không trở thành implementation thay thế. Chạy kiểm tra cần thiết, lưu artifact/provenance, báo rõ đã đo hay chưa. Tiếp tục mọi task không phụ thuộc blocker. Sau P0, xuất acceptance report và proposal PREREG-A; chỉ thu bank pilot khi guard cho phép. Chưa có ngân sách GPU hay measurement machine thì hoàn tất CPU/code tasks và báo chính xác các trường còn thiếu; không tự thuê máy hoặc đoán thông số. Báo phase theo mẫu ở cuối guide.

**Đây là yêu cầu implement và nghiệm thu, không chỉ viết một kế hoạch khác.**

## 1. Thứ tự ưu tiên và quyền sửa

1. Research Spec v1.1.1: scientific configuration, objective, data, baseline, thresholds, claims.
2. Guide này: runtime contract và acceptance requirements.
3. Handoff/contract/brief v1.1: giữ phần tương thích; các lỗi đã nêu ở v1.1.1 phải sửa.
4. Repo/code hiện tại: kiểm rồi reuse; không tin một toy test là bằng chứng runtime correctness.

Không sửa grammar, reward, scoring, seed scheme, split, cap, threshold hoặc paid-information rules bằng quyết định engineering. Ghi câu hỏi cụ thể vào `docs/QUESTIONS.md`, nêu phương án và ảnh hưởng; chỉ dừng task phụ thuộc quyết định đó.

Không cần research approval riêng cho CLI, typing, caching không đổi hành vi, vectorization hay sửa bug trong scope. Ownership không đồng nghĩa hardware: job GPU chạy trên measurement machine đã chỉ định, ai có quyền truy cập cũng có thể chạy, owner vẫn review code.

## 2. Repo và phân công

```text
adaptive-psp/
  third_party/psp/                 pinned read-only submodule
  reference/budget_dp.py           legacy reference, read-only
  apsp/
    contracts/                    Codex
      hashing.py manifest.py schema.py identity.py
      ledger.py paid_state.py access.py policy_api.py prereg.py
    data/                         Codex
      prompts.py geneval_text.py synthetic_bank.py
    sim/                          Codex
      budget_dp.py env.py features.py schedules.py oracles.py caps.py
    policies/                     Codex
      fixed.py threshold.py selectors.py screen_resume.py
      b10_predictor.py reinforce.py artifacts.py
    stats/                        Codex
      variance.py bootstrap.py power.py frontier.py decisions.py
    reports/                      Codex
      anchor.py headroom.py pilot_report.py replay_vs_online.py ctrl_costs.py
    gpu/                          Claude Code
      env_check.py model.py hooks.py validity.py decode.py scoring.py
      engine.py profile.py collect_bank.py online.py
      evaluators/                 HPS and GenEval adapters
  configs/
    pilot.yaml                    Research scientific values
    machines.yaml                 actual machine + Research spending budget
    env.lock                      Claude Code creates; Research freezes
    primary_contrasts.yaml         Research preregisters
  docs/
    RESEARCH_SPEC.md IMPLEMENTATION_GUIDE.md PREREG.md DECISIONS_LOG.md
    QUESTIONS.md RUNBOOK.md PREREG_PROPOSALS/
  tests/
    contract/                     shared review
    sim/                          Codex
    gpu/                          Claude Code
  data/                           gitignored
    prompts/ banks/ costs/ artifacts/ runs/ evaluation/
  reports/                        measured/generated reports
```

Claude Code tạo skeleton/root packaging/submodule. Codex tạo file trong ownership của mình. Tạo `.gitignore` cho weight/data/run caches; không gitignore config/provenance nhỏ cần review. Không đưa token hay credential vào log.

Nếu repo đã có implementation, inspect diff và reuse thay vì tạo bản thứ hai. Thay đổi interface phải được hai owner ghi nhận trong QUESTIONS/DECISIONS và bump version; không âm thầm đổi tên/cost semantics.

## 3. Config cố định và trường cần điền

`pilot.yaml` phải biểu diễn tối thiểu các giá trị sau, đọc trực tiếp từ Research Spec; mọi module dùng cùng một config loader:

```yaml
spec_version: apsp-v1.1.1
psp_commit: 590f59f58384169c719e431dc01d14006fb0fc0c
model_repo: stable-diffusion-v1-5/stable-diffusion-v1-5
model_revision: null             # resolve once, pin actual revision
scheduler: DDIM
eta: 0.0
steps: 64
delta: 4
cfg: 7.5
negative_prompt: ""
height: 512
width: 512
grammar: G-842-v1
k0: 8
keep: [4, 2]
reference_cuts: [4, 8]
alpha: [0.25, 0.5, 1.0, 1.5]
primary_alpha: [0.5, 1.0]
seed_scheme: apsp-v1-sha256-cpu
scoring_path: ir_pil_v1
preview_dtype: null              # proposed after M3, research freezes
deliver_dtype: fp16
vae_decode_chunk: null           # selected in P0
calibration_prompt_count: 50
throughput_subset_count: 20
max_val_total: 300
max_val_looks: 2
cnn_enabled: false
```

Machine config ghi actual CPU, GPU model/VRAM, RAM/cgroup limit, driver, CUDA, machine_id, agent access, measurement role và budget. Không hard-code rằng máy là 4090. Agent có thể phát hiện facts, nhưng research chọn measurement machine và điền `pilot_gpu_hours_budget`; không tự tạo spending authorization.

Không freeze số timestep theo một dãy giả định. Dump scheduler config và timesteps của actual pipe. Pin tokenizer, text encoder, RM và evaluator revisions/artifacts.

## 4. Task DAG và điều kiện bắt đầu

| Phase | Claude Code | Codex | Điều kiện kết thúc |
|---|---|---|---|
| P0 | Env/model/hooks/scoring/core engine, primitive profile, validity, parity, collector equivalence, throughput | Contracts, prompt pool, DP/replay/oracle/features/stats, synthetic fixtures, baseline code | Required P0 checks đạt, prompt pool duyệt, PREREG-A signed |
| P1 | Thu bank; chạy online G1 theo fixtures/specs freeze | Validate bank; fit train anchor/PCA/norm; proposal B; G1 report | PREREG-B signed; G1 status rõ |
| P2 | Online gate runs nếu replay không đủ parity | Headroom/breadth; dual + primal; gate report | G2-evaluated; không STOP nghiên cứu trước train thật |
| P3 | Online confirm; đo controller thật cùng script Codex | Train/tune baselines/RL trên train; freeze candidates; G3/G3b/G4, ablation | Controller costs không provisional; report đúng statistical protocol |
| P4 | GenEval/HPS/online test; R0 reference reproduction | Frozen report/effect CIs/frontier | PREREG-C signed và test budget authorized |
| P5 | Backbone/grammar/verifier extensions nếu được duyệt | B10-CNN, stronger no-query control và các extension | Research decision riêng |

Code baseline/RL và test synthetic làm được từ P0. **Không yêu cầu controller final để thu bank.** Core generation code phải tồn tại và freeze trước EPH/bank; `engine.py` không được trì hoãn toàn bộ đến P1.

## 5. Contracts trước tích hợp GPU

### 5.1 Identity và stable rows

```python
CandidateId = tuple[int, int]  # bank_id, seed_idx

# Target interface, to implement in apsp/contracts/identity.py
class RowMap:
    rows: tuple[CandidateId, ...]
    def retained_indices(self, keep_ids: set[CandidateId]) -> tuple[int, ...]: ...
    def apply_keep(self, indices: tuple[int, ...]) -> None: ...
```

Pool 16 order: `(0,0)…(0,7),(1,0)…(1,7)`. Rank decreasing finite score, then increasing CandidateId. Invalid ranks last. Sau chọn subset, tensor hàng vẫn theo thứ tự hàng cũ, không theo rank score.

Assert sau mọi prune:

- `len(rows)==latents.shape[0]==state.k`.
- `prompt_embeds.shape[0]==2*state.k` với CFG `[uncond rows; cond rows]`.
- state.alive, hook tensors và scheduler candidate state cùng RowMap.
- Không dùng seed_idx làm pool-row khi có hai bank.

### 5.2 State và views

`EpisodeState` mutable thuộc env/runner, không giữ reference tới toàn bank. `PolicyView` immutable chứa:

- grammar id, j/k, fresh/pruned/preview;
- alive CandidateIds, remaining/cap/L_here/C_ref/n_verify;
- paid histories của survivor, mỗi score có j và ba validity flags;
- optional paid-for trajectory history;
- prompt PCA feature read-only.

State methods: `arrive(j)`, `charge(us)`, `record_verify(j,scores)`, `record_prune(keep)`, `view()`. `arrive` không cộng phí. `charge` reject negative/non-integer fee và overflow. `record_verify` reject wrong j/candidate/duplicate query; `record_prune` reject missing fresh scores hoặc keep không hợp lệ.

Main observation chỉ đọc PolicyView. B10 predictor nhận `CandidateFeatureView` riêng chứa feature hiện tại đã được trả `lfeat`, không nhận whole-bank accessor.

### 5.3 Policy, schedule, delivery

```python
CONT, VER, PRUNE = 0, 1, 2

class Policy(Protocol):
    method_id: str
    grammar_id: str
    uses_traj: bool
    cph: str
    def reset(self, prompt_feat: np.ndarray, cap_us: int) -> None: ...
    def act(self, obs: np.ndarray, mask: tuple[int, ...]) -> int: ...

@dataclass(frozen=True)
class Cut:
    j: int
    keep: int
    source: Literal["paid", "pred"]

@dataclass(frozen=True)
class ScheduleSpec:
    candidates: tuple[CandidateId, ...]
    queries: tuple[int, ...]
    cuts: tuple[Cut, ...]
    final_k: int

class Scheduler(Protocol):
    method_id: str
    cph: str
    def plan(self, view: PolicyView, mc: MethodCosts, cap_us: int) -> ScheduleSpec: ...
```

Schedule validation: candidate unique; 1≤cut.j≤15, tăng ngặt; keep giảm ngặt; paid cuts có query cùng j; không duplicate query; final_k đúng keep cuối. B8 plan sau query phải biểu diễn full trace từ j=0, gồm query đã mua, để audit chi phí; executor không chạy lại các event quá khứ.

`Delivery` chứa winner CandidateId hoặc failure, image/validity, RM-used flag, optional inference score, actual ledger spend và wall-clock. F(1) hợp lệ không bị coi thất bại chỉ vì inference score chưa chấm. Evaluator chấm IR sau inference khi cần.

### 5.4 CostTable và API chung

```python
class MethodCosts:
    den: dict[int, int]
    ver: dict[int, int]
    fin: dict[int, int]
    prune: dict[tuple[int, int], int]
    feat: dict[int, int]
    lfeat: dict[int, int]
    disp: int
    once: int
    pred: dict[int, int]
    fixed: dict[int, int]          # startup configurations
    uses_traj: bool

def cost_gen(q, action, mc, delta=4) -> int: ...
def cost_decision(mc) -> int: ...
def cost_action(q, action, mc, delta=4) -> int: ...  # cost_gen + cost_decision
def reserve_once(cap_us, mc) -> int: ...
def schedule_cost(spec, mc, delta=4) -> int: ...
def solve_L(grammar, mc) -> Mapping[GrammarState, int | float]: ...
def mask(q, remaining_us, L, grammar, mc) -> tuple[int, ...]: ...
def build_obs(view: PolicyView, norm: NormStats, variant="main") -> np.ndarray: ...
```

**Nguồn cộng phí duy nhất là ledger.py.** GPU runner gọi API này, không tự copy công thức.

## 6. Ledger, DP và ordering thực thi

### 6.1 Mapping sửa từ reference

Không sửa `reference/budget_dp.py`; port/generalize sang `apsp/sim/budget_dp.py`.

| Reference concept | Implement mới |
|---|---|
| `c_den[k]` | den[k] một forward, không gồm feature |
| CONT block | Δ·den[k] + uses_traj·feat[k] + disp |
| `c_ctrl` | disp cho từng action closed-loop |
| `c_ver` | ver |
| `c_prune` | prune[(k,k′)] gồm ranking+tiebreak+gather |
| Terminal cost | fin[final_k] |
| Initial budget | cap−once trước mọi feasibility check |

Test regression: Δ=4, den=10, feat=3 thì CONT generation fee=43, không phải 52; thêm disp riêng.

### 6.2 Decision ordering duy nhất

Tại state q, budget r **trước dispatch hiện tại**:

1. Tạo view với L(q) đã gồm future dispatch costs.
2. Tính mask bằng r; mask action cost đã gồm dispatch hiện tại.
3. Build observation và gọi policy.act, kể cả forced action.
4. Validate action thuộc mask.
5. Charge `cost_action(q, action, mc)` đúng một lần; đây là `cost_gen + disp`.
6. Engine thực hiện action, cập nhật state/rows/info.

Không charge disp rồi dùng lại mask trên r đã giảm trong khi action cost vẫn gồm disp. Không expose budget khác nhau giữa replay và online.

FINAL tự động khi (j=16,k=2): charge fin(2), không thêm disp. States (j=16,k>2) invalid. Mask reserve đúng terminal cost và hai prune ở checkpoint khác nhau.

### 6.3 Schedule cost

Schedule runner không gọi policy ở từng checkpoint, disp=0. Tính cost toàn trace trước chạy, gồm once, các đoạn den, unique queries, cuts theo paid/pred source và fin.

Paid cut: ver tại query + prune. Pred cut: lfeat + pred + prune, không ver. B10 vẫn terminal RM để chọn top cuối khi k=2.

B8 preflight trước query:

\[
\min_{j_1\ge j_c,\;j_2>j_1}C_{B8}(j_c,j_1,j_2)\le B.
\]

Nếu không có completion, ghi infeasible **trước khi trả query**. Nếu j₁=j_c, reuse score tại query, không trả ver(8) lần hai. Nếu j₁>j_c, có extra ver(8) tại j_c. Query feature/selector cost thuộc once đã dự trữ, không miễn phí.

## 7. GPU engine — task Claude Code

### CC-0: skeleton, dependency và reproducible environment

- Kiểm working tree và instructions local trước thay đổi; tạo skeleton nếu thiếu.
- Submodule PSP pin đúng commit, kiểm sạch, read-only.
- Đọc `third_party/psp/Fk-Diffusion-Steering/requirements.txt`; root requirements rỗng theo audit cũ, phải kiểm lại.
- Python 3.10/setuptools 69.5.1 là starting recipe của v1.1; resolved dependency thực tế ghi env.lock. Không silent-upgrade diffusers làm đổi scheduler.
- Tạo GPU env cho generator/ImageReward/HPS và evaluator env riêng nếu GenEval dependency xung đột. Pin cả hai, ghi liên kết phiên bản.
- `env_check` ghi actual devices, CPU, RAM và cgroup limit, CUDA/driver/torch, attention backend, thread counts.
- `cudnn.benchmark=False`, `cudnn.deterministic=True`; eval/no_grad hoặc inference_mode. Nếu kernel không deterministic, ghi rõ và xử lý parity thay vì giả định flag là guarantee.

**Acceptance:** clean install/import smoke; RUNBOOK dựng lại được; không sửa third_party; actual machine report, chưa đoán budget.

### CC-1: model loading, scheduler và provenance — M1

- Load mirror SD1.5, resolve exact revision, lưu repo/revision và sha256 weights UNet/VAE/text encoder/tokenizer configs.
- Dùng wrapper/subclass tương thích FKDStableDiffusion; DDIM config như base; η=0.
- Không rely `runwayml/stable-diffusion-v1-5` hard-code hay floating revision.
- Dump toàn scheduler config, timesteps, ᾱ và prediction_type; assert64 steps, negative prompt rỗng, CFG/resolution đúng.
- Đọc actual terminal update để giải thích vì sao preview@T khác delivered latent; report coefficients đo/tính từ config, không hard-code hệ số từ trao đổi cũ.
- Load RM theo base nhưng kiểm missing/unexpected keys; missing_keys phải rỗng, unexpected nếu có phải giải thích/resolve.

**Acceptance:** manifest inputs đủ; hash stable với cùng configuration; repeat load không đổi revision/weights; report M1.

### CC-2: hook không thêm forward — M0

Hook lấy `pred_original_sample` từ scheduler result và noise predictions cond/uncond đã tính. Mở rộng callback wrapper nếu cần; không sửa submodule và không chạy thêm UNet để tạo preview.

| Mode | Work |
|---|---|
| none | Giữ current x0 reference cho possible VERIFY; không tính norm/drift/gap |
| traj | Tại checkpoint, tính drift/gap và chuyển reduced features; charged feat |
| lfeat | Giữ previous checkpoint x0; chỉ cut mới tính feature B10; charged lfeat |
| collect | Tính mọi checkpoint feature; thu bank, không được đem overhead collector thành phí policy main |

Giữ tensor reference an toàn; không để in-place mutation làm đổi previous x0. Gather hook state theo RowMap khi prune. Reduced statistics thực hiện FP32. M0 kiểm same batch repeat; hook x0 so hàm collector gốc; η=0 DDIM inversion chỉ chẩn đoán.

**Acceptance:** hook không thêm forward; none không chạy trajectory-feature kernels; aligned rows; report max error và exact-match flags, không chỉ assert chung chung.

### CC-3: validity, decode, PIL scoring — M3/M4

Tạo một scoring module dùng cho collector, preview và terminal k>1:

1. Check `isfinite` per latent row.
2. Decode bằng path cố định, đúng latent scaling theo base.
3. Check decoded tensor trước clamp/PIL.
4. Chỉ valid image rows được postprocess PIL và chấm RM batched.
5. Map score về CandidateId ổn định; record finite score flag; invalid lưu NaN trong bank, None trong view.

Preview/delivered decode là hai hàm riêng nếu base khác scaling/preprocessing; scoring/PIL/RM chung. Không đổi sang tensor interpolation path cho nhanh mà không research decision và EPH mới.

FP32 preview nghĩa VAE weights **và phép tính/activation** dùng FP32. `x0.float()` trước VAE FP16 chưa tạo decode FP32. Cách giữ hai VAE instance hoặc cast instance phải được ghi vào memory/timing protocol; không cast mỗi query rồi bỏ phí. Đồng bộ preview/delivery dtype theo approved execution profile.

F(1): decode+validity+CPU delivery, không gọi RM. Collector vẫn chấm mọi delivered seed để có bank labels; online evaluator chấm ảnh một seed sau inference timer.

M4 so PIL với tensor path upstream trên ≥1.000 preview: mean/max abs difference, correlation, top-k agreement, không dùng khác biệt đó để tune val.

**Acceptance:** repeat scoring trên cùng ảnh; invalid stages có test; main path duy nhất; RM missing keys check; RM spy xác nhận F(1) không gọi.

### CC-4: primitive profiling — M2/M2b/M3/M8

Timer host `perf_counter` với cuda synchronization trước/sau primitive. Warm-up 10; n≥50. Lưu raw samples, mean/p50/p95/CV và stage dimensions. Đo qua cùng engine functions mà online sẽ dùng, không qua một toy kernel khác.

| Primitive | Cấu hình |
|---|---|
| den | k=1…16, CFG batch2k, one forward+step; representative timesteps đã khai báo |
| ver / fin | k=1…16; fin(1) skip RM; preview versus delivered path đúng |
| prune | Tất cả source→target pairs mà grid/main/S&R sử dụng; gồm rank/tie/gather |
| feat / lfeat | k∈{2,4,8,16} cần dùng; per checkpoint/cut |
| fixed | Startup k₀=1…16 hoặc mọi cấu hình baseline sử dụng |

Prune profile nhận score/prediction có sẵn; không double count RM/predictor. Profile các phần synchronize/transfer ở đúng primitive, không tính cùng transfer lần hai trong disp.

Preview cost có thể đổi theo checkpoint/validity. Single ver[k] table chỉ là model mean timing; report spread theo j. Dùng cost model đã preregister cho mask; không hứa wall-clock cap.

Chunk VAE thử {1,2,4,8,16}; chọn phương án theo measured throughput và memory. Tie khi latency không phân biệt được thì chọn chunk nhỏ hơn; rule/measurement ghi trong proposal A. Dùng headroom VRAM theo cấu hình đã approved, không tự ép pool16 bị OOM.

M3: 50 prompt calibration×8seed×16checkpoint cho mỗi precision thử; báo failure riêng latent/image/score theo j và delivered. Escalate >10⁻³ theo bất kỳ mức hoặc delivered failure. Giữ trajectory khi so decode precisions để tránh confound; không tự sửa latent NaN bằng VAE cast.

M8: peak allocated/reserved VRAM và actual RAM limit, cho collector/online các batch. M2b: B2 actual wall-clock so sum primitive trên ≥20 prompt calibration episodes; báo phần dư.

**Acceptance:** finite/nonnegative costs, đủ keys, units us; ceil mean ở loader; CV>0.15 có diagnostic; generator costs dùng cuối không provisional; lỗi M3 xử lý trước bank.

### CC-5: batch invariance — M6

So cùng prompt/seed ở batch1,2,4,8,16; cfg row layout đúng. Report latent max error, preview/delivered score difference và winner flips. η=0 không bảo đảm bitwise invariance qua batch.

Không dùng padding để giả tạo parity mà không tính phí padding. Batch1 chỉ diagnostic; actual main prune vẫn dùng batch thực 8→4→2. Với pool16 so exact candidate ordering từ hai bank.

**Acceptance:** report có numeric deltas và implications cho replay; không khẳng định bitwise khi chỉ tolerance test pass.

### CC-6: collector, bank validation và M9

Thu 8 seed/bank cho một prompt, tất cả16checkpoint. Không prune khi thu bank. Seed CPU generator theo seed scheme, lưu initial latent hash.

Tại j ghi preview score/validity/features; nếu enabled train-only, lưu x0 latent. Cuối lưu PNG mọi valid delivered seed và IR riêng. Đừng lấy preview@T làm delivered.

Collector viết shard tạm + atomic finalize; lưu config hash, shard checksum và completed key ranges. Resume chỉ skip shard đã validate với cùng EPH/config; không skip partial files hoặc đổi seed sau fail. Preserve failed artifact và reason; không overwrite silently.

`--test-equivalence`: 10 prompt calibration, same prompt/seed/config/dtype. Reference collector có dead hard-code checkpoint thì dùng loader wrapper pin mirror; không edit algorithm hay third_party. So exact hashes nếu cùng precision/numeric path; nếu approved FP32 tạo khác biệt có chủ đích, tách comparator và báo tolerance/score effect, không ép về FP16 để qua hash check.

M9 trên subset20 prompt calibration đã định trước: seconds/prompt-bank, GPU-hours cho pilot nhỏ/lớn gồm hai bank mỗi prompt, peak RAM/disk size. Không xem quality để chọn pilot. Tách thời gian collection, profiling và G1 trong budget dự báo.

**Acceptance:** `validate_bank` của Codex pass; đủ keys; repeat/resume cho cùng bank identity; collector equivalence report; M9 extrapolation; actual storage/memory report.

### CC-7: shared engine và online runner — M7/M10

Core engine cần có trước EPH/P0 profiling. Bổ sung decider integration sau khi Codex API sẵn sàng.

```python
# Algorithmic pseudocode, not an alternative implementation of ledger/DP.
mc = load_method_costs(...) 
state.charge(mc.once)
r0 = cap - state.spent
assert r0 >= L[q0]             # otherwise mark infeasible before generation
prepare_request()             # C_fixed outside cap, inside e2e timer

while not terminal(state):
    view = state.view()        # budget before current dispatch
    allowed = shared_mask(view, L, grammar, mc)
    obs = shared_build_obs(view, frozen_norm)
    action = policy.act(obs, allowed)
    require(action in allowed)
    state.charge(cost_action(state.q, action, mc))
    engine.execute(action, state, row_map)

state.charge(mc.fin[state.k])
delivery = engine.finalize(state, row_map)
```

Bookkeeping once có thể trả trước dù một phần selector chạy ở j_c; timer e2e vẫn đo tại vị trí work thật. Không đếm setup lần hai.

VERIFY: engine gọi scorer, state.record_verify. PRUNE: common ranking/gather, state.record_prune. CONT: đúng4forward, state.arrive mới; traj nếu enabled chỉ một feature event ở checkpoint. Không trả private labels cho policy.

Scheduled runner dùng cùng engine; upfront validate full cost; VERIFY trước paid cut tại cùng j; pred cut trả lfeat/pred/prune. B8 có đoạn prefix/query, gọi scheduler.plan rồi thực thi suffix với events đã mua được reuse.

G1 fixtures từ train-dev, fixed methods và frozen controller representatives. Codex phân tích Δ_disc. Final selected policy phải có G1 validation phù hợp, không dùng parity của một stub cho tất cả controller.

Wall-clock steady-state request bắt đầu trước once/C_fixed work, kết thúc winner ra CPU; không gồm model load/warm-up hoặc PNG/log I/O. Save files sau timer. Báo median/p95, real ledger-spend, fraction W>B, unaccounted residual; ghi có resident-RM setup hay không.

**Acceptance:** replay/online identical API use; no double charges; ledger trace sum khớp spend; effect discrepancy report; controller final profiles matching artifacts.

### CC-8: evaluators

HPS/GenEval adapters đặt ngoài env/policy modules. Trainer không đọc `eval/`. Evaluator chuẩn bị bằng prompt tự tạo trước PREREG-C; không chạy GenEval sớm.

Pin official GenEval repo/weights, dùng official per-tag aggregation. Synthetic evaluator smoke được phép. F(1) post-inference IR không được feedback vào decision. Failure giữ trong denominator; report all failed requests.

**Acceptance:** PREREG-C guard thật; official aggregation test fixtures; evaluator manifest/versions; image identity joins đúng.

## 8. Contracts, replay và method — task Codex

### CX-0: schema, hashing, ledger, access

Implement §§5–6 trước. Dùng bank synthetic đúng schema, có rank ties và invalid cases. Export versioned interfaces để Claude Code import. Validate costs không âm, finite, integer after conversion, đủ primitive keys.

### CX-1: prompt pool

Metadata-only DiffusionDB loader; explicit source revision và normalization. Prompt_uid=sha1(text chuẩn hóa NFC/whitespace theo rule frozen), không lowercase/reorder prompt tự ý. Ghi mapping raw→normalized và UID collision checks.

CLIP token filter theo actual tokenizer, đếm content tokens không gồm BOS/EOS; near-duplicate groups bằng connected components của relation cosine>0.95. Group-disjoint splits; lọc GenEval text cosine>0.9/template. Không normalize PCA embedding bằng công thức dedup rồi nhầm với raw pooler_output.

Chuẩn bị 50 prompt calibration và subset 20 cố định, tách tất cả train/dev/val/test. Val_ext sealed220 hoặc100 tùy pilot; tổng val≤300. Research duyệt prompt pool/split trước collection.

### CX-2: budget DP và schedule enumeration

G-842 valid actions như Research Spec, no prune at j16, no two prune same j. L/Mask method-specific với once prepaid. Cache key gồm grammar, costs/hash và method flags; không reuse L sau đổi ver/disp/feat.

A1 `G-A1-v1`: CONT và VP atomic, VP trả một disp + ver + prune, state prevents second VP same checkpoint. Traj CONT nếu enabled vẫn có feature fee. Solve riêng; không giả lập bằng miễn phí một dispatch trong grammar cũ.

Enumerate105pairs; schedule cost exact. GridB4b lấy actual search space base code, pin grid definition và config count, không gọi một grid nhỏ tùy chọn là “full PSP grid”. `required_prune_pairs()` xuất pairs để CC profile.

### CX-3: replay env và no-leak accessor

Replay transition thay GPU execution bằng privileged bank read, còn state, rowmap, ledger, mask, obs dùng common modules. Không để policy closure giữ bank accessor.

`PrivilegedAccessor` chỉ env/oracle/training-label/reward. `PolicyView` chỉ thông tin đã mua/cho phép. Log read gồm purpose, current j, CandidateIds, key, split, event. `EvalAccessor` tách module.

Whitelist no-leak đúng:

- Score trung gian chỉ khi VER đúng j, candidate còn sống.
- Traj tại checkpoint hiện tại chỉ khi enabled và charge feat.
- B10 candidate feature chỉ tại cut, charge lfeat.
- Prompt feature qua approved setup/once.
- IR_deliv vào terminal reward/evaluation hoặc privileged training labels, không vào runtime view.
- Oracle có thể đọc whole bank bằng purpose `oracle`; không được expose ra policy.

Không viết test “mọi dữ liệu phải qua record_verify”: rule đó sẽ cấm cả prompt/trajectory hợp lệ. Future checkpoint, pruned candidate và `eval/` vẫn bị cấm với runtime policy.

Training modules được dùng private labels chỉ train_fit; dev evaluation terminal scores không trở thành fit labels. Tách training/runtime modules để import-graph check không cấm training hợp lệ.

### CX-4: feature definitions — đóng băng với config

Các công thức dưới đây hoàn thiện những chi tiết engineering chưa rõ trong v1.1; ghi vào obs-version/PREREG-A/B trước ảnh thật, dùng y hệt collector và replay.

**Drift** của candidate tại j>1:

\[
d_i(j)=\frac{RMS(\hat x_{0,i}^{j}-\hat x_{0,i}^{j-1})}
{RMS(\hat x_{0,i}^{j-1})+10^{-6}}.
\]

**CFG gap:**

\[
g_i(j)=\frac{RMS(\epsilon_{text,i}^{j}-\epsilon_{uncond,i}^{j})}
{RMS(\epsilon_{uncond,i}^{j})+10^{-6}}.
\]

Use FP32 reductions. Bank drift j1=NaN; feature view converts unavailable drift to0 with known checkpoint condition. Invalid feature excluded from aggregates; no finite values→0. Mean/std channel over spatial dims, std ddof0 for consistency; all implementation paths use same reducer.

Paid-score normalization: μ_j/σ_j fit valid train_fit scores, σ floor1e−6, normalize by checkpoint of purchased score, not current j. Sorted valid scores of alive candidates s₁≥…≥s_v; cut h=k/2:

\[
z_{cut}=\frac{s_h-s_{h+1}}{SD(s_{valid})+10^{-6}},\qquad
z_{top}=\frac{s_1-s_2}{SD(s_{valid})+10^{-6}}.
\]

Theo v1.1, chỉ bật các gap nếu v≥h+1; ngược lại bằng 0. Clip standardized score/gap features [−10,10]. Tính Δz_cut trên tập candidate còn sống có score hợp lệ chung giữa hai VERIFY gần nhất; unavailable→0. Kendall τ-b trên ≥2 common valid candidates, degenerate→0. `has_two=1` khi có hai lần VERIFY và ít nhất hai candidate hợp lệ chung; ngược lại bằng 0. Ghi rõ trường hợp gap chưa tính được dù Kendall đã tính được.

Main38dims đúng thứ tự Research Spec; traj42dims thêm mean/std drift/gap hiện tại. Mean/std aggregates ddof0. Score absent→finite zeros+has_score0; all-invalid→has_score1, frac_invalid1, finite aggregates0. `np.isfinite(obs).all()` bắt buộc.

PCA16 raw SD text pooler_output768; fit train_fit only, deterministic solver/sign convention recorded. Immutable normalized artifact. Không refit PCA vì thêm val_ext.

B10 feature12dims: drift, cfg_gap, channel means4, channel std4, within-current-pool z-score drift/gap2. Invalid predictor outputs rank last, obs không có −∞.

### CX-5: baseline implementation

| Method | Implementation requirements |
|---|---|
| B0/B1/B2 | Nested same-bank seeds; B1 pick feasible N by measured cost, B2 reference(4,8) |
| B3 | Evaluate all105 fixed schedules train; choose valid theo từng cap; no theo từng prompt oracle selection |
| B4b | Actual PSP grid, pool16, verifier charges, arbitrary keep, ≤4cuts |
| B5 | M8/16, one paid screen j_s then resume q1/2; cost full prefix/suffix/Fq |
| B6 | Threshold τ_s(j)=a_s+b_sj/16, stage8/4; j_start1..8, verify cadence m1/2/4 |
| B7 | Predict schedule outcomes from prompt feature; select only affordable schedules |
| B8 | Query j_c1..8; features prompt+current paid score summaries; commit affordable future schedule |
| B9 | Uniform over feasible mask, reproducible eval seed |
| B10 | Percheckpoint ridge/GBM predictor, fixed cut search, no intermediate RM |

**B6 action rule:** j0/structural forced→allowed action. If fresh/unpruned and PRUNE allowed, prune when z_cut≥τ; else CONT if allowed. If not fresh, query at cadence only when VER allowed; otherwise CONT if allowed. When CONT would destroy completion, follow mask-forced VERIFY/PRUNE. Never choose action outside mask.

B6 tuning ≤2.000objective evaluations/cap, a_s,b_s∈[−3,3], train_fit search and train_dev selection. Two stage pairs+a shared j_start+m = six parameters. Log budget and actual calls.

B7/B8 labels privileged train_fit schedule delivered Y; mask infeasible schedules before argmax. Ridge regularization13log-spaced values10⁻³…10³; GBM depth≤3/trees≤200 only when group5fold better. Fold by prompt group, banks of a prompt together. B8 current query info available, future score labels never runtime features.

B10 percheckpoint labels within(prompt,bank) z-score delivered Y, train_fit only. Feature availability currentj; ridge default, GBM as above; fixed cut schedule tune train. B10-CNN disabled pilot. No claim its minimal predictor is the strongest possible no-query method.

### CX-6: oracle và frontier

O1 max over affordable105 schedules, fixed ranking. Assert upper-bound relation only against methods in identical paid-prune grammar; not B10/pool16.

O2-inst diagnostic. O2-avg implement **two distinct outputs**:

- `dual_lower_bound_total`, λ, numeric convergence status;
- `primal_feasible_assignment`, verified constraints, `primal_cost_total`, optimality gap/status if known.

Primal one choice per instance discrete schedule problem; MILP optional. No solver⇒dual diagnostic still usable, cost GO unavailable without witness. Any returned witness re-evaluate quality/cost from original bank. Handle target infeasibility before unbounded λ bracketing. Do not label a finite sampled λ search result exact optimum.

Use same δ_IR target in primal/dual. Divide total cost by instance count for mean cost; average bank weights equal within prompt. Pool16 endpoints use one pooled episode per prompt.

Operating point dev: among quality-qualified feasible points, argmin mean actual e2e cost, predefined tie-break. Freeze α/artifact before val. Report both e2e and variable-ledger savings. No interpolation claim without actual mixture implementation.

### CX-7: REINFORCE

Implement Research Spec loss, separate value network, masked categorical distribution. Terminal reward uses delivered IR difference vs B3 same instance/cap; environment computes baseline privately, no runtime leakage.

Sample prompts/banks train_fit and uniformly feasible caps; mark infeasible caps in report. Evaluate greedy, action-ID tie-break. Default MLP2×128/Adam3e−4/entropy.01/valueweight.5/gradclip1.0. Tune≤24configs×3init, earlystop dev. Store selected artifact and all configs/seeds; no val-informed learning rate or stopping.

Synthetic sanity environments with known optimum verify gradient/action-mask behavior; no requirement RL win every noisy tiny fixture. Correctness tests compare return/actions to analytic optima where appropriate. Actual experiment requires `G2-evaluated` marker and no research STOP.

### CX-8: stats, anchor, prereg guards và gate report

Anchor script only train_fit/train_dev; no import oracle/adaptive policy. Compute valid train_fit failure penalty, σ_R, Δ_anchor/τ and NormStats/PCA hashes. Write proposal B, never signed marker.

Primary contrasts registry frozen before first val; one global M across gates/caps/endpoints. Two look α.025each, contrast CI1−.025/M. Cost has distinct10%GO and5%STOP thresholds; don't reuse single-threshold `three_way` for cost.

Cluster bootstrap prompt; keep all banks/methods paired. Bootstrap primal/dual diagnostics if CIs used, clearly distinguish feasible primal statistic and optimistic dual statistic. Enough resamples for adjusted tails; retain bootstrap seed/raw summary. Statistical synthetic checks at actual alpha as well as95%, not only “95% coverage passed”.

Power adjusted alpha, report SE/MDE/P_needed. M12 used planning, not to quietly change prereg threshold. `val_access.jsonl` logs contrast/config/look/cph; eval helper refuses third look or config changed after first look.

G3 includes dev-selected candidate fromB6/B7/B8/RL; best-simple separate. G4 opponent selected fromB6/B8 theo từng cap ondev, no theo từng prompt max. G3b best-paid dev-selected vsB10-min. Gate proposal follows Research Spec, retains INC after second look.

## 9. Bank schema và file contracts

Version mới `apsp-bank-v1.1.1`; old v1.1 có thể migrate nếu generation data unchanged và validator chứng minh đủ fields. Không đổi sampling để migrate.

| File | Schema/data |
|---|---|
| manifest.json | schema_version, bank_name, eph/body, bmh/body, machine_id, commit, created_at; source config và code hashes |
| prompts.parquet | prompt_uid str; text str; split enum; source str; group_id str; n_clip_tokens int16 |
| prompt_emb.npy + index.parquet | float16[N,768] raw pooler_output, UID→row unique |
| seeds.parquet | UID str; bank_id int8; seed_idx int8; seed_global int64; init_latent_sha str |
| previews.parquet | UID/bank/seed; j int8; forward_idx int16; t_in int16; ir float32; latent_ok/img_ok/score_ok bool; drift/cfg_gap float32; mean_c0..3/std_c0..3 float32 |
| delivered.parquet | UID/bank/seed; ir_deliv float32; three validity flags; image_path str nullable; image_sha str nullable; valid PNG or explicit failure |
| eval/eval_scores.parquet | separate: HPS float32, GenEval test-only columns |
| latents/x0_<uid>_<bank>.npy | optional float16[8,16,4,64,64], train_fit/train_dev only; latent storage dtype distinct from decode dtype |

Seed mapping: sha256 UTF8 `apsp-v1|{uid}|{bank_id}|{seed_idx}`, first4bytes little-endian &0x7FFFFFFF; CPU torch.Generator. Seed SHA collision check; seed identity includes full tuple, so equal integer seed is logged rather than identity collision. Save actual initial latent bytes hash and dtype/shape.

`validate_bank` checks uniqueness, required rows8×16theo từng prompt-bank, contiguousj, forward_idx=4j, timesteps mapping, seed derivation, embedding mapping, scores finite iffscore_ok, invalid chain implications, actual PNG/latent checksums and hashes. It excludes `eval/` from training-bank BMH so adding evaluator results does not invalidate generation data; evaluator has separate manifest.

## 10. Five-hash provenance

| Hash | Body includes |
|---|---|
| EPH | base commit; GPU generation/scoring/core code_sha; env lock; model/revision/weights; scheduler/timesteps; CFG/η/T/Δ/resolution; dtypes/chunk; RM/scoring/validity; numeric runtime |
| BMH | EPH; prompt/seed collection schema; collector batch/mode; train-only latent policy; content checksums of input bank files |
| GCH | EPH; generator primitive samples/summary; units/timer/protocol; profiling code_sha; measurement CPU/GPU/runtime/thread profile |
| CPH | method/grammar; artifact; observation/norm/PCA/predictor hashes; behavior code_sha for policy/features/DP/ledger/runtime accessor; training bank IDs |
| CCH | CPH+GCH; measured disp/once/pred; controller profiling code; actual host CPU/thread/device runtime |

Canonical serializer preserves types; numeric1.0 must not collide with string"1". Dictionary keys must be strings; NFC normalization must not merge distinct keys silently. Reject nonfinite hash-body floats, unordered sets and unsupported types. Sorted keys/relative paths, stable list order. Use type-tagged canonical representation or strict schema-typed equivalent; golden fixtures pin exact bytes/hash.

No hash includes its own value, absolute paths, timestamps or free notes. Hash relevant code files, not merely a human policy_version string. BMH includes actual generation files, excludes eval artifacts and temp files.

Cross-check: bank.eph==GCH.eph; method.cph==CCH.cph; CCH.gch==selectedGCH; online.eph==bank.eph when claiming parity. Matching GPU model alone does not authorize reuse of CPU/controller/PIL timings on another host.

| Change | Recompute |
|---|---|
| Policy artifact/norm/PCA/features behavior | affected CPH/CCH and runs; bank stays if EPH unchanged |
| DP/ledger behavior | affected runs/CPH/CCH and gate reports; bank may stay |
| Stats-only code | reports/statistics, keep original generated data |
| Generation/decode/scoring behavior | EPH and dependent banks/costs/runs; qualify or recollect |
| New prompt/shard | new BMH; preserve previous artifact |

Run manifest stores EPH/BMH/GCH/CPH/CCH, git commit/code versions, machine_id, method/grammar, split/cap, episode/failure counts, PREREG hashes and immutable artifact IDs.

## 11. Measurement registry

| ID | Measurement | Owner |
|---|---|---|
| M0 | Repeat determinism, hook/reference x0 differences | CC |
| M1 | Scheduler/timesteps/terminal coefficients, model provenance | CC |
| M2 | Generator costs incl method startup variants | CC |
| M2b | Primitive sums versus B2 request wall-clock | CC |
| M2c | Controller disp/once/pred actual artifacts | Codex script, measurement machine |
| M3 | Validity rates by checkpoint/precision, ≥6.400preview per tested precision | CC |
| M4 | PIL versus upstream tensor score differences | CC |
| M5 | Preview@T versus delivered ranking agreement | Codex |
| M6 | Batch-size invariance1/2/4/8/16 | CC |
| M7 | Paired-effect replay-online discrepancy | CC executes, Codex analyses |
| M8 | PeakVRAM/RAM/reserved and actual cgroup limits | CC |
| M9 | Calibration throughput/storage/GPU-hour forecast | CC |
| M10 | Final wall-clock median/p95/residual/over-budget rate | CC |
| M11 | Train anchor/threshold/failure reward proposals | Codex |
| M12 | Variance, multiplicity-adjusted MDE/P_needed | Codex |

Controller profile stub allowed during development but `provisional=true`. Final scientific comparison requires actual measured artifact CPH/CCH, not MLP stub timing. Method-specific CPU build_obs/mask/selector work measured too.

Trước chọn candidate cuối trên train-dev, profile các artifact shortlist thật, cập nhật L/mask/feasible caps và đánh giá lại train-dev với chi phí cuối. Không chọn operating point bằng stub cost rồi chỉ đổi ledger khi chạm val. CPH/CCH và protocol chọn candidate đóng băng trước đánh giá G3; G2 chỉ quyết định đi tiếp/dừng, không được dùng để sửa feature, search space hoặc hyperparameter.

## 12. Acceptance tests có ý nghĩa

### 12.1 Contract/simulation

- Hash: deterministic, type separation, source-change invalidation, no self-reference; actual files validated.
- DP: compare L/mask with brute force small grammars; budget conservation for every reachable teststate/budget; action set neither admits impossible completion nor excludes affordable valid completion.
- Regression: CONT feature once/block; once prepayment; dispatch forced actions; no terminal dispatch; source→target prune; F1 versusF2; B8 preflight/reuse-query.
- A1: own DP versus brute force, one VP fee; no second VP samej.
- Leak proxy: thousands random legal traces, sentinel future/pruned/terminal/eval values; permitted prompt/traj routes still work.
- Obs: exact38/42 shapes, finite under invalid/all-invalid/ties, candidate-order invariance, frozen normalization.
- Oracle: O1 compare brute force; dual≤known small exactprimal; primal witness constraints; counterexample O2-inst not mean-cost lower bound; no unsupported O1 relation forB10.
- Stats: clustered dependence, repeated-look/multiplicity rules, correct cost greyzone, INC retained, frozen candidate selection, selection by actual cost rather thanalpha.

### 12.2 GPU/integration

- RowMap survivor paths, CFG embeds and hook state aligned.
- Hook no extra forward; main no free trajectory norm.
- Validity stages beforePIL; controlled NaN fixture; FP32 trulyFP32.
- F1 RM spy; k2 terminal selection usesdelivered score, notpreview.
- M0/M6 actual deltas, not assumed bitwise parity.
- Collector equivalence/resume; online/replay events/spend/views align.
- Primitive timer boundaries and sum consistency; actual method artifacts matchcosthashes.
- PREREG guards and evaluator isolation.

Run necessary targeted checks and required suite. Do not add tests merely asserting the same formula twice. The numerical/theory tests above are necessary because incorrect masks, leakage and gate directions can change conclusions.

## 13. CLI và runbook yêu cầu triển khai

Các command là **target CLI**, chưa được khẳng định tồn tại. Agents triển khai argparse tương ứng hoặc ghi mapping rõ trong RUNBOOK.

```bash
# P0: no val/test outcomes
python -m apsp.gpu.env_check --machine-id MEASUREMENT_ID --out reports/env.json
python -m apsp.data.prompts --config configs/pilot.yaml --out data/prompts
python -m apsp.gpu.profile --config configs/pilot.yaml --stage model
python -m apsp.gpu.profile --config configs/pilot.yaml --stage hook
python -m apsp.gpu.profile --config configs/pilot.yaml --stage validity
python -m apsp.gpu.profile --config configs/pilot.yaml --stage generator
python -m apsp.gpu.profile --config configs/pilot.yaml --stage batch-parity
python -m apsp.gpu.collect_bank --config configs/pilot.yaml --test-equivalence
python -m apsp.gpu.collect_bank --config configs/pilot.yaml --calib-throughput
python -m pytest tests/contract tests/sim
python -m pytest tests/gpu
```

Sau research ký A:

```bash
python -m apsp.gpu.collect_bank --config configs/pilot.yaml --split train_fit train_dev val --resume
python -m apsp.contracts.schema --bank data/banks/PILOT_BANK --validate
python -m apsp.reports.anchor --bank data/banks/PILOT_BANK --proposal-only
```

Sau research ký B, theo orderG1→G2→P3:

```bash
python -m apsp.gpu.online --config configs/pilot.yaml --task replay-parity --split train_dev
python -m apsp.reports.replay_vs_online --config configs/pilot.yaml
python -m apsp.reports.headroom --config configs/pilot.yaml --look 1
python -m apsp.policies.reinforce --config configs/pilot.yaml --train
python -m apsp.reports.ctrl_costs --config configs/pilot.yaml --artifacts data/artifacts/FROZEN
python -m apsp.reports.pilot_report --config configs/pilot.yaml --look 1
```

Baseline train/tune commands theo từng module cần thêm trong RUNBOOK. P3 report không tự train lại khi chạm val. G2 headroom không trở thành dataset feature cho policy.

Look 2 chỉ qua plan/guard đã preregister; thêm subsetval_ext đã authorized, tổng≤300. Sau C và test budget:

```bash
python -m apsp.gpu.online --config configs/frozen_test.yaml --split test
python -m apsp.gpu.evaluators.geneval --run data/runs/FROZEN_TEST_RUN
python -m apsp.gpu.evaluators.hps --run data/runs/FROZEN_TEST_RUN
```

Mọi CLI hỗ trợ dry-run/preflight, confighash dump, explicit outputdir, deterministic runseed, idempotent resume và nonzero exit khi guard/schema sai. Test-equivalence/calib exceptions chỉ cho pool calibration, không mở cửa val/test.

## 14. P0 sign-off checklist trước thu bank

- [ ] Measurement machine thật, budget và access rõ; không dựa thông số giả định.
- [ ] Submodule/weights/revisions/envlock pinned, coreengine freeze.
- [ ] Promptpool group-disjoint duyệt, 50 prompt calibration +20subset, pilot size quyết theo M9.
- [ ] Shared contracts/ledger/DP/views/obs ready với meaningful tests.
- [ ] Scoring/validity, F1 và preview/delivered distinction đúng.
- [ ] M3 escalation xử lý; preview dtype và chunk được research chốt.
- [ ] M0/M6/collector equivalence có báo cáo; không giấu batch drift.
- [ ] Generator costtable/caps proposal nonprovisional, enough primitive pairs.
- [ ] Throughput/storage/memory forecasts có actual samples.
- [ ] Collector atomic/resume, validation/hash chain đúng.
- [ ] PREREG-A signed bởi research, not agent.

Không chặn bank vì controller final chưa train hoặc CCH còn stub. Những phần đó phải hoàn tất trước báo kết quả P3.

## 15. OOM, lỗi, resume và reporting

Preflight RAM/cgroup và VRAM; concurrency phải dựa measurement, không mặc định4 workers. Nếu OOM, preserve attemptdirectory/log/config/reason; sửa concurrency/batching chỉ khi không đổi scientific config. Kiểm output invariance hoặc ghi enginechange ảnh hưởng EPH. Không overwrite run đã fail thành “pass”.

Research escalation cần thiết: đổi dtype/scoring/sampler, M3 failure excessive, no valid training data, replay drift lớn, budget không đủ pilot, muốn thay scientific design. Làm tiếp independent tasks trong khi chờ; không dừng cả project vì một module chưa có artifact.

Mỗi phase report `reports/Px_<owner>_summary.md`:

```text
Status: implemented / executed / measured / blocked (separate each)
Spec/config/EPH/BMH/GCH/CPH/CCH and commit:
Machine/runtime and units:
Tasks finished; artifacts and exact commands:
Checks executed; result and limitations:
Measurement: n, mean/p50/p95/CV, uncertainty where relevant:
Scientific data accessed: split, purpose, val look if any:
Failed attempts, resumed ranges, changed operational settings:
Outstanding blocker; precise decision needed:
Next authorized tasks:
```

Never report toy test as image quality result, estimated VRAM as measured VRAM, or unsigned proposal as signed PREREG. Gate results retain GO/STOP/INC definitions; agent may recommend but does not approve scientific claim itself.

## 16. Definition of done cho team

**P0 done:** repo dựng lại được, contracts thống nhất, calibrated generation core có reports và đủ điều kiện kýA.

**Pilot done:** validated bank, artifacts/profile matching, parity-qualified evaluation mode, frozen primary contrast registry, all baselines under actual cost accounting, gate report+frontier+failure analysis, provenance và rawdata tái lập.

**Paper experiments done:** frozen method qua independent online test với evaluator official, quality/cost claims có CI đúng, training/setup cost riêng, limitations/negative results đầy đủ. Pilot dương không tự động có nghĩa paper đã đủ thuyết phục.

## 17. Reference routing

- Research formulation/proofs/stats decisions: Research Spec §§6–11.
- PSP base: https://github.com/rogerioagjr/PSP @ pinnedcommit.
- PSP paper/author page: https://arxiv.org/abs/2607.21591 ; https://www.vision.caltech.edu/psp/
- Model mirror: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- GenEval official: https://github.com/djghosh13/geneval
- DiffusionDB metadata/card: https://huggingface.co/datasets/poloclub/diffusiondb
- Flash-BoN reference, not mandatory direct fork: https://github.com/flash-bon/flash-bon
- ELASTIC comparison: https://arxiv.org/html/2606.31132v1

Tài liệu này kế thừa codeaudit v1.1 nhưng implementation phải kiểm file thật tại commit. Không cần mở thêm literature round để bắt đầu P0; noveltyaudit mới dành trước final paper submission.
