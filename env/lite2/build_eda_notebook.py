"""Generate the committed, output-free Lite2 EDA notebook using only stdlib."""
from pathlib import Path
import json
import textwrap

ROOT=Path(__file__).resolve().parents[2]
CELLS=[]


def md(text):
    CELLS.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip().splitlines(True)})


def code(text):
    CELLS.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                  'source':textwrap.dedent(text).strip().splitlines(True)})


md('''
# APSP Lite2 — EDA toàn diện sau collection trên công ty

**Dataset:** `apsp-lite2-fast2/lite2-r1`, export `1792706875990190337`.
**Theo receipt người vận hành cung cấp:** 4.800 task chính + 120 calibration,
`core_latent_qa`, `RELEASE_VERIFIED`. Đây là đầu vào cần đối chiếu, không phải
kết quả notebook đã đo. Bấm **Run All** trên máy công ty để tạo bằng chứng mới.

Notebook chạy CPU bằng runtime cũ; không tải model, không gọi scorer, không
cấp GPU, không sửa release/backup/queue/job. Chỉ thư mục `OUT` mới được ghi.
Không sửa `apsp/` hoặc runtime để làm EDA. Các phép phân tích dùng toàn bộ Core;
native-value audit và gallery có phạm vi mẫu được khai báo riêng.

**Cách đọc:** integrity → coverage → split/leakage → scores/diversity →
trajectory/pruning → paired contrasts → feature/embedding → native/provenance →
cost/storage → manual QA → quyết định chuẩn bị training. Mỗi phần nêu câu hỏi,
đơn vị phân tích, cách đọc và giới hạn. File CSV giữ toàn bộ hàng, notebook chỉ
hiển thị một phần để tránh làm treo trình duyệt.
''')
md('''
## 0. Đường dẫn, phạm vi I/O và tái lập

Các đường dẫn đã điền theo node `duytx-snipe-dgx06-attempt-4-0-0` bạn cung cấp.
Chạy từ checkout EDA riêng. Không update checkout đang được worker sử dụng.
Kernel cần Python có sẵn `numpy`, `pandas`, `pyarrow`, `safetensors`, `Pillow`.
`matplotlib` là tùy chọn cho biểu đồ: thiếu thì vẫn có bảng/HTML, không tự cài.
Không cần Torch, CUDA, `nbclient` hoặc `nbformat` cho lệnh headless đi kèm.

- `FULL_HASH=True`: stream toàn bộ file export, có thể đọc ~50–80 GB; RAM không
  tăng theo kích thước release. Đổi `False` để chỉ hash Core/provenance/metadata;
  báo cáo sẽ ghi rõ các extension chưa hash, không gọi toàn bộ release PASS.
- `NATIVE_VALUES_PER_PROFILE=4`: đọc từng tensor của 4 task/profile theo thứ tự
  task ID, kiểm finite/RMS/raw-byte hash. Header được đọc cho **mọi latent**.
- `BACKUP_FULL_HASH=False`: đối chiếu marker + ACK toàn bộ backup; bật True nếu
  muốn đọc và hash toàn bộ payload backup thêm một lượt. Không đồng nhất ACK với
  backup bền vững độc lập.
- `BOOTSTRAP=1000`, seed cố định. CI chỉ exploratory. Không tuning bằng test.

Nếu I/O dùng chung đang bận, chạy chế độ metadata trước hoặc qua lịch CPU của
công ty. Notebook không tự chọn CPU/GPU/priority của hệ thống.
''')
code('''
import os, sys, json, importlib.util
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/workspace/storage-shared/nlp/tungks/apsp_all')
RUN = ROOT / 'apsp-lite2-fast2/lite2-r1'
COLLECTION = Path(os.environ.get('APSP_EDA_COLLECTION', str(RUN / 'release')))
EXPORT = Path(os.environ.get('APSP_EDA_EXPORT', str(COLLECTION / 'exports/1792706875990190337')))
MANIFESTS = Path(os.environ.get('APSP_EDA_MANIFESTS', str(COLLECTION / 'manifests')))
BACKUP = Path(os.environ.get('APSP_EDA_BACKUP', str(RUN / 'backup')))
OUT = Path(os.environ.get('APSP_EDA_OUT', str(ROOT / 'eda-v2' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))))
SOURCE = Path(os.environ.get('APSP_EDA_SOURCE', os.environ.get('APSP_SOURCE', Path.cwd())))
FULL_HASH = os.environ.get('APSP_EDA_FULL_HASH', '1') == '1'
BACKUP_FULL_HASH = os.environ.get('APSP_EDA_BACKUP_FULL_HASH', '0') == '1'
NATIVE_VALUES_PER_PROFILE = int(os.environ.get('APSP_EDA_NATIVE_VALUES', '4'))
BOOTSTRAP, RNG_SEED = 1000, 20261023

# Locate only the explicitly supplied checkout / notebook's parents, never scan storage.
helpers = [SOURCE / 'env/lite2', *[p / 'env/lite2' for p in [Path.cwd(), *Path.cwd().parents]]]
HELPER = next((p for p in helpers if (p / 'eda_v2.py').is_file()), None)
if HELPER is None:
    raise FileNotFoundError('Set APSP_EDA_SOURCE to the new EDA checkout containing env/lite2/eda_v2.py')
sys.path.insert(0, str(HELPER))
from eda_v2 import Audit, describe, PROFILES, KEY, SEED
import numpy as np
import pandas as pd
try:
    from IPython.display import display, HTML
except ImportError:
    display = print
    HTML = str
pd.set_option('display.max_columns', 24)
pd.set_option('display.max_rows', 35)

required = ['numpy','pandas','pyarrow','safetensors','PIL']
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise RuntimeError('Use existing APSP runtime; missing packages: ' + ', '.join(missing))
for p in [EXPORT / 'RELEASE_MANIFEST.json', MANIFESTS / 'manifest.json', BACKUP]:
    if not p.exists(): raise FileNotFoundError(p)
print('Python:', sys.executable, '\\nExport:', EXPORT, '\\nOutput:', OUT)
print('Full export hash:', FULL_HASH, '| backup full hash:', BACKUP_FULL_HASH)
A = Audit(EXPORT, OUT, full_hash=FULL_HASH, native_values=NATIVE_VALUES_PER_PROFILE,
          seed=RNG_SEED, bootstrap=BOOTSTRAP, collection=COLLECTION,
          manifests=MANIFESTS, backup=BACKUP, backup_full_hash=BACKUP_FULL_HASH)
''')
md('''
## 1. Kiểm checksum và snapshot trước khi tin thống kê

Đối chiếu tập file thực với `RELEASE_MANIFEST.json`, index `SHA256SUMS`, hash
manifest và hash payload đã chọn. Chặn symlink/đường dẫn thoát root. Thiếu file,
thừa file hoặc sai hash: dừng thống kê, giữ `checks.csv` và báo cáo lỗi.

Hash đúng chứng minh snapshot khớp receipt, không chứng minh recipe đúng về
khoa học hoặc ảnh đẹp. Export qualification 20 task/profile không thay export
4.920 task chính này. Cột `NOT_HASHED_EXTENSION` là **chưa kiểm**, không phải PASS.
''')
code('''
integrity = A.integrity()
display(integrity.groupby('status').agg(files=('file','size'), bytes=('bytes','sum')))
display(A.sections['integrity_scope'])
''')
md('''
## 2. Schema, khóa, cardinality và foreign keys

Mỗi logical task `(profile_id, prompt_uid, bank_id)` phải có 8 candidate,
8 delivered, 64 preview với `seed_idx=0..7`, `j=1..8` chính xác. Không chỉ so
tổng số hàng: mất 1 task và lặp 1 task khác vẫn có thể giữ nguyên tổng.
Embedding có shape `[unique_prompt_uid,384]`, float32, finite, L2 norm≈1 và
index ánh xạ đúng UID. Train chỉ bank0; mọi sinh dữ liệu test đều là lỗi.

Theo plan đầy đủ, gồm calibration: 4.920 task / 39.360 candidate / 314.880
preview / 39.360 delivered. Main riêng: 4.800 / 38.400 / 307.200 / 38.400.
''')
code('''
display(A.load())
display(pd.DataFrame(A.checks).query("status != 'PASS'"))
print('Unique prompts:', len(A.data['prompts']), '| embedding:', A.embedding.shape)
''')
md('''
## 3. Coverage và trạng thái terminal

So sánh planned → collected → complete_valid / model failure / failed_infra /
pending, theo profile × split × bank × source. Model failure khác infrastructure
failure. Có COMPLETE không đồng nghĩa mọi candidate valid. Missing không phải
reward=0 và không được âm thầm bỏ khỏi mẫu.

Canonical target mỗi profile: train2.000 bank0, dev200 × 2bank = 400; calibration
60task/profile, 50 UID (10 UID đầu có cả 2 bank). Quota train theo source:
DiffusionDB1.000 / COCO400 / structured600; dev100/40/60. Phân biệt task count,
candidate count, UID count và source-group count.
''')
code('''
display(A.coverage())
display(A.tables['canonical_plan_coverage'])
display(A.tables['source_coverage'])
display(A.data['task_plan'].groupby(['profile_id','split']).durable_ack_verified.agg(['size','sum']))
''')
md('''
## 4. Frozen manifests → exported task → live source → backup ACK

Hash mọi JSONL theo manifest đóng băng. Đối chiếu **nội dung task đầy đủ**, không
chỉ ID. Đối chiếu `source-complete.json` trong export với COMPLETE thật trong
`release/main|calibration/tasks/`, ACK và COMPLETE của backup cùng task.

Không theo `source_task_uri`/ACK tới root bất kỳ: chỉ đọc collection/backup đã
chỉ định. Mặc định kiểm marker + ACK hiện tại, chưa hash payload backup. Bật
`BACKUP_FULL_HASH` ở config cho kiểm byte backup. Cả hai chế độ đều không chứng
minh storage có failure domain độc lập hay chính sách retention phù hợp.

40 calibration task được reuse từ qualification có source
`qualification/<profile>/fast/tasks/<id>`, copy ở `qualification/<profile>/backup/<id>`
và **không có worker ACK theo thiết kế**. Audit nhận diện đúng đường dẫn đóng băng
này trong collection; ghi `qualification_reuse`/`qualification_local_copy` và
`NOT_EMITTED_QUALIFICATION_REUSE`, không giả ACK và không tính là mất task.
''')
code('''
live = A.source_and_backup()
display(live.groupby(['source_marker','backup_marker','ack'],dropna=False).size().rename('tasks'))
display(A.tables['frozen_manifest_integrity'])
display(A.sections['live_backup_scope'])
''')
md('''
## 5. Trùng lặp và leakage giữa split

Kiểm UID từ normalized text, UID/text/group_id có vượt split không; liệt kê
parent_group/image_id/template_family giao nhau để điều tra. Template family
giống nhau có thể hợp lệ nếu split recipe cho phép; không tự gọi đó là leakage.
Đọc `test-locked.jsonl` **chỉ text/metadata** để đối chiếu UID/group với train/dev/
calibration, không sinh/chấm hay dùng outcome test. Gần nhau theo embedding ở §15
là tín hiệu kiểm thủ công, không thay audit ngữ nghĩa/nguồn.
''')
code('''
display(A.leakage().head(40))
display(A.sections['test_leakage_scope'])
display(A.data['prompts'].groupby(['split','source'],dropna=False).agg(
    prompts=('prompt_uid','nunique'),groups=('group_id','nunique')))
''')
md('''
## 6. Validity, missingness, failure và finite contract

Kiểm `valid => img_ok & latent_ok & finite(ir, pickscore_raw) & pixel_hash`.
Invalid phải có failure_code; giữ riêng tỷ lệ preview/final và mẫu số từng profile.
**Valid100% chỉ nói pipeline số học không phát hiện lỗi**, không xác nhận alignment,
counting, composition hay aesthetic. Scorer finite vẫn có thể sai về chất lượng.
''')
code('''
display(A.validity())
display(A.tables['previews_failure_reasons'])
display(A.tables['delivered_failure_reasons'])
display(A.preview.groupby(['profile_id','split','j']).valid.agg(['size','sum','mean']))
''')
md('''
## 7. Reward distribution, tails và prompt difficulty

Xem mean/std/min/p01/p10/p50/p90/p99/max, histogram, ECDF theo từng profile ×
split × source. IR và PickScore khác thang; không cộng/so trực tiếp hai số raw.
So model còn lẫn resolution/recipe: Hyper512 vs Lightning1024 không phải ablation
chỉ thay kiến trúc. Xem cả stratified tables để tránh aggregate che mix prompt.

Seed rows dùng để mô tả phân phối; CI ở §12 dùng source-group, không dùng ~39k
seed như mẫu độc lập. Low tail phải đọc prompt/ảnh nếu ảnh được giữ; thiếu PNG
không phải lỗi Core.
''')
code('''
display(A.tables['score_distribution'])
display(A.tables['score_tails'][['profile_id','split','source','prompt_uid','seed_idx','ir','pickscore_raw','text']].head(30))
''')
md('''
## 8. Seed diversity, collapse và seed-position bias

Mỗi task: std/range reward, best/mean, số valid seed, số pixel hash riêng.
Pixel trùng là tín hiệu collapse/bug; pixel khác không chứng minh semantic diversity.
Reward spread thấp có thể do scorer saturation, prompt dễ hoặc seed gần nhau về
chất lượng; chưa đủ để kết luận ảnh giống nhau. So seed_idx0..7 giúp phát hiện
ordering/RNG bug, nhưng phải xem phân phối theo prompt và repeatability.
''')
code('''
display(A.diversity().groupby(['profile_id','split']).agg(
    tasks=('prompt_uid','size'),ir_std=('ir_std','mean'),ir_range=('ir_range','mean'),
    unique_pixels=('unique_pixels','mean'),valid_seeds=('valid_seeds','mean')))
display(A.tables['seed_position'])
print('Rows sharing a final pixel hash:', len(A.tables['duplicate_final_pixels']))
''')
md(r'''
## 9. Best-of-k chính xác: cần bao nhiêu seed?

Với task có reward cuối \(r_1,\ldots,r_8\), tính cho mỗi \(k=1..8\):

\[B_k=\binom{8}{k}^{-1}\sum_{S:|S|=k}\max_{s\in S}r_s.\]

Đây là kỳ vọng **mọi tổ hợp** (tối đa70 tổ hợp/k), không dùng prefix seed0..k−1
và không Monte Carlo. Xem gain \(B_k-B_1\), marginal gain, theo profile/split.
Aggregate chuẩn chỉ dùng all-eight-valid; bảng vẫn giữ task ít valid với cờ riêng.
Đây là headroom chất lượng trong bank đã sinh, chưa tính latency và chưa phải
chính sách có thể quan sát final reward miễn phí.
''')
code('''
budget=A.tables['seed_budget_exact']
display(budget[budget.all_eight_valid].groupby(['profile_id','metric','k']).agg(
    tasks=('prompt_uid','size'),expected_best=('expected_best','mean'),gain=('gain_vs_random_one','mean')))
''')
md('''
## 10. Trajectory và khả năng dự đoán winner cuối

Tại từng checkpoint j: reward preview, rank correlation **trong task** với
delivered; tỷ lệ defined-rank phải đi cùng mean vì tied/constant reward không có
Spearman xác định. Pooled correlation thường bị prompt difficulty chi phối.
Đo delta giữa checkpoint, tỷ lệ đảo chiều và plateau; trajectory không đơn điệu
không tự động là collector bug. Không giả x0_j8 và final giống hệt trên mọi model.
''')
code('''
display(A.trajectories())
display(A.tables['trajectory_increments'])
display(A.preview.groupby(['profile_id','j'])[['drift_available','cfg_gap_available']].mean())
''')
md(r'''
## 11. Khi nào prune? Oracle survival và delivered regret

Tại từng j, giữ top-k preview theo active reward (k1/2/4), đo winner cuối còn
sống không và \(\max_s r_s^{final}-\max_{s\in S_j}r_s^{final}\). Chỉ dùng task
đủ mọi valid để chẩn đoán selection; so mẫu số với coverage để thấy survivorship.
Tie-break cố định seed_idx nhỏ. Bảng j8 là **terminal diagnostic**; legal preview
VERIFY chỉ j1..7. Các phép này dùng final cho evaluator, tuyệt đối không đưa
oracle/future/inactive scorer vào actor features.
''')
code('''
prune=A.tables['pruning_diagnostics']
display(prune.groupby(['profile_id','metric','j','k']).agg(
    tasks=('prompt_uid','size'),oracle_survival=('oracle_survives','mean'),regret=('final_regret','mean')) if len(prune) else prune)
''')
md('''
## 12. Bank repeatability và paired model contrasts với cluster CI

Dev có bank0/1 cùng UID: so paired bank1−bank0 cho mean/best reward. So profile
trên cùng prompt, cùng bank; Lightning−Hyper. Trước bootstrap, lấy mean trong
`group_id` để tránh đếm seed, bank, prompt cùng source-group là độc lập.

CI95% percentile bootstrap1000 nhóm, seed cố định. Estimand là mean **đều trọng
số source-group**, không phải mean seed hoặc đều prompt nếu group size khác.
Group<2 không xuất CI. Không chọn threshold/model thắng bằng khoảng tin cậy của
hàng chục contrast exploratory; chưa có confirmatory test hay multiplicity claim.
''')
code('''
display(A.paired_analysis())
display(A.sections['statistical_scope'])
''')
md('''
## 13. Scorer agreement và tính hữu ích của feature

IR/PickScore: rank agreement trong task, winner agreement, IR mất bao nhiêu nếu
chọn winner PickScore và ngược lại. Reward conflict hữu ích để xem controller
đang tối ưu mục tiêu nào; không coi một scorer là ground-truth con người.

Feature `drift`/`cfg_gap`: availability, distributions, within-task Spearman tới
final IR ở từng j. Lightning guidance0 phải CFG unavailable/null, drift j1 thiếu
là expected. Không fill null bằng0 để làm giả feature. Correlation này là phân
tích privileged cho người nghiên cứu, không phải chứng minh feature mang thông
tin tăng thêm sau khi kiểm soát prompt/checkpoint hoặc có giá trị online.
''')
code('''
display(A.scorer_and_feature_diagnostics().groupby(['profile_id','split']).agg(
    tasks=('prompt_uid','size'),winner_agreement=('winner_agrees','mean'),
    rank_agreement=('within_task_spearman','mean'),ir_regret=('ir_regret_of_ps_winner','mean')))
display(A.tables['feature_distribution'])
display(A.tables['feature_rank_associations'].groupby(['profile_id','j','feature']).agg(
    task_rows=('prompt_uid','size'),defined=('within_task_spearman_to_final_ir','count'),
    rho=('within_task_spearman_to_final_ir','mean')))
''')
md('''
## 14. Baseline offline với lịch prune hợp lệ, không tuning

Chạy full8 và lịch fixed2/5,3/6,4/7, random3/6; prune8→4→2. Actor chỉ dùng
reward active của alive seeds ở checkpoint hiện tại. Final chọn trong survivors,
inactive scorer chỉ dùng evaluator. Seed random cố định; một repeat không phải
kỳ vọng random-policy. Task invalid bị loại **có số đếm rõ**, không impute.

Đếm candidate-forward và VERIFY units để so cấu trúc workload. **Không đổi các
units này thành GPU speedup/USD**: kernel batching, VAE/scorer, I/O, concurrency
và cost K2/4/8 phi tuyến. Full-batch trajectories chưa chứng minh bằng trajectories
online của batch đã prune. Dev chỉ để mô tả; chưa chọn cấu hình hay train controller.
''')
code('''
policies=A.offline_policies()
display(policies.groupby(['profile_id','split','active','policy']).agg(
    tasks=('prompt_uid','size'),score=('final_active_score','mean'),regret=('oracle_regret','mean'),
    forward_units=('candidate_forward_units','mean'),preview_units=('preview_verify_units','mean'),
    terminal_units=('final_verify_units','mean')) if len(policies) else policies)
display(A.sections['offline_policy_scope'])
''')
md('''
## 15. Core384: tokens, truncation, embedding và gần trùng train

Kiểm norm/dtype/index ở §2. Ở đây xem original/used tokens và tỷ lệ truncation
theo source/split; max256 gồm special tokens. PCA để nhìn cấu trúc, không dùng
2D để phán leakage hoặc separability. Với mỗi dev/calibration UID, tìm nearest
train cosine **chính xác**, block256 để hạn chế RAM. Pair cosine cao phải đọc
text/nguồn, không tự xóa hay resplit dữ liệu sau khi nhìn reward.

MiniLM text embedding không phải image embedding; không dùng nó để kết luận
image diversity hoặc chất lượng tạo ảnh. Không tải encoder hay re-encode.
''')
code('''
display(A.embeddings())
display(A.tables.get('truncation'))
display(A.tables['near_duplicate_train_neighbors'].sort_values('cosine',ascending=False).head(35))
''')
md('''
## 16. Native latent, RNG và provenance

Đọc header safetensors toàn bộ task, so tensor keys/dtype/shape với provenance;
x0_01..08 giữ native dtype riêng, không stack để ép dtype. Đã đo ở collector:
Hyper x0/final fp16; Lightning x0 fp32, final fp16. Bất thường phải đối chiếu
recipe/receipt, không upcast v1 để giả native. Header đúng chưa chứng minh tensor
finite hoặc giữ nguyên byte; sample value audit so SHA256 từng tensor với hash
trước serialization, kiểm finite/RMS/min/max, một tensor trong RAM mỗi lượt.

Đồng thời kiểm deterministic CPU noise seed từ UID/bank/seed_idx, provenance
runtime/source/registry/scorer/encoder có bị trộn phiên bản không. Không import
Torch hay gọi VAE re-decode. GPU re-decode đầy đủ là audit khác, chưa chạy ở đây.
''')
code('''
native=A.native_and_provenance()
display(native.groupby(['profile_id','tensor','dtype']).size().rename('tasks') if len(native) else native)
display(A.sections['native_scope'])
prov=A.tables['provenance_identities']
display(prov.groupby('profile_id')[['registry_hash','runtime_identity','encoder_identity','scorer_identity']].nunique())
display(native[native.values_checked].head(30) if len(native) else native)
''')
md('''
## 17. Timings theo profile và cost calibration K2/4/8

Profile lấy từ task top-level / core `profile_id`, không lookup nhầm trong
`prompt` như EDAv1. Xem stage mean/p50/p90/p99 theo profile/split. Fused path có
timing stage gộp ở frame đầu; không chia đều rồi coi mỗi preview cost giống nhau.
Qualification cost sample native-batch K2/4/8 tách generation/decode/IR/PickScore.
Scorer cost chưa có = unavailable, không phải0.

Tổng task-wall là work, không phải fleet makespan. Không suy GPU idle/throughput/
ETA từ task timing khi thiếu timeline worker/scheduler và overlap. EDA này không
restart job hay tự mở benchmark để lấp chỗ thiếu. PickScore diagnostic không được
tính như scorer active IR nếu recipe thực thi không dùng nó.
''')
code('''
display(A.costs_storage())
display(A.tables['qualification_cost_samples'])
display(A.sections['storage_and_cost_scope'])
''')
md('''
## 18. Storage, retention và Core-only portability

So bytes thực (decimalGB và GiB), Core/latent/QA/provenance/qualification, cap80GB;
working storage có raw+backup+parts không nằm trong cap export. Core phải tự mang
prompt text/embedding/candidate/preview/final/mask/provenance references để dùng
không cần PNG/latent. Không xóa extension để kiểm: phân tích Core ở notebook chỉ
mở các bảng và embedding, §16/§20 mới mở extension rõ ràng.

Xem retention.json và coverage.json; retention chưa xác nhận không được gọi là
durable. Không lấy hash/restore PASS làm bằng chứng backup khác failure domain.
''')
code('''
display(A.tables['storage_components'].assign(GB=lambda x:x.bytes/1e9))
display(json.loads((EXPORT/'coverage.json').read_text()))
retention=pd.DataFrame(json.loads((EXPORT/'retention.json').read_text()))
display(retention.groupby(['native_exported','durability'],dropna=False).size().rename('tasks'))
''')
md('''
## 19. Biểu đồ tĩnh để lưu/chia sẻ

Matplotlib là tùy chọn. Nếu có: histogram/ECDF, trajectory/rank, exact best-of-k,
PCA và pruning headroom; ảnh PNG lưu cùng report. Không cần renderer browser.
Thiếu thư viện thì ghi UNAVAILABLE và giữ toàn bộ CSV, không tự cài dependency
vào runtime đang dùng. Những đồ thị không có dev trong dataset subset sẽ trống
và phải được đọc cùng coverage, không nội suy dữ liệu.
''')
code('''
display(A.plots())
if A.figures:
    try:
        from IPython.display import Image
        for name in A.figures: display(Image(filename=str(OUT/name)))
    except ImportError: print('Figures saved:', A.figures)
''')
md('''
## 20. Manual QA gallery: low/mid/high trong tập ảnh đã giữ

Chỉ ảnh calibration đã được preselect trước reward; main không giữ PNG đại trà.
Chọn low/middle/high IR trong mỗi profile của tập saved-QA, thumbnail≤320px,
tối đa48ảnh, xuất HTML tự chứa và CSV reviewer trống. **Đây không phải sample
đại diện cho quality toàn dataset.** Không gọi model để lấp ảnh thiếu.

Reviewer đọc prompt rồi ghi: alignment, counting, composition, artifacts,
scorer mismatch, ghi chú. So IR thấp nhưng ảnh đúng / IR cao nhưng ảnh sai để
phát hiện scorer failure. Giữ raw scores, không tự relabel valid hoặc bỏ tail.
Muốn estimate quality population cần mẫu UID random độc lập và protocol riêng.

Trước gallery, CPU decode mọi PNG đã giữ và đối chiếu pixel SHA256 với Core,
đo channel std và tỷ lệ pixel gần đen/trắng để điều tra ảnh blank. Đây là heuristic
và kiểm byte ảnh, chưa phải phân loại chất lượng hay VAE re-decode từ latent.
''')
code('''
pixels=A.qa_pixel_audit()
display(pixels.groupby(['profile_id','table']).agg(images=('image_uri','size'),
    hash_matches=('pixel_hash_matches','sum'),black_fraction=('black_fraction','mean')) if len(pixels) else pixels)
display(A.qa_gallery())
display(A.tables['qa_manual_review'].head(12))
print('Open:', OUT/'qa_gallery.html')
''')
md('''
## 21. Chốt kiểm tra, báo cáo và gói trả lại

`summary.json`, `checks.csv` là kết quả kiểm thực sự của lần chạy này. FAIL
coverage/hash/RNG/native không được đổi threshold để xanh. Statistical EDA
không tự trở thành collector qualification hoặc online scientific reproduction.
`REPORT.md` có Material Passport, phạm vi đã kiểm/chưa kiểm và11 considerations
về sai lầm thống kê; `report.html` gom bảng/biểu đồ; artifact hashes giúp chuyển
báo cáo nguyên vẹn. Không tự upload data công ty ra mạng.

Quyết định đọc theo thứ tự:
1. Structural/hash/native/source/backup lỗi? Giữ artifact, sửa nguyên nhân trước training.
2. Coverage/split/leakage đạt? Kiểm source-group và external locked test.
3. Có seed headroom và checkpoint hữu ích? Xem regret theo profile/source/dev.
4. Reward có conflict/QA lệch? Xem reviewer labels, chưa kết luận bằng valid rate.
5. Cost samples đủ và đúng runtime? Sau đó mới thiết kế controller benchmark.

Gửi lại thư mục báo cáo, ưu tiên summary/checks/coverage/paired/trajectory/
scorer/native/timings/QA review; chưa cần chuyển latent/ảnh hàng chục GB.
''')
code('''
display(A.finish())
failed=pd.DataFrame(A.checks).query("status != 'PASS'")
display(failed.head(60))
print('Report:', OUT/'report.html')
print('Markdown:', OUT/'REPORT.md')
print('Artifacts:', OUT/'ARTIFACT_SHA256.json')
if len(failed): print('ATTENTION: failures remain; this is not a dataset acceptance PASS.')
''')


def main():
    notebook={'nbformat':4,'nbformat_minor':5,
              'metadata':{'kernelspec':{'display_name':'Python 3 (existing APSP runtime)',
                    'language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},
                    'apsp':{'schema':'apsp-core-lite2-v1','purpose':'read-only company post-collection EDA'}},
              'cells':[{**cell,'id':f'lite2-eda-{i:03d}'} for i,cell in enumerate(CELLS)]}
    path=ROOT/'notebooks/APSP_LITE2_EDA_COMPANY.ipynb'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
