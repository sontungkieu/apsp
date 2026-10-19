# GitHub source và Hugging Face server distribution

Canonical GitHub: https://github.com/sontungkieu/apsp, branch `main`.
HF public dataset: https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128.
HF là kênh phân phối cho server; không phải Git remote hai chiều hoặc automatic sync.

## Publish sau mỗi milestone source

Commit đã ký và kiểm tra trước. Từ checkout Windows sạch, có GitHub credential và
HF_TOKEN trong env tạm hoặc credential HF đã cấu hình, chạy:

```powershell
python env/lite/publish_source.py
python env/lite/publish_source.py --publish
```

Lệnh đầu chỉ check. Lệnh thứ hai push `main` lên đúng GitHub, đọc ngược ref,
tạo full-history `source/apsp.bundle` + `source/source-manifest.json` và upload
cùng `source/update_source.py`, `source/update-apsp.sh` bằng một HF commit có
parent guard. Sau đó tải lại toàn bộ bốn file không token, kiểm SHA và kiểm lại
GitHub. Chỉ `DUAL_PUBLISH_PASS` mới hoàn tất. Receipt nằm trong
`remote_artifacts/source-<commit>-*/receipt.json`; không commit receipt vào chính
source nó mô tả, tránh vòng lặp commit/revision.

Không đưa model/dataset/runtime vào GitHub. Chỉ đồng bộ source nhỏ mỗi lần;
hai part model/data và runtime đã public giữ nguyên bytes và revision bàn giao.
`APSP_RELEASE.json`/root `SHA256SUMS` mô tả payload cũ; source channel có manifest
riêng. HF revision khác source Git commit. Snapshot source trong companion là
bootstrap c7d5130; source main mới tải qua kênh bundle này.

## Update server

Dùng bootstrap ghim HF revision/hash trong receipt bàn giao đầu tiên. Sau khi đã
có helper đáng tin, operator chạy (không cần cài HF/Modal SDK):

```bash
(
set -euo pipefail
bash /path/to/update-apsp.sh \
  --target /workspace/storage-shared/nlp/tungks/APSP-git
)
echo '--- het block APSP update, shell con song ---'
```

Proxy lấy từ PROXY/HTTPS_PROXY/HTTP_PROXY sẵn có, không in giá trị. `--revision`
có thể ghim một HF commit cụ thể; nếu bỏ thì resolve HF main đúng một lần rồi
tải manifest và bundle tại cùng immutable revision. Có lock riêng ngoài checkout.
Lần đầu clone bundle với `--branch main`; lần sau kiểm clean/main/hash/ref/ancestry
rồi `merge --ff-only`. Dirty, diverged hoặc rollback phải dừng, không reset/stash.
Giữ working receipt và bundle trong staging cạnh checkout để chẩn đoán. Không
update checkout đang có job dùng; operator chọn thời điểm giữa các run.

Sau khi activate companion/runtime, trỏ source mới rõ ràng để tránh dùng snapshot cũ:

```bash
export APSP_SOURCE=/workspace/storage-shared/nlp/tungks/APSP-git
export PYTHONPATH="$APSP_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
```

Output/model/cache nằm ngoài source checkout. Đổi source không rebuild runtime;
chỉ rebuild/qualify lại khi dependency lock/ABI/CUDA/base image đổi. Mỗi run ghi
Git source commit, HF source revision, runtime archive hash và model/data revision.
Việc phân phối source không thay cho company-node/GPU/scientific qualification.
