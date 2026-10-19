# APSP Lite — hai instance,12 luồng mỗi part, rồi ghép

Source trong companion là bootstrap snapshot. Sau restore, dùng
[SOURCE_DISTRIBUTION.md](SOURCE_DISTRIBUTION.md) để lấy GitHub `main` mới qua HF
bundle và đặt `APSP_SOURCE`/`PYTHONPATH` về checkout đó. Runtime/parts bên dưới
giữ nguyên revision, không tải lại model chỉ vì source đổi.

HF public: https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/tree/a13a190c0d29ed44a8ebff6b9c78d9e51f65e820
Revision ghim: `a13a190c0d29ed44a8ebff6b9c78d9e51f65e820`. Không thay bằng `main`.

| Payload | Bytes | SHA256 |
|---|---:|---|
| part01 | 22992424293 | `18722c698015991b4fc95d9c57fe6edcc41e2db10d8185383dea0a57aa17bada` |
| part02 | 22992424292 | `a7b463d4c0781aad3ee8e65ac8f97c034aef0e8ea6b8ae91ed9bf9931911eafb` |
| runtime.tar.gz | 4571656042 | `d944669e3211621416e5e017fdc010ea04d8726a89ec35cea2eb30d686bfda84` |

Hai part nén lossless còn khoảng22.99GB mỗi file, dưới cap30GB. Mỗi instance tải
một part bằng12 kết nối curlRange, tổng tối đa24 kết nối tải. Runtime là file riêng.
Đây là lệnh cung cấp cho operator; chưa chạy trên hai instance/node công ty.

Đường tải theo PROXY_DOWNLOAD_GUIDE.md §6 và DPCA §0b.2. Chỉ cần bash/curl/python3,
không cần cài SDK Modal hoặc HF. Proxy lấy từ env sẵn có, không in giá trị. Nếu
thiếu, nạp theo PROXY_DOWNLOAD_NOTE.md trước. Hai instance cần storage chung hoặc
chuyển hai part đã verify về cùng join node. Shared destination mặc định là staging
`/workspace/storage-shared/nlp/tungks/tmp/apsp-lite-b200-cu128`; có thể đặt
`APSP_PAYLOAD_DIR` phù hợp instance. Dự trù250GB nếu giữ cả range chunks,parts,tar và
payload bung. Không tự xóa cache/ranges, không dùng GPU hoặc đụng process khác.

Block APSP-D1 — chạy trên instance A do người dùng chọn — tải part01; tên node thực tế chưa được cung cấp.

```bash
(
set -euo pipefail
PROXY="${PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"
[ -n "$PROXY" ] || { echo 'TU_CHOI: THIEU proxy; nap env theo note'; exit 2; }
DEST="${APSP_PAYLOAD_DIR:-/workspace/storage-shared/nlp/tungks/tmp/apsp-lite-b200-cu128}"
BOOT="$DEST/bootstrap-part01"
mkdir -p "$BOOT"
BASE='https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/resolve/a13a190c0d29ed44a8ebff6b9c78d9e51f65e820'
for F in download_part_hf.sh COMPANION_PARTS.json; do
  echo "tai helper $F"
  curl --silent --fail --location --retry 3 --connect-timeout 20 \
    --proxy "$PROXY" --noproxy '' "$BASE/$F" -o "$BOOT/$F"
done
printf '%s  %s\n' \
  'f2f398f3a7c6aba955350e2e71e9479d2c576d052a0f98d96f854e00a0504df1' "$BOOT/download_part_hf.sh" \
  '6cf333969b6aaaf515eca59642d49f75550c5c14cbc5b4d88a7822de76cb729d' "$BOOT/COMPANION_PARTS.json" | sha256sum -c -
APSP_DOWNLOAD_THREADS=12 bash "$BOOT/download_part_hf.sh" \
  "$BOOT/COMPANION_PARTS.json" 1 "$DEST" 'codemaivanngu/apsp-lite-b200-cu128' 'a13a190c0d29ed44a8ebff6b9c78d9e51f65e820'
)
echo '--- het Block APSP-D1, shell con song ---'
```

Block APSP-D2 — chạy trên instance B do người dùng chọn — tải part02; tên node thực tế chưa được cung cấp.

```bash
(
set -euo pipefail
PROXY="${PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"
[ -n "$PROXY" ] || { echo 'TU_CHOI: THIEU proxy; nap env theo note'; exit 2; }
DEST="${APSP_PAYLOAD_DIR:-/workspace/storage-shared/nlp/tungks/tmp/apsp-lite-b200-cu128}"
BOOT="$DEST/bootstrap-part02"
mkdir -p "$BOOT"
BASE='https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/resolve/a13a190c0d29ed44a8ebff6b9c78d9e51f65e820'
for F in download_part_hf.sh COMPANION_PARTS.json; do
  echo "tai helper $F"
  curl --silent --fail --location --retry 3 --connect-timeout 20 \
    --proxy "$PROXY" --noproxy '' "$BASE/$F" -o "$BOOT/$F"
done
printf '%s  %s\n' \
  'f2f398f3a7c6aba955350e2e71e9479d2c576d052a0f98d96f854e00a0504df1' "$BOOT/download_part_hf.sh" \
  '6cf333969b6aaaf515eca59642d49f75550c5c14cbc5b4d88a7822de76cb729d' "$BOOT/COMPANION_PARTS.json" | sha256sum -c -
APSP_DOWNLOAD_THREADS=12 bash "$BOOT/download_part_hf.sh" \
  "$BOOT/COMPANION_PARTS.json" 2 "$DEST" 'codemaivanngu/apsp-lite-b200-cu128' 'a13a190c0d29ed44a8ebff6b9c78d9e51f65e820'
)
echo '--- het Block APSP-D2, shell con song ---'
```

Block APSP-D3 — chạy trên join node do operator chọn, sau khi có đủ hai part đúng hash.

```bash
(
set -euo pipefail
DEST="${APSP_PAYLOAD_DIR:-/workspace/storage-shared/nlp/tungks/tmp/apsp-lite-b200-cu128}"
cd "$DEST"
printf '%s  %s\n' \
  '18722c698015991b4fc95d9c57fe6edcc41e2db10d8185383dea0a57aa17bada' apsp-lite-companion-v1.tar.gz.part01 \
  'a7b463d4c0781aad3ee8e65ac8f97c034aef0e8ea6b8ae91ed9bf9931911eafb' apsp-lite-companion-v1.tar.gz.part02 | sha256sum -c -
if [ ! -e apsp-lite-companion-v1.tar ]; then
  [ ! -e apsp-lite-companion-v1.tar.joining ] || { echo 'TU_CHOI: joining file da co; kiem tra truoc'; exit 2; }
  cat apsp-lite-companion-v1.tar.gz.part01 apsp-lite-companion-v1.tar.gz.part02 \
    | gzip -dc > apsp-lite-companion-v1.tar.joining
  printf '%s  %s\n' 'd11d1ecf591ae59e6e63d80d767e74c3659ded51ec93792139aa34eb667b3b96' apsp-lite-companion-v1.tar.joining | sha256sum -c -
  mv apsp-lite-companion-v1.tar.joining apsp-lite-companion-v1.tar
fi
printf '%s  %s\n' 'd11d1ecf591ae59e6e63d80d767e74c3659ded51ec93792139aa34eb667b3b96' apsp-lite-companion-v1.tar | sha256sum -c -
)
echo '--- het Block APSP-D3, shell con song ---'
```

Sau đó dùng `restore_companion.py restore --archive ... --target ... --sha256 ...`
với SHA của tar ở trên. Helper đã kiểm từng file/hash, traversal/links và preserve
thư mục đích có sẵn. `ensure_runtime.py` và `runtime-manifest.json` dùng cho runtime
riêng. Tải chúng và LICENSES/NOTICE từ cùng HF revision; kiểm SHA trong SHA256SUMS
(SHA256 của chính file checksum: `2e318c09e27057f724d7cdf9fbdd3af003b1a1bfa1e51f344de8a4ca870f6961`).

Đã chạy: archive full hash readback, hai-part gzip roundtrip, restore companion,
offline CPU config/tokenizer/scorer-path/shard checks và common encoder768d. Template
curl kiểm12 request song song/resume/last-range/refuse server bỏ Range bằng fixture;
HF public anonymous Range được kiểm trực tiếp. Chưa kiểm proxy/node công ty trong
lượt này và chưa bật hai instance thay người dùng. Không dùng kết quả transport làm
bằng chứng controller hoặc full dataset. Train/dev generated candidate vẫn0;
SDXL50 cross-container drift và manual prompt QA còn pending.

Block APSP-D4 — tải runtime riêng và restore CPU sau D3; cần quyền ghi `/opt`.
Chỉ chạy khi join node đã được operator chọn cho runtime này.

```bash
(
set -euo pipefail
DEST="${APSP_PAYLOAD_DIR:-/workspace/storage-shared/nlp/tungks/tmp/apsp-lite-b200-cu128}"
PROXY="${PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"
[ -n "$PROXY" ] || { echo 'TU_CHOI: THIEU proxy'; exit 2; }
BOOT="$DEST/bootstrap-restore"; mkdir -p "$BOOT"; cd "$BOOT"
BASE='https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/resolve/a13a190c0d29ed44a8ebff6b9c78d9e51f65e820'
for F in SHA256SUMS download_part_hf.sh restore_companion.py ensure_runtime.py runtime-manifest.json; do
  curl --silent --fail --location --retry 3 --connect-timeout 20 \
    --proxy "$PROXY" --noproxy '' "$BASE/$F" -o "$F"
done
printf '%s  %s\n' '2e318c09e27057f724d7cdf9fbdd3af003b1a1bfa1e51f344de8a4ca870f6961' SHA256SUMS | sha256sum -c -
for F in download_part_hf.sh restore_companion.py ensure_runtime.py runtime-manifest.json; do
  awk -v f="$F" '$2==f { print }' SHA256SUMS | sha256sum -c -
done
python3 - <<'PY'
import json
json.dump({'parts':[{'file':'runtime.tar.gz','bytes':4571656042,'sha256':'d944669e3211621416e5e017fdc010ea04d8726a89ec35cea2eb30d686bfda84'}]},open('runtime-download.json','w'))
PY
APSP_DOWNLOAD_THREADS=12 bash download_part_hf.sh "$BOOT/runtime-download.json" 1 "$DEST" 'codemaivanngu/apsp-lite-b200-cu128' 'a13a190c0d29ed44a8ebff6b9c78d9e51f65e820'
python3 ensure_runtime.py --archive "$DEST/runtime.tar.gz" --manifest "$BOOT/runtime-manifest.json" --cpu-only
python3 restore_companion.py restore --archive "$DEST/apsp-lite-companion-v1.tar" \
  --target "$DEST/companion-v1" --sha256 'd11d1ecf591ae59e6e63d80d767e74c3659ded51ec93792139aa34eb667b3b96'
source "$DEST/companion-v1/activate_companion.sh"
/opt/apsp-runtime/bin/python "$APSP_SOURCE/env/lite/verify_companion_offline.py" --root "$APSP_COMPANION_ROOT"
)
echo '--- het Block APSP-D4, shell con song ---'
```

D4 là restore/import CPU, chưa thay gate B200/model/scorer trên node được giao.
Runtime dùng hai prefix cố định `/opt/apsp-python` và `/opt/apsp-runtime`; helper
từ chối ghi đè prefix không có marker đúng checksum.
