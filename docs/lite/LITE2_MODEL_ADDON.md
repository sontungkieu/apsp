# Lite2 model add-on — tải phần thiếu trên node

Gói `apsp-lite2-model-addon-v1.tar`: **4133734400 bytes**, khoảng4.13GB.
SHA256: `3ba6fc1a2cd47ee9e42b7857146c6f967ed56c4c9dbb6d28ae9cbffcdc306533`.
HF revision: `246ca32ce4dcb287296a740ed7cbd17998d19024`; repo: https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/tree/246ca32ce4dcb287296a740ed7cbd17998d19024.

Có UNet/VAE/config SD1.5 mirror451f4fe, Hyper-SD15-8steps-CFG LoRA bc08d97,
MiniLM-L6-v2 1110a24 (384d, mask-mean pooling + L2, max256 tokens) và upstream licenses.
Không có runtime mới, không đóng lại Lightning/ImageReward/PickScore hoặc text
encoder/tokenizer SD1.5 đã có. Sáu file cũ được kiểm hash trước khi tạo cache view.
Base weights dùng default safetensors upstream, không convert/quantize file.

Restore tạo HF cache view riêng, tham chiếu companion cũ bằng symlink; phải giữ
companion cũ tại chỗ. Không sửa companion v1, snapshot, job hoặc source đang chạy.
Shared storage chỉ cần một bản add-on cho4 worker nếu cả4 node mount cùng path.
Dự trù16GB trống cho ranges + tar + payload restore; không tự xóa ranges/cache.

Block APSP-L2M1 — dùng proxy env đã nạp theo DSH note, không cần SDK mới:

```bash
(
set -euo pipefail
PROXY="${PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"
[ -n "$PROXY" ] || { echo 'TU_CHOI: THIEU proxy'; exit 2; }
ROOT=/workspace/storage-shared/nlp/tungks/apsp_all
DEST="$ROOT/lite2-addon-download"; BOOT="$DEST/bootstrap"
mkdir -p "$BOOT"
BASE='https://huggingface.co/datasets/codemaivanngu/apsp-lite-b200-cu128/resolve/246ca32ce4dcb287296a740ed7cbd17998d19024'
for F in download_part_hf.sh LITE2_ADDON_DOWNLOAD.json restore_lite2_addon.py; do
  curl --silent --fail --location --retry 3 --connect-timeout 20 \
    --proxy "$PROXY" --noproxy '' "$BASE/$F" -o "$BOOT/$F"
done
printf '%s  %s\n' \
  'f2f398f3a7c6aba955350e2e71e9479d2c576d052a0f98d96f854e00a0504df1' "$BOOT/download_part_hf.sh" \
  'a1c5af144605ba68feff92095453f52935e09e33cd168e35b5fd346c8ffec9f5' "$BOOT/LITE2_ADDON_DOWNLOAD.json" \
  '15c18911a42260bf86c04ee8802df428a9eb24d6badc03736b3547539d536c89' "$BOOT/restore_lite2_addon.py" | sha256sum -c -
APSP_DOWNLOAD_THREADS=12 bash "$BOOT/download_part_hf.sh" \
  "$BOOT/LITE2_ADDON_DOWNLOAD.json" 1 "$DEST" 'codemaivanngu/apsp-lite-b200-cu128' '246ca32ce4dcb287296a740ed7cbd17998d19024'
)
echo '--- het Block APSP-L2M1, shell con song ---'
```

Block APSP-L2M2 — restore riêng và kiểm CPU offline; không chạy inference trên B200:

```bash
(
set -euo pipefail
ROOT=/workspace/storage-shared/nlp/tungks/apsp_all
DEST="$ROOT/lite2-addon-download"
COMPANION="${APSP_COMPANION_ROOT:-$ROOT/apsp-lite-b200-cu128/companion-v1}"
python3 "$DEST/bootstrap/restore_lite2_addon.py" restore \
  --archive "$DEST/apsp-lite2-model-addon-v1.tar" \
  --companion "$COMPANION" --target "$ROOT/lite2-model-addon-v1" \
  --sha256 '3ba6fc1a2cd47ee9e42b7857146c6f967ed56c4c9dbb6d28ae9cbffcdc306533'
source "$ROOT/lite2-model-addon-v1/activate_lite2_models.sh"
/opt/apsp-runtime/bin/python "$DEST/bootstrap/restore_lite2_addon.py" check \
  --root "$APSP_LITE2_MODELS"
)
echo '--- het Block APSP-L2M2, shell con song ---'
```

Sau đó trong shell/worker V2 mới, source `activate_companion.sh` trước, rồi:

```bash
source /workspace/storage-shared/nlp/tungks/apsp_all/lite2-model-addon-v1/activate_lite2_models.sh
```

Exports: `HF_HOME`, `APSP_SD15_SNAPSHOT`, `APSP_HYPER_SD15_LORA`, `APSP_MINILM_SNAPSHOT`.
Source này không thay `APSP_SOURCE` về snapshot cũ. Các worker v1 đang chạy không
bị đổi env. Không cần tải add-on lặp lại cho từng GPU nếu dùng shared mount.

## Bằng chứng và giới hạn

Đã chạy trên Modal CPU: upstream file size/LFS SHA, toàn archive/member hash,
restore sang root riêng, chặn network, load full SD1.5 + fuse đúng CFG8 LoRA,
DDIM trailing8 timesteps và MiniLM forward `[1,384]` L2-normalized. HF đọc ngược
metadata size/LFS SHA, anonymous one-byte206 và hash toàn bộ helper/manifest.
Không tải model về laptop. Runtime/dependency lock giữ nguyên.

**Receipt trong tài liệu này chỉ nghiệm thu add-on model.** Collector V2 hiện
đã có; xem [RUNBOOK_LITE2.md](RUNBOOK_LITE2.md) và
[qualification của source V2](B200_LITE2_QUALIFICATION.md). Node công ty vẫn phải
chạy qualification V2 riêng trước bulk. Không dùng receipt model/collector v1
thay V2 hoặc dùng `hyper_flux8` cho checkpoint SD15.
