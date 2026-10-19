# Companion bundle ngoài runtime

**Đường tải node hiện hành:** [DSH_DOWNLOAD_WORKFLOW.md](DSH_DOWNLOAD_WORKFLOW.md).
Hai part gzip lossless dưới30GB/part, mỗi instance dùng12 curl range workers qua
proxy/HF. Phần Modal CLI bên dưới chỉ là truy xuất Volume của operator, không phải
đường tải dùng cho node công ty.

Một archive riêng `apsp-lite-companion-v1.tar` chứa source/Git bundle, pinned model
weights/adapters, tokenizer/config, hai scorer, common prompt encoder, text/task
manifests và raw metadata/caption annotations đã dùng để chuẩn bị prompt. Dependency
runtime4.57GB giữ riêng. Archive không chứa token/credential, runtime dependencies,
test images, full generated image/latent bank hoặc các job output cũ.

Đóng gói trên CPU Modal/private Volume `apsp-lite-runtime-v1`, profile
`billionaireproject8`, trong cùng cap task7USD. Không tải weights về laptop.
Chỉ lấy đúng loader inputs từ các revision đã qualify; không thêm model mới.
SDXL50, Lightning và FLUX dùng shared snapshots; bỏ alternate ONNX/OpenVINO/raw
checkpoint exports và bản PickScore.bin trùng safetensors. HF snapshot symlinks
được materialize thành regular files bên trong archive, không phụ thuộc blob paths
của Volume gốc. Không chuyển dtype/quantize hoặc đổi byte weights.

Inventory ban đầu:60,623,709,730 bytes model/scorer/tokenizer, cộng metadata/text/source.
Tar không nén để tránh tốn CPU cho tensor khó nén; checksum từng file được kiểm khi
ghi và đọc lại toàn archive trước publication. Thiếu hoặc sai pinned hash thì fail,
giữ `.partial` làm bằng chứng và không công bố archive hoàn tất.

## Restore trên Linux node được cấp quyền

Đường tải hiện hành là [NODE_DOWNLOAD_COMMANDS.md](NODE_DOWNLOAD_COMMANDS.md).
Lệnh Modal dưới đây chỉ dành cho operator truy xuất Volume nếu cần:

```bash
mkdir -p /shared/apsp-lite/payload
modal volume get apsp-lite-runtime-v1 /export/apsp-lite-companion-v1.tar \
  /shared/apsp-lite/payload/apsp-lite-companion-v1.tar --profile billionaireproject8
modal volume get apsp-lite-runtime-v1 /export/apsp-lite-companion-v1.tar.sha256 \
  /shared/apsp-lite/payload/apsp-lite-companion-v1.tar.sha256 --profile billionaireproject8
cd /shared/apsp-lite/payload
sha256sum -c apsp-lite-companion-v1.tar.sha256
```

Lấy `env/lite/companion.py` từ source handover đã có và dùng Python>=3.10 stdlib:

```bash
python3 env/lite/companion.py restore \
  --archive /shared/apsp-lite/payload/apsp-lite-companion-v1.tar \
  --target /shared/apsp-lite/companion-v1 \
  --sha256 SHA256_FROM_VERIFIED_RECEIPT
source /shared/apsp-lite/companion-v1/activate_companion.sh
```

Helper kiểm archive SHA, từng member/hash, chỉ nhận regular files, chặn traversal/
absolute paths/links/devices, preserve target có sẵn và publish restore bằng rename.
Restore lặp lại đúng marker trả `already_restored`, không đọc lại toàn file; muốn
full recheck dùng `companion.py verify --archive ...`. Cần chỗ trống cho archive và
payload sau giải nén, dự trù ít nhất130GB ngoài runtime/output/backup.

`HF_HOME` trỏ `companion-v1/hf`, offline flags bật và `PYTHONPATH` trỏ source đúng
checkpoint. Dependency runtime vẫn restore qua `ensure_runtime.py` riêng. Sau đó:

```bash
/opt/apsp-runtime/bin/python "$APSP_SOURCE/env/lite/verify_companion_offline.py" \
  --root "$APSP_COMPANION_ROOT"
```

CPU offline loader check chặn socket network, đọc generator configs/tokenizers,
resolve adapter/scorer paths, kiểm safetensors/shard indices và common encoder
forward768d; không gọi nó là GPU rerun. Company-node gate, SDXL50 drift và full
collection/backup limitations trong LITE_DATA_REPORT.md vẫn có hiệu lực.

Text có2850 UID và13800 task, không phải110400 generated candidates. Raw inputs
gồm DiffusionDB metadata, COCO caption annotations, Parti TSV và GenEval exclusion
texts. Full dataset chưa thu; manual language/content/truncation QA vẫn pending.
Đã public theo chỉ định người dùng tại HF dataset `codemaivanngu/apsp-lite-b200-cu128`.
Attribution/revision/hash và bản license/NOTICE upstream đi kèm; không thay license
bằng cách đóng gói offline. Không có credential trong companion.
