# apsp-lite-mixed-v1 — preparation candidate

Trạng thái: **prompt/task manifests đã tạo và kiểm checksum; chưa có train/dev bank**.
Đây chưa phải release dữ liệu DONE của plan. Full/v1 giữ nguyên; Lite là manifest
độc lập vì Full chưa khóa. Không dùng số lượng kỳ vọng làm số ảnh đã sinh.

| Split | UID | DiffusionDB | COCO | Structured | Parti |
|---|---:|---:|---:|---:|---:|
| train | 2000 | 1000 | 400 | 600 | 0 |
| dev | 300 | 150 | 60 | 90 | 0 |
| calibration | 50 | 25 | 10 | 15 | 0 |
| test ID | 300 | 150 | 60 | 90 | 0 |
| test source-shift | 200 | 0 | 0 | 0 | 200 |

Train structured có đúng 120 row/nhóm chính và 100 template family. Generator cấu
trúc xác định bằng code, không lấy output từ LLM hay reward. Family hash template
chưa điền thuộc tính, gồm câu/context/rendering clause cố định; không khẳng định
100 family là 100 cấu trúc ngữ pháp độc lập. Domain/naturalness còn phải review.

Nested 250 ⊂ 1000 ⊂ 2000 giữ quota nguồn 50/20/30 phần trăm. Ba profile dùng chung
UID/split. Hai bank × 8 seed/profile; seed v2 không phụ thuộc worker/rank.
Task file có đúng 13.800 train/dev task, tương ứng **kỳ vọng** 110.400 candidate,
1.177.600 preview row, 2.576.000 score value. Không có test task trong collection file.

Nguồn:
- DiffusionDB metadata `poloclub/diffusiondb@fb620fbe49fa4420e0734bd9c0df11f51176b61f`.
  Chỉ đọc metadata prompt, không tải ảnh. Candidate lấy theo thứ tự metadata,
  lọc độ dài, NSFW<0.1 và ASCII heuristic. Không phải random sample toàn corpus
  hay xác nhận mọi câu là tiếng Anh. Manual/language review còn pending.
- COCO2017 caption annotations từ bucket chính thức
  `https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`.
  Chọn một caption/image trong pool; giữ annotation id và image_id, không tải ảnh.
- Structured generator `apsp/lite/prepare_text.py`, version structured-lite-v1.
- PartiPrompts TSV chính thức tại commit `5a657978134374ce28973948331b319adef164bd`.
  Chọn 200 theo category round-robin; gọi `PartiPrompts-200`, không toàn benchmark.

SHA256 nguồn đầy đủ nằm trong manifest. Cần kiểm điều khoản/attribution từng
nguồn trước public redistribution; hiện workspace/Volume private, chưa publish.

UID=SHA256(UTF8(NFC+collapse whitespace)). Exact alias giữ mọi parent provenance;
COCO image/structured family được gom trước split. Near detector dùng lexical
3-gram Jaccard>=0.85; không đo semantic coverage. Release thực có 2.850 UID duy nhất
và group overlap=0 theo detector này.

Exclusion: 553 GenEval text, 1.632 Parti text, 370 v1 text khả dụng và một prompt
qualification riêng. CSV v1 còn 220 row thiếu text; manifest v1 khác chưa có.
Không tuyên bố đã loại hết overlap với toàn v1 hoặc mọi tương đồng ngữ nghĩa.

Frozen common encoder là SD1.5 CLIP pooler 768 chiều; không trộn native SDXL/FLUX
embeddings. Encoder/norm theo subset, per-encoder truncation và feature integration
đầy đủ vẫn cần trước training. Chưa fit PCA/normalizer trên test, chưa train controller.

Text local: `data/lite-v1/prepared/manifests/`; remote:
`apsp-lite-runtime-v1:/text/lite-v1-20261019/manifests/`. Hash audit:
`reports/lite/PROMPT_MANIFEST_AUDIT.json`. Test chỉ có text manifest, chưa sinh/chấm ảnh.
