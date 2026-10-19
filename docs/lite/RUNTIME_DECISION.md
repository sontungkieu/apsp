# Runtime decision và audit bundle

Quyết định: **runtime Adaptive PSP mới từ lock bundle, không nâng chồng TTS**.

| Thành phần | TTS portable | APSP Lite |
|---|---|---|
| Source | thanhlamauto/TTS-reserach, Static BFS | DMRL bundle396b313 + Lite namespace |
| Python | 3.10 | 3.11.6 |
| Torch | 2.14.1+cu130 | 2.9.1+cu128 |
| Transformers | 4.38.2 | 4.57.6 |
| Diffusers | git af28ae2d… | 0.39.0 |
| Generator protocol | SD1.5, Static BFS100/eta1 | Lightning8, Hyper-FLUX8, SDXL50 |
| Verifiers | ImageReward | ImageReward + PickScore |

Đây là thay đổi ABI/dependency thật, đủ lý do dựng runtime riêng. cu128 có thể chạy
trên driver hỗ trợ CUDA13; cần kernel/driver gate thực, không đổi Torch chỉ vì tên
CUDA trên node. Không downgrade toàn bộ stack mới để giữ TTS. Có thể tái dùng model
snapshot hoặc archive có cùng hash; không coi cache cùng tên là cùng nội dung.

ZIP đầu vào SHA256 `21f148b678db3dda8df06c04123cd681e6765d59c466fbfe2b8cfb004273b123`.
Git bundle HEAD `396b3132eeb35bf7c98e0e4b1a851bf88268cc1b`, branch `b200/inference-env`.
Branch triển khai `feat/apsp-lite-data-v1`. Origin hiện là **Git bundle local**;
chưa thiết lập GitHub remote cho nhánh mới, chưa push.

Bundle tự ghi rõ chỉ lock check pass, chưa clean install/B200. Audit không dựa
vào phần README cũ từng ghi trạng thái quá sớm. Registry cũ chưa có checkpoint
Lightning/Hyper, không có PEFT, default seed v1, checkpoint grid mặc định không đủ
J của Lite. Collector cũ skip theo part tồn tại và thiếu contract-bound durable ack.
Lite bổ sung riêng, giữ nguyên các đường code Full.

Model-card loading được kiểm từ nguồn chính:
- https://huggingface.co/ByteDance/SDXL-Lightning
- https://huggingface.co/ByteDance/Hyper-SD

Lightning dùng full 8-step UNet, Euler trailing, guidance0. Hyper-FLUX dùng 8-step
LoRA, fuse0.125, guidance3.5. Revisions mới pin tại `env/lite/profiles.json`.
Hyper-SD15 held-out giữ disabled cho tới khi freeze checkpoint/scheduler contract.

Lỗi đã gặp khi build: Modal build inject index mặc định khác PyPI của lock;
explicit `--default-index https://pypi.org/simple` sửa được mà không đổi lock.
Remote resolve bằng uv0.11.17 xác nhận lock từ uv0.10.2 không đổi; chênh phiên bản
uv không phải nguyên nhân. ImageReward import helper relocation được xử lý bằng
exact function aliases trong Lite source; third-party package không bị sửa.

Hai lần CPU BERT harness đầu thiếu cấu hình cross-attention của BLIP; đã sửa
harness thành input/flag đúng rồi mới gọi pass. Không coi lỗi harness đó là lỗi
ImageReward checkpoint. Các receipt thật và trạng thái cuối nằm ở báo cáo triển khai.
