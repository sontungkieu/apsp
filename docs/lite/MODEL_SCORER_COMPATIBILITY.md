# Model/scorer compatibility — Lite v1

Các kết quả dưới đây là runtime checks trên Modal B200 sm100, Torch2.9.1+cu128,
Python3.11.6, Diffusers0.39.0, Transformers4.57.6, PEFT0.17.1. Mỗi profile chạy một
prompt qualification riêng × tám seed; không thay study50 prompt hoặc test500.

| Profile | Frozen loading contract | Previews | ImageReward / PickScore | Fresh Ubuntu22.04 B200 restore | PNG SHA giống build/restore |
|---|---|---:|---|---|---:|
| sdxl_lightning8 | full8-step UNet, Euler trailing, guidance0, fp16 | 64 | PASS / PASS | PASS | 8/8 |
| hyper_flux8 | FLUX.1-dev + Hyper8-step LoRA fused0.125, guidance3.5, bf16 | 64 | PASS / PASS | PASS | 8/8 |
| sdxl_base50 | SDXL base50-step, guidance5, fp16 | 128 | PASS / PASS | PASS | 0/8 |
| Hyper-SD15 held-out | chưa freeze checkpoint/scheduler | chưa chạy | chưa chạy | chưa chạy | chưa chạy |

Mỗi profile có 8/8 final valid; từng run kiểm final latent re-decode cho max pixel
diff=0. Restore subset/hash drill đạt, nhưng backup trên cùng Volume chưa là hai
failure domains độc lập. Scorer failure được phân lớp riêng với generator failure.

SDXL50 functional PASS không chứng minh cross-container determinism. Trên tám ảnh,
max channel abs diff=100 (0–255), max mean channel abs diff=0.2115631103515625;
max IR abs delta=0.04556381702423096; max raw PickScore abs delta=
0.04733085632324219. Nguyên nhân chưa xác định. Không trộn hai môi trường cho một
bank mà bỏ qua drift qualification. Lightning/FLUX chỉ xác nhận tám seed đang đo.

ImageReward dùng exact aliases của ba helper Transformers đã đổi module; xem
`apsp/lite/compat.py`. Không patch checkpoint hay package. Profile/revision/source
fingerprint ở `env/lite/profiles.json`, receipts ở `reports/lite/`; receipt remote
không có Git metadata, được ràng buộc với source hash thay vì đoán commit.
