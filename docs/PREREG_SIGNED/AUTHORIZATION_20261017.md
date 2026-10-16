# Authorization record (2026-10-17)

Source: two messages from the project owner (git user `Mmmjnh`) in the Claude Code session
`c8081ba0-a4d8-4c0d-baa8-d73f41b08105`, received 2026-10-16T22:0xZ–22:2xZ UTC (instance clock; laptop date 2026-10-17).
Copied verbatim below by the Claude Code agent (claude-opus-5-5). This file is the approval source cited by the
PREREG markers in this directory. It is not a signature by anyone else, and the agent signs nothing in its own name.

## Message 1 — PREREG-A approval

> Tôi phê duyệt PREREG-A theo nội dung báo cáo P1 trên instance56272061 và các quyết định dưới đây:
>
> 1. Giữ generator, grammar/cap nghiên cứu, seed, scoring, bộ lọc và split như đã mô tả trong gói A và báo cáo P1. Không chuyển hay bỏ prompt theo kết quả sinh ảnh.
> 2. Chọn preview_dtype=fp16 và vae_decode_chunk=1.
> 3. A16: lưu x0 latent float16 của train_fit/train_dev theo schema hiện hành để phục vụ B10/ablation. Không mở hay lưu latent val/test ngoài protocol.
> 4. A17: chấp nhận Amendment1 rev.2. Replay là simulator xấp xỉ; kết luận xác nhận chạy online. O1-replay chỉ là cận trong simulator.
> 5. Chấp nhận M2 làm bảng accounting ban đầu, giữ nhãn provisional cho phần chưa hoàn tất. Profile controller và xác nhận chi phí trước đánh giá cuối; thiếu controller cost không chặn thu bank.
>
> Thực hiện ngay:
> - Ghi cấu hình fp16/chunk1, đo và đóng băng EPH trên máy.
> - Kiểm cost table và equivalence gắn đúng cấu hình này. Nếu khác, chỉ đo lại phần bị ảnh hưởng; không chạy lại toàn bộ P1/smoke.
> - Hoàn tất proposal A theo các quyết định này, tính SHA256 và tạo docs/PREREG_SIGNED/PREREG-A.json. Tôi ủy quyền ghi nhận phê duyệt từ chính tin nhắn này; lưu nguồn/timestamp/hash, không giả chữ ký của người khác.
> - Chạy detached/resume P2 → các công việc P3 đủ điều kiện ngay sau đó.
>
> Trong lúc P2 chạy, gửi nguyên nội dung gói B/C và gom mọi quyết định còn thiếu thành một lần duyệt. Tiếp tục chuẩn bị evaluator và các phần CPU độc lập; không tự ký B/C.
>
> Không Stop/Destroy instance, không thử Stop. Giữ máy chạy giữa các phase và khi chờ research.
>
> Cập nhật dự toán toàn suite, ghi riêng SDXL, external baselines và evaluator nếu chưa nằm trong 32.4 giờ. Đó là dự toán, không phải thời gian đã chạy hay trần chi phí.

## Message 2 — full-suite authorization, A/B/C as protocol-lock snapshots

> Tôi yêu cầu chạy toàn bộ suite Adaptive PSP trên instance56272061 bằng một pipeline tự tiếp tục đến cuối.
>
> Tôi phê duyệt các quyết định PREREG-A đã thống nhất: fp16, VAE chunk1, lưu latent train_fit/train_dev, và Amendment1 về replay xấp xỉ/đánh giá xác nhận online.
>
> Tôi ủy quyền hoàn thiện các chi tiết protocol còn thiếu theo HANDOFF và kế hoạch v1.3, với các điều kiện:
> - Chỉ dùng train/dev để fit, tune, tính ngưỡng và chọn checkpoint.
> - Các công thức đã thống nhất được áp dụng tự động khi đủ dữ liệu.
> - Khóa methods, operating points, margins, sample size, contrasts và quy tắc thống kê trước khi xem kết quả xác nhận tương ứng.
> - Test chỉ dùng đánh giá, không quay lại chọn model.
> - Ghi đầy đủ quyết định, timestamp, hash và nguồn ủy quyền từ tin nhắn này.
>
> Cập nhật quy trình A/B/C thành các snapshot khóa protocol theo ủy quyền trên, không yêu cầu tôi ký lại từng phase. Không giả chữ ký của người khác.
>
> Chạy tiếp: bank → baseline/dev → controller/RL → validation → GenEval → ablation → SDXL và đối chứng ngoài phù hợp → báo cáo cuối. Tự xử lý các lựa chọn triển khai thường lệ và lỗi có thể phục hồi; báo cáo theo mốc nhưng không dừng để chờ tôi trả lời.
>
> Giữ kết quả âm và inconclusive trung thực. Không ép RL thắng hay đổi tiêu chí sau khi xem kết quả.
>
> Không Stop/Destroy instance trước khi hoàn tất toàn bộ thí nghiệm. Không có trần tài chính; tiếp tục ghi chi phí.

## How the agent applies it

- PREREG-A: marker written now from Message 1 + 2 (proposal hash pinned).
- PREREG-B and PREREG-C: no separate user signature per phase (Message 2). The agent writes each marker as a
  **protocol-lock snapshot** only when (a) every B/C value is computed from train_fit/train_dev (or fixed in advance by
  an agreed formula), (b) the snapshot file is committed with its sha256 and timestamp, and (c) this happens before
  any val (B) or test/GenEval (C) outcome is read. Nothing is changed after the corresponding confirmatory results
  are seen; negative and inconclusive results are reported as such.
