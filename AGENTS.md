# APSP source distribution

User đã chỉ định repo GitHub `https://github.com/sontungkieu/apsp.git` làm nguồn
quản lý chính và duy trì cùng source trên HF để server tải, theo mô hình SimCT.
Canonical branch là `main`. Quy tắc này chỉ áp dụng repo APSP; yêu cầu mới cụ thể
hơn của user và hướng dẫn global có ưu tiên.

- Review/status vẫn chỉ đọc. Với milestone source đã được user giao sửa, kiểm
  tra phù hợp, commit đã ký đúng phạm vi rồi đồng bộ GitHub và HF, trừ khi user
  yêu cầu local-only/uncommitted hoặc thay đổi quyền. Không force-push, đổi
  signing/identity hay cuốn edits không thuộc task.
- Dùng `python env/lite/publish_source.py` để check, thêm `--publish` để push
  `main`, xác minh GitHub ref, upload full Git bundle và manifest bằng một HF
  commit, đọc ngược anonymous và đối chiếu GitHub lần nữa. Chỉ
  `DUAL_PUBLISH_PASS` là cả hai kênh đã kiểm chứng. Nếu một bước lỗi, báo riêng
  commit/ref đã lên GitHub và HF pending; không tuyên bố sync hoàn tất.
- HF dataset được phép: `codemaivanngu/apsp-lite-b200-cu128`; source channel
  `source/`. Không đưa model/data/runtime vào GitHub hoặc re-upload các part
  lớn chỉ vì source đổi. Không sửa snapshot provenance của companion cũ.
- Server dùng `env/lite/update-apsp.sh`, kiểm pinned HF revision, bundle hash,
  branch/ref, clean checkout và ancestry rồi `merge --ff-only`. Không reset,
  stash hoặc update source đang có job dùng. Server mới cần bootstrap helper
  từ revision/hash đã xác minh; xem `docs/lite/SOURCE_DISTRIBUTION.md`.
- Quyền phân phối source không cấp thêm quyền launch server/GPU, tăng ngân
  sách, rebuild runtime hoặc tải dependency/model lớn về local. Không in secrets.
- Kết thúc báo Git source commit, GitHub ref, HF revision, test và phần chưa
  kiểm chứng. Receipt publish ở `remote_artifacts/source-*/receipt.json`, ngoài
  commit nó mô tả để tránh vòng lặp revision.
