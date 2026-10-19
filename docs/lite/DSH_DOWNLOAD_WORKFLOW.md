# Đường tải chuẩn từ DSH research_vai

Nguồn được đọc trực tiếp ngày20/10/2026:

- `D:/dev/dsh/research_vai/PROXY_DOWNLOAD_GUIDE.md`, đặc biệt §2 và §6.
- `D:/dev/dsh/research_vai/PROXY_DOWNLOAD_NOTE.md`, §4.
- `D:/dev/dsh/research_vai/dpca_b200/PROJECT.md`, §0b–§0b.2: lỗi retry/append,
  phần cuối không bằng CHUNK và resume tại byte đứt đã xảy ra thật.
- `D:/dev/dsh/research_vai/RUNTIME_PORTABLE_RUNBOOK.md`, §8–§9; publish harness TTS.
- `D:/dev/dsh/research_vai/AGENTS.md`, §1–§3 và §7: block/subshell, proxy,
  persistent shared staging, không đoán node/GPU hoặc can thiệp job khác.

Quyết định: **node công ty dùng curl+proxy+Range từ HF revision đã khóa**. Không
dùng SDK Modal, không cài dependency mới trên node chỉ để tải file. Downloader SDK
tự viết trước đó đã bỏ khỏi đường triển khai; smoke receipt cũ chỉ giữ như bằng
chứng lịch sử, không phải hướng dẫn node hiện hành.

`env/lite/download_part_hf.sh` áp dụng template DSH với N=12 mặc định. Mỗi part
archive do một instance tải qua12 range workers; hai instance có thể có tối đa24
kết nối tải file. Đây là concurrency được cấu hình, không phải kết quả đo băng
thông của node hoặc lời hứa tốc độ gấp đôi.

Giữ các guard từ note: proxy tường minh và không in giá trị; pinned HF commit;
Range0–0 phải trả206; CHUNK=ceil(WANT/N) và phần cuối tính riêng; resume từ số byte
đã có; curl ghi file tạm không dùng--retry/>>; chỉ nối byte khi HTTP206 và size
không vượt phần còn thiếu; ghép range theo danh sách0..11, không glob; SHA256
toàn part phải đúng trước khi publish file đích. Giữ range/log hỏng để chẩn đoán.

Template đã được kiểm bằng curl/bash sẵn có qua proxy fixture loopback:12 kết
nối thực đồng thời, resume từ13 byte cũ, phần cuối dư, reuse file đúng hash và
từ chối server bỏ qua Range. Đó là kiểm implementation/protocol trên dữ liệu nhỏ,
không phải company-node qualification hoặc benchmark throughput.

Hai transport part có22,992,424,293 và22,992,424,292 bytes; đều dưới30GB. Nén gzip
lossless từ tar61,109,442,560 bytes; kiểm roundtrip của cả hai part đạt SHA256
`d11d1ecf591ae59e6e63d80d767e74c3659ded51ec93792139aa34eb667b3b96`.
Chúng là hai nửa của cùng một gzip stream, không bung độc lập. Mỗi instance tải
đúng một part vào storage bền vững; cần cùng storage hoặc đưa cả hai part về cùng
join node. Dự trù khoảng250GB nếu giữ đồng thời ranges, parts, tar và payload bung.

Người dùng đã cho phép tạo HF repo hợp lý và public toàn bộ. Đích:
`codemaivanngu/apsp-lite-b200-cu128`, loại dataset. Upload chạy trực tiếp từ Modal
CPU, token project-local qua ephemeral Secret như TTS, kiểm whoami trước write;
repo public trước upload theo quyền đã cấp (tránh quota private storage); đọc ngược
size/LFS SHA và kiểm anonymous Range/metadata. LICENSES
và NOTICE đi kèm; không đổi quyền upstream bằng cách gắn license chung.

Chỉ dùng revision/hashes trong publication receipt đã PASS. Block node đầy đủ sẽ
được sinh từ receipt đó; chưa có receipt thì không bịa REV hoặc báo publish xong.
Chưa có tên node/SSH được giao cho lượt này: các block là hướng dẫn để operator
chạy, chưa phải lệnh đã chạy trên máy công ty. Không dùng GPU cho bước tải/ghép.
