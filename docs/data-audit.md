# Kiểm kê dữ liệu thô trên MinIO

**Ngày kiểm: 05/08/2026.** Bucket `hit-mira-media`, prefix `raw/google-drive/data/`.
**Phương pháp: full scan** — `list_objects(recursive=True)` cho toàn bộ 2.204
object, đọc header từng file ảnh và atom `moov` từng video bằng ranged GET. Không
lấy mẫu, không suy từ metadata trừ chỗ ghi rõ là "theo khai báo".

Sinh lại toàn bộ số dưới đây bằng một lệnh (chỉ đọc, không có `--apply` vì
không sửa gì):

```bash
python scripts/audit_minio_dataset.py --deep    # ~14s: đọc header ảnh, atom moov, SHA-256 cả 1,99 GB
python scripts/audit_minio_dataset.py --census-out API/tests/fixtures/minio_dataset_census.txt
```

> Nhắc: đây là trạng thái **dữ liệu thô**. Ngày kiểm, Qdrant trên máy này còn
> trống và PostgreSQL chưa có schema (`relation "media" does not exist`), nên
> các số về collection ở `handoff.md` §2 nói về một môi trường khác.

---

## 1. Toàn cảnh

| Thứ | Số thật |
|---|---|
| Object | 2.204 |
| Dung lượng | 1.987.398.725 byte (1,99 GB) |
| Ảnh | 1.628 (1.554 `.jpg` + 74 `.png`) |
| Video | 59 `.mp4`, chiếm 68% dung lượng |
| Bài có ít nhất một media | 496 |
| Bài có ảnh | **452** |
| Bài có video | **45** |
| File cấp bộ dữ liệu | `posts.jsonl`, `manifest.json`, `crawl.log` |

Bố cục key — **quy ước duy nhất** mà `scripts/minio_registration.py` đọc:

```text
raw/google-drive/data/<facebook_post_id>/post.json
raw/google-drive/data/<facebook_post_id>/media/<filename>
```

Nguồn: page `CLB Tin học - Đại học Công nghiệp Hà Nội` (`213347192060163`),
crawl trải **28/06/2021 → 21/06/2026**, đều ~8–9 bài/tháng, không có khoảng trống.
Loại bài: photo 291, album 161, video 46, status 16, link 1.

Text: 505/514 bài có `message`, trung vị 1.259 ký tự, dài nhất 6.336, tổng ~667k
ký tự tiếng Việt dày emoji.

Video: tổng **230 phút**, trung vị 191s, dài nhất 806s. **Một video không có
track audio** — `.../213347192060163_1320967086518440/media/video_07.mp4`; đúng ca
"video câm → transcript `[]`" mà pipeline phải chịu được.

## 2. Đếm theo dòng hay đếm sau dedupe — phân biệt cho rõ

`posts.jsonl` có **515 dòng nhưng chỉ 514 post**: ID
`213347192060163_1036672484947903` xuất hiện hai lần. Mọi con số dưới đây vì
thế có hai phiên bản, và trộn hai phiên bản là nguồn sai lệch:

| Đại lượng | Theo dòng JSONL | Sau dedupe theo ID |
|---|---|---|
| Bản ghi post | 515 | **514** |
| Media có `*_error` | 211 (khớp `manifest.json`) | **210** |
| Bài có đúng 1 media | 334 | **333** |

**Hai bản ghi trùng ID không giống nhau**: `created_time` lần lượt là
`2026-05-05T14:00:01+0000` và `2024-05-30T13:00:47+0000`, trong khi `post.json`
của thư mục dùng mốc **2024**. Vì vậy không được `set(id)` rồi giữ bừa một bản
— phải giữ bản khớp `post.json` và ghi lại conflict.

Đường ingest hiện tại không dính bẫy này: `scripts/minio_registration.py` đọc
`<prefix>/<post_id>/post.json` theo thư mục, **không đọc `posts.jsonl`**. Quy
tắc trên áp cho script phân tích/eval.

## 3. Bốn cái bẫy khi đọc bộ dữ liệu này

**a. Không dùng `str.splitlines()` cho `posts.jsonl`.** Có đúng một ký tự
`U+2028` (LINE SEPARATOR) trong `message` của bài
`213347192060163_4670327683028736`. `splitlines()` cắt cả ở đó → 516 mảnh, 2
mảnh JSON hỏng. `split("\n")` parse đủ 515 bản ghi.

**b. `width`/`height` trong `posts.jsonl` không đáng tin.** Full scan 1.628 ảnh:
**1.259 ảnh lệch (77,33%)**, trong đó 1.258 file thật **lớn hơn** khai báo, chỉ
1 file nhỏ hơn. Mẫu lệch phổ biến nhất là `1080×720 → 2048×1365` (321 lần). Cần
kích thước thật thì đọc header file.

**c. 210 media có `*_error` vẫn tải được file, nhưng độ phân giải có lệch.**
Lỗi chủ yếu là `FB code=12 singular statuses API is deprecated`. Đối chiếu lưu
trữ: 1.687 media khai `local`, **thiếu trên filesystem 0, thiếu trên MinIO 0**
(thêm 1 video external không khai `local`). Nhưng theo **file thật**:

| Nhóm | n | p50 cạnh dài | ≤512px |
|---|---|---|---|
| Không lỗi | 1.418 | **2.048** | 24 (1,69%) |
| Có `*_error` | 210 | **1.080** | 6 (2,86%) |

Nói hai nhóm "tương đương" là kết luận rút từ metadata — tức là dẫm đúng bẫy
(b). Hệ quả thật: nhúng CLIP hạ về 512px nên vector gần như không đổi, còn
Qwen Vision nhận nguyên file nên **OCR chữ nhỏ là nơi chịu ảnh hưởng rõ nhất**.
Vì file đã qua kiểm local + MinIO + checksum, nên coi `resolve_error` là
**warning**, không phải lỗi mất dữ liệu.

**d. 33 bản dư trong 65 file thuộc 32 nhóm trùng nội dung** (31 nhóm đôi, 1
nhóm ba), toàn bộ là ảnh.

Con số này phải xác nhận bằng **SHA-256 trên nội dung thật**, không được tin
ETag: **41 object mang ETag dạng `<md5>-<số phần>`** vì tải lên nhiều phần —
đó không phải MD5 nội dung, và toàn bộ 41 object đó là video, tức đúng nhóm
file lớn mà phép so ETag yếu nhất. Băm lại cả 1.687 file (1,99 GB, ~14 giây
trên MinIO cục bộ) cho **kết quả trùng khít với ETag**: 32 nhóm / 65 file / 33
bản dư, không nhóm nào bị ETag bỏ sót (`sha256_khop_etag: true`). Dedupe
**nhúng** theo SHA-256 tiết kiệm được `33 × 4.000 = 132.000` token Jina (1,32
phút quota), còn với embedding server tự host thì đây là tiết kiệm GPU/thời
gian chứ không phải tiền token. **Vẫn phải giữ đủ quan hệ ảnh–post**, nếu không
sẽ mất provenance cho trích dẫn.

## 4. Nghi vấn về độ đầy đủ của corpus — chưa kết luận

Phân bố media/bài (sau dedupe): 17 bài không media, 333 bài 1 media, **0 bài 2
media**, 16 bài 3 media, … và **53 bài đúng 13 media**.

- **Không bài nào có đúng 2 media.** Album 2 ảnh là chuyện thường trên
  Facebook, nên khoảng trống này nhiều khả năng là do crawler.
- **53 bài dừng đúng ở 13**, và 13 cũng là số media lớn nhất của cả tập — dấu
  hiệu rất mạnh của giới hạn phân trang.

Chưa có source của crawler nên **chưa kết luận nguyên nhân**. Đánh dấu 53 album
này là `possibly_truncated` và recrawl thử một bài có pagination đầy đủ trước
khi coi corpus là hoàn chỉnh.

## 5. Bug prefix — đã vá 05/08/2026

`scripts/minio_registration.py` bản trước hardcode `events/` ở hai chỗ và có
một fallback im lặng `Path(key).parent.name`. Với layout `raw/google-drive/data/`:

1. fallback suy ra `event_id = "media"` cho **mọi** object;
2. `load_post_metadata` tìm `events/media/post.json`, không thấy, trả `{}`;
3. toàn bộ **1.628 ảnh và 59 video** dồn vào một post giả
   `facebook_post_id="media"`, không content, không permalink;
4. script in ra "thành công".

Đã sửa thành prefix-aware, chạy được cả hai layout:

```bash
python scripts/register_minio_images.py --prefix raw/google-drive/data   # bộ hiện tại
python scripts/register_minio_images.py                                  # bộ events/ cũ
```

Bốn lớp bảo vệ đi kèm:

- **Không còn fallback.** Key sai quy ước là `ObjectKeyError`, và key được soi
  hết **trước** khi mở kết nối DB — truyền nhầm `--prefix` thì dừng ngay với
  danh sách key sai, không ghi dòng nào.
- **Listing rỗng cũng là lỗi** (`EmptyPrefixError`). Prefix *lệch hẳn* không
  sinh ra key sai nào để mà soi: `list_objects` lọc từ trước nên nó trả về danh
  sách rỗng, và "tìm thấy 0 ảnh" trông y hệt một lượt chạy sạch. Ai thật sự cần
  chạy trên prefix rỗng thì bật `--allow-empty`.
- **Chạy thử cũng kiểm `post.json`**, không đợi tới `--apply`, và kiểm cả với
  media **đã đăng ký** — nếu không, lượt chạy lại sẽ im lặng đúng ở những bài
  đã vào DB, tức là mất khả năng phát hiện `post.json` biến mất về sau.
- **Thiếu metadata thì bỏ qua media và ghi lỗi**, không tạo post giữ chỗ, trừ
  khi tự bật `--allow-missing-metadata`.

Kiểm trên MinIO thật sau khi vá: 1.628 ảnh → **452 bài**, 59 video → **45 bài**,
không bài nào tên `"media"`, và **cả 496 bài đều đọc được `post.json`** (0 bài
thiếu metadata). Chạy nhầm prefix, đo bằng chính CLI chứ không phải bằng hàm
parse:

| Lệnh | Kết quả |
|---|---|
| `--prefix events` | exit **2** — `Không có media nào dưới prefix 'events/'` |
| `--prefix raw` | exit **2** — `1628/1628` (ảnh), `59/59` (video) key sai quy ước |
| `--prefix raw/google-drive/data` | qua được cửa prefix |

> Bản đầu của mục này ghi "`events` dừng với 1687/1687 key sai" — **sai**. Số
> đó lấy từ việc gọi thẳng hàm parse trên danh sách key của prefix *đúng*, tức
> là không đi qua `list_objects`. Đúng đường đi thật thì `events` cho listing
> rỗng, và trước khi có `EmptyPrefixError` thì nó **không** dừng.

Khoá lại bằng `API/tests/test_minio_registration.py` (30 test): cả hai layout,
prefix có/không `/` cuối, key ngoài prefix, key thiếu segment `media`, prefix
lệch hẳn cho listing rỗng, `post.json` thiếu/hỏng/không phải object, chạy lại
vẫn kiểm metadata của media đã đăng ký, chạy thử không ghi gì, và mốc hồi quy
452/45 dựng từ `API/tests/fixtures/minio_dataset_census.txt`.
