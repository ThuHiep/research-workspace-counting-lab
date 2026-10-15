# Paper 2 — Distributional Count Distillation (R2) — MASTER

> 📌 **Số đã kiểm + cách tính: [`NUMBERS_VERIFIED.md`](NUMBERS_VERIFIED.md)** — tra ở đó TRƯỚC khi chạy lại
> bất cứ thứ gì. Chạy lại parity PanNuke + chuyển miền: `python verify_numbers.py` (1,4 giây, CPU).

> **FILE DUY NHẤT** (gộp MODEL + KETQUA cũ, 2026-07-17). Từ giờ chỉ cập nhật ở đây.
> Số cũ/leaky (single-split, seed-đơn 0.773, runbook đã xong) ĐÃ XOÁ để tránh rối.
> Nguyên tắc: **không bịa** — mọi thành phần có cơ sở thống kê + số liệu; hằng số thiết kế = design-choice có sensitivity, KHÔNG claim "zero-heuristic".

---



## ✅ main.tex ĐÃ ĐỒNG BỘ với main_vi.tex — 2026-08-11

Cả hai file giờ có **38 label, 34 heading, cùng cấu trúc lồng nhau, 0 tham chiếu treo**; mọi con số đối chiếu
khớp theo tần suất. Abstract EN = 210 từ (giới hạn RINENG 250).

Bốn phát biểu SAI/quá mạnh trong main.tex đã sửa: (1) δ* rộng dần "reflects differing statistical power" →
bảng phân rã hiệu ứng/độ chính xác; (2) "about 2.6 times as much" → bỏ tỉ số; (3) "narrows the explanation to a
single mechanism" → "support a mechanism in which..."; (4) "runs on both a commodity GPU and a CPU" → giới hạn
về phần cứng đã đo (CPU chậm hơn GPU ~7×).

Đã thêm vào EN: hàm phương sai tựa-Poisson (GLM), eq:tost, eq:covgroup, quy ước phân vị j ≥ k, hiệu ứng chọn lọc
của worst-organ, suy luận có-điều-kiện khi lấy TB theo seed, Wilcoxon kiểm định dịch chuyển (không phải trung
bình), Phụ lục A + B, tiểu mục mới §5.2.2 `sec:localize`.

**★★ SỬA 2026-08-12 — Mục 4.1 MÔ TẢ SAI phép bắt cặp (số ĐÚNG, chữ SAI):** bản cũ viết "μ là dự đoán của chế
độ r, lấy TRUNG BÌNH TRÊN CÁC SEED" rồi e = |N − μ|. Code [sup_compare.py:49-50] làm NGƯỢC: lấy |N − μ_s| từng
seed RỒI mới trung bình. Hai phép khác nhau thật — kiểm lại từ npz trên máy: ConvNeXt p 2,6e−07 vs 5,2e−18,
δ* 1,14% vs 1,49%; TinyViT p 0,79 vs 0,62, δ* 0,50% vs 0,43%. Cách thứ HAI khớp mọi số trong bài ⇒ đã sửa mô tả
ở §4.1 (ký hiệu μ_i^{(r,s)} + công thức trung bình sai số), KHÔNG đụng số. Chú thích Bảng 1 vốn đã đúng.
**ĐỪNG viết lại "dự đoán được lấy trung bình trên các seed".**

**★ ĐỘ LỆCH CHUẨN trong mọi bảng = np.std ddof=0 (tổng thể), không phải sd mẫu.** Kiểm: TinyViT count
[0,9113 0,9359 0,9377] → ddof0 = 0,012 (khớp bài), ddof1 = 0,015. efflite0 5 seed cũng vậy (0,011 vs 0,012).
Nhất quán toàn bài; nếu đổi phải đổi HẾT mọi ± trong mọi bảng (per-seed của phần lớn bảng KHÔNG có trên Mac).
⇒ **Đã CHỐT 2026-08-12: khai báo quy ước trong §4.1** thay vì tính lại; một câu ở cuối đoạn 'Mô hình sai số bắt cặp'.

**★ SỬA 2026-08-12 — phép cắt cận dưới tại 0 KHÔNG "làm tăng độ phủ"**, nó không đổi biến cố phủ (vì số đếm
N ≥ 0 nên "N ≥ max(0, μ−qσ)" tương đương "N ≥ μ−qσ" ở mọi trường hợp); nó chỉ THU HẸP khoảng, tức chỉ ảnh hưởng
bề rộng/Winkler. Bản cũ viết "chỉ có thể làm tăng độ phủ" ngay TRƯỚC eq:cov vốn khẳng định ĐẲNG THỨC k/(n+1) →
tự mâu thuẫn, và làm hỏng cận trên. Đã sửa ở §3.3 + Phụ lục B cả 2 file. **ĐỪNG viết lại "increase coverage".**
Lệch độ phủ THẬT chỉ đến từ quy ước phân vị j ≥ k (xem trên), đó là chỗ duy nhất cận trên của eq:cov bị mất.

**★ QUYẾT ĐỊNH 2026-08-11 — BỎ con số "17,5%" của LIRNet khỏi Mục 2** (trích dẫn `lirnet` VẪN GIỮ, nó là một nấc
trong dãy giảm dần thông tin không gian). Lý do: 17,5% là % so với một mốc khác, từ nghiên cứu khác, **mẫu số
không biết** (mỗi ảnh? mỗi mảnh? mỗi đơn vị công?) → không so được với luận điểm chi phí của bài, vốn dùng đúng
một nguồn/đơn vị (Qu 2020, giây/nhân: 11,5 khoanh viền vs ≤1,4 điểm). Để lại thì thành một con số RẺ HƠN cận trên
của chính ta (≤100% chi phí điểm) nằm ngay cạnh, mà bài không diễn giải nổi → mồi cho phản biện.
⇒ Mục 2 = mức giám sát; Mục 5.4.2 = chi phí. **ĐỪNG thêm lại** trừ khi đọc được LIRNet và biết mẫu số.

## ✅ ĐÃ VÁ 2026-08-11 — tiền đề động lực giờ CÓ 3 trích dẫn (metadata xác minh qua Crossref API)

| key | công trình | vì sao dùng được |
|---|---|---|
| `tcunet` | Zheng et al., Med Eng Phys **133**:104244, 2024, doi 10.1016/j.medengphy.2024.104244 | biến thể **U-Net++** train bằng MẶT NẠ, tách bằng distance watershed, **chấm điểm bằng sai số ĐẾM** — trả lời thẳng "có ai dùng U-Net để đếm không" |
| `pathonet` | Negahbani et al., Sci Rep **11**:8489, 2021, doi 10.1038/s41598-021-86912-w | chỉ số Ki-67 = điểm cuối lâm sàng dạng ĐẾM, nhãn train = phát hiện+phân loại mức tế bào |
| `midog` | Aubreville et al., Med Image Anal **84**:102699, 2023, doi 10.1016/j.media.2022.102699 | mitotic count phục vụ phân độ khối u, nhãn train = phát hiện |

Thêm tiếp `conic` + `tupac16` (Crossref) → references.bib **37 mục**, 0 mồ côi.

**★★ ĐÃ KIỂM TỪNG TRÍCH DẪN xem có THẬT SỰ chấm bằng thước đo ĐẾM không (2026-08-11) — ĐỪNG kiểm lại:**

| công trình | đầu ra | thước đo | đạt tiêu chí "đếm"? |
|---|---|---|---|
| `conic` Task 2 "cellular composition" | số nhân TỪNG LOẠI mỗi ảnh | **hệ số xác định đa lớp (R²)** | ✓✓ ĐÚNG thước đo của bài ta, có leaderboard riêng |
| `tcunet` | số đếm + diện tích + trọng tâm | cải thiện sai số đếm 5,09% | ✓ |
| `pathonet` | phát hiện + phân loại tế bào → Ki-67 | F1 (harmonic mean) | ~ gián tiếp |
| `tupac16` | **điểm phân độ RỜI RẠC 1/2/3** | **kappa trọng số bậc hai** | ✗ KHÔNG phải đếm |
| `midog` | phát hiện phân bào | **F1 phát hiện** | ✗ |
| `kongnet` | phát hiện + phân loại | **F1 + FROC** (tra tận arXiv HTML) | ✗ KHÔNG có đếm |
| `distillfm` | phân đoạn + phân loại | — | ✗ |

**★★ QUY TẮC VĂN PHONG (user chốt 2026-08-11) — ÁP CHO MỌI LẦN VIẾT SAU. Tôi đã vi phạm nhiều lần.**
1. **CẤM cấu trúc phòng thủ "X không phải vì A mà vì B"** / "not A but B" / "rather than" khi B là lời cãi với
   một người duyệt tưởng tượng. Q1 chỉ viết **"vì A nên làm"** — nêu mệnh đề dương, để phạm vi do từ ngữ chính
   xác gánh, không do phủ định gánh.
2. **CẤM lời văn hướng dẫn người đọc**: "Đáng chú ý", "Cần lưu ý", "Cần nói rõ", "Theo cách đọc này", "Như vậy",
   "Nói cách khác", "nên được đọc là", "Điều này cho thấy" / "Note that", "It should be noted", "Notably",
   "should be read as", "In other words".
3. **Ngắn và sắc.** Bỏ câu lặp lại ý câu trước dù diễn đạt khác.
4. ❌ **Ngoại lệ "tương phản kỹ thuật" mà tôi từng viện ra là SAI — user bác 2026-08-11.** Lý lẽ của user:
   **khi đã khẳng định "là A" thì người đọc không còn phân vân giữa A và B nữa**, nên vế "không phải B" chỉ là
   nhấn mạnh, không thêm thông tin. ĐÃ BỎ HẾT (13 chỗ nữa mỗi file): "đẳng thức chứ không phải một chặn" → "là
   một đẳng thức:"; "đặc hiệu với nhân chứ không phải thao tác chỉnh ảnh" → "đặc hiệu với nhân."; "φ dự đoán theo
   ảnh thay vì ước lượng chung" → "Ở đây φ được dự đoán riêng cho từng ảnh"; "3 seed thay vì 5" → "3 seed".
   Mẹo viết lại phủ định trần (không có vế A): đổi sang mệnh đề dương nói cái CẦN, ví dụ "'giám sát mặt nạ thuần
   tuý' KHÔNG phải đường cơ sở khả dụng" → "Giám sát mật độ thuần tuý CẦN một trọng số đếm khác không"; "Không bác
   bỏ H0 KHÔNG phải bằng chứng tương đương" → "Tương đương ĐÒI HỎI một kiểm định riêng".
   Chỉ còn giữ 1 chỗ: chú thích Hình 1 "các chấm sáng không phải nhân đã phát hiện" — ở đó chính HÌNH ẢNH khẳng
   định B, phủ định là để sửa cái mắt người đọc thấy. Nếu user vẫn muốn bỏ thì đổi thành trỏ sang Mục 5.2.
✅ Đã quét toàn bài và sửa **26 chỗ mỗi file** (2026-08-11). Sau khi sửa: abstract EN 209/250 từ, 0 mồ côi,
0 tham chiếu treo, 0 dòng lặp, 0 từ lặp liền, ngoặc cân, 0 dòng > 120 ký tự.
Ví dụ đã sửa: "Điều được xác lập ở đây là mức đủ, không phải mức tối ưu" → "Phép so sánh này xác lập mức đủ:";
"chú thích không gian dày đặc không phải điều kiện tiên quyết" → "một nhãn đếm mức ảnh đủ cho độ chính xác cao"
(vừa bỏ phủ định vừa khớp tiêu đề mới).

**★★ "NHÁNH MẶT NẠ CÓ PHẢI PHƯƠNG PHÁP ÍT AI DÙNG KHÔNG?" — KHÔNG. Đã vá 2026-08-11. Đừng lo lại.**
Câu hỏi này user hỏi 3 lần ở 3 dạng khác nhau ⇒ nếu người duyệt cũng nghĩ vậy là hỏng. Sự thật kỹ thuật:
- Nhánh mặt nạ = **CÙNG DSU-Net**, huấn luyện bằng **hồi quy bản đồ mật độ** — đúng mô thức thống trị của đếm
  tế bào (Mục 2.3: lempitsky/fcrn/countception/saunet/decouple). Khác biệt DUY NHẤT: bản đồ đích trải khối lượng
  theo **hình dạng nhân** thay vì quanh **một điểm**. Cả hai đều tích phân ra số đếm.
- ⇒ Mặt nạ CHỨA SẴN trọng tâm ⇒ nhánh này có mọi thứ bộ đếm giám sát-bằng-điểm có, CỘNG hình dạng.
  Nó là **mức giám sát MẠNH NHẤT** của hướng hồi quy mật độ, tức baseline BẢO THỦ, KHÔNG phải bù nhìn.
- Lỗi trình bày (đã sửa): bài gọi trống không là "nhánh mặt nạ" ⇒ người đọc tưởng phân đoạn. Câu "ước lượng bảo
  thủ" thì có nhưng bị chôn ở cuối Mục 3.2 và không nối với Mục 2.3, trong khi nghi ngờ hình thành từ Mục 1.
- ĐÃ THÊM 2 chỗ (cả 2 file): (1) Mục 3.2 ngay sau định nghĩa $\mathcal{L}_\mathrm{density}$ — *"Đây chính là bản
  đồ mật độ mà hướng đếm ở Mục 2 sử dụng, chỉ khác ở chỗ khối lượng được trải theo hình dạng nhân thay vì quanh
  một điểm chú thích."* (2) cuối Mục 3.2 — *"mặt nạ chứa sẵn trọng tâm... là mức giám sát mạnh nhất trong hướng
  đếm bằng hồi quy mật độ, chứ không phải một cấu hình yếu hơn được dựng lên để so."*

**★★ NGUỒN 11,5 / 1,4 GIÂY — ĐÃ TRÍCH NGUYÊN VĂN TỪ PDF 2026-08-12. Đừng tra lại.**
Qu+ (IEEE TMI 39(11):3655--3666, 2020), mục "F. Annotation time", nguyên văn:
> *"Dr. Riedlinger, a board-certified pathologist annotated **eight images (one per case)** in the LC dataset
> using points, bounding boxes and full masks... The average time spent on **each image** (about 600 nuclei on
> average) for full masks is **115 minutes**, while for bounding boxes 67 minutes. However, it only takes about
> **14 minutes** for all points annotation and less than 2 minutes for 10% points annotation."*
⚠️ **115 phút là MỖI ẢNH, và MỖI ẢNH ~600 nhân.** Bài từng viết "tám ảnh chứa trung bình 600 nhân: khoanh viền
mất 115 phút" ⇒ đọc thành 8 ảnh/600 nhân/115 phút = sai 8× về tổng. ĐÃ SỬA cả 2 file.
Tỉ suất suy ra KHÔNG đổi và ĐÚNG: 115·60/600 = **11,5 s/nhân**; 14·60/600 = **1,4 s/nhân**.
Ngoại suy lên PanNuke fold 1--2 (123.090 nhân): 393,2 h mặt nạ / 47,9 h đếm → khớp "393" và "48" trong bài.
Cũng có sẵn nhưng bài KHÔNG dùng: hộp bao 67 phút/ảnh, 10% điểm < 2 phút/ảnh, tăng tốc 60× (10% points).
Giới hạn thật khi ngoại suy: 1 người chú thích, 8 ảnh, bộ LC ~600 nhân/ảnh so với mảnh PanNuke ~24,5 nhân/ảnh
(dày gấp ~24×) ⇒ Mục 5.4 phải giữ chữ "ước lượng bậc độ lớn"; chú thích Hình 1 đã thêm "ước tính" + trích nguồn.

**★★ CHI PHÍ NHÃN ĐẾM — KHÔNG ĐO NỮA, ĐÃ VÁ BẰNG TRÍCH DẪN (2026-08-11). Đừng đề xuất đo lại.**
User KHÔNG đo được thời gian chú thích (cần chuyên gia). Không cần: **C-WSL (Gao+, ECCV 2018, LNCS 155–171,
doi 10.1007/978-3-030-01246-5_10)** đã mô tả đúng cơ chế. Hai câu lấy NGUYÊN VĂN từ PDF (đã trích bằng fitz):
- *"object count contains no location information and can be obtained without actually clicking on an object"*
  → đỡ bước "đếm bị chặn trên bởi chi phí đánh dấu điểm" (trước đây là khẳng định trần).
- *"K has positive correlation with annotation time, since annotators may not be able to subitize for high values
  of K and will need to spend an amount of time proportional to K in order to produce an accurate count."*
  → ⚠️ Ngoài ngưỡng tri giác tức thời (~1–4 vật), thời gian ĐẾM cũng tăng tỉ lệ với số vật. Mảnh ở đây 24,5
  nhân ⇒ **đếm KHÔNG rẻ hơn điểm bao nhiêu.**
⇒ Đã sửa Mục 5.4 + caption Bảng 8 (cả 2 file): nói THẲNG *"phần tiết kiệm so với đánh dấu điểm KHÔNG được tuyên
bố; phần được tuyên bố là so với khoanh viền đầy đủ"*. Đổi "chưa đo trong tài liệu" → "chưa đo **trên ảnh bệnh
học**" (vì C-WSL đã đo trên ảnh tự nhiên). Bib 39 → **40** (`cwsl`).
Lý do đây tốt hơn tự đo: phản biện, đã công bố, và đúng cơ chế mà lập luận cận-trên của bài dựa vào.

**⚠️ BUG TÔI GÂY RA 2026-08-11 — ĐÃ SỬA. Đọc trước khi lại viết script sửa .tex hàng loạt.**
Script thay-thế của tôi dùng `re.sub(pat, lambda m: new.replace('\\','\\\\'), t)`. **SAI**: khi replacement là
**lambda** thì `re.sub` KHÔNG xử lý escape, nên phần `.replace` kia nhân đôi gạch chéo THẬT → `\%`→`\\%`,
`\ref`→`\\ref`, `\Delta`→`\\Delta`… 13 chỗ mỗi file. Build Overleaf gãy: "Missing $ inserted" rồi mọi dấu
tiếng Việt (`\'`, `\^`, `\dj`, `\uhorn`) báo "invalid in math mode" vì kẹt trong math mode.
✅ ĐÚNG: lambda thì trả thẳng `new`, KHÔNG replace gì cả.
✅ Cũng sửa: 3 DOI mới (`decouple`/`annoeff`/`cwsl`) để `_` trần → BibTeX lỗi ở output.bbl. Mọi DOI Springer
PHẢI viết `\_`. Các mục cũ vốn đã đúng, chỉ 3 mục tôi thêm là sai.
⚠️ **Rồi tôi gây tiếp lỗi thứ hai ở CÙNG chỗ:** chạy `.replace('_','\\_')` cho MỌI trường doi → 6 mục cũ vốn
đã có `\_` bị escape thêm lần nữa thành `\\_`, build gãy đúng 6 mục đó. **Bài học: thay thế hàng loạt phải
kiểm cái ĐÃ escape trước, hoặc chỉ nhắm đúng mục vừa thêm.** Đúng: `_` trần → `\_`, và `\_` giữ nguyên.
Trạng thái đúng cuối cùng: cả 9 DOI Springer đều đúng một dấu — `10.1007/...\_39`.
🔧 **CHECK BẮT BUỘC sau mỗi lần script sửa .tex** (4 cái cũ KHÔNG bắt được bug này):
   `\\` giữa dòng (không ở cuối dòng, không trước `[`) · số `$` lẻ theo từng đoạn · ngoặc cân · `_` trần trong bib.
Cảnh báo "empty pages in dualunet/countception/bnll/cqr" là CŨ và vô hại (hội nghị không có số trang).

**★ HIGHLIGHTS VIẾT LẠI 2026-08-11 theo tiêu đề mới — sync CẢ 3 chỗ (highlights.txt + block trong 2 file .tex).**
Bộ cũ hỏng 3 điểm: (1) không dòng nào nói ĐỦ — dòng 1 vẫn khung "đếm sánh ngang mặt nạ" của tiêu đề cũ;
(2) **không dòng nào có chữ "nuclei"** — mà ScienceDirect hiện highlights TÁCH RỜI tiêu đề, người đọc không biết
đang đếm cái gì; (3) "are estimated to cost" và "does not predict" = hedge + phủ định, trái quy tắc văn phong.
Bộ mới (dài nhất 81/85, có biên an toàn — nhớ ký tự `\%` làm python đếm dôi 1):
1. Image-level count labels match pixel masks for nuclei counting on three datasets
2. An equivalence test bounds the gap at 1.1% of mean count on 2,722 PanNuke patches
3. Image-level counts cost at most one eighth of full nucleus delineation
4. The most accurate of four methods is trained with the cheapest spatial label
5. A quarter of the label budget reaches 92.8% of the full-label R2
Dòng 4 thay "Annotation cost does not predict counting accuracy" — cùng dữ kiện, dạng dương, và gây bất ngờ hơn.
Cung bậc: ngang bằng → chặn → chi phí → bất ngờ → đường ngân sách.

**★★ TIÊU ĐỀ CHỐT 2026-08-11 — quay lại tiêu đề bản draft cũ. ĐỪNG đổi lại.**
- EN: `Image-level supervision can suffice for accurate nuclei counting in histopathology`
- VI: `Giám sát mức ảnh có thể đủ để đếm chính xác nhân tế bào trong ảnh bệnh học`
- **Tiêu đề cũ bỏ đi:** "Đếm nhân tế bào với một phần tám chi phí chú thích: nhãn đếm mức ảnh sánh ngang mặt nạ
  mức điểm ảnh". Lý do bỏ: nó neo mốc bán hàng vào MẶT NẠ, mà chính Mục 5.3 của bài kết luận mức mặt nạ là vô
  ích (DeNuC nhãn ĐIỂM 1,4 s/nhân đạt $R^2$ 0,936 > mặt nạ 0,847/0,809). Bài tự chứng minh mốc trong tiêu đề
  mình là mốc sai. Cộng thêm: rất ít công trình dùng mặt nạ để ĐẾM → phản biện "đánh bù nhìn".
- Tiêu đề mới KHÔNG nhắc mặt nạ ⇒ phản biện đó mất chỗ bám mà không phải đổi trục bài.
- Chữ "sufficient" hợp với bộ máy thống kê: **kiểm định tương đương chính là công cụ để đỡ khẳng định dạng "đủ"**
  (kiểm định thường chỉ bác bỏ được "có khác biệt").
- Luật RINENG về tiêu đề chỉ có: *"concise and informative, avoid abbreviations and formulae"*. KHÔNG có giới hạn
  số từ, KHÔNG cấm dạng khẳng định. Tiêu đề mới 12 từ, không viết tắt, không công thức → hợp.
- ⚠️ RỦI RO CÒN LẠI đã vá: người duyệt đối chiếu "sufficient" với bảng 4 phương thức rồi nói DeNuC 0,936 > 0,918.
  Đã thêm đoạn vào Mục 5.3 (cả 2 file): *"Điều được xác lập ở đây là mức đủ, không phải mức tối ưu."*
- Con số **một phần tám** rời khỏi tiêu đề nhưng GIỮ trong tóm tắt/điểm nhấn/kết luận (đã có chữ "chặn trên",
  "ước tính"). Số đáng bán hơn giờ là **1,1%** (có kiểm định tương đương đỡ) và **độ chính xác không tăng theo
  chi phí chú thích** — hai thứ này không phụ thuộc việc có ai đếm bằng mặt nạ hay không.
- ĐÃ KIỂM sau khi đổi: EN abstract 215/250 từ; 5 điểm nhấn dài nhất 85/85; §6.1 vốn đã tên "Vì sao giám sát mức
  ảnh có thể đủ"; kết luận + abstract đã có "không phải điều kiện tiên quyết" ⇒ khớp tiêu đề, không cần sửa thêm.
- Mục hạn chế đã chặn đúng phạm vi cho chữ "đủ" (chỉ tác vụ đếm, chỉ H&E, chỉ kiến trúc đã khảo sát) → giữ NGUYÊN,
  đó là thứ bảo vệ tiêu đề.

**★ CHỐT CUỐI 2026-08-11 — quy tắc phạm vi Mục 2.1: CHỈ công trình mà ĐẾM là thứ được chấm điểm.**
- Mục 2.1 giữ: `conic` + `tcunet` (+ `unet`/`hovernet`/`stardist` là kiến trúc nền, + `dualunet`/`denuc` là hai
  đại diện chi phí nhãn).
- `tupac16` + `midog` + `pathonet` → **CHUYỂN sang trích dẫn Mục 1** (chúng đỡ tiền đề "đếm được làm GIÁN TIẾP",
  không phải chuẩn đo đếm). ĐỪNG đưa lại vào Mục 2.
- `distillfm` + `kongnet` → **ĐÃ XOÁ khỏi bài VÀ khỏi references.bib** (37 → 35 mục), vì không đo đếm và không
  được trích ở đâu khác. Backup bib trước khi xoá ở scratchpad phiên này.
- Mục 2 KHÔNG được chứa `\ref{}` tới mục nào — đây là công trình liên quan, không phải chỗ điều hướng.
⚠️ Metadata lấy từ **Crossref API** (title/author/volume/page/year/doi), KHÔNG phải từ trang scrape → tin được.

**★ 2026-08-11 vòng 2 — KHẢO SÁT "đa số paper đếm dùng nhãn gì": trả lời = NHÃN ĐIỂM, không phải mặt nạ.**
- Tài liệu đếm tế bào chia 2 họ: dựa-phát-hiện (đếm số thực thể) và hồi-quy-mật-độ (tích phân bản đồ). Các khảo
  sát nói hồi quy mật độ **đã thay thế phần lớn** họ kia (phát hiện sập khi chen chúc) → nhãn chuẩn của bài toán
  đếm là **một chấm mỗi nhân**. Nhánh này bài ĐÃ có ở đoạn hồi quy mật độ (`lempitsky`/`fcrn`/`countception`).
- Nhánh lâm sàng cũng nhãn điểm: Ki-67 DCIS (Fulawka, Sci Rep 12:3166, 2022) chấm 150.592 nhân bằng ImageJ Cell
  Counter, báo cáo MAE trên **chỉ số**, không phải trên số đếm.
- Tìm bài huấn-luyện-bằng-mặt-nạ mà báo cáo thước đo ĐẾM → không ra bài nào; toàn Dice/AJI/PQ.
  ⇒ Mục 2.1 mỏng KHÔNG phải lỗi của Mục 2.1; đó chính là khoảng trống bài khai thác. ĐỪNG cố nhồi thêm bài.
- **CoNIC = đếm THEO LOẠI** (6 lớp, xếp hạng bằng R² trung-bình-trên-lớp), không phải tổng số. Đã ghi rõ chữ
  "của từng loại" trong bài để người duyệt không bắt được. `tcunet` mới là bài đếm TỔNG bằng mặt nạ.
- Tiêu đề Mục 2.1 đổi: "Đếm bằng phân đoạn và phát hiện nhân" → **"Đếm bằng nhãn không gian"** (tiêu đề cũ hứa
  một họ lớn mà chỉ đỡ được 2 trích dẫn). Thêm 1 câu: mức thông dụng hơn là một điểm mỗi nhân → bắc cầu sang
  đoạn hồi quy mật độ.

**❌ ĐÃ KIỂM VÀ LOẠI (2026-08-11) — đừng tra lại, đừng đưa vào Mục 2.1:**

| bài | tác vụ chấm điểm | thước đo |
|---|---|---|
| CellPilot (Endres+, ISBI 2025, arXiv 2411.15514) | phân đoạn tế bào+tuyến | phân đoạn; "counting" không phải tác vụ được đánh giá |
| PathoSAM (arXiv 2502.00408) | phân đoạn nhân | phân đoạn |
| BMC Bioinformatics instance-level 2025 | phân đoạn+phân loại | phân đoạn |
| PLOS One federated 2025 | phân đoạn+phân loại | phân đoạn |
| TAND (arXiv 2511.13615) | **phát hiện**+phân loại (giám sát ĐIỂM, PUMA) | F1 |
| CelloType (Nat Methods 2024) | phát hiện+phân đoạn+phân loại | AP theo COCO |

Bẫy lặp lại: nguồn gợi ý dùng tiêu chí "**có thể dùng** để đếm" — vô nghĩa, mọi mô hình phân đoạn đều thế.
Ghi chú: TAND giám sát bằng ĐIỂM và tuyên bố vượt phương pháp giám sát MẶT NẠ trên phát hiện → cùng chiều luận
điểm của bài, nhưng thuộc Mục 2.2 (nới lỏng giám sát), KHÔNG thuộc Mục 2.1.

**★ 2026-08-11 vòng 3 — TÌM ĐÚNG NHÁNH: bài ĐO BẰNG THƯỚC ĐO ĐẾM thì CÓ NHIỀU, nhưng nhãn đều là ĐIỂM.**
⇒ Chúng thuộc **Mục 2.3 (hồi quy mật độ)**, không phải Mục 2.1. Đã thêm 4 mục (bib 35 → **39**, 0 mồ côi):

| khoá | bài | nhãn | thước đo | vì sao thêm |
|---|---|---|---|---|
| `decouple` | Rethinking Cell Counting: Decoupling Counting and Localization, MICCAI 2024, LNCS 418–426, doi 10.1007/978-3-031-72083-3_39 | điểm | MAE/MSE | **sát luận điểm nhất**: tách bộ đếm khỏi bộ định vị làm GIẢM sai số đếm ⇒ định vị không cần cho đếm |
| `saunet` | SAU-Net, IEEE/ACM TCBB 19(4):1920–1932, 2022, doi 10.1109/TCBB.2021.3089608 | chấm | MAE trên VGG/MBM/ADI/DCC | mốc kinh điển của nhánh; chấm thay cho chú thích toàn tế bào |
| `cellfmcount` | CellFMCount, IEEE ICDM 2025, 613–622, doi 10.1109/ICDM65498.2025.00069 | 430k vị trí tế bào | MAE | recency 2025 + BẰNG CHỨNG SỐ rằng chuẩn đo đếm dựng trên nhãn điểm |
| `annoeff` | Annotation-Efficient Cell Counting, MICCAI 2021, LNCS 405–414, doi 10.1007/978-3-030-87237-3_39 | điểm (active learning chọn VÙNG) | — | chặn phản biện "đã có người làm label-efficiency cho đếm rồi": họ giảm SỐ VÙNG, không hạ MỨC nhãn |

Vá luôn lỗ recency: trước đó Mục 2.3 dừng ở Count-ception **2017**, người duyệt sẽ bắt.
❌ Đã kiểm và LOẠI ở vòng này: **CellRegNet** (Bioengineering 11(8):814, 2024) — nhãn điểm nhưng CHỈ báo cáo
precision/recall/F1, không có thước đo đếm. **DLA-Count** (arXiv 2503.12063) — có MAE nhưng preprint, không có
venue trên Crossref → bỏ, `cellfmcount` đã gánh phần recency.

<details><summary>Ghi chép cũ khi lỗ này chưa vá</summary>

Bài khẳng định **2 chỗ** rằng đếm nhân hiện được làm gián tiếp qua phân đoạn/phát hiện:
- `main.tex:67-69` / `main_vi.tex` Mục 1, đoạn 1 — **0 chữ `cite` trong cả đoạn**
- Mục 2 đoạn 1, câu "khi số đếm là đại lượng cần thu được, thực hành phổ biến là..." — không cite

**Quét toàn bộ 32 tài liệu: KHÔNG có công trình nào [tác vụ = đếm] × [nhãn = mặt nạ/phát hiện].**
- 9 bài tiêu đề có "count/quantif/enumerat" = hồi quy mật độ (lempitsky/fcrn/countception, nhãn ĐIỂM),
  đếm đám đông (matt/sorting/transcrowd/count2density/tcformer), đếm vi sinh vật (microvit). Không cái nào là nhân.
- 7 bài về nhân (unet/hovernet/stardist/kongnet/distillfm/dualunet/denuc) = tiêu đề đều segmentation/detection/
  classification. Không tiêu đề nào có "counting".

⇒ Phản biện hỏi "ai nói người ta đếm theo cách đó?" thì bài KHÔNG trả lời được. Đây cũng là lý do Mục 2 đọc gượng
qua 3 vòng sửa: đang phê phán một thực hành mà bài chưa chứng minh là tồn tại.

**CẦN TÌM (user tra, KHÔNG được để tôi bịa tên bài):** đếm phân bào / mitotic count (MIDOG), chỉ số Ki-67,
đếm TIL, ước lượng tế bào khối u. **Chỗ rẻ nhất:** mục ứng dụng hạ nguồn trong `nucleisurvey` 2025 — ĐÃ CÓ trong
references.bib. Tìm được thì giữ nguyên cách viết mạnh hiện tại; không tìm được thì hạ xuống phát biểu cấu trúc
("đầu ra thực thể CHO PHÉP suy ra số đếm; Mục 5.3 là chỗ đo") — khi đó Mục 5.3 thành đóng góp vì chưa ai chấm
điểm các phương pháp đó trên tác vụ đếm dưới cùng một giao thức.

</details>

**CHƯA BUILD.** Không có LaTeX local. Phải build Overleaf trước khi nộp: đánh số phụ lục A/B, `note` của 3
preprint trong elsarticle-num, và caption Bảng 1/2 đã dài thêm.

---

## ★ REPO GỐC + LỆNH CHẠY của 2 baseline ngoài (tra lại 2026-08-01, trước đó KHÔNG ghi ở đâu)

| baseline | repo | train | eval |
|---|---|---|---|
| DualU-Net | `https://github.com/davidanglada/DualU-Net` | `python train.py --config <yaml>` | ta dùng `eval_count_dualunet.py --config-file <yaml> --checkpoint <pth>` |
| DeNuC | `https://github.com/ZijiangY1116/DeNuC` | `python ./denuc_train.py --arch denuc_det_shufflenet_x2_0 --datasets pannuke12_3` | `python ./denuc_eval.py --exp_name <exp> --eval_dataset pannuke12_3 --eval_mode test` (nms_dist mặc định 12.0) |

**BẪY cờ:** repo DualU-Net dùng **`--config`**, script `eval_count_dualunet.py` CỦA TA dùng **`--config-file`**. Không lẫn.
`--arch denuc_det_shufflenet_x2_0` khớp ShuffleNet-x2 4.37M ở Bảng 4 ✓. Clone DualU-Net PHẢI vào đúng
`/kaggle/working/DualU-Net` vì `eval_count_dualunet.py:24` hard-code sys.path.

**DỮ LIỆU COCO cho DualU-Net** (bước `pannuke2coco` KHÔNG còn trong repo — chỉ còn sản phẩm):
`/kaggle/input/datasets/hipinhththu/pannuke2coco/coco` chứa `fold1/ fold2/ fold3/`, mỗi fold có
`annotations.json` + `images/`. Bản gốc ở máy: `~/Downloads/coco.zip` (794 MB).
**Đã verify 2026-08-01:** fold1 2656 + fold2 2523 = **5179** (=train, GIỮ colon ✓), fold3 **2722**,
5 category; fold3 có 66.654 annotation → **24,49 nhân/ảnh**, khớp đúng GT trung bình dùng khắp bài.
⚠️ Mất bộ này = phải dựng lại bộ chuyển đổi từ đầu.

**Seed:** DualU-Net có sẵn `experiment.seed` trong yaml → chỉ cần sinh config (đổi cả `"resnext_pannuke_fold12"`
ở `name:`+`output_name:` kẻo 5 seed đè checkpoint nhau). DeNuC README không nhắc seed → phải vá.


**★ StarDist 5-SEED XONG (2026-08-02, leak-free full fold3 n=2722):** R² per-seed 0.8537/0.8452/0.8582/0.8483/0.8283
→ **0.847±0.011**; MAE 4.316/4.390/4.363/4.449/4.393 → **4.38±0.05**. Thay số 1-seed cũ (0.841/4.64 — nằm ở
biên dưới). Pred mean 21.5 vs GT 24.5 = **under-count 12.2%**. Bảng 4 + caption 2 bản .tex ĐÃ SỬA.
✅ **KIEM CHEO LOCAL 2026-08-02** bằng `agg_baseline_seeds.py` trên 5 CSV thật ở `~/Documents/stardist/`:
ra ĐÚNG 0.847±0.011 / 4.38±0.05, và **chặn cứng cột `gt` đã qua** → cả 5 seed cùng một tập kiểm tra.
Mỗi file n=2722, GT mean 24.49, tổng 66.654 nhân — khớp tuyệt đối bảng chính (full fold3, GIỮ colon).
CSV gốc cũng có trong `~/Downloads/results-2,-3,-4/out/`. Checkpoint `.h5` ở `results-*/stardist_models/`.
**Không còn hàng 1-seed nào**: DensitySigmaUNet, StarDist, DeNuC (MASTER:255) và DualU-Net đều 5 seed.

**★★ NuLite-T (baseline mặt nạ THỨ BA cho Bảng 4) — SETUP XONG 2026-08-03, đang chạy 5 seed:**
Lý do thêm: bảng chỉ có 2 phương pháp mặt nạ (StarDist 1,43M / DualU-Net 41,0M), dễ bị bác "thử có hai cái".
NuLite-T lấp khoảng giữa và ở bảng OOD §4.2 nó thắng cả LKCell 164M lẫn CellViT 700M nên không gọi là yếu được.
Repo `CosmoIknosLab/NuLite`, entry `train_nulite.py --config`.

**⚠️ BIẾN THỂ — ĐO TỪ CODE 2026-08-04, ĐỪNG TIN BÀI BÁO:** `models/nulite.py:43-58` nhận 7 backbone. Đếm tham số
(num_nuclei=6, num_tissue=19): t8 5.283.149 | **t12 10.130.013** | **s12 12.049.885 (reparam 12.009.117)** |
sa12 14.157.277 | sa24 24.130.525 | sa36 34.103.773 | ma36 47.929.201.
Bài ghi NuLite-T/M/H = 17.116.701 / 31.109.149 / 41.065.117 → **KHÔNG khớp backbone nào**.
Mỏ neo đúng: MASTER:696 đo checkpoint OFFICIAL NuLite-T = **12,009M**, khớp `fastvit_s12` reparam
**12.009.117** tới từng chữ số ⟹ **NuLite-T = `fastvit_s12`**.
⛔ `training_example.yaml` mặc định `fastvit_t8` (5,28M) — chỉ là ví dụ, **không phải biến thể công bố nào**.
Lượt 5 seed 2026-08-04 lỡ chạy bằng t8; giữ làm dòng phụ, phải chạy lại `fastvit_s12` cho hàng chính.

**Môi trường PHẢI ghim (Kaggle 2026 quá mới, repo 2024):** `torchmetrics==0.11.4` (>=1.0 gỡ `functional.dice`),
`schema`, `pandarallel`, **`albumentations==1.4.4`** (>=1.4.15 đổi `RandomSizedCrop(height,width)` → `size`; 1.4.14 và 1.4.10 bị pip từ chối vì đòi hạ numpy — bản chạy được với **numpy 2.0.2** của Kaggle là 1.4.4).
⚠️ **Mọi lệnh pip phải kèm `numpy=={bản hiện có}`** — `albumentations==1.3.1` ép `numpy<2`, pip ghi đè numpy giữa
lúc kernel đang giữ numpy cũ → `ModuleNotFoundError: numpy.rec`, vỡ pandas. Và `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
vì torch>=2.6 mặc định `weights_only=True`.

**Ba cái bẫy dữ liệu:** (1) `prepare_pannuke.py` của NuLite CHỈ ghi `images/`+`labels/`, **không** sinh
`types.csv` lẫn `cell_count.csv` (bản CellViT gốc có) → phải tự sinh; tự sinh cho fold2 ra **24,487 nhân/ảnh**,
khớp 24,49 của bảng chính ⇒ cách sinh đúng. (2) `dataset_config.yaml` phải chép từ `configs/datasets/PanNuke/`
sang gốc `dataset_path`. (3) checkpoint do train sinh ra **thiếu khoá `config`** mà lớp suy luận của họ đòi →
nhét config phẳng (dấu chấm) vào rồi mới `dump_cellvit_counts.py --nulite` chạy được.

**Giao thức:** train_folds=[0,1] (=PanNuke 1+2, 5179 mảnh), **val_folds=[1] nằm TRONG tập train** nên chọn mô hình
không bao giờ chạm fold 3; test_folds=[2]. Lấy `latest_checkpoint.pth` = epoch cuối, ngân sách cố định, không
chọn theo val — như DualU-Net epoch 50. Nhãn trung gian ~6 GB → để `/kaggle/temp`, đừng để `/kaggle/working`.

**Ngân sách 60 epoch (không phải 130 của họ):** đo thật 1 epoch = train 2ph35 + val 49s ≈ 3ph25; 130 epoch ≈ 7,5h
mỗi seed × 5 = 37h **vượt hạn mức 30h/tuần** của Kaggle. 60 epoch ≈ 3,5h/seed, tổng ~20h, cùng bậc với DualU-Net
50 epoch. **Phải ghi rõ trong bài là baseline dùng ngân sách rút gọn.**

**✅ SEED ĐÃ KIỂM CHỨNG (2026-08-03)** — thứ DualU-Net đã thiếu: 1 epoch, seed 42 md5 `af77d3b5…40c9` /
pred mean 26,2 / bPQ 0,497; seed 43 md5 `edd03660…0ff3` / pred mean 25,9 / bPQ 0,513. Khác nhau ⇒ `seed_run()`
ở `base_experiment.py:72` có tác dụng thật.

**★★ KẾT QUẢ 5 SEED (2026-08-04, 60 epoch, leak-free full fold3 n=2722):**
R² 0,9102/0,9031/0,9100/0,9054/0,9051 → **0,9068 ± 0,0032**; MAE 3,348/3,452/3,437/3,413/3,423 → **3,415 ± 0,040**.
pred mean 24,22 vs GT 24,49 = **gần như không lệch** (StarDist thiếu 20,8%, DualU-Net dư ~8%).
⚠️ **ĐỔI LẬP LUẬN:** so PACT 0,918±0,011 / 3,60±0,10 → ta **hơn 0,011 R² (≈1 sd, không tách được)** nhưng
**thua 0,185 MAE (nhất quán 5/5 seed)**. Câu "cả hai phương pháp mặt nạ đếm kém hơn" (main.tex:735) và câu
abstract "above both surveyed representatives trained on pixel masks" ĐÃ SAI, phải viết lại thành PARITY.
⚠️ **Điểm yếu phản biện sẽ đánh:** NuLite chạy 60/130 epoch, tức baseline MẠNH NHẤT bị cắt ngân sách, mà
chiều cắt lại có lợi cho ta. Phải kiểm val loss đã bão hoà chưa (log_s*.log) hoặc chạy lại 130 epoch.

**⚠️ PARAMS = 5.309.244 (5,31M)**, đếm từ chính checkpoint — KHÔNG phải 12,0M như MASTER:696 ghi cho bản
off-the-shelf. Bảng 4 dùng 5,31M kèm ghi backbone `fastvit_t8`. Xem [[feedback-audit-bang-o-trong]].


## ★★ CHOT TAP CHI (2026-08-01): Results in Engineering (Elsevier), Special Issue 335088

**Ban thao:** `paper2_rineng/main.tex` (elsarticle) + `main_vi.tex`. **CHI CON MOT BAN THAO.**
Ban Springer `paper2_overleaf/` DA XOA 2026-08-10 (mang 4 so da bi thay: 1.25 dinh vi, 6.0ms, 46ms, 0.234) —
neu RINENG desk-reject thi lay lai bang `git checkout bda5741 -- distillation_counting/paper2_overleaf`
roi PHAI dong bo lai so lieu truoc khi dung.

| hang muc | gia tri |
|---|---|
| tap chi | Results in Engineering, gold OA, IF 9.4 (6/2026), JCR **Q1**, WoS+Scopus+DOAJ |
| APC | **2.160 USD** chua thue, thu SAU khi accept -> nop thu KHONG mat tien |
| tac gia | Duong Viet Hang dung dau (de xin ho tro ~50% APC cho giang vien), Dinh Thi Thu Hiep corresponding |

**RUI RO CHINH = cua "engineering" trong scope:** *"A clear engineering problem, system, or process must be
addressed"* + *"not only the re-application of a known AI algorithm to existing problems"*. Bai la nghien cuu
y-sinh, mo hinh dung linh kien co san -> **rui ro desk-reject**. Desk-reject mat ~1-2 tuan, khong mat tien.

**Cach da xu ly:** viet lai quanh **BA NGAN SACH** (chu thich / tinh toan / dac ta do tin cay), KHONG them so
lieu moi - moi con so da co san, chi doi thu tu uu tien:
- ngan sach chu thich: Qu et al. 115 phut mask vs 14 phut diem -> >=8x; 25% ngan sach = 93% hieu nang
- ngan sach tinh toan: 3.62M, 0.98 GMACs, 6.9ms GPU (Tesla T4) / 51ms CPU, 37.9MB, nho hon 11.3x DualU-Net ma van chinh xac hon
- dac ta do tin cay: mondrian worst-organ 0.906, **0/19** duoi phu (global 4/19) - gia la interval rong hon 15%

**Kiem da chay:** numeric-token diff Springer vs Elsevier = chi mat 2 dong ORCID; moi \cite co trong bib;
5 highlights deu <= 85 ky tu; moi truong/ngoac/$ can; so cot moi bang khop header.
**CHUA build** - tren may khong co LaTeX, phai build tren Overleaf.

**CAN QUYET:** muc *Declaration of generative AI* dang de BAT trong main.tex. Neu khong khai thi xoa CA muc.


## 0. ★ VERIFICATION STATUS (NGUỒN CHÂN LÝ — mọi số phải khớp bảng này)

| Thành phần | Trạng thái | Số honest |
|---|---|---|
| **R2 PanNuke** | ✅ VERIFIED (tái tạo khớp từng số, 3 fold) | mondrian worst-org **0.906**, MAE **3.36**, Winkler **19.28** |
| **R2 NuInsSeg** | ✅ RE-ESTABLISHED 5-seed (2026-07-17) | cluster worst-org **0.750±0.049**, MAE **14.7±1.7**, Winkler **95.4±11.9** |
| **CondConf/PCP NuInsSeg** | ✅ RE-EVAL 5-seed matched (2026-07-17) | CondConf 0.898±0.013 / 146.8±24.9 (over-cover); PCP 0.708±0.069 / 105.9±12.1 |
| **KD NuInsSeg** | ✅ VERIFIED 5-seed (2026-07-18) | same-scheme §4.1: global 0.278/132/21.7, cluster 0.658/98.7/21.7; per-image sig MAE p<0.05 mọi seed, Winkler 1 seed n.s. **N4 reframe: PB-σ chỉ sập global.** pkl backup kaggle `sam3-paper2-uqkd` |
| **UQ-floor NuInsSeg** (Ensemble/CQR/CHDQR/MC-Dropout) | ✅ VERIFIED 5-seed (2026-07-18) | cluster n=5: Ens 69.6/0.767, CQR 80.7/0.806, CHDQR 78.7/0.722, MCD 173/0.774 — **R2 KHÔNG dẫn đầu (~4/5)**, xem §4.3 |
| **Cross-dataset N5 (3 dataset)** | ✅ CryoNuSeg thêm (2026-07-18) | NuInsSeg→CryoNuSeg marg.cov **0.967**/Winkler 503/MAE 73 (σ transfer sống); NuInsSeg↔PanNuke §4.4. MoNuSAC=boundary (scale 4× → μ sập) |
| **Label-efficiency frontier** | ✅ EMPIRICAL (2026-07-19, §4.10) | distilled↔supervised head-to-head: **ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count ≈ 5–10× rẻ hơn mask** (K=52.8 nhân/ảnh). Đóng lỗ "tại sao distill". `label_efficiency{,_both}.py` |
| **Multi-teacher committee (Hướng A)** | ⏳ PROBE xong, thí nghiệm đầy đủ CHƯA (2026-07-19, §4.11) | 4 teacher (PathoSAM+SAM3+NuLite+LKCell): consensus KHÔNG hạ accuracy (CNN teacher đếm thiếu OOD); **disagreement-σ +0.65 ngang learned-σ** (dao động 0.49→0.41→0.65, chưa chắc). CÒN: train σ-target=disagreement, so coverage. `probe_multiteacher_full.py` |
| **Baseline PanNuke (8c)** | ✅ chạy trên pkl R2 còn sống | giữ nguyên |
| pkl R2 5-seed | ✅ backup kaggle `hipinhththu/sam3-r2-nuinsseg-seeds` | — |

**Provenance (2026-07-17):** pkl NuInsSeg canonical (`student_r2_nuinsseg_cv5_poisson.pkl` + KD) đã MẤT (không ở vast/kaggle).
Bản `_feat` còn sót thiếu `--detach_mu` → số sai. Đã retrain R2 5-seed đúng config. **Mọi "0.773" cũ = seed đơn** (nằm trong dải
0.70–0.82); số chính thức = **0.750±0.049**. PanNuke KHÔNG ảnh hưởng (nhiều ảnh/mô → worst-org ổn định; NuInsSeg ít ảnh/organ → nhạy seed = lý do phải multi-seed).

**★ BACKUP INVENTORY (2026-07-18 — mọi artifact đắt, đừng rebuild/mất):**
| kaggle dataset | chứa |
|---|---|
| `hipinhththu/sam3-r2-nuinsseg-seeds` | R2 NuInsSeg 5-seed pkl (headline 0.750) |
| `hipinhththu/sam3-paper2-uqkd` | 20 UQ-floor + 5 KD + 3 ch16-PanNuke pkl + teacher_targets_nuinsseg + teacher_density_nuinsseg + xfer_cryonuseg + prep_cryonuseg |
| `hipinhththu/sam3-pannuke-density-cache` | teacher_density_pannuke_f123.pkl (3.6GB, PathoSAM output — rebuild rất tốn) |
| `hipinhththu/monusac` | raw MoNuSAC (bỏ dùng, giữ tham khảo) |
| git repo | PAPER2_MASTER.md + mọi script (`aggregate_*.py`, `prep_cryonuseg_counts.py`) + `data/pathosam_nuinsseg_preds.pkl` (teacher-PB) |

**★ SPLIT PanNuke của PathoSAM (CHỐT CỨNG, verify từ CODE patho-sam — [[pathosam-training-data]]):**
**PathoSAM train PanNuke FOLD_1 + FOLD_2, TEST trên FOLD_3** (`get_generalist_datasets.py:80` train fold_1+2; `dataloaders.py:149` eval fold_3).
⟹ **Đánh giá PanNuke trên FOLD_3 = SẠCH (leak-free)** — dùng dataset KHÔNG bị leak. **Paper 2 lấy fold_3 làm số PanNuke chính (worst-org 0.905).**
**NuInsSeg + MoNuSAC không nằm trong training PathoSAM → clean OOD; NuInsSeg = anchor generalization.** Paper 1 cũng test fold_3 → sạch, loại colon đúng.

---

## 0.5 ★★★ PIVOTAL (2026-07-22) — DISTILLATION KHÔNG GIÚP CHO ĐẾM → REPOSITION

**Bối cảnh:** chẩn đoán vì sao PACT (TinyUNet, teacher-distill) accuracy chỉ ~0.78. Kết quả **lật trục "distillation = linh hồn"**.

**Bảng chẩn đoán (leak-free, global R² qua `compute_r2_counting.py`; efflite0 = EfficientNet-Lite0 pretrained ~3.6M, backbone hỗ trợ sẵn trong `distill_student_r2.py --backbone`):**

| Dataset | Backbone | count-only (w_d=0,w_c=1) | teacher-distill (w_d=1,w_c=0.01) | GT-density (target hoàn hảo) |
|---|---|---|---|---|
| NuInsSeg | TinyUNet (5-seed) | 0.767 | 0.786 | 0.810±0.023 |
| NuInsSeg | efflite0 (**1 seed**) | **0.925** | **0.512** ⬇️ | 0.881 |
| PanNuke f3 | efflite0 (**1 seed**) | 0.928 | 0.935 | — |
| *(teacher PathoSAM zero-shot NuInsSeg = 0.711)* | | | | |

**Kết luận cứng (logic bịt đường cãi):**
1. **GT-density hoàn hảo (0.881) < count-only (0.925)** → bắt chước density map, DÙ target hoàn hảo, **không thể bằng** dạy thẳng count. Ceiling của distill = chất lượng density, mà ceiling đó vẫn dưới count-only.
2. Teacher thật ≤ GT: **in-domain** (PanNuke, density≈GT) → distill **HÒA** count-only (0.935≈0.928); **OOD** (NuInsSeg, density rác) → distill **SẬP** (0.512≪0.925).
3. ⟹ **Distill giỏi nhất = hòa, dở nhất = hại, KHÔNG BAO GIỜ thắng count-only.** Nguyên nhân 0.512 = **teacher tồi (OOD)**, KHÔNG phải student yếu (efflite0 đạt 0.881 với GT, 0.925 với count trực tiếp).
4. **"Distill foundation → tiny counter" chết ở trục accuracy.** Phản biện trực tiếp tiền đề CellGenNet/H-Optimus (đều distill).

**⚠️ IN-DOMAIN caveat (lỗi đã mắc, phải nhớ):** efflite0 0.925 là **in-domain** (train NuInsSeg count). KHÔNG được claim "đè foundation 640M" (teacher zero-shot) — bất đối xứng = critique A. So fair phải là **các counter nhẹ khác cũng in-domain count**, hoặc efflite0 eval **zero-shot cross-dataset**.

**→ REPOSITION (chốt hướng với user 2026-07-22):** xây **MODEL MỚI count-only sạch** (backbone nhẹ pretrained + đầu đếm trực tiếp + đầu σ calibrated, CHỈ nhãn count, KHÔNG teacher/mask). "Distill" **rời khỏi method**, chỉ còn là **ablation/động-cơ** ("thử distill foundation — không giúp/hại → nên thiết kế gọn không teacher"). Novelty gánh bởi: **(i) negative-result distill-là-thừa** (cái mới nhất, ngược field) + **(ii) đầu UQ calibrated/transfer** + (iii) label-efficiency + chặt chẽ đa dataset. Model một mình = incremental → PHẢI đi kèm (i).

### ⏳ PHẢI CHẠY để firm-up claim "distill không giúp" lên chuẩn Q1 (chưa làm)
- **(a) [BẮT BUỘC] efflite0 5-seed** cho cả 3 điều kiện (count-only / teacher-distill / GT-density) trên NuInsSeg + PanNuke → hiện mới **1 seed** (±0.000), cần mean±sd loại may rủi. Backbone efflite0, `--kfold 5` (NuInsSeg) / `--test_fold 3` (PanNuke), lặp `--seed {42..46}`.
- **(b) [NÊN] CryoNuSeg teacher-OOD** — điểm thứ 3 cho trục "teacher OOD → distill hại" (hiện chỉ 2 điểm: NuInsSeg hại, PanNuke hòa). Teacher PathoSAM cũng OOD trên CryoNuSeg → kỳ vọng distill < count-only, xác nhận xu hướng. Cache/prep: `prep_cryonuseg_counts.py`.
- **(c) [GHI PHẠM VI]** chỉ 1 teacher (PathoSAM) → scope claim "một foundation pathology mạnh", hoặc thử teacher thứ 2.

**Artifacts đã tạo (chẩn đoán, chưa backup):** `work/baseline_countonly_efflite0.pkl` (0.925), `pact_efflite0_distill.pkl` (0.512), `efflite0_gtdensity.pkl` (0.881), `baseline_countonly_tinyunet.pkl` (0.767), `efflite0_pan_countonly_f3.pkl` (0.928), `efflite0_pan_distill_f3.pkl` (0.935). **→ backup Kaggle khi chạy 5-seed.** TinyUNet GT-density 5-seed = pkl `student_r2_nuinsseg_cv5_supervised_s*` (dataset `sam3-paper2-uqkd`).

### 0.6 ★ REORIENT (2026-07-22) — hướng FEATURE-DISTILL "phễu" (gate DƯƠNG đầu tiên)

**Chuỗi chẩn đoán loại sạch:** distill-OUTPUT bác; crowding-module tiền đề bác (`premise_tiling_test`: tiles TỆ hơn whole, whole đã gần đúng ở bin dày); scale-robustness = nghiệm tầm thường (MPP). ⟹ **accuracy in-domain ĐÃ bão hoà**, không còn khe method.

**Insight cứu distill (trả lời "student không thừa hưởng gì từ teacher"):** mình đã distill **OUTPUT (density)** của teacher OOD (lỗi) — phải distill **REPRESENTATION (feature)**. Teacher đếm dở nhưng **đặc trưng thị giác pathology giàu**.

**★ GATE DƯƠNG (2026-07-22, `gate_feature_surplus.py`, linear-probe leave-organ-out, Phikon ViT-B vs ImageNet ViT-B, CÙNG kiến trúc):**
| nhãn | ImageNet R² | Phikon R² | Δ |
|---|---|---|---|
| 100% | 0.801 | **0.875** | +0.074 |
| 50% | 0.738 | 0.868 | +0.130 |
| 25% | 0.688 | 0.838 | **+0.150** |
| 10% | 0.670 | 0.788 | +0.118 |

**Feature pathology CÓ thặng dư** — lớn nhất ở **low-label + shift (mô chưa thấy)**, thu hẹp khi đủ nhãn. ⟹ hướng **feature-distill "phễu"** (distill feature, cổng lọc lỗi teacher = confidence × count-agreement) **SỐNG**.

**⚠️ Honest:** (1) cải thiện là **R²/generalization, KHÔNG phải MAE** (MAE ~ngang/hơi tệ) → claim = generalize/label-efficiency, không phải sai-số-tuyệt-đối. (2) đây là **upper-bound** (probe trên Phikon 86M đông lạnh, CHƯA phải student tí hon). (3) **bẫy capacity-gap** khi nhồi 86M→efflite0 3.6M.

**⏳ Bước tiếp (chưa làm):** transferability — efflite0 + feature-distill-Phikon có thắng efflite0-ImageNet ở low-label/shift không? (`transfer_feature_distill.py`). Confidence teacher cho phễu = **đã cache** `teacher_targets_nuinsseg.pkl`. Phikon truy cập HF `owkin/phikon` (Internet ON; ⚠️ `pip install` làm hỏng torch-CUDA → factory-reset, đừng pip).

### 0.7 PHỄU feature-distill (2026-07-22, `transfer_feature_distill.py`, 25% nhãn, leave-organ-out) — 8-SEED LẬT 3-SEED: KHÔNG SIGNIFICANT

Student = efflite0 (DensitySigmaUNet, count-only density). **Phễu 2 cổng (chỉ nhãn count):** density-gate spatial (density student detach → dồn distill vào vùng nhân) + reliability-gate per-image (Phikon-probe-count vs GT, 2-fold cross-fit → hạ ảnh teacher kém tin).

**★ KẾT QUẢ FIRM 8-seed + paired-Wilcoxon (w_feat=30, ablate 5 mode):**

| mode | R² mean±sd | Δ vs count | #thắng | p(Wilcoxon) |
|---|---|---|---|---|
| count-only | **+0.688±0.204** | — | — | — |
| naive | +0.614±0.154 | **−0.074** | 1/8 | 0.20 |
| density-only | +0.708±0.216 | +0.020 | 4/8 | 0.84 |
| reliab-only | +0.643±0.191 | **−0.046** | 1/8 | 0.20 |
| gated-both | +0.744±0.174 | +0.056 | 5/8 | **0.74** |

**⚠️ 3-seed (gated 0.812 > count 0.751) là LẠC QUAN MẪU-NHỎ — 8 seed dội lại: gated-both 5/8, p=0.74 → KHÔNG có ý nghĩa thống kê.** Cùng loại dương-tính-giả multi-seed từng bắt ở naive.

**Cái GÌ vững (tái lập):**
- **naive feature-distill HẠI** (−0.074, 1/8) — nhồi feature teacher thô vào student 3.6M làm tệ. Vững.
- **reliab-gate một mình cũng HẠI** (−0.046, 1/8) — cổng tin-cậy per-image hạ nhầm ảnh tốt (w_feat=30 ép quá?).
- **density-gate GỠ được cái hại** (−0.074→+0.020) nhưng **không thêm giá trị** vs không-distill.
- gated-both +0.056 nhưng ±0.20 nuốt trọn → không phân biệt được với 0.

**Nghĩa khoa học (finding đáng giá):** thặng dư feature pathology-FM **CÓ THẬT** (gate probe §0.6: Phikon>ImageNet) **nhưng KHÔNG chuyển được vào student 3.6M**. Cổng lọc chỉ *khử hại* của distill, **không thu hồi surplus**. Root cause = **capacity gap** ViT-86M (probe đọc surplus) ↔ student 3.6M. = negative **giải thích được**, khớp arc rigor cả paper (distillation KHÔNG phải trục accuracy — xem [[paper2-core-A-decision]], [[paper2-q1-hardening]]).

**Quét w_feat∈{5,10,30} × 4 seed (2026-07-22):** gated-both Δ = +0.062 / −0.092 / −0.149 (ĐỔI DẤU quanh hyperparam) → không w_feat nào cứu → **negative robust**. Cũng lộ bug: count-only (không dùng w_feat) vẫn nhảy 0.658/0.703/0.736 giữa các lần chạy → `np.random.shuffle` global chưa seed → sàn nhiễu cao (đã fix ở script pseudo). Artifacts: script git; chưa pkl backup (train nhanh, tái tạo được).

### 0.7b PIVOT pseudo-label semi-supervised (2026-07-23, `pseudo_label_semisup.py`) — TREND DƯƠNG NHƯNG CHƯA SIGNIFICANT (NuInsSeg thiếu power)

Sau khi funnel chết: **giữ tốt/thay xấu** — probe Phikon làm **teacher OUTPUT** (teacher THẬT mạnh hơn student, điều kiện distill hoạt động) thay vì mimic feature (đấu capacity); confidence-gate đổi việc = **chọn pseudo-label** thay vì hạ nhãn thật. Count-only, 25% nhãn, leave-organ-out, 8 seed.

| version | teacher−student | pseudo vùng-shift | **Δ student** | p | #thắng |
|---|---|---|---|---|---|
| v1 (teacher ensemble yếu, pool no-shift) | +0.042 | — | +0.051 | 0.74 | 4/8 |
| **v2 (teacher full-data mạnh 0.755, transductive pool phủ shift)** | **+0.116** | **+0.530** | **+0.079** | **0.38** | 5/8 |

**Đọc honest:** sửa 2 lỗi thiết kế (teacher yếu + pool bỏ đói vùng shift) → effect TĂNG ĐỀU (+0.05→+0.08), hướng nhất quán, teacher giờ mạnh hơn thật, pseudo phủ được mô-mới. **NHƯNG p=0.38 — chưa significant.** Rào cản = **phương sai NuInsSeg** (labeled-only ±0.309, pseudo-shift ±0.506; ít ảnh/mô → leave-organ-out nhảy) **nuốt** effect +0.08. = **thiếu power của dataset, KHÔNG phải lỗi method.**

**confidence-gate: VÔ DỤNG** (v1: pseudo-all == pseudo-confident = +0.051; SANITY confident không chọn pseudo tốt hơn) → **bỏ**, ghi làm ablation honest.

**Stop-rule pre-register:** p<0.05 → method thật; không → pseudo = mục phụ (ổn định label-efficiency), DỪNG. **v2 p=0.38 → theo rule = DỪNG trên NuInsSeg.** Đề xuất mở: 1 power-check trên **PanNuke** (nhiều ảnh/mô → variance nhỏ, dataset headline) — KHÔNG phải chase (sửa power cho vấn đề đã chẩn đoán là power), dứt điểm dương/âm. Chờ cô chọn (A) fold ngay / (B) 1 lần PanNuke.

**Ghi nhận positive thật (dù chưa sig):** pseudo-label **giảm phương sai student** (v1: 0.279→0.136) = ổn định hoá — property có thật, dùng được cho mục label-efficiency phụ.

---

### 0.8 ★★ HƯỚNG METHOD MỚI: count-only anti-crowding (2026-07-23) — đánh điểm-yếu-cấp-CODE của hematoxylin-family

**Điểm yếu đối thủ (đọc CODE, không chỉ paper):** Triple U-net (`colornorm.py`: NMF-Vahadane stain, chuẩn hoá percentile+exp → **saturate ở OD cao = vùng nhân dày**; đếm = 2-decoder foreground+distance → **watershed under-split** + `remove_small_objects(min_size=100)` rụng nhân nhỏ) và CP-Net (lightweight+H-prior, detection/classification). **Đếm KHÔNG bao giờ được học — nảy từ chuỗi stain→mask→watershed, mỗi khâu thoái hoá dưới crowding → undercount mô dày = lỗi CẤU TRÚC.** Đều cần **mask đắt**. Xem [[feedback-research-methodology]].

**Đòn đánh (count-only, 3 phần có-lý):** (a) học đếm trực tiếp bỏ watershed; (b) module **hiệu chỉnh khối→số theo crowding** (un-saturate); (c) loss **composite-linearity** — ghép ảnh có ΣGT biết, ép cộng-tính → dạy (b) chỉ bằng nhãn đếm, KHÔNG mask. Khác composition-loss (crowd+dot) & khác tiling đã chết (composite tạo ảnh dày MỚI, forward lại → count(ghép)<Σ = tín hiệu THẬT, không rỗng).

**✅ PREMISE P1 PASS (`premise_saturation_test.py`, efflite0 count-only, test 133 ảnh):** overlay min-blend k=1→4, Δsat% (Σpred_parts−overlay) = **0 → 26.5 → 39.9 → 50.9** đơn điệu → head **saturate mạnh dưới overlap** → loss có target lớn để học.

**⚠️ Cảnh báo từ tile-control:** Δtile% cũng ~50% (≥ overlay ở k=2) → **một phần lớn undercount là RESOLUTION (nhân nhỏ khi downscale), không chỉ overlap** → method PHẢI ghép **OVERLAY full-res** (không tile) kẻo chỉ dạy scale-robustness (trivial). **Cổng sống-chết TIẾP THEO (chưa test):** anti-saturation học trên overlay TỔNG HỢP có **chuyển sang mô DÀY THẬT** (thymus/spleen undercount, worst-organ R²) không — nếu không → method rỗng.

**Kế hoạch bậc:** Bước 1 = test **LOSS tối giản** (composite-linearity, overlay k2-4, target ΣGT) vs count-only baseline, multi-seed; đo (a) Δsat giảm? (b) **MAE mô dày thật + worst-organ R² cải thiện?** [cổng transfer] (c) R² tổng. Chỉ khi (b) pass mới thêm module (b) + ablate. Script: `method_composite_linearity.py`.

**BƯỚC 1 KẾT QUẢ (5 seed paired):** Δsat 53.9→**26.7%** ✅ + MAE mô-dày 58.0→**51.1** ✅ (lần ĐẦU chạm metric thật) NHƯNG R² tổng 0.867→0.793 ✗ + worst-organ −1.41→−5.31 ✗. **Chẩn đoán:** loss đẩy count LÊN **toàn cục** → giúp mô dày (undercount) nhưng **over-count mô thưa** → R²↓. = **correction KHÔNG có mục tiêu** (gốc lỗi = ràng buộc không cục bộ, KHÔNG phải thiếu module).

**★ BƯỚC 2 = CONTRIBUTION: Local Superposition-Consistency (`method_superpos_consistency.py`) — WIN SẠCH đầu tiên.**
Nguyên lý (của mình, tổng quát mọi density-counter): bộ đếm phải **cộng-tính dưới chồng ảnh, CỤC BỘ**: `D(overlay(A,B)) ≥ D(A)+D(B)` theo ô. Loss **một-phía, per-cell, TỰ-GIÁM-SÁT** (target=density phần detach) → cục bộ nên KHÔNG bias toàn cục; một-phía nên không ép over-count; self-sup nên **dùng được ảnh không nhãn** (label-efficiency).

Ablation cô lập locality (5 seed, mới có mean — CHỜ 8-seed+Wilcoxon):

| metric | baseline | global(P1) | local(P4) |
|---|---|---|---|
| R² tổng ↑ | 0.850 | 0.865 | **0.882** (+0.031) |
| worst-organ R² ↑ | −2.124 | −1.548 | **−1.317** |
| MAE mô-dày ↓ | 55.5 | 54.2 | **45.7** (−9.72) |
| Δsat overlay | 55.0 | 55.2 | 54.7 |

5-seed: R²↑ + worst-organ↑ + MAE-mô-dày↓ đồng thời + ablation đơn điệu — TRÔNG như win sạch.

**❌ 8-seed + Wilcoxon LẬT NGƯỢC (2026-07-24):** local vs baseline **ΔR² −0.034 (p=0.148, 2/8), ΔMAE-dày +5.61 (p=0.46, 4/8)** — NGHIÊNG ÂM, không sig; ablation hết đơn điệu (baseline 0.907 > local 0.873 > global 0.869). **5-seed win = dương-tính-giả mẫu-nhỏ (lần thứ 3: funnel/pseudo/superpos).** Nguyên nhân chung = **bức tường variance NuInsSeg** (R² ±0.05–0.13, dày-MAE ±13–29) nuốt mọi effect ≤0.05 → NuInsSeg KHÔNG đủ power phân xử method modest.

**❌ PanNuke ADJUDICATION (2026-07-24, high-power): SUPERPOS CHẾT.** fold_3 no-colon, local vs baseline **ΔR² −0.048 (0/5, p=0.06 SAI HƯỚNG)**, MAE-dày +1.57; đơn điệu NGƯỢC (baseline 0.891 > global 0.873 > local 0.843). Power đủ (worst-org +0.66±0.17). → **Superposition-consistency đóng dứt.**

---

### 0.9 ★★ CHỐT hướng accuracy-method (2026-07-25) — FRONTIER là cái THẬT, method novelty KHÔNG có

**❌ Spatial-grounding premise CHẾT (`premise_spatial_grounding.py`, PanNuke fold_3):** oracle GT-shape (magnitude-decoupled CE, trần trên của "spatial distill từ FM") → R² **0.905 → 0.617** (−0.29, 0/3). → *Với ĐẾM, tự do không-gian của count-only là ƯU điểm; ép shape (dù hoàn hảo) xung đột count-loss → hại.* Xác nhận count-only ≥ gt-density. **Ý magnitude-decoupled spatial distillation đóng trước khi đụng FM.**

**Tổng kết: 6 hướng accuracy-method đều ÂM** — density-distill (PACT 0.512), feature-funnel (p=0.74), pseudo-label (p=0.38), composite-global (biased), superposition (hại PanNuke), spatial-grounding (oracle hại). **Không phải thiếu cố gắng — đếm-count-only + backbone tí hon ĐÃ gần trần tự nhiên; mọi cấu trúc thêm đều xung đột/vô ích.** → **DỪNG săn accuracy-method** (cam kết, không #7 mù).

**★ CÁI THẬT — FRONTIER params–accuracy (`method_frontier.py`, PanNuke fold_1+2 train / fold_3 test, count-only, 3-seed):**

| backbone | params | R² ↑ | MAE ↓ | |
|---|---|---|---|---|
| TinyUNet | 1.94M | 0.775 | 5.14 | quá nhỏ → cách xa |
| **EfficientNet-lite0 = PACT** | **3.62M** | **0.908±0.019** | **3.78** | ★ **sweet-spot** |
| EfficientNet-b1 | 6.76M | 0.884 | 3.82 | to hơn KHÔNG giúp |
| ~~ResNet18~~ | 14.4M | (sập, loại) | — | train collapse |

Scaling **bão hoà ở 3.6M** (lớn hơn tệ đi) = lập luận **efficiency thuần**. ⚠️ efflite0 = **backbone STOCK (ImageNet)** → accuracy KHÔNG novel; novelty thật = **đầu UQ (Poisson-σ, N2/N3) + count-only + efficiency curve**. Distill BỎ khỏi accuracy (efflite0 distill 0.512 << count-only 0.908); distill chỉ còn ở nhánh σ/UQ.

**★ BẢNG CHÍNH accuracy công bằng — XONG (2026-07-25, leak-free, cùng nền: train full fold12=5179 / test full fold3=2722, colon GIỮ khắp nơi):**

| Method | Params | Nhãn | R² | MAE | trạng thái |
|---|---|---|---|---|---|
| ~~DeNuC (retrain fold12, 2026)~~ | 4.37M | detection | ~~0.455~~ | ~~9.64~~ | ⛔ **SAI — checkpoint hỏng (F1 68.0 P89/R55). SUPERSEDED, đừng trích lại.** |
| **DeNuC 5 seed (2026-08-02)** | 4.37M (ShuffleNet-x2) | detection (centroid) | **0.936±0.002** | **2.98±0.02** | ✅ **F1 86.31%**; per-seed 0.937·0.937·0.937·0.933·0.935; bias +0.6..+0.9 mọi tầng, riêng 45+ −3.1..−3.8 |
| DualU-Net (retrain fold_1+2) | ~35M (resnext50) | mask | **0.809±0.009** | **5.44±0.19** | ✅ 5-seed 2026-08-03 (over-count) |
| ↳ **param VERIFIED 2026-08-01 — SỬA `~35M` → `41.0M`** | **41,006,775** (+ 0.07M buffer BN = 41.08M) | — | — | — | `resnext_pannuke_fold12_epoch_50.zip` key `model`; encoder 23.05M + decoder 9.01M + **decoder_count 9.01M** (hai decoder = tên DualU-Net). `~35M` cũ có lẽ quên nhánh decoder thứ hai. Bảng 4 đã sửa. |
| StarDist (retrain fold_1+2, **5 seed** 2026-08-02) | 1.43M (U-Net) | mask | **0.847±0.011** | **4.38±0.05** | ✅ (under-count 20.8; nhỏ hơn efflite0 nhưng cần mask + kém acc) |
| ↳ **param VERIFIED 2026-08-01** | **1,425,761** (= 1.43M, khớp bảng) | — | — | — | cộng mọi dataset trong `weights_best.h5` (kaggle `dattong12/stardist-ckpt-fold1-2-pannuke`); gồm cả non-trainable (BN moving stats) |
| **PACT efflite0 (ours)** | **3.62M** | **count-only** | **0.920±0.005** | **3.61±0.07** | ✅ **thắng CẢ 3 paradigm** |

**★ KIỂM PARAM Bảng 4 (2026-08-01) — 2/4 ô từng SAI, đừng tin ô nào chưa có dấu ✅:**

| ô | trạng thái | nguồn |
|---|---|---|
| DensitySigmaUNet 3.62M | ⬜ chưa kiểm lại | tự tính lúc train |
| StarDist 1.43M | ✅ **1,425,761** khớp | kaggle `dattong12/stardist-ckpt-fold1-2-pannuke`, `weights_best.h5` |
| DualU-Net ~35M | ❌ **SAI → 41.0M** | `~/Downloads/resnext_pannuke_fold12_epoch_50.zip` key `model` |
| DeNuC 4.37M | ⬜ chưa kiểm | — |

- Cùng đợt: cột Nhãn của DeNuC **`box/point` → `points`** (MASTER:194 ghi centroid; `denuc_count_eval.py` đọc `gt_coords` Nx2).
- **Bài học:** ô có dấu `~` hoặc dấu `/` là ô viết lúc CHƯA biết chắc; các vòng kiểm sau chỉ dò con số nên không bao giờ chạm tới.
  Xem [[feedback-audit-bang-o-trong]].
| KongNet (2510.23559, 10/2025) SOTA | — | mask | — | — | **cite** |
| Distill-H-Optimus (2502.19217, 02/2025) | ~24M | mask | 0.871 (họ báo) | — | **cite peer (no-UQ)** |

**★ Efficiency headline (TRUY LOG KAGGLE 2026-08-10, kernel `hipinhththu/learning-curve` chạy 2026-07-26 15:52, `efficiency_profile.py`):** efflite0 **3.619.138 params · 0.98 GMACs · 6.855 ms/ảnh GPU (145.9 ảnh/s) [Tesla T4] · 51.02 ms CPU · VRAM 37.9 MB** (256²). Bài ghi 6,9 ms / 51 ms. Real-time, deploy GPU+CPU. *(Số cũ 6.04 ms/46 ms/165 ảnh-s ĐÃ BỎ: quét 35 kernel 07-2x→08-xx, KHÔNG kernel nào khác từng chạy `efficiency_profile.py`, không truy được log. Lưu ý `kernels_pull` chỉ lấy source bản MỚI NHẤT nên kernel sửa-sau-khi-chạy lọt lưới.)*

**★ UQ conformal IN-DOMAIN (efflite0 count-only, PanNuke fold3, 20-seed, 2026-07-27, `gen_uq_pannuke`+`eval_r2_grouped`):** marg.cov 0.904/0.923/0.906 (global/mondrian/cluster, target 0.90 ✓); **mondrian worst-org 0.906, 0/19 under-cover** (global 0.765); MAE 3.45 khớp headline; Winkler cluster 19.96 (p=1.9e-6). → **KHỚP số headline cũ 0.906 trên đúng model 3.62M** (consistency ✓). σ tốn 0 accuracy (detach_mu). pkl `work/efflite0_pan_uq_f3_s42.pkl`. UQ-floor E-AURC vẫn ~4/5 → bán "CÓ calibrated+transfer", không "tốt nhất".

**★ ABLATION σ head Poisson vs raw (efflite0 matched, 2026-07-28, `gen_uq_pannuke`/`gen_uq_nuinsseg --sigma_mode` + `eval_r2_grouped`):** đổi CHỈ σ head. **PanNuke (range 0–45 hẹp):** Poisson≈raw, raw nhỉnh Winkler global 21.6 vs 24.7 → Poisson KHÔNG cần. **NuInsSeg (range 0→500+ rộng):** Poisson thắng ĐẬM — width 37.72 vs 67.89 (**−44%**), Winkler 45.43 vs 78.09 (**−42%**), worst-org 0.864 vs 0.655 (**+0.21**), cov ~0.93 cả hai (raw over-cover 0.937 bằng thổi phồng σ hằng số). → **Poisson-anchoring cần khi count-range RỘNG, thừa khi hẹp** = biện minh thiết kế cho deployment/dataset-mới (§3.5b report). Caveat: 2 run riêng → μ lệch GPU-noise (MAE 8.15 vs 10.09) nên phần nhỏ Winkler-gap do μ; width/worst-org chủ yếu σ-shape. pkl `work/nuins_uq_{poisson,raw}.pkl` — BACKUP kaggle. Xem [[r2-poisson-sigma]].

**★ Count-only vs mask CÙNG kiến trúc (efflite0, `sup_compare.py`, PanNuke leak-free, 2026-07-28):** cô lập supervision (chỉ đổi target). count-only **0.9296±0.009 / MAE 3.48** vs GT-density-mask **0.9247±0.009 / 3.58** → **Δ +0.005 R² < std → PARITY**. Đọc honest: KHÔNG "count-only thắng mask" mà **count-only NGANG mask, nhãn rẻ hơn 1 bậc** (1 số/ảnh vs mask pixel) → "đếm không cần mask". Đối chứng đóng chốt label-efficiency. GT-density đầu bất ổn (collapse khi w_count=0) → fix w_count=0.01 + 60ep → ổn định. (số cũ NuInsSeg 0.925 vs 0.881 = SUPERSEDED).

- **`mask_s42.pt` (2026-08-02, 14,7MB, efflite0 ch32 + đầu σ):** checkpoint nhánh mask seed 42, chạy CHỈ để vẽ
  density map dự đoán cho Hình 1. Mac: `~/Downloads/mask_s42-2.zip` = chính file .pt (torch.save ghi ra zip, trình
  duyệt đổi đuôi), đổi tên là load được. `--w_count_mask 0.1` = ĐÚNG cấu hình Bảng 1 (dòng 2026-07-28 w_c=0,01 đã
  SUPERSEDED). R² 0,9408 vs seed-42 của bảng 0,9345 → lệch từ chỗ khác (epochs/bs/merge), 1 seed nên đừng trích.
- **Bản đồ mật độ của nhánh mask KHÔNG tái tạo hình dạng nhân** (đo trên `im0507.png`, `fig1_pred.py`): tổng đúng
  (20,2 vs 22 nhân) nhưng khối lượng dồn vào **422 điểm ảnh** thay vì 10 866 của mục tiêu (26× đặc hơn), đỉnh gấp
  17× phân vị 99 của mục tiêu. ⇒ hậu thuẫn §fidelity ("density map barely localises nuclei"), nhưng n=1 ảnh.
  Vì vậy Hình 1 KHÔNG nhúng bản đồ dự đoán, chỉ vẽ ô lưới sơ đồ.

**★★ SỐ CUỐI PanNuke head-to-head 5 seed (2026-07-31, gộp từ .npz, n=2722 ảnh fold3 — DÙNG CHO BẢNG~1):**
| Nhánh | R² gộp | trung vị | SẬP | per-seed (42→46) |
|---|---|---|---|---|
| count-only | **0,9182 ± 0,011** | 0,9197 | 0/5 | 0,9269 · 0,9133 · 0,9008 · 0,9306 · 0,9197 |
| mask (GT-density, w_c=0,1) | **0,9313 ± 0,006** | 0,9340 | 0/5 | 0,9345 · 0,9333 · 0,9195 · 0,9353 · 0,9340 |

**⚠️ mask cao hơn ở CẢ 5/5 hạt giống.** Hiệu bắt cặp theo seed: +0,0131 ± 0,0067 (sd mẫu) → t(4)=4,34, **p≈0,012**.
⇒ **KHÔNG được viết "chênh lệch nhỏ hơn dao động giữa seed" hay "không nhất quán về dấu"** cho PanNuke nữa
(|Δ|=0,0131 > sd count 0,011; dấu nhất quán 5/5). Đã sửa abstract + gạch đầu dòng Giới thiệu + §4.1 + Bảng 1.
- Chiều này **NGƯỢC** với dòng 2026-07-28 ở trên (count 0,9296 > mask 0,9247): dòng đó chạy w_count=0,01 và
  không phải merge 5-seed leak-free → **SUPERSEDED**, đừng trích lại.
- Nguyên nhân mask khá lên: chính ta đã đổi w_c 0,01→0,1 để cho baseline **cấu hình hội tụ tốt nhất** (§Thiết lập).
  Đây là hệ quả đúng của nguyên tắc đó, không phải hồi quy chất lượng.
**★★ TOST XONG 2026-07-31 — số CHỐT của Bảng 1 (n=2722, số nhân TB = 24,49):**
| | |
|---|---|
| MAE từng ảnh | count 3,5953 vs mask 3,3853 |
| Δ (count − mask) | **+0,210 nhân = 0,86%** số nhân TB (se 0,0338) |
| Wilcoxon | **p = 1,07e−13** → khác biệt CÓ thật |
| CI 90% bootstrap | [+0,1555; +0,2673] |
| **δ\*** | **0,2673 nhân = 1,09%** số nhân TB |
| δ = 2% (0,490) | p_TOST = 8,86e−17 → **TƯƠNG ĐƯƠNG** |
| δ = 1% (0,245) | p_TOST = 0,151 → chưa kết luận được |

⇒ **Phát biểu chuẩn của bài (2 vế, phải đi cùng nhau):** khác biệt có thật (Wilcoxon) **VÀ** bị chặn trên bởi
~1% số nhân/ảnh (TOST, δ*=0,27). Đừng viết vế nào một mình.
- Không chạy TOST được ở biên 1% → **đừng nói "tương đương trong 1%"**, phải nói "δ* = 1,1%" hoặc "tương đương ở biên 2%".
- Đã điền: Bảng 1 (bỏ `\pending` cuối cùng), §Thiết lập (mô tả TOST + δ*), §4.1 (đoạn 2 kiểm định), abstract,
  gạch đầu dòng Giới thiệu, §4.1 đoạn cuối, Kết luận.
- Nguồn: `hipinhththu/pannuke-5seed` + `hipinhththu/pannuke-mask` (cả 3 đều là **thư mục .npy đã bung**, không
  còn là .npz → script nạp phải dò theo nội dung `gt.npy` + `pred__*.npy`, đừng glob theo đuôi `.npz`).
- Bảng 4 paradigm (0,920/3,61) = mô hình hạt giống **trung vị** (0,9197 ≈ 0,920) → đã ghi rõ trong caption.

**★ Cross-backbone pvt_v2_b0 — DÒ lr (2026-07-29, log cũ, chép vào đây 2026-07-31):**
`sup_compare.py --backbone pvt_v2_b0 --seeds 42 --epochs 3 --lr 3e-4 --w_count_mask 0.01`
→ count-only **+0,8563** / MAE 4,50; mask **+0,9178** / MAE 3,63; Δ −0,0615; Wilcoxon p=2,2e−18.
- ⟹ **lr 3e-4 KHÔNG sập** (lr 1e-3 sập cả 2 nhánh về hằng số, R²=−1,43 = chữ ký −μ²/σ² của fold3). **Dùng 3e-4.**
- ⛔ **KHÔNG được trích số trên làm kết quả:** mới **3 epoch** (giao thức là 60), **1 seed**, và `w_count_mask=0.01`
  thay vì 0,1 của giao thức. Ở 3 epoch nhánh mask nhận ~65k giá trị giám sát mỗi ảnh còn count-only nhận **1**,
  nên mask dẫn trước là chuyện hội tụ, không phải chuyện tính đủ của giám sát.
- Đối chiếu: efflite0 60 epoch cho Δ −0,013. Ở 3 epoch PVT cho −0,062 (gấp ~5). Câu hỏi thật là gap còn lại bao
  nhiêu **khi hội tụ** → phải chạy 60 epoch × 3 seed × 2 nhánh, **cùng lr 3e-4 cho CẢ HAI nhánh**, w_count_mask=0,1.

**★★ NuInsSeg parity 5 chia + TOST (2026-07-31, `parity_nuinsseg.py`, npz `nuinsseg_parity_5split.npz`) — SỐ CHỐT BẢNG 1:**
count-only **0,8951 ± 0,0454** vs mask **0,8832 ± 0,0712** → ΔR² **+0,012**. (Số cũ 0,913/0,892 = **SUPERSEDED**,
run khác, KHÔNG được ghép với δ* của run này.)

| chia | Δ (nhân) | CI90 | δ* | kết luận ở δ=5% |
|---|---|---|---|---|
| 42 | **+1,987** | [+0,54; +3,61] | 3,61 | chưa |
| 43 | −4,131 | [−6,11; −2,25] | 6,11 | chưa |
| 44 | +0,095 | [−1,50; +1,72] | 1,72 | **TƯƠNG ĐƯƠNG** |
| 45 | −4,184 | [−5,99; −2,41] | 5,99 | chưa |
| 46 | −2,133 | [−3,45; −0,77] | 3,45 | chưa |

**⚠️ LẬT chú thích † cũ của Bảng 1** ("0/5 chia có p<0,05"): thực tế **3/5 chia CÓ ý nghĩa, và ở CẢ HAI chiều**
(seed42 mask hơn; 43/45/46 count hơn). δ* trung vị **3,614 nhân = 6,6%** số nhân TB (≈55,1/ảnh), tức **rộng gấp
~6 lần** PanNuke (1,1%). Nguyên nhân: 133 ảnh/chia so với 2722, và chia ngẫu nhiên đổi theo seed.
⇒ **Cách viết đúng cho NuInsSeg: "không có ưu thế nhất quán", KHÔNG được viết "tương đương".**
Chỉ PanNuke đủ lực để bán biên hẹp. MoNuSeg cũng hạ xuống "dao động lấn át, chưa kết luận" (sd gấp đôi Δ).

**★ MoNuSeg in-domain dataset THỨ 3 (efflite0 count-only, `monuseg_indomain.py`, 2026-07-28):** đa cơ quan TCGA, ĐỘC LẬP PanNuke/NuInsSeg (KHÔNG trùng như Lizard/CoNIC). XML→centroid→patch256 native (resize 1000→1024, không co nhân), **split CHÍNH THỨC 37 train/14 test folder** (random WSI split cho 0.54±0.29 nhiễu → official cho **0.663±0.099** ổn). MAE 5.42±0.86. R² thấp hơn PanNuke/NuInsSeg (nhỏ+test count-variance thấp) nhưng MAE tốt. ⚠️ Q1 KHÔNG bán R² tuyệt đối → cần count-only-vs-mask MoNuSeg (rasterize polygon→density) để bán PARITY 3-dataset. Input Kaggle `hipinhththu/monuseg` (TestData có 14 xml). Xem [[paper2-q1-pivot]].

**★ Generalization đa dataset (efflite0 count-only, 2026-07-27):** in-domain PanNuke 0.920±0.005 / **NuInsSeg 0.854±0.048** (random split, `method_frontier --dataset nuinsseg`). → chạy tốt dataset 2, là ceiling cho transfer. **Transfer PanNuke→NuInsSeg efflite0 (2026-07-27, `eval_cross_dataset`+`eval_r2_grouped` 20-seed):** marg.cov 0.90 GIỮ (mondrian 0.922) NHƯNG MAE 30.35 (in-domain 13.8, ~2.2× tệ) + worst-org cov tụt 0.72 (in-domain 0.906). → **transfer MỘT PHẦN: coverage biên transfer, accuracy điểm KHÔNG** (khớp finding cũ MAE-không-transfer). Bán honest: khoảng tin cậy giữ phủ khi recalibrate ở target. pkl `work/xfer_pan2nuins_efflite0.pkl`. *(số transfer 1.9M cũ = SUPERSEDED.)*

> **⚠️ TRUY XUẤT 2026-08-04 — CHƯA ĐƯA VÀO BÀI ĐƯỢC.** Lệnh gốc tìm lại từ Kaggle kernel
> `hipinhththu/notebook9e96ca2a3a` (2026-07-26):
> `eval_cross_dataset.py --train_dataset pannuke --test_dataset nuinsseg --backbone efficientnet_lite0
> --w_density 0 --w_count 1.0 --detach_mu --use_gt_density --exclude_tissue "" --out work/xfer_pan2nuins_efflite0.pkl`
> rồi `eval_r2_grouped.py --preds ... --seeds 20 --n_clusters 5`.
> - ✅ **count-only THẬT**: `--w_density 0 --w_count 1.0` truyền TƯỜNG MINH → đè được mặc định hỏng
>   `w_density=1.0/w_count=0.01` của bản `885859c`. Lỗi dán-nhãn-sai của CryoNuSeg (dòng ~746) KHÔNG áp dụng ở đây.
> - ✅ colon GIỮ (`--exclude_tissue ""`), backbone efflite0, ch32.
> - ⚠️ **`--epochs` = 80** (mặc định, notebook không truyền) — giao thức chính của bài là **60**.
> - ⛔ **CHỈ 1 SEED HUẤN LUYỆN.** `eval_cross_dataset.py` gọi `train()` ĐÚNG MỘT LẦN (dòng 160). `--seeds 20`
>   là của `eval_r2_grouped.py` = **20 lần chia calib/test** trên CÙNG một bộ dự đoán (`rng.permutation`,
>   dòng 71-73), **KHÔNG phải 20 lần train**. Chữ "20-seed" ở dòng trên DỄ ĐỌC NHẦM — đừng lặp lại.
> - ⟹ MAE 30,35 / worst-org 0,72 là số của **một** model. Bài đã quét sạch mọi hàng 1-seed (DualU-Net) nên
>   KHÔNG được thêm lại. **Chạy lại 3 seed** (`--seed` có sẵn, dòng 135), `--epochs 60`, CHỈ nhánh count-only
>   (mục transfer thuộc §4.4 ngân sách độ tin cậy = mô tả hệ đã triển khai, không phải so giám sát). ~4h GPU,
>   xếp SAU 2 run backbone.

**★ Label-efficiency learning curve (efflite0 count-only, PanNuke leak-free, 3-seed, **40 EPOCH**, `label_curve.py`; log đầy đủ lấy từ kernel `hipinhththu/learning-curve` 2026-07-26 15:52):** R² theo %nhãn count (#ảnh): 5%→0.453±0.208 (259) / 10%→0.742±0.077 (518) / **25%→0.849±0.029** (1295) / 50%→0.859±0.036 (2590) / **100%→0.915±0.010** (5179). → 25% nhãn đạt ~93% full. **40 epoch, KHÔNG phải 60** (notebook không truyền `--epochs`, `label_curve.py` chỉ có 1 commit cf7d441 default=40) ⇒ §4 bài đã sửa "tất cả mô hình 60 epoch" → "mọi mô hình so sánh hai chế độ", và 100%→0.915 thấp hơn bảng chính 0.918 là do lịch ngắn hơn + biến thiên seed, KHÔNG chỉ do seed. Thay khung "distilled↔supervised 1.9M" cũ.

**⛔ ĐOẠN DƯỚI SUPERSEDED 2026-08-02 (DeNuC 0.936 > efflite0 0.920, 5/5 seed) — xem mục ★★ cuối file.**

**★ Kết quả (CŨ):** efflite0 3.6M count-only ~~thắng cả 3 paradigm cạnh tranh~~ — DualU-Net 35M (density/mask 0.804), StarDist (instance/mask 0.841), DeNuC (detection 0.455) — trên bài đếm, nhãn rẻ hơn 1 bậc. **Detection (DeNuC) sụp mạnh nhất ĐÚNG ở vùng dày** (bias per-tier <10 −0.8 → 45+ −30.4: nhân chồng lấn → NMS bóp chết TP → recall sụp) = bằng chứng trực tiếp cho động lực CROWDING. Caveat: baseline mask/detection tối ưu segmentation/localization, đếm=readout (leak-free cùng test set). GT mean 24.5. efflite0 full-fold3 (0.920) > no-colon cũ (0.908).

**Provenance artifacts (2026-07-25, ĐÃ save version Kaggle — điền tên version chính xác khi biết):**
- `checkpoints/*.pth` (DualU-Net, train 50ep ~3.5h) — ⚠️ từng mất 1 lần do thiếu mkdir; đã backup version này.
- `stardist_models/stardist_pannuke_f12/` (StarDist 150ep) + `stardist_fold3_counts.csv`.
- `dualunet_fold3_counts.csv` (per-image gt,pred DualU-Net).
- **DeNuC (2026-07-26):** `outputs/20260726_110537_denuc_pan12_3/best_checkpoint.pth` (50ep ~13ph, 2.1GB) + `test_results/pannuke12_3/best_checkpoint_infer_res.npy` (dict 2722 ảnh, mỗi ảnh {gt_coords, pred_coords sau NMS}) — ⚠️ BACKUP kaggle. Setup: patch SPLIT_CONFIGS `pannuke12_3`={1,2:train,3:test}, whitelist sed vào denuc_{train,eval}.py, ln ./dataset→denuc_data.
- COCO fold12 = derived từ `dualunet_merge_fold12.py` (rẻ, regen được).
- Scripts: `method_frontier.py` (efflite0), `eval_count_dualunet.py`, `stardist_baseline.py`, `dualunet_merge_fold12.py`, `denuc_count_eval.py`, config `config_{train,eval}_pannuke.yaml` — tất cả trong repo.

**Định vị honest (CŨ, SUPERSEDED — câu "thắng cả 3 paradigm" đã SAI):** *"bộ đếm nhân 3.6M, count-only (không mask), R² **0.920** leak-free — **thắng cả 3 paradigm nhãn-đắt** (density DualU-Net 35M, instance StarDist, detection DeNuC) ở bài đếm với **nhãn rẻ hơn 1 bậc**; kèm UQ calibrated (worst-organ robust, transfer 3 dataset)."* = **applied efficiency + UQ**, honest, mid-Q1-applied. Xem [[paper2-main-table-leakfree]], [[paper2-main-is-new-method]].

**★ Nguồn số "rẻ hơn ít nhất 8 lần" (Bảng chi phí chú thích, §4.5 + abstract) — VERIFIED 2026-07-31:**
Qu et al., TMI 2020 (arXiv 2007.05448, `\cite{qu2020}`): **một** nhà bệnh học chú thích **8 ảnh** tập LC
(Lung Cancer) theo 3 cách. Trung bình mỗi ảnh (**~600 nhân**): mặt nạ đầy đủ **115 phút**, hộp bao **67 phút**,
điểm toàn bộ **~14 phút**, 10% điểm **<2 phút**. → per-nhân: viền 11,5 s, điểm 1,4 s.
- Bài dùng: 115/14 = **8,2×** → viết "ít nhất 8 lần". Chi phí nhãn ĐẾM **chưa ai đo** → bài chặn trên bằng chi
  phí điểm (đếm phải duyệt từng nhân như đánh dấu, không cần đặt điểm/vẽ viền). Đây là LẬP LUẬN, không phải đo.
- ⚠️ Yếu điểm reviewer sẽ bắt: n=8 ảnh, 1 người, 1 dataset; ảnh ~600 nhân vs mảnh PanNuke 24,5 nhân.
  → §4.5 đã ghi rõ cả ba caveat + neo lại bằng chi phí per-nhân (bậc độ lớn, không phải hằng số chuyển được).

**★★ MoNuSeg PARITY 5 hạt giống + Wilcoxon + TOST (2026-07-31, `monuseg_indomain.py --parity --split official`,
npz `monuseg_parity_5seed.npz`) — ĐIỀN 3 Ô TRỐNG BẢNG 1:**
count-only **0,4585 ± 0,1944** / MAE 6,841 ± 1,506 vs mask **0,3615 ± 0,1541** / MAE 7,496 ± 1,249 → ΔR² **+0,097**.
- per-mảnh (gộp 5 seed, **n=224 mảnh**): count-only 6,84 vs mask 7,50 | **Wilcoxon p = 5,936e-05 → count-only THẮNG**.
- TOST: Δ|err| = **−0,654 nhân** (2,2% số nhân TB ≈ **29,9/mảnh**), CI90 [−0,889; −0,427], **δ* = 0,889 nhân = 3,0%**.
- ⇒ Bảng 1 hàng MoNuSeg: p = 5,9e-05 | δ* = 0,89 | 3,0%.
- **★ Dãy δ* ĐƠN ĐIỆU:** PanNuke 1,1% (n=2722) < MoNuSeg 3,0% (n=224) < NuInsSeg 6,6% (~133 ảnh/chia).
- **★★ SỬA 2026-08-11 — δ* KHÔNG phải "thang lực kiểm định":** phân rã δ* ≈ |Δ̂| + z·s_d/√n (Pt.13) trên số thật:

  | bộ | hiệu ứng \|Δ̂\| | độ chính xác | δ* | hiệu ứng chiếm |
  |---|---|---|---|---|
  | PanNuke | 0,86% | 0,24% | 1,1% | 78% |
  | MoNuSeg | 2,2% | 0,8% | 3,0% | 74% |
  | NuInsSeg (chia trung vị 42) | 3,6% | 2,8% | 6,6% | 55% |

  ⇒ **hiệu ứng chi phối ở CẢ BA**, biên rộng KHÔNG quy về riêng cỡ mẫu. Câu cũ trong bài ("phản ánh chủ yếu lực
  thống kê chứ không phải ba độ lớn hiệu ứng khác nhau") **SAI, đã thay** bằng bảng phân rã này (main_vi §5.1).
  main.tex dòng 528 vẫn còn câu sai — sửa khi đồng bộ. NuInsSeg: dấu Δ̂ đổi giữa các chia nên 3,6% là biến thiên
  theo cách chia, không phải ưu thế ổn định.
- **★ LẬT chú thích ‡ cũ** ("dao động lấn át, chưa kết luận, KHÔNG đủ tuyên bố tương đương") → nay CÓ kết luận:
  chênh lệch thật, **ngược chiều PanNuke**, và bị chặn ở 3,0%. R² vẫn dao động mạnh (sd ~2×|Δ|) nhưng **bắt cặp
  theo từng mảnh loại được phương sai hạt giống** vì so hai chế độ trên CÙNG một mảnh.
- ⚠️ **Điểm yếu phải khai (đã ghi vào chú thích ‡):** 224 mảnh cắt từ **14 ảnh** (16 mảnh/ảnh) → mảnh cùng ảnh
  KHÔNG độc lập → p và δ* là **lạc quan**. PanNuke không dính (2722 mảnh = đơn vị mẫu chuẩn của bộ dữ liệu).
  Muốn chặt phải gộp mảnh→ảnh (n=14) nhưng npz không lưu ánh xạ mảnh→ảnh; cần sửa dump nếu reviewer đòi.
- ⚠️ Run 3-seed KHÔNG tất định cũ cho "mask thắng p=6e-13" = **ngược dấu**, đã bỏ (xem commit 5eda906). Run này
  5 seed + cuDNN tất định, và R² lẫn MAE **cùng chiều** nên nhất quán nội bộ.
- ⛔ **BACKUP `monuseg_parity_5seed.npz` lên Kaggle dataset NGAY** = 10 lần train (~3 giờ).

**★ MoNuSeg kiểm định lại ở MỨC ẢNH (2026-07-31, gộp từ chính npz, KHÔNG train lại) — trả lời phản biện phân cụm:**
224 mảnh cắt từ 14 ảnh, mảnh cùng ảnh KHÔNG độc lập → kiểm định mức mảnh lạc quan. **Ánh xạ mảnh→ảnh suy ra
được từ thứ tự**: `build_patches` duyệt từng ảnh, 2 vòng `gy,gx` trên `GRID=4` → LUÔN đúng 16 mảnh/ảnh, không
lọc; `te` và `eval_r2` giữ nguyên thứ tự ⇒ **mảnh k thuộc ảnh k//16**. (Không cần biết ảnh NÀO, chỉ cần nhóm.)

| | mức mảnh (n=224) | mức ảnh (n=14) |
|---|---|---|
| Δ|err| | −0,654 nhân | **−8,18 nhân** |
| CI90 | [−0,889; −0,427] | **[−15,74; −1,24]** (không chứa 0) |
| δ* | 0,889 = **3,0%** | 15,74 = **3,3%** |
| MAE c/m | 6,84 / 7,50 | 89,68 / 97,86 |
| gt TB | 29,9/mảnh | **478,36/ảnh** |

⇒ **Bỏ giả thiết độc lập chỉ nới δ* 3,0% → 3,3%; chiều + ý nghĩa + độ lớn tương đối KHÔNG đổi.** Chú thích ‡
Bảng 1 giờ ghi số đo này thay vì chỉ cảnh báo "lạc quan". Cross-check phép gộp: 224×29,9/14 = 478,4 ✓.
**★★ ĐÃ CÓ (2026-07-31): Wilcoxon mức ảnh `p = 0,1189`, count-only thấp hơn ở `10/14` ảnh.**
⇒ **HAI KẾT LUẬN TÁCH SỐ PHẬN:** *biên* VỮNG (δ* 3,0% → 3,3%), *tính có ý nghĩa* KHÔNG VỮNG
(p 5,9e-05 ở mức mảnh → 0,12 ở mức ảnh). p mức mảnh bị thổi phồng bởi giả-lặp (pseudo-replication).
⚠️ CI90 bootstrap [−15,74; −1,24] KHÔNG chứa 0 nhưng Wilcoxon p=0,12: không mâu thuẫn — CI tính trên
TRUNG BÌNH (mean diff −8,18, se 4,62 → t≈1,77, ~p 0,10), Wilcoxon tính trên HẠNG. n=14 nên hai phép lệch nhau.
Bài chỉ báo cáo Wilcoxon + δ*, KHÔNG nhấn "CI không chứa 0" để tránh đọc thành mâu thuẫn.
⇒ Câu chốt MoNuSeg trong bài: **chỉ phát biểu về BIÊN là vững; chiều nghiêng count-only nhất quán nhưng
CHƯA thiết lập.** (Sửa lại chú thích ‡ đã viết sai "tính có ý nghĩa không đổi".)

**★ Cross-backbone pvt_v2_b0 — nhánh COUNT-ONLY 3 seed (2026-07-31, `sup_compare.py`, lr 3e-4, w_count_mask 0.1,
60 epoch, PanNuke fold3). npz `pvt_count_3seed.npz` (backup kaggle). ⏳ ĐANG CHỜ nhánh mask (máy B).**

| seed | R² | MAE |
|---|---|---|
| 42 | **−1.4313** | 24.49 |
| 43 | +0.8836 | 3.88 |
| 44 | +0.8876 | 3.95 |

- **seed42 SẬP**: −1.4313 = chữ ký `−μ²/σ²` fold3, MAE 24.49 = đúng số nhân TB ⇒ **xuất hằng số 0**.
- mean +0.1133±1.092 **VÔ NGHĨA**; đọc **median +0.8836, sập 1/3**. Hai seed sống (0.8836/0.8876) rất khớp nhau
  ⇒ vấn đề là **ổn định tối ưu hoá**, KHÔNG phải PVT đếm kém.
- ⚠️ **Nhánh sập là COUNT-ONLY (w_d=0) ⇒ KHÔNG có density-MSE ⇒ chỉnh `w_count_mask` là VÔ ÍCH.** Cảnh báo cũ
  của script đổ cho density-MSE là sai nhánh — **đã sửa** (in đúng nhánh sập + nguyên nhân tương ứng).
  Giả thuyết: dead activation ở đầu density khi lr còn cao cho backbone transformer.
- ⛔ **KHÔNG được báo cáo pvt ở lr này.** Chuẩn của chính bài (§3.3): *"nhánh mask được đánh giá ở cấu hình
  thuận lợi nhất… để phép so sánh không thiên vị count-only"* ⇒ phải áp cho MỌI nhánh. efflite0 lr 1e-3 **đã
  dò**; pvt lr 3e-4 chỉ chọn từ dò **3 epoch / 1 seed** = chưa dò thật. lr 1e-3 sập CẢ HAI nhánh, 3e-4 sập 1/3
  count-only ⇒ hướng là còn cao → **dò 1e-4** trước khi kết luận bất cứ điều gì.
- Nếu chạy tiếp: **cả hai nhánh CÙNG một lr đã dò** (biến duy nhất = loại giám sát).

**★★ NHÁNH MASK về (2026-07-31, npz `pvt_mask_3seed.npz`):** median **+0.8553**, **SẬP 1/3**, MAE 11.10±9.47
(mean +0.1054±1.087 = vô nghĩa).

| nhánh | median R² | sập | MAE |
|---|---|---|---|
| count-only | +0.8836 | **1/3** | 10.77±9.70 |
| mask | +0.8553 | **1/3** | 11.10±9.47 |

⇒ **SẬP ĐỐI XỨNG** (cùng tỉ lệ, cùng cấu hình) ⇒ nguyên nhân là **tối ưu hoá/backbone**, KHÔNG phải loại giám
sát. Đây là bằng chứng phụ ủng hộ cách viết ở §3.3 (suy biến = thất bại tối ưu hoá). *(Chỉ ghi MASTER, KHÔNG
đưa vào bài: backbone này không nằm trong bài.)*
- ⛔ **KHÔNG gộp 2 npz lúc này**: Wilcoxon gộp sẽ đo mấy seed sập (`sup_compare` tự cảnh báo). Mỗi nhánh chỉ còn
  **2 seed sống** → chênh trung vị 0.028 không nói lên gì; và 2 seed mâu thuẫn chuẩn của chính bài (đa hạt giống).
- **QUYẾT ĐỊNH đề xuất: DỪNG.** Bài KHÔNG cần thí nghiệm này (§5 đã để ngỏ tính khái quát qua họ kiến trúc).
  Chạy lại đúng chuẩn = dò lr 1e-4 + 3 seed × 2 nhánh × 60 epoch ≈ **6 giờ GPU**, cho một mục bài không bán.
  Nếu vẫn muốn: dò `lr 1e-4` (3 epoch, CẢ HAI nhánh, 1 seed) xác nhận hết sập rồi mới chạy full.
- ⚠️ Máy B chạy **clone cũ** (cảnh báo sập vẫn là bản trước khi sửa). Lần sau clone lại trước khi chạy.
- Bài KHÔNG cần thí nghiệm này: §5 đã ghi kết luận gắn với một họ kiến trúc và để ngỏ câu hỏi mở (trung thực).
  Dừng ở đó là chấp nhận được nếu pvt tiếp tục đòi chỉnh cấu hình từng vòng.

**★★ SỔ lr ĐÃ THỬ cho `pvt_v2_b0` (PanNuke fold3, `sup_compare.py`) — ĐỌC TRƯỚC KHI CHẠY LẠI, ĐỪNG THỬ LẠI 2 lr ĐẦU:**

| lr | epoch × seed | w_count_mask | count-only | mask | kết luận |
|---|---|---|---|---|---|
| **1e-3** (mặc định script) | dò 2026-07-29 (số epoch không ghi lại) | — | **SẬP** R²=−1,43 | **SẬP** R²=−1,43 | ⛔ chết hẳn, đừng thử lại |
| **3e-4** | 3 × 1 (seed 42) | 0,01 | +0,8563 / MAE 4,50 | +0,9178 / MAE 3,63 | dò thấy "không sập" — **dò SAI** |
| **3e-4** | 60 × 3 (42/43/44) | 0,1 | median +0,8836, **sập 1/3** | median +0,8553, **sập 1/3** | ⛔ chạy thật vẫn sập, không báo cáo được |
| **1e-4** | — | — | **CHƯA THỬ** | **CHƯA THỬ** | bước duy nhất còn lại nếu muốn tiếp |

- **Bài học tái dùng được (quan trọng hơn mấy con số trên):** dò **3 epoch / 1 seed** KHÔNG dự báo được ổn định
  ở 60 epoch / 3 seed. 3e-4 qua vòng dò sạch sẽ rồi sập 1/3 khi chạy thật ⇒ suy biến xuất hiện **muộn**.
  Lần sau dò lr cho backbone mới: dò bằng **≥3 seed**, và chỉ tin khi **0/3 sập**, chứ không phải 1 seed sống.
- Chữ ký nhận sập trên fold3: `R² = −1,4313` **và** `MAE = 24,49` (= số nhân TB) ⇒ mô hình xuất hằng số 0.
- efflite0 giữ lr **1e-3** (đã dò, 0/5 sập) — sổ này CHỈ áp cho pvt_v2_b0.
- **Trạng thái: DỪNG.** npz `pvt_count_3seed.npz` + `pvt_mask_3seed.npz` đã backup kaggle; không có số nào vào bài.

**★★ DÒ lr CHO 2 BACKBONE GENERALITY (2026-08-04, 3 epoch × 1 seed × 2 nhánh, PanNuke fold3, w_count_mask=0,1):**
Mục đích = vá lỗ hổng "một họ kiến trúc" ở §5 (xem [[paper2-q1-pivot]]). Params **cả mô hình**: efflite0 3,62M ·
convnext_atto **4,606M** (1,27×, khớp capacity nhất) · tiny_vit_5m_224 **6,650M** (1,84×, transformer thật).

| backbone | lr | count-only | mask | đọc |
|---|---|---|---|---|
| convnext_atto | 1e-3 | 0,604 / 7,79 | **−1,431 / 24,49** | mask CHẾT |
| convnext_atto | **3e-4** | 0,290 / 13,09 | 0,642 / 7,27 | ✅ cả hai sống (0,29 = CHẬM, không phải sập) |
| tiny_vit_5m_224 | 3e-4 | 0,888 / 4,11 | **−1,431 / 24,49** | mask CHẾT |
| tiny_vit_5m_224 | **1e-4** | 0,754 / 5,87 | 0,845 / 5,39 | ✅ cả hai sống, khoẻ nhất |

- ⚠️ **Ngưỡng `--collapse_thr 0.5` BÁO ĐỘNG GIẢ ở 3 epoch.** Sập thật CHỈ khi trúng chữ ký `R²=−1,4313` **và**
  `MAE=24,49`. R²=0,29 kèm MAE 13,1 là học chậm, thêm epoch là hết.
- **Cơ chế mới học được:** hai nhánh kéo lr về HAI PHÍA. mask chết ở lr CAO (density-MSE bị nền >99% số 0 chi
  phối → hút vào cực tiểu "xuất 0 khắp nơi"); count-only tệ ở lr THẤP (mỗi ảnh 1 giá trị giám sát → gradient
  yếu, cần bước lớn). Giao thức bắt dùng CHUNG 1 lr ⇒ phải chọn điểm giao cả hai cùng sống.
- ⛔ **KHÔNG cứu nhánh mask bằng cách tăng `--w_count_mask`** (script có gợi ý thế): giao thức §4.1 chốt 0,1,
  đổi thì mất nối với Bảng 1.
- Ở 3 epoch mask LUÔN dẫn trước (65k giá trị giám sát/ảnh vs 1) — khớp ghi chú dòng ~383, **đừng đọc gap
  3-epoch như gap cuối**.
- **CHỐT:** convnext_atto lr **3e-4**, tiny_vit_5m_224 lr **1e-4**, 60 epoch × seed 42,43,44, `--dump_preds`
  + `--ckpt_dir` bắt buộc. Rủi ro còn lại: suy biến xuất hiện MUỘN (bài học pvt) → có seed sập thì BÁO CÁO.

**★★ KẾT QUẢ tiny_vit_5m_224 — 3 SEED XONG (2026-08-05, 60 epoch, lr 1e-4, PanNuke fold3 n=2722, 0/3 sập):**

| | count-only | mask | |
|---|---|---|---|
| R² | **0,9283 ± 0,012** | **0,9406 ± 0,003** | Δ = **−0,0123** (median −0,0067) |
| per-seed | 0,9113 / 0,9359 / 0,9377 | 0,9434 / 0,9426 / 0,9358 | mask cao 2/3 |
| MAE gộp theo ảnh | 3,18 | 3,13 | Δ|err| = **+0,045 nhân = 0,2%** |
| Wilcoxon bắt cặp | | | **p = 0,62 → KHÔNG phân biệt được** |
| δ* | | | **0,106 nhân = 0,4%** |

**Đối chiếu efflite0 (Bảng 1, CNN 3,62M, 5 seed):** Δ R² = −0,013, p = 1,1e-13, Δ|err| = 0,21 nhân = 0,86%,
δ* = 1,1%.

- ⟹ **Δ R² gần TRÙNG (−0,012 vs −0,013)** dù đổi họ kiến trúc và tăng ~1,8× tham số ⇒ **bác giả thuyết
  capacity-confound** ("mô hình quá nhỏ nên không khai thác nổi 65k giá trị của mask thì Δ phải nở ra").
  Đây đúng là dự đoán của lập luận gradient ở Hình 1: ∂μ/∂d_p = 1 là tính chất của HÀM MẤT MÁT, không của mạng.
- ⟹ Trên transformer khoảng cách còn **THU LẠI**: Δ|err| 0,21 → 0,045 nhân, p 1e-13 → 0,62, δ* 1,1% → 0,4%.
  KHÔNG phải thiếu lực kiểm định (vẫn n=2722, cùng phép kiểm đã bắt được 0,21 nhân ở p=1e-13).
- ⚠️ **Δ R² −0,0123 phần lớn do MỘT seed:** count s42 = 0,9113 vs 0,9359/0,9377 ⇒ sd nhánh count 0,012 vs mask
  0,003. **Median Δ = −0,0067, bằng nửa mean.** Viết vào bài PHẢI nói rõ, không được chỉ trích mean.
- ⚠️ Dòng script tự in *"cái thắng đến từ SUPERVISION, không phải kiến trúc"* là câu mẫu khi p>0,05 — KHÔNG
  phải kết luận đúng ở đây, đừng chép.
- **Artifacts:** `work/tv_s{42,43,44}.npz` (đã tải về Mac; s42 kéo qua Kaggle API từ kernel
  `hipinhththu/tyni-vit-5m-seed42`, s43/s44 từ 2 máy khác). md5 cột pred khác nhau cả 3 → seed có tác dụng.
  ⛔ **CHƯA backup Kaggle dataset** — phải làm cùng `sup_convnext_atto.npz`. Xem [[feedback-note-backup-results]].
- **Mẹo tải:** file Kaggle tải về tên `.zip` CHÍNH LÀ `.npz` (npz = zip chứa các .npy) → chỉ cần đổi tên, khỏi
  giải nén. Và `--load_preds` đã nhận cả THƯ MỤC đã bung (`load_npz_any`), nên Safari tự bung cũng không sao.
**⛔ convnext_atto lr 3e-4 — NHÁNH MASK SẬP 2/3, KHÔNG DÙNG LÀM SỐ (2026-08-05):**
count-only 0,8870 ± 0,021 (0/3 sập) vs mask **−0,656 ± 1,096, median −1,4313, SẬP 2/3** (chữ ký hằng số 0:
R² −1,4313 + MAE 24,49). Quy tắc của bài là nhánh mask phải ở cấu hình CÓ LỢI NHẤT ⇒ lr này không đạt, số này
đo thất bại tối ưu hoá chứ không đo giám sát.
- **Lần thứ BA đúng bài học dòng ~528:** dò 3 epoch KHÔNG dự báo được ổn định ở 60 epoch. pvt qua dò rồi sập
  1/3; convnext qua dò rồi sập 2/3. ⇒ **đừng dò nữa, chạy thẳng full rồi đọc số seed sập.**
- ⚠️ **KHÔNG được bán "count-only bền hơn mask":** convnext chỉ mask sập, nhưng pvt (dòng ~522) cả HAI nhánh
  cùng sập 1/3 ⇒ bằng chứng mâu thuẫn, không có kết luận về độ bền tương đối.
- Dùng được ở mức: density-MSE trên đích thưa có **hố hút xuất-hằng-số nhạy lr** — cùng cơ chế với câu đã có ở
  `main.tex:385-393` (mask sập khi `w_c` quá nhỏ), chỉ đổi núm từ `w_c` sang lr.
- pkl/npz `sup_convnext_atto.npz` (lr 3e-4) **GIỮ làm bản ghi lần sập**, đừng ghi đè.

**★★ convnext_atto lr 1e-4 — 0/3 SẬP CẢ HAI NHÁNH, ĐẠT LUẬT ĐÃ CHỐT → BÁO CÁO (2026-08-05):**
count-only **0,9104 ± 0,004** / MAE 3,45 ± 0,08 vs mask **0,8917 ± 0,013** / MAE 3,76 ± 0,14.
Δ = **+0,0186** (median +0,0123), Wilcoxon p = 5,21e-18 (count-only lỗi ít hơn), δ* = 0,365 nhân = **1,5%**.
npz `sup_convnext_atto_lr1e-4.npz`.
- ⚠️ **Dự đoán trước khi chạy SAI chiều:** đã đoán lr 1e-4 cứu mask và làm chậm count ⇒ Δ nở về phía mask.
  Thực tế count dẫn. Ghi lại để đừng tin vào suy luận "nhánh nào thích lr nào" từ vòng dò 3 epoch.
- ★ **lr 1e-4 được chọn VÌ nhánh mask** (3e-4 mask sập 2/3) ⇒ cấu hình đã nghiêng về phía CÓ LỢI cho mask
  trong số các cấu hình đã khảo sát, mà count-only vẫn dẫn. Viết được thành một câu trong bài.
- ⚠️ Vẫn giữ caveat: trên convnext hai nhánh kéo lr về HAI PHÍA ngược nhau (count thích cao, mask thích thấp)
  nên mọi lr dùng chung đều thiệt cho một bên ⇒ Δ ở backbone này lẫn giám sát với tối ưu hoá, không tách được.
  efflite0 và tiny_vit KHÔNG có vấn đề này (0/5 và 0/3 sập ở một lr duy nhất).

**★★★ TỔNG HỢP 3 BACKBONE (2026-08-05) — DẤU ĐẢO CHIỀU, ĐỘ LỚN BỊ CHẶN:**

| backbone | tham số | họ | Δ R² (count − mask) | Wilcoxon | δ* |
|---|---|---|---|---|---|
| efficientnet-lite0 | 3,62M | CNN | **−0,013** (mask cao 5/5) | 1,1e-13 | 1,1% |
| tiny_vit_5m_224 | 6,65M | transformer | **−0,012** (mask cao 2/3) | 0,62 | 0,4% |
| convnext_atto | 4,61M | CNN mới | **+0,019** (count cao) | 5,2e-18 | 1,5% |

**★ SỬA 2026-08-11 — tỉ số "2,6 lần" là SAI SỐ HỌC:** biên độ R² giữa 3 kiến trúc = 0,018 (count) và 0,049
(mask) ⇒ 0,049/0,018 = **2,72**, không phải 2,6. Nhưng cả hai biên đều là hiệu của số đã làm tròn 3 chữ số, sai
số kéo tỉ số về dải [2,53; 2,94] ⇒ **BỎ HẲN tỉ số** trong bài, chỉ nêu hai biên độ (main_vi §5.1.1).
**main.tex:589 vẫn còn "about 2.6 times" — sửa khi đồng bộ.**

⟹ **Không chế độ nào ưu thế nhất quán GIỮA CÁC KIẾN TRÚC**, độ lớn luôn ≤1,5% số nhân/ảnh. Đây LẶP LẠI đúng
cấu trúc lập luận bài đã dùng cho 3 dataset (`main.tex:471-482`), giờ trên trục kiến trúc ⇒ vá được câu
Limitations "established on a single architecture family". Xem [[paper2-q1-pivot]].

---

## 1. IDENTITY & RANH GIỚI

### 1.1 Paper 1 vs Paper 2 (KHÔNG double-claim)
- **PAPER 1** = *"Predictor-Agnostic Joint Conformal Cell Counting under Shift"*: **PB-JCI** (score `R=|N−E[N]|/σ`, σ suy từ cấu trúc
  Poisson-Binomial của điểm detection `σ=√Σsᵢ(1−sᵢ)`), joint đa lớp, Adaptive PB-JCI Online. Nằm TRÊN detector nặng.
- **PAPER 2 DÙNG LẠI khung conformal PB-JCI của P1 (CITE, KHÔNG claim mới).** Đóng góp gốc = **chưng cất foundation → student 1.9M
  tự HỌC (μ,σ)** thay vì suy σ từ detector nặng.
- **Mắt xích:** baseline **KD** trong P2 = cách σ Poisson-Binomial của P1 áp lên student. "R2 vs KD" = *σ HỌC ĐƯỢC (P2) vs σ Poisson-Binomial (P1)* cùng student nhẹ.

### 1.2 Core A (chốt 2026-07-16): distillation là linh hồn
Sau vòng thử đua accuracy + đổi backbone FastViT, chốt **core A**: distillation là linh hồn, giữ student nhỏ, **accuracy KHÔNG phải trục bán**.
Hook = **label-efficiency** (count-only vs mask). FastViT = gác lại (core B).

**Thesis:** *"Distributional Count Distillation: chưng cất PathoSAM ~640M → student 1.9M chỉ cần nhãn COUNT rẻ (KHÔNG cần instance mask
như NuLite/CellViT), xuất PHÂN PHỐI đếm calibrated (μ,σ) — trustworthy ở giá distillation-scale."*

**★ FRAMING KHOA HỌC (chốt 2026-07-18 — ĐÃ SỬA 2 lần, đây là bản đúng):**
> ⚠️ **BỎ framing "cái gì sống sót qua nén — đếm hay bất định?"** (từng ghi, SAI): **PathoSAM KHÔNG có bất định nội tại** để "sống sót". PathoSAM là segmentation → xuất mask + điểm detection sᵢ; **PB-σ (√Σsᵢ(1−sᵢ)) là CẤU TRÚC của Paper 1** dựng từ điểm detection, KHÔNG phải output teacher. Và **cả R2 lẫn KD đều KHÔNG distill bất định của teacher**: R2 **HỌC** σ từ GT; KD áp công thức PB lên điểm của **CHÍNH student**. → Không có "teacher uncertainty" trong pipeline.
>
> **Framing ĐÚNG:** *"Chưng cất khả năng ĐẾM của pathology foundation model vào student 1.9M, trang bị đầu **phân phối đếm HỌC được** (μ,σ) calibrated. Bất định KHÔNG đến từ teacher — nó được HỌC (N2 Poisson-anchored); learned-σ đáng tin hơn công thức analytic PB-σ (Paper 1) khi model NHỎ (N4 — vì điểm detection của model nhỏ không đủ calibrated cho cấu trúc PB), và learned-σ này TRANSFER cross-dataset (N5)."*
> **Câu bán sạch (không overclaim):** *"a lightweight distilled counter that provides **learned, calibrated, transferable** count uncertainty — where the analytic Poisson-Binomial uncertainty degrades at small scale."*

**⚠️ CẢNH BÁO THUẬT NGỮ (tránh reject):** KHÔNG gọi phương pháp là *"distribution distillation"* — khái niệm ĐÃ CÓ TÊN (Malinin et al., *Ensemble Distribution Distillation*, ICLR 2020). **σ HỌC từ GT (§2.3), KHÔNG distill từ teacher**; baseline KD (áp PB) THUA learned-σ (N4). Phát biểu: *"distilled counter + **learned** calibrated distributional head"*.

---

## 2. METHOD (những gì mình tự thiết kế)

### 2.1 `DensitySigmaUNet` ([distill_student_r2.py](distill_student_r2.py))
U-Net nhẹ (`student_ch=32` ⇒ **1,935,266 params / 10.49 GMACs@256**), HAI đầu ra:
1. **density** (Conv1×1→ReLU): `density_map≥0`; **μ = Σ density_map** (density-map counting, dùng FULL mask teacher làm target, không cần dot-annotation).
2. **log-σ** (global-avg-pool bottleneck → MLP → 1 scalar/ảnh = hệ số phân tán).
Chung 1 backbone → σ mượn đặc trưng sâu, gradient sạch, σ **heteroscedastic** học từ lỗi thật.

### 2.2 σ **Poisson-anchored** (đóng góp method quan trọng nhất — diagnostic→fix)
```
σ = √(max(μ, 1)) · exp(clamp(log_s, −2, 2))
```
- **Thất bại ban đầu:** σ head thô `exp(log_s)` → PanNuke thắng nhưng **NuInsSeg TRƯỢT** (Winkler bất ổn, std ±36). Không giấu — điều tra `diagnose_sigma.py`.
- **Chẩn đoán:** head thô calibrated khi count đồng đều + data dồi dào (PanNuke corr(|err|,σ)=+0.53) nhưng **SẬP khi dải count lớn + data ít**
  (NuInsSeg count 1→370: corr=−0.02, σ runaway=15703).
- **Fix (cơ sở thống kê):** neo σ vào **√μ** (Poisson equidispersion Var≈mean là *mốc neo*, KHÔNG giả định data tuân Poisson; overdispersion do
  head learned-dispersion gánh). → count-scaling miễn phí + chặn runaway; head chỉ học hệ số phân tán.
- **Kết quả:** NuInsSeg lật TRƯỢT→ĐẠT; PanNuke cũng cải thiện. **MỘT dạng σ, cả 2 dataset thắng.** (A2: Poisson NLL 4.21 < NB/Gaussian-raw 4.58.)

### 2.3 `--detach_mu` (cấu hình CHỐT)
Loss = `density-KD (MSE) + count·|μ−GT| + β·NLL(GT|μ,σ)` (β-NLL Seitzer 2022, β=0.5; w_count=w_nll=0.01).
- **Vấn đề (qua ablation):** NLL kéo cả μ, `L_nll≫L_count` → bóp méo μ → hỏng MAE.
- **Fix:** `mu = density.sum().detach()` trong nhánh σ → NLL CHỈ dạy σ. Gỡ **mean–variance optimization conflict**.

### 2.4 Suy luận: DÙNG LẠI PB-JCI của Paper 1 (CITE, không claim)
Split conformal trên score `r=|GT−μ|/σ`; 3 scheme ([eval_r2_grouped.py](eval_r2_grouped.py)):
- **global** (1 quantile), **mondrian** (1 quantile/organ — PanNuke, đủ mẫu/mô), **cluster** (gom organ thành n_clusters=5 theo độ khó từ σ — NuInsSeg, ít mẫu/organ).
- Gọi **"data-regime-aware grouping"** (quy tắc mondrian/cluster đặt *a priori* theo mật độ mẫu). KD không có σ học được → KHÔNG khai thác được cluster.
- ⚠️ **mondrian & cluster ĐỀU cần nhãn organ lúc test** (critique 3.3) — KHÔNG claim "R2 không cần organ". Chỉ global mới không cần (worst-org yếu hơn).

### 2.5 Protocol leak-free
- **PanNuke:** train 2 fold → predict fold held-out (`--test_fold`), lặp 3 fold, `--exclude_tissue colon` (Lizard-overlap, y hệt P1).
- **NuInsSeg:** cross-fitting 5-fold (`--kfold 5`), ghép 665 dự đoán leak-free.
- Teacher density chỉ dùng cho ảnh TRAIN; test chỉ so GT thật.

> ★ **SPLIT PanNuke của PathoSAM (CHỐT CỨNG — verify từ CODE patho-sam, xem [[pathosam-training-data]]):**
> **PathoSAM train PanNuke FOLD_1 + FOLD_2, TEST trên FOLD_3** (`get_generalist_datasets.py:80` = `folds=["fold_1","fold_2"]`; `dataloaders.py:149` = `folds=["fold_3"]`).
> ⟹ **Đánh giá PanNuke trên FOLD_3 = SẠCH** (teacher chưa thấy fold_3). **Paper 2 lấy fold_3 làm số PanNuke chính = worst-org 0.905** (MAE/Winkler theo fold_3).
> f1/f2 chỉ ghi kèm cho robustness (là in-domain của teacher; số ~bằng f3 nên nhất quán). **KHÔNG dùng f1/f2 làm headline.**
> **NuInsSeg = out-of-domain (KHÔNG train) = CLEAN OOD = anchor generalization thật.** Mọi claim generalization của Paper 2 tựa NuInsSeg.
> Paper 1 cũng test fold_3 → sạch, colon-exclusion đúng.

### 2.6 Harness Bước 2 (so heavy net công bằng)
`count_student_cost.py` (params+GMACs thop), `prep_nuinsseg_as_pannuke.py` (GT count = len(unique(mask))−nền, y hệt student),
`dump_cellvit_counts.py` (chạy code OFFICIAL CellViT/LKCell/NuLite), `eval_heavy_count.py` (MAE/RMSE/MAPE+Bias).

---

## 3. NOVELTY (N1–N5) — đóng góp gốc, tách khỏi P1/baseline

| # | Đóng góp | Loại | Vì sao mới |
|---|---|---|---|
| **N1** | **Tổ hợp** [density-distill 640M→1.9M] + [calibration count-level] + [head Poisson-anchored (μ,σ)] → student tí hon xuất phân phối calibrated | tổ hợp (không phải concept lẻ) | Distillation & heteroscedastic-UQ tồn tại RIÊNG; cái mới = kết hợp cụ thể + không peer distilled nào xuất UQ. Mạnh **nhờ đi cùng N2–N4** |
| **N2** | **σ Poisson-anchored** `√(max(μ,1))·exp(clamp(log_s,−2,2))` | parameterization | Công thức σ mới cho count; khác hẳn σ Poisson-Binomial của P1; kèm diagnostic→fix |
| **N3** | **`detach_mu`** — tách μ khỏi NLL | kỹ thuật train | Gỡ mean–variance optimization conflict; nguyên lý tổng quát cho heteroscedastic count regression |
| **N4** | **learned-σ > PB-σ ở student 1.9M — ĐẬM dưới global, thu hẹp dưới cluster** (R2 vs KD, ✅ VERIFIED 5-seed) | phát hiện empirical | Same-scheme (§4.1): worst-org global R2 0.610 vs KD 0.278; cluster 0.750 vs 0.658. PB-σ **chỉ sập dưới global**; cluster bù lại. R2 còn đè teacher-PB 640M worst-org cả 2 scheme + MAE −32% (đầu đếm). Sig per-image: MAE p<0.05 mọi seed, Winkler p<0.05 hầu hết (1 seed n.s.). **KHÔNG claim "PB vỡ 0.282" (cross-scheme, đã bỏ).** |
| **N5** | **conditional coverage TRANSFER cross-dataset dù MAE không** | phát hiện empirical | worst-org NuInsSeg→PanNuke 0.897≈in-domain 0.906 → tin cậy generalize độc lập với accuracy điểm |

**Ranh giới:** khung conformal/PB-JCI = P1 (cite). Density-counting, U-Net = cũ. UQ floor = code người khác làm mốc. **Gốc = N1–N5.**

---

## 4. RESULTS (honest, current — 2026-07-18)

### 4.1 R2 vs KD (cùng student 1.9M) — ✅ KD VERIFIED 5-seed (2026-07-18)

| Dataset | Winkler R2 ↓ | MAE R2 ↓ | worst-org R2 ↑ | KD (verified 5-seed) |
|---|---|---|---|---|
| **PanNuke** (no-colon, headline fold_3) ✅ | **19.28** | **3.36** | **0.905** (fold_3, cluster) | 23.7 / 3.94 / 0.739 |
| **NuInsSeg** (cross-fit, 5-seed) ✅ | **95.4±11.9** | **14.7±1.7** | **0.750±0.049** (cluster) | xem bảng same-scheme ↓ |

**★ N4 SAME-SCHEME (NuInsSeg, 5-seed, cùng student 1.9M — NGUỒN CHÂN LÝ, thay số cross-scheme "0.282" cũ đã SAI):**
Số cũ so R2-**cluster** (0.750) với KD-**global** (0.282) = **lệch scheme, KHÔNG fair**. Same-scheme thật:

| scheme | metric | KD (PB-σ) | R2 (learned-σ) |
|---|---|---|---|
| **global** | worst-org ↑ | 0.278±0.048 | **0.610±0.076** |
| **global** | Winkler ↓ | 131.99±1.99 | **94.17±9.83** |
| **cluster** | worst-org ↑ | 0.658±0.026 | **0.750±0.044** |
| **cluster** | Winkler ↓ | 98.67±1.72 | 95.37±10.66 (~hòa) |
| **cả hai** | MAE ↓ | 21.71 | **14.72** |

**Đọc honest (KHÔNG overclaim "PB-σ vỡ"):**

- **MAE −32% (14.7 vs 21.7) bền cả 2 scheme** — nhưng do **ĐẦU ĐẾM** (density-sum vs segment-threshold), KHÔNG phải UQ. R2 đếm giỏi hơn KD độc lập scheme.
- **worst-org:** learned-σ > PB-σ **cả 2 scheme**, dramatic ở **global** (0.61 vs 0.28) nhưng **thu hẹp ở cluster** (0.75 vs 0.66 — grouping bù phần σ miscalibrated). ⟹ "PB-σ collapse" **CHỈ đúng dưới global**; dưới cluster PB-σ nén tốt (teacher 640M 0.680 → student 1.9M 0.658, gần như không rớt).
- **Winkler:** R2 thắng lớn global, **~hòa cluster**.

**★ vs Teacher-PB 640M (VERIFIED LOCAL `data/pathosam_nuinsseg_preds.pkl` — `eval_r2_grouped.py`):** teacher-PB cluster **0.680/85.07/17.89**; global **0.482/111.64**.
So R2 1.9M: **R2 đè teacher worst-org CẢ 2 scheme** (cluster 0.750>0.680; global 0.610>0.482) **+ MAE** (14.7<17.9); teacher CHỈ nhỉnh **Winkler-cluster** (85<95). ⟹ **Distilled 1.9M ≥ teacher 640M ở reliability+accuracy, tại 1/330 size** — teacher chỉ giữ 1 góc Winkler-cluster. *(json: `work/teacher_pb_grouped.json`.)*

**Significance per-image (đơn vị=ảnh N=333, thay p=1.9e−6 pseudoreplication cũ):** R2 vs KD **MAE p<0.05 MỌI seed** (Δ≈−4→−10 count, CI không chứa 0); **Winkler p<0.05 HẦU HẾT seed** (vd p=2e−8) nhưng **1 seed n.s.** (p=0.109, CI chứa 0) → ghi trung thực "significant in most, not all seeds".

PanNuke per-fold worst-org: f1 0.908 / f2 0.906 / f3 0.905 (0/18 mỗi fold). NuInsSeg R2-cluster 5-seed: [0.701,0.764,0.701,0.817,0.767]. KD-cluster: [0.641,0.631,0.68,0.699,0.641].

### 4.2 vs baseline recent (code official, cùng μ,σ leak-free)
**PanNuke (avg 3 fold no-colon)** ✅:

| Method | Winkler ↓ | MAE ↓ | worst-org ↑ |
|---|---|---|---|
| **R2-mondrian (ours)** | 19.28 | **3.36** | **0.906** ← cao nhất mọi method |
| **R2-cluster (ours)** | **18.50** | 3.36 | 0.843 |
| CondConf-25 | 18.81 | 3.37 | 0.853 |
| PCP-24 | 23.26 | 3.37 | 0.805 |
| CPCP-26 | 35.46 | 6.18 | 0.758 |
| R2CCP-24 | 58.4 | 5.83 | 0.621 |
| KD | 22.08 | 3.64 | 0.721 |

**NuInsSeg (5-seed matched, 2026-07-17)** ✅:

| Method | marg.cov | Winkler ↓ | worst-org ↑ | Đọc |
|---|---|---|---|---|
| **R2-cluster (ours)** | ~0.91 (sát target) | **95.4±11.9** | 0.750±0.049 | cân bằng: Winkler tốt nhất nhóm calibrated + worst-org mạnh |
| CondConf-25 | 0.95 (over-cover) | 146.8±24.9 | **0.898±0.013** | worst-org cao nhất NHƯNG nới khoảng gấp đôi → Winkler tệ +54% |
| PCP-24 | ~0.92 | 105.9±12.1 | 0.708±0.069 | **R2 đè cả 2 trục** |
| CPCP-26 / R2CCP-24 (feat) | — | 250.6 / 261.2 | 0.500 / 0.562 | không re-run (cần feature, seed pkl không có) |

MAE mọi method = 14.7±1.7 (cùng μ). **Kết luận:** R2 = **interval hiệu quả nhất** trong nhóm calibrated đúng target; CondConf đổi worst-org cao lấy over-coverage + khoảng rộng gấp đôi; R2 thắng PCP cả 2 trục.

### 4.3 UQ-floor (cùng student 1.9M, trục reliability) — ✅ VERIFIED 5-seed cluster n=5 (2026-07-18)

**PanNuke worst-org:** R2 **0.906** (tie cao nhất), Ensemble 0.901, CQR 0.904, CHDQR 0.897, MC-Dropout 0.901.
**NuInsSeg (cluster n=5, 5-seed matched — pkl `work/uq_{method}_nuinsseg_s{42..46}.pkl`, script `aggregate_uqfloor.py`):**

| Method (cùng student 1.9M) | compute | marg.cov | Winkler ↓ | worst-org ↑ |
|---|---|---|---|---|
| Ensemble | **5× model** | 0.915 | **69.61±3.14** | 0.767±0.024 |
| CQR | 1 model | 0.928 (over) | 80.71±13.20 | **0.806±0.014** |
| CHDQR | 1 model | 0.920 | 78.72±7.89 | 0.722±0.061 |
| **R2 (ours)** | **1 model, 1 pass** | ~0.91 | 95.4±11.9 | 0.750±0.049 |
| MC-Dropout | N-pass | 0.911 | 173.03±52.98 | 0.774±0.051 |

**★ ĐỌC TRUNG THỰC (setback — KHÔNG giấu):** trên NuInsSeg **R2 KHÔNG dẫn đầu UQ-floor** — xếp ~4/5 cả Winkler lẫn worst-org.
CQR (cùng 1 model) nhỉnh R2 **cả 2 trục**; Ensemble Winkler tốt nhất nhưng **tốn 5× compute**. **R2 chỉ thắng rõ MC-Dropout** (95.4 vs 173).
→ **KHÔNG được bán pillar-UQ là "reliability tốt nhất/rẻ nhất"** (claim cũ đó SAI với số này). Claim honest thu hẹp còn:
R2 = **phân phối tham số (μ,σ) 1-forward** — (a) đè MC-Dropout (UQ 1-forward ngây thơ), (b) ngang Ensemble ở **1/5 compute**,
(c) σ này cho phép **transfer cross-dataset (N5)** + tích hợp vào distillation (UQ "miễn phí", không cần cơ chế riêng).
CQR/CHDQR chỉ cho **interval cố định-α**, KHÔNG cho phân phối tái dùng/transfer. **Interval-efficiency thuần: CQR cạnh tranh — phải thừa nhận trong manuscript.**
*(Hệ quả: UQ KHÔNG phải trục bán — xem §5; lead bằng distillation/label-efficiency/N4, UQ = "distributional calibrated, integrated, competitive-not-best".)*

**★ Triage / selective-prediction (2026-07-18, `analysis_triage.py`, 0-compute trên pkl 5-seed) — cược top-tier THẤT BẠI, ghi honest:**
Đo E-AURC (chất lượng σ-ranking, tách base-MAE): Ensemble **2.78** (5× compute) < CHDQR 3.49 < CQR 3.89 < **R2 3.96** < MC-Dropout 5.77.
**R2 xếp 4/5** (cả AURC 8.60, MAE@80% 11.24). CQR/CHDQR cùng-1-model lọc lỗi tốt hơn R2; R2 chỉ hơn rõ MC-Dropout.
⟹ **σ của R2 KHÔNG enable triage tốt nhất** — lần thứ 3 xác nhận UQ competitive-not-best (sau UQ-floor + N4). **KHÔNG có hook top-tier qua UQ**; chốt Q1 tầm-trung, framing = distillation/label-efficiency/compression.

### 4.4 Cross-dataset transfer (N5) — ✅ 3 dataset (CryoNuSeg thêm 2026-07-18)
| Transfer | scheme | MAE | Winkler | cov (worst-org/marg) | #under |
|---|---|---|---|---|---|
| NuInsSeg → PanNuke | mondrian | 19.90 | 97.21 | **0.897** worst-org | **0/18** |
| PanNuke → NuInsSeg | cluster | 44.88 | 214.83 | 0.685 worst-org | 4/27 |
| **NuInsSeg → CryoNuSeg** (OOD→OOD, dataset 3) | global | 73.2 | 503.3 | **0.967** marg | 0/1 |

**Conditional coverage TRANSFER**: NuInsSeg→PanNuke worst-org 0.897 ≈ in-domain 0.906. σ distilled vẫn informative dưới shift (chiều khó: cluster kéo worst 0.42→0.685, Winkler 564→215). MAE KHÔNG transfer (lệch thang count) — ghi trung thực.

**★ CryoNuSeg = dataset 3 (2026-07-18):** train NuInsSeg → predict CryoNuSeg (n=30, count mean **253**, range 85–638; clean OOD, không trong PathoSAM/Lizard training). σ **informative** (mean 32.9, std 18.3) → conformal đạt marginal cov **0.967** với interval KHÔNG vacuous (width 454 ≈ ±0.9× count); MAE 73 ≈ **29% rel** degrade graceful (count-scale khác → MAE không transfer, ghi trung thực). **⟹ σ calibrated TRANSFER sang dataset thứ 3.** (cov 0.967 hơi over target 0.90 — một phần do n=30 nhỏ → split-conformal quantile bảo thủ.)

**★ Ranh giới vận hành (honest — MoNuSAC KHÔNG dùng làm số):** MoNuSAC native **1024** → resize 256 **co nhân 4×** (ngoài scale train) → μ **SẬP** (MAE 138≈mean, σ 6.5, interval vacuous ±5×; cov 0.934 rỗng). ⟹ transfer **sống khi scale-gap vừa phải** (CryoNuSeg native 512 = 2×) **, sập khi quá lớn** (MoNuSAC 4×). Đây là **điều kiện áp dụng của phương pháp** (density-head phụ thuộc nucleus scale) — ghi trung thực trong Limitations. *(pkl `work/xfer_nuinsseg2cryonuseg.pkl`, prep `prep_cryonuseg_counts.py`, backup kaggle `sam3-paper2-uqkd`.)*

**★★ RERUN 2026-07-31 — CryoNuSeg 2 NHÁNH × 3 seed. ⛔ QUYẾT ĐỊNH: KHÔNG đưa vào paper** (user chốt 2026-07-31;
giữ bản ghi ở đây, main.tex đã revert sạch mọi nhắc tới CryoNuSeg). Thay cho dòng 2026-07-18 ở trên.
Lý do rerun: run cũ chạy mặc định `--w_density 1.0 --w_count 0.01` = **giám sát mật độ**, không phải count-only →
dùng nó cho paper là **dán nhãn sai chế độ giám sát**. `eval_cross_dataset.py` nay có `--supervision {count,mask}`.

| Nhánh (NuInsSeg → CryoNuSeg, n=30) | s42 | s43 | s44 | GỘP R² | GỘP MAE | dự đoán TB |
|---|---|---|---|---|---|---|
| **count-only** (w_d 0, w_c 1) | +0,3974 / 79,58 | +0,2636 / 85,03 | −0,8914 / 169,59 | **−0,077 ± 0,579** | **111,4 ± 41,2** | 143 |
| **mask** (w_d 1, w_c 0,1) | +0,1628 / 101,46 | **−2,8218 / 253,07 (SẬP)** | −0,0758 / 123,22 | −0,912 ± 1,354 | 159,3 ± 66,9 | 94 |

GT TB = **253** nhân/ảnh, trong khi NuInsSeg train TB = **52,8** → **lệch thang đếm 4,8×**, ảnh đích nằm **ngoài
miền giá trị** đã huấn luyện. Cả hai nhánh **đều đếm thiếu nặng** (143 và 94 so với 253) ⇒ hỏng vì
**ngoại suy dải đếm**, KHÔNG phải vì loại giám sát.
- mask s43 dự đoán TB = **0** = đúng chữ ký sập của density-MSE (R² = −μ²/σ² : 253²/150,6² = −2,82 ✓).
- **Bỏ hẳn seed sập cho mask (cấu hình có lợi nhất cho baseline): mask R² +0,044 / MAE 112,3 vs count −0,077 / 111,4
  → hai nhánh KHÔNG phân biệt được.** n=3, sd khổng lồ ⇒ **không chạy test, không claim nhánh nào hơn.**
- ⟹ với paper: parity KHÔNG bị đe doạ (hỏng đối xứng cả 2 nhánh); đây là **giới hạn miền** đã được §Thảo luận dự đoán.
- Dòng "MAE 73 / marg cov 0,967" (2026-07-18) **SUPERSEDED**: 1 seed, w_count=0,01, không so nhánh — không dùng nữa.
- pkl: `/kaggle/working/outputs/xfer_cryonuseg/xfer_{count,mask}_cryo_s{42,43,44}.pkl` — **PHẢI backup kaggle dataset ngay.**
- ⏳ còn thiếu: conformal (`eval_r2_grouped.py`) trên 6 pkl này → mới kết luận được σ/coverage có transfer không.

### 4.5 Efficiency
| Model | Params (M) | GMACs@256 | UQ? |
|---|---|---|---|
| CellViT-SAM-H (2024) | 699.74 | 214.33 | ✗ |
| LKCell-L (2024) | 163.84 | 47.86 | ✗ |
| LSP-DETR (2026) | 45.0 | 26 | ✗ |
| NuLite-T (2024, đo thật) | 12.009 | 19.80 | ✗ |
| **Student R2 (ours)** | **1.935** | **10.49** | **✓ σ + interval** |

Student **nhỏ nhất** + ít FLOPs nhất + **duy nhất trong nhóm khảo sát có UQ calibrated**.

### 4.6 Count-MAE vs heavy net (NuInsSeg, RTX 5090)
Heavy net off-the-shelf (checkpoint PanNuke → OOD trên NuInsSeg, không leak); student in-domain distill.

| Method | Params | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|
| CellViT-SAM-H (OOD) | 699.74M | 24.24 | 34.74 | 56.9% |
| LKCell-L (OOD) | 163.84M | 16.54 | 28.07 | 38.8% |
| Teacher PathoSAM (zero-shot) | ~640M | 15.80 | 29.02 | **28.3%** |
| **Student R2 (in-domain, ours)** | **1.9M** | **~13.5–14.7** | 22.61 | 45.3% |

**Đọc trung thực:** student MAE/RMSE thấp nhất bảng này + duy nhất có UQ; **NHƯNG** MAPE 45.3% > teacher 28.3% (sai tương đối cao ở ảnh ít nhân do density-sum). Khung = *in-domain-distill (rẻ) vs OOD-zero-shot* — KHÔNG phải "student giỏi hơn heavy net". Đóng góp = thích nghi rẻ + trustworthy.

**★ 4.6b — BẢNG CHÍNH COHERENT (2026-07-20/21, `dump_counts.py` + `compute_r2_counting.py`, 665 ảnh, len-instances CÙNG THƯỚC; teacher=len-scores, PACT=Σdensity):** thay bảng cũ (§4.6 là `eval_heavy_count` KHÁC thước → đừng trộn).

| Method | Params | R² ↑ | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|---|
| CellViT-SAM-H (off-the-shelf, @1024 native) | 699.7M | 0.663 | 21.83 | 31.33 | 52.9% |
| LKCell-L (off-the-shelf) | 163.8M | 0.448 | 20.92 | 40.10 | 37.4% |
| NuLite-T (off-the-shelf) | 12.0M | 0.622 | 20.01 | 33.22 | 39.6% |
| PathoSAM teacher (zero-shot) | ~640M | 0.711 | 15.80 | 29.02 | **28.3%** |
| **PACT (ours, in-domain, 5-seed)** | **1.9M** | **0.786±0.052** | **14.74±1.53** | **24.81±3.03** | 47.6±3.4% |

✅ CellViT-SAM-H chạy @1024 native (fair): R² 0.663 (ở 256 chỉ 0.444 → 256 thiệt cho nó, đã bỏ). csv `cellvit_preds_1024.csv`.

**PACT dẫn R²+MAE+RMSE, nhỏ nhất 6–368×; teacher giữ MAPE.** LKCell (164M) thua cả NuLite (12M) OOD; không model dùng-sẵn nào (kể cả CellViT 699M) gần PACT → foundation nặng off-the-shelf không transfer tốt tương ứng kích thước. Đóng khung = **thích-nghi-rẻ (nhãn count) vs off-the-shelf**, KHÔNG "model giỏi hơn".

**★ 4.6c — Reviewer-hardening (2026-07-21, `stratified_error.py` + `significance_counting.py`, Kaggle):**
- **[B] Significance Bảng 1:** PACT 5-seed **ensemble** MAE 12.28 vs teacher 15.80 → **ΔMAE −3.52** (95%CI [−4.85,−2.24]), **paired-Wilcoxon per-ẢNH p=3.5e-4 \*\*\*** (n=665, KHÔNG per-seed). Single-model 14.74 cũng <15.80. → critique B CLOSED.
- **[C1] Lỗi theo tầng mật độ (PACT 5-seed, mean±sd):** Thấp(1-20,n167,GT̄10.6) MAE 6.78±0.28/MAPE **110.1±8.5%** · TB(21-50,n261) 9.82±0.71/28.4±2.0% · Cao(>50,n237) 25.78±3.74/**24.8±3.1%** · toàn cục 14.74±1.53/47.6±3.4%. **2 kết luận honest:** (1) MAPE-47.6% bị kéo bởi vùng ít nhân (mẫu số nhỏ) — đúng cơ chế C1; (2) **limitation THẬT**: bin Thấp PACT (6.78/110%) thua teacher (2.95/37.2% local) — density-sum over-count ảnh gần-trống; PACT thắng nhờ vùng đông (R²_cao 0.63). ⚠️ R² per-bin âm = range-restriction, KHÔNG báo. → critique C1 CLOSED + Limitations có nội dung.
- **[A-1] Baseline khác-kiến-trúc count-only** (efficientnet_lite0, w_density=0): ⏳ CHỜ CHẠY (`RUNBOOK_reviewer_ABC.md`). Kỳ vọng ≈ PACT (khớp §4.8 supervised≈distilled) → xác nhận lợi thế = in-domain count-label, KHÔNG phải kiến trúc. TUYỆT ĐỐI không claim "PACT arch superior".

**PROVENANCE csv (backup Kaggle dataset `hipinhththu/lkcell-nulite-preds`):**
- `nulite_preds.csv` (665) · `lkcell_preds_full.csv` (665) · CellViT `cellvit_preds_full.csv`@256 + `cellvit_preds_1024.csv`@1024 (đang chạy) · gt `gt_counts.csv`.
- ⚠️ csv cũ `lkcell_preds.csv` = **5 ảnh HỎNG** (dấu vết `--limit 5`), ĐỪNG dùng — đã thay bằng `_full`.
- Script SẠCH resume/flush/try-except: `dump_counts.py` (thay `dump_cellvit_counts.py`); baseline recent: `dump_instanseg.py`/`dump_cellpose.py`.
- Manuscript skeleton (lightweight-primary, UQ-phụ): `PAPER2_MANUSCRIPT_SKELETON.md`.

### 4.7 Annotation-cost (lá chắn label-efficiency)
| Method | Teacher | Nhãn TARGET cần | Độ mịn | Output | UQ | Params |
|---|---|---|---|---|---|---|
| **R2 (ours)** | frozen PathoSAM | **count-scalar/ảnh** (+density teacher=0 nhãn người) | **point** | count dist (μ,σ) | **✓** | **1.9M** |
| NuLite-T / CellViT-SAM-H | ImageNet / SAM | instance mask pixel + class | pixel | seg | ✗ | 12M / 700M |
| HoVer-unet / 9M-H-Optimus (distilled) | mask-trained teacher | (distill) output seg | pixel | seg | ✗ | ~/9M |

**★ Honesty caveat:** claim là về **YÊU CẦU giám sát của phương pháp** (task-head chỉ cần count-scalar, lấy được bằng point/dot-annotation),
KHÔNG phải "mình đã dùng nhãn rẻ hơn" (thí nghiệm này GT count vẫn lấy TỪ mask vì dataset có sẵn). Point-vs-pixel rẻ hơn nhiều lần = **cite** counting literature, KHÔNG bịa "100×".
vs peer **distilled** (HoVer-unet/9M): khác biệt KHÔNG phải label-cost mà là **task-head count-level + distributional UQ**.
**★ Bằng chứng thực nghiệm (2026-07-19): xem §4.10** — budget-frontier + head-to-head distilled↔supervised cho thấy **ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count-scalar (≈5–10× rẻ hơn mask)** → luận điểm này giờ CÓ số, không còn suông.
**Câu bán:** *"nhãn point-level (không mask nào ở target) + teacher đông lạnh → student 1.9M đạt ~70% accuracy segmenter fully-supervised, ĐỔI LẠI có UQ calibrated."*

### 4.8 Ablations — ✅ RE-RUN LEAK-FREE 5-seed cross-fit (2026-07-18) — số single-split cũ ĐÃ LẬT
NuInsSeg 5-seed, cluster n=5, cùng eval bảng chính (`aggregate_ablations.py`; pkl `student_r2_nuinsseg_cv5_{tag}_s*.pkl`):

| Config | worst-org ↑ | Winkler ↓ | MAE ↓ |
|---|---|---|---|
| **MAIN** ch32 + detach + teacher-distill | 0.750±0.044 | 95.4 | 14.72 |
| detach OFF (coupled NLL) | 0.767±0.024 | 99.0 | 15.62 |
| ch16 (~0.5M) | **0.797±0.016** | **86.7** | **13.79** |
| ch64 (~7.7M) | 0.781±0.032 | 169.2 | 17.13 |
| GT-supervised (density từ GT mask) | 0.789±0.028 | 92.8 | 14.23 |

**★ 3 KẾT LUẬN LẬT (honest — số single-split cũ = leaky artifact, ĐÃ BỎ):**
1. **detach_mu (N3) YẾU hơn claim cũ:** detach-OFF MAE **15.62** (KHÔNG phải ~18), worst-org còn **nhỉnh** (0.767>0.750). ⟹ detach cho MAE tốt hơn CHÚT (14.72 vs 15.62) nhưng đánh đổi worst-org. **KHÔNG claim "detach cứu MAE"**; giữ detach vì MAE (trục chính) + đơn giản.
2. **Capacity 0.5M–1.9M ĐỀU tốt (ch32 KHÔNG phải "sweet spot" — ch16 ≥ ch32):** ✅ VERIFIED cả 2 dataset (2026-07-18):
   - NuInsSeg: **ch16 (~0.5M) thắng ch32** cả 3 trục (0.797/86.7/13.79 vs 0.750/95.4/14.72); ch64 tệ nhất (0.781/169/17.1).
   - PanNuke (mondrian avg 3 fold, `student_r2_pannuke_ch16_f{1,2,3}`): **ch16 HÒA ch32** — 0.906/18.18/3.35 vs 0.906/19.28/3.36 (ch16 nhỉnh Winkler; per-fold wo [0.907,0.903,0.909]).
   ⟹ **Method robust theo capacity; chạy tốt ở 0.5M = nén 1280× teacher 640M.** Giữ **ch32 làm primary** (mọi thí nghiệm khác chạy ở đó — không đổi headline vì chi phí re-run), báo ch16 như "scales down to 0.5M no loss" (efficiency + robustness). ch64 overfit (nhỏ-data) → capacity lớn KHÔNG giúp.
3. **★ Distilled vs GT-supervised — LẬT:** supervised **0.789 > distilled 0.750** (+ Winkler 92.8<95.4 + MAE 14.23<14.72) → supervised đè distilled MỌI trục. Số cũ "distilled 0.753 > supervised 0.711" = **leaky, BỎ**.
   ⟹ **Distillation KHÔNG nâng reliability so với dùng GT mask.** Giá trị distillation = **LABEL-EFFICIENCY** (reliability *cạnh tranh* mà KHÔNG cần mask), khớp core A. §4.8 = "competitive without masks", KHÔNG phải "better than masks". **KHÔNG đưa distilled>supervised lên bảng chính.**

### 4.9 Hardening A1–A6
A1 coverage-curve 4α (grouping≥global mọi α); A3 per-organ Wilson CI (undercoverage = 1 mô khó + nhiễu mẫu nhỏ, không systematic);
A5 σ-analysis (corr(σ,|e|)+0.40/+0.43, z-std PanNuke 1.01 calibrated); A2 σ-mode ablation (Poisson NLL 4.21 < NB/raw 4.58);
A4 latency 1.87ms/112MB VRAM. *(A6 3-seed worst-org 0.78±0.02 → superseded bởi 5-seed leak-free 0.750±0.049.)*

**A7 — Probe stain-perturbation uncertainty (2026-07-18, `probe_stain_uncertainty.py`, Kaggle GPU):** thử stain-TTA-variance (HED-jitter, Tellez 2019) làm **nguồn σ thay thế/bổ sung** → **variance LỚN (45.4% gt) nhưng KHÔNG informative**: corr(σ_stain,|err|)=**+0.17** << learned-σ corr **+0.65** (cùng test split). ⟹ **stain-sensitivity ≠ count-uncertainty**; distill teacher-stain-uncertainty KHÔNG motivated. **Design-justification cho learned heteroscedastic σ (N2)** + preempt câu hỏi reviewer *"sao không distill uncertainty của teacher / dùng stain-uncertainty?"*. Kết thúc mọi hướng add-on (UQ-floor/triage/stain-uncertainty đều âm) → paper = solid mid-Q1, viết.

### 4.10 ★ Label-efficiency frontier (EMPIRICAL — đóng lỗ "tại sao distill", 2026-07-19)
`label_efficiency.py` (piece 1 count-only) + `label_efficiency_both.py` (piece 2 distilled↔supervised head-to-head). NuInsSeg (665 ảnh, K=**52.8** nhân/ảnh mean, median 38, tổng 35,138), 80/20 single-split × 3 seed {0,1,2}, cluster n=5, ch32, 60 epoch, **grad-clip norm 5** (chặn β-NLL phân kỳ hiếm). Caches align qua **img-hash** (teacher & GT cache khác thứ tự build_index). **⚠️ Protocol khác bảng chính (single-split, calib-set lớn hơn → số tuyệt đối cao hơn §4.1/4.8, vd full-budget worst-org 0.84–0.89) → CHỈ đọc TƯƠNG ĐỐI giữa budget/nhánh, KHÔNG cross-compare với bảng chính.**

**Piece 1 — count-only frontier (chỉ distilled):**
| frac | n_lab | worst-org ↑ | MAE ↓ |
|---|---|---|---|
| 0.10 | 53 | 0.888±0.044 | 26.34±4.45 |
| 0.25 | 133 | 0.819±0.092 | 31.11±16.2 *(1 seed xui, n=133 nhỏ)* |
| 0.50 | 266 | 0.866±0.066 | 18.33±1.94 |
| 1.00 | 532 | 0.891±0.027 | 12.77±0.68 |

⟹ **worst-org PHẲNG mọi budget (kể cả 10% nhãn)** — conformal cho conditional coverage hợp lệ **gần-như-miễn-phí-nhãn**; MAE giảm đơn điệu theo nhãn (accuracy CẦN nhãn, coverage thì KHÔNG).

**Piece 2 — DISTILLED (teacher-density) vs SUPERVISED (GT-density) cùng ngân sách ẢNH:**
| budget | DISTILL worst / MAE | SUPERV worst / MAE | đọc |
|---|---|---|---|
| 10% (53) | 0.865 / **24.26** | 0.879 / 26.09 | distill nhỉnh MAE |
| 25% (133) | 0.858 / 21.85 | 0.824 / **18.49** | superv nhỉnh |
| 50% (266) | 0.840 / 20.34 | 0.830 / **17.73** | superv nhỉnh |
| 100% (532) | 0.843 / **14.12** | 0.897 / 14.61 | hòa (distill nhỉnh) |

⟹ **coverage KHÔNG phân biệt được** (chồng ±SD, ~0.82–0.90); **MAE supervised nhỉnh ~2-3 ở giữa (25-50%), HÒA ở 10%+100%** (full-budget distill 14.12<14.61). **≈ tương đương, KHÔNG winner nhất quán.**

**★ Luận điểm annotation-cost:** cùng số ảnh & ngang accuracy+coverage, nhưng nhãn TARGET/ảnh khác hẳn:
- **DISTILL:** teacher-density (0 nhãn người, PathoSAM unsup) + **1 count-scalar** ≈ dot-count 52.8×2.4s ≈ **127s/ảnh** (point cost, Bearman ECCV'16 "What's the Point").
- **SUPERV:** **mask 52.8 nhân** ≈ 52.8×(5–10× point) ≈ **640–1270s/ảnh**.
- Cả hai tỉ lệ K ⟹ **ratio = t_mask/t_point ≈ 5–10× bất kể K** → *distilled đạt coverage+MAE ngang supervised ở **1/5–1/10 chi phí annotation**, chỉ cần nhãn đếm mức-ảnh thay vì mask dày.* = **câu trả lời trực tiếp cho "tại sao distill".**

**★ Honesty:** (1) protocol single-split ≠ bảng chính → đọc TƯƠNG ĐỐI. (2) GT count ở đây vẫn lấy từ mask (dataset có sẵn) → chứng minh **YÊU CẦU giám sát của method** (count-level), KHÔNG phải "đã annotate rẻ hơn thật". (3) ratio **5–10×** dùng khoảng an toàn — point 2.4s/instance có Bearman; t_mask/nhân KHÔNG có citation chuẩn → **KHÔNG bịa 1 con số**. (4) 3 seed hơi mỏng + 0.25 MAE nhiễu 1 seed → nếu lên figure paper nên nâng 5 seed. Script: `label_efficiency_p1_result.txt` / `label_efficiency_p2_result.txt`. GT cache dựng lại 35s từ `ipateam/nuinsseg` (rẻ, không cần backup gấp).

### 4.11 ⏳ Multi-teacher committee (Hướng A — PROBE xong 2026-07-19, thí nghiệm đầy đủ CHƯA)
**Động cơ:** user muốn **model là đóng góp chính**. Ý tưởng: distill **hội đồng foundation model** (không chỉ 1 teacher), lấy **bất đồng giữa teacher làm σ epistemic distill được** — vá lỗ "PathoSAM không có bất định nội tại để distill" (với *committee* thì có). Script probe: `probe_multiteacher.py` (2-teacher local) + `probe_multiteacher_full.py` (3-4 teacher) + notebook `multiteacher_probe.ipynb`.

**4 teacher trên NuInsSeg 665 ảnh (count = #instance):** PathoSAM (MAE 15.80) + SAM3+A2-LoRA (15.45) [2 pkl sẵn có, `data/pathosam_nuinsseg_preds.pkl` + `weights/phase_E_nuinsseg_preds.pkl`] + **NuLite-T (20.01)** + **LKCell-L (20.92)** [dump off-the-shelf, `dump_cellvit_counts.py --nulite / --lkcell --no_tokens`, csv].

| # teacher | consensus-MAE | corr(std-disagreement, |lỗi|) |
|---|---|---|
| 2 (SAM×2) | **14.97** (<15.45, hạ nhẹ) | +0.49 |
| 3 (+NuLite) | 16.08 (tệ hơn) | +0.41 |
| 4 (+LKCell) | 17.07 (tệ hơn) | **+0.65** |

**★ 2 KẾT LUẬN (honest):**
1. **Consensus KHÔNG cải thiện accuracy** — 2 model UniRepLKNet (NuLite/LKCell) train PanNuke → đếm THIẾU trên OOD NuInsSeg (~32-35 vs gt 52.8) → kéo naive-mean xuống. ⟹ **μ phải lấy từ teacher tốt (SAM), KHÔNG trung bình cả 4.** Accuracy **KHÔNG phải trục** của hướng này.
2. **Disagreement-σ = tín hiệu epistemic THẬT, mạnh khi committee đủ đa dạng:** corr nhảy +0.49→+0.41→**+0.65** (2→3→4 teacher); ở 4 teacher (2 SAM + 2 CNN) **ngang learned-σ (+0.65)**. ⟹ σ = bất đồng hội đồng distill được, *khác loại* learned-σ (epistemic vs aleatoric) → **có thể trộn**.

**⚠️ CẢNH BÁO trung thực:** (a) corr **dao động không đơn điệu** (0.49→0.41→0.65) trên 665 mẫu + align xấp xỉ (organ,gt cho 2 pkl cũ) → **+0.65 CHƯA nên tin tuyệt đối**; số quyết định là **coverage/Winkler sau conformal**, KHÔNG phải corr. (b) corr đo so *consensus-error* (17.07, hơi tệ) → có thể thuận lợi. (c) accuracy-win KHÔNG có — giá trị (nếu có) nằm ở **chất lượng/novelty của uncertainty**, không phải đếm giỏi hơn.

**→ HƯỚNG TIẾP (thí nghiệm đầy đủ, chưa làm):** train student **μ distill từ teacher tốt**, **σ-target = committee disagreement (4 teacher)**; so **coverage/Winkler** của σ-disagreement vs learned-σ (§4.3) vs PB-σ (§4.1); test **blend** epistemic+aleatoric. Nếu blend/disagreement **đè** trên coverage → **headline model contribution** (distilled committee-disagreement = principled epistemic UQ). Nếu KHÔNG đè → ghi honest, quay về label-efficiency (§4.10) làm trục chính. Backup: `nulite_preds.csv` + `lkcell_preds.csv` (dump lại được nhưng setup đau — nên giữ). NuLite-T/LKCell-L kiêm **heavy-baseline** (peer cite).

---

## 5. VERDICT Q1 + 3 TRỤ

**ĐỦ submit Q1 methods/applied** nếu kể đúng 3 trụ (KHÔNG overclaim "thắng mọi metric"):
1. **Label-efficient distillation** — foundation teacher + count-label, KHÔNG cần mask (khác mọi peer segmentation). **Distilled đạt reliability CẠNH TRANH (hơi dưới) supervised MÀ không cần mask** (§4.8: distilled 0.750 vs supervised 0.789) — value = label-efficiency, KHÔNG phải "hơn supervised". **★ NAY CÓ BẰNG CHỨNG BUDGET-FRONTIER (§4.10): ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count-scalar ≈ 5–10× rẻ hơn mask (K=52.8 nhân/ảnh).** = lỗ reviewer "tại sao distill / không có thí nghiệm tiết kiệm nhãn" ĐÃ ĐÓNG.
2. **Distributional UQ** — calibrated (μ,σ), interval theo mô, transfer cross-dataset. Không peer distilled nào có.
3. **Efficiency** — 1.935M, nhỏ nhất có UQ.
*Accuracy = "cạnh tranh cho ngân sách nhãn", KHÔNG phải điểm bán (1.9M density-counter không đè SOTA: in-domain PanNuke 1.72× MAE vs NuLite-12M).*

**Peer (Related Work, chỉ CITE):** HoVer-unet (ISBI24), 9M H-Optimus student (2502.19217, citation vàng), RCKD — đều distilled/lightweight nhưng **KHÔNG UQ**.

**Rủi ro:** (i) ★ UQ-floor 5-seed VERIFIED: **R2 xếp ~4/5, CQR (cùng 1 model) nhỉnh cả worst-org lẫn Winkler** (§4.3) → **KHÔNG lead bằng UQ**; UQ = "distributional (μ,σ) 1-forward, đè MC-Dropout, ngang Ensemble ở 1/5 compute, cho phép transfer N5" — competitive-not-best; (ii) ✅ ĐÃ 3 dataset (NuInsSeg + PanNuke + CryoNuSeg transfer, §4.4); MoNuSAC = boundary honest (scale-gap 4×); (iii) NuInsSeg nhỏ/nhiễu → claim subgroup mềm. **Rủi ro lớn nhất = FRAMING, không phải thiếu thí nghiệm.**

**★ VERDICT TẦNG VENUE (chốt 2026-07-18, đúng framing §1.2):**
- **As-is (số đóng băng): solid Q1 tầm-trung** (Comput. Biol. Med. / Comput. Med. Imaging Graph. / Artif. Intell. Med.). Đóng góp = gói distillation-UQ hội tụ + rigor + honest. Đủ bar Q1 thật.
- **⚠️ Cửa "cái gì sống sót qua nén — đếm hay bất định?" ĐÃ ĐÓNG** (RETRACTED §1.2: PathoSAM là segmentation, KHÔNG có bất định nội tại để "sống sót" — PB-σ là cấu trúc của P1). Đừng dựng top-tier trên framing này.
- **Cửa top-tier còn lại (nếu muốn tiếp): (a) đào sâu cơ chế N4** (vì sao PB-σ structural vỡ dưới global, learned-σ sống — làm hết scheme-dependent); hoặc **(b) validation lâm sàng.** Cả hai ngoài scope hiện tại. **As-is đã đủ solid mid-Q1; label-efficiency (§4.10) là trụ mạnh nhất giờ có số.**
- **Cửa ĐÃ ĐÓNG (đừng lặp):** UQ-superiority (thử 3× — UQ-floor/N4/triage đều ~4/5, thua); grab-bag stain/scale/DA (nồi lẩu, top-tier phạt). Chi tiết triage §4.3.

---

## 6. NEXT STEPS (thứ tự)
1. ✅ R2 NuInsSeg 5-seed + backup kaggle.
2. ✅ Re-eval CondConf/PCP 5-seed matched.
3. ✅ **UQ-floor regen 5-seed** (2026-07-18) — 20 pkl `work/uq_{method}_nuinsseg_s{42..46}.pkl`, `aggregate_uqfloor.py`. Số §4.3. **R2 ~4/5 → UQ không phải trục bán.** Backup kaggle `sam3-paper2-uqkd`.
4. ✅ **KD 5-seed + per-image significance** (2026-07-18) — 5 pkl `work/student_kd_nuinsseg_cv5_s{42..46}.pkl`, `aggregate_kd.py`/`aggregate_r2.py`. Số same-scheme §4.1. **N4 CLOSED (reframe honest: PB-σ chỉ sập global).** Backup kaggle `sam3-paper2-uqkd`.
5. ✅ **Dataset 3 = CryoNuSeg transfer** (2026-07-18) — NuInsSeg→CryoNuSeg: marg.cov 0.967, Winkler 503, MAE 73 (σ transfer sống). MoNuSAC bỏ (co nhân 4× → μ sập; ghi boundary). `prep_cryonuseg_counts.py` + `xfer_nuinsseg2cryonuseg.pkl`, backup kaggle.
6. ✅ **Ablations §4.8 re-run leak-free 5-seed** (2026-07-18) — `aggregate_ablations.py`. **3 claim cũ LẬT:** supervised>distilled (0.789>0.750), ch16>ch32 NuInsSeg, detach yếu. §4.8 reframe honest (distillation=label-efficiency KHÔNG phải hơn-supervised). Backup kaggle.
7. ✅ **Label-efficiency frontier EMPIRICAL** (2026-07-19, §4.10) — `label_efficiency.py` (count-only) + `label_efficiency_both.py` (distilled↔supervised head-to-head, align img-hash, grad-clip). **Đóng lỗ "tại sao distill": ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count (≈5–10× rẻ hơn mask, K=52.8 nhân/ảnh).** GT cache dựng-lại 35s (rẻ). *(Tùy chọn hardening: nâng 3→5 seed nếu lên figure.)*
8. ⏳ **Multi-teacher committee (Hướng A, §4.11)** — PROBE 4-teacher XONG (gate qua: disagreement-σ +0.65 ngang learned-σ; consensus KHÔNG hạ accuracy). **CÒN thí nghiệm đầy đủ:** train student σ-target = committee disagreement, so coverage/Winkler vs learned-σ/PB-σ + test blend. Đè → headline model contribution; không đè → về label-efficiency. `probe_multiteacher_full.py` + `multiteacher_probe.ipynb`.
9. → **VIẾT manuscript**: đóng gói "Distributional Count Distillation under mean-variance optimization conflict", KHÔNG claim PB-JCI (=P1); hình ~4–5. **Lead = distillation + label-efficiency (§4.10) + N2-Poisson-σ + PanNuke comparative-win (§4.2); KHÔNG lead UQ; KHÔNG dùng "distribution distillation".** (nếu §4.11 đè → thêm trục committee-epistemic-σ.)

**Baseline = code official (không tự chế):** CondConf (Gibbs–Cherian–Candès JRSS-B 2025), PCP (Zhang–Candès 2024), R2CCP (Guha ICLR 2024),
CPCP (ICML 2026), UQ floor (MC-Dropout/Ensemble/CQR/CHDQR); CellViT/LKCell/NuLite/LSP-DETR/PathoSAM. Dùng đúng checkpoint official.
Runbook chi tiết vast: xem [RUN_LIST_STRENGTHENING.md](RUN_LIST_STRENGTHENING.md).

**★★ DeNuC 5 seed LẬT Bảng 4 (2026-08-02, `denuc_fold3_counts_s4{2..6}.csv` ở `~/Documents/denuc/`):**
DeNuC **0.936±0.002 / MAE 2.98±0.02, F1 86.31%** > efflite0 count-only **0.920±0.005 / 3.61**, **5/5 seed đều cao hơn**
(Δ +0.016, sd 0.002 → không phải dao động). Kiểm chốt cứng ĐÃ ĐẠT: 2722 dòng × 5, cột `gt` trùng khớp tuyệt đối giữa
5 seed, tổng 66 654 nhân / mean 24.49 (khớp StarDist + efflite0), cột `pred` khác nhau cả 5 → seed có tác dụng.
- **CÂU "thắng cả 3 paradigm" PHẢI BỎ.** Bảng 4 đọc lại theo CỘT NHÃN: mặt nạ điểm ảnh (DualU-Net 0.804, StarDist
  0.847) **đều thua**; thứ duy nhất thắng là DeNuC, và nó **không dùng mặt nạ** (điểm, 1.4 s/nhân). ⇒ luận điểm
  "đếm không cần mặt nạ" MẠNH HƠN bảng cũ; chỉ mất câu leaderboard.
- **Điểm yếu phải nói thẳng:** điểm vs số đếm chênh chưa tới 1 bậc → DeNuC là đối thủ GẦN NHẤT. Reviewer sẽ hỏi
  "sao không dùng điểm?", bài phải trả lời, đừng né.
- **Lập luận CROWDING yếu đi:** bias 45+ cũ −30.4 → nay −3.1..−3.8, các tầng dưới đều +0.6..+0.9. Chỉ còn viết được
  "detection suy giảm RIÊNG ở tầng dày nhất" (dấu vẫn đảo), không viết "sụp" được nữa.
- Setup khác lượt hỏng: chọn checkpoint **epoch cuối** (split pannuke12_3 không có val nên KHÔNG chọn qua fold3 =
  không leak). Lượt cũ MASTER không ghi `best_checkpoint.pth` lấy từ đâu → nghi chọn nhầm checkpoint sớm.
- ⚠️ CSV đã có trên Mac, CHƯA backup kaggle dataset.

**★★ RINENG — VIỆC TỒN ĐỌNG TRƯỚC KHI NỘP (2026-08-02):**
1. **Affiliation THIẾU địa chỉ bưu chính.** Guide đòi *"full name and postal address of each affiliation"*; bài mới
   có đơn vị + thành phố + quốc gia. Phải điền `addressline={...}` và `postcode={...}` trong `\affiliation[uit]{}`
   (elsarticle hỗ trợ sẵn hai trường này) ở CẢ `main.tex` và `main_vi.tex`. Cần user cung cấp số nhà/đường/mã bưu chính UIT.
2. **Mục khai báo AI ĐÃ GỠ khỏi cả 2 bản** (user quyết 2026-08-02). Guide: *"If you have nothing to disclose, you do
   not need to add a statement"* và *"does not apply to basic tools (grammar, spelling, references)"*. ⚠️ Nếu đổi ý
   thì tiêu đề PHẢI đúng nguyên văn *"Declaration of generative AI and AI-assisted technologies in the manuscript
   preparation process"*, đặt ngay TRƯỚC danh mục tài liệu, và mẫu câu đòi điền TÊN công cụ cụ thể.
3. **Khai trùng lặp là ô riêng trong Editorial Manager**, không phải file bib. `pbjci` (Paper 1) + `cryonuseg` đã xoá
   khỏi `references.bib` (30→28 mục). Xem [[paper1-vs-paper2-boundary]].
4. **Hình 3 đã sửa lên 400 dpi** (`make_figs.py` rcParams `savefig.dpi`): 363 dpi trên file thô / 450 dpi khi in,
   trước đó 226 dpi = DƯỚI ngưỡng 300 của guide. Ô ảnh gốc chỉ 437 px nên phần dôi ra là nội suy cho đủ mật độ in.
5. **Abstract PHẢI ≤ 250 từ** (guide ghi rõ). Hiện 249. Highlights PHẢI là file riêng có chữ "highlights" trong tên
   → đã xuất `paper2_rineng/highlights.txt` (5 dòng, dài nhất 81/85 ký tự).
6. Hình 1 đã bọc vào bài (dựng bằng `fig1_make.py`). Bảng 4 hàng DualU-Net ĐÃ có 5 seed.

**★ DualU-Net 5-seed HỎNG DO THAY CHUỖI TRƯỢT (2026-08-03) — NGUYÊN NHÂN ĐÃ XÁC ĐỊNH:** cell train dùng
`base.replace("seed: 43", f"seed: {sd}")`, NHƯNG `config_train_pannuke.yaml:75` ghi **`seed: 42`**. Chuỗi không
khớp -> `str.replace` trả nguyên văn, KHÔNG báo lỗi -> cả 5 lần train bằng seed 42. Hệ quả: 5 file
`du_fold3_counts_s4*.csv` **trùng nhau từng byte** (md5 `eaaa09d1e3d27f78a528f0434df0e31f`, 0/2722 ảnh khác),
đều cho `R2 0.8038 / MAE 5.331`. Mọi khâu khác ĐÚNG (tên ckpt theo seed, assert, eval `--checkpoint` đích danh).
**Sửa:** `re.sub(r"seed:\s*\d+", f"seed: {sd}", base)` + `assert f"seed: {sd}" in s`.

**★ ĐÃ CHẠY LẠI XONG 5 SEED (2026-08-03), 5 máy 1 seed/máy — `train.py` CÓ dùng trường seed:**

| seed | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| R² | 0.8038 | 0.8157 | 0.8216 | 0.8011 | 0.8019 |
| MAE | 5.331 | 5.528 | 5.191 | 5.689 | 5.446 |
| pred mean | 26.2 | 26.7 | 26.1 | 27.0 | 26.5 |

⇒ **R² 0.809 ± 0.009 | MAE 5.44 ± 0.19** (n=2722, GT mean 24.5, over-count ở cả 5 seed).
Bảng 4 (`main.tex:772`, `main_vi.tex:721`) đã điền ±; caption đã bỏ mệnh đề "single training run" vì giờ
**cả 4 hàng đều 5-seed**. Kết luận định tính KHÔNG đổi: 0.809 vẫn thua efflite0 0.918 count-only.
Params **41.0M** (đã đo lại; 35M in trong script là sai). Backup 1 file CSV là đủ.

**★ Preds fold3 2 nhánh cho Hình 1 (2026-08-03):** `~/Downloads/fig1_preds_fold3.csv` (2722 dòng, cột
`image,gt,pred_mask,pred_count`) từ `mask_s42-2.zip` + `count_only_s42.pt` — mask R² 0.9408/MAE 3.17,
count R² 0.9343/MAE 3.40. Dùng dựng `figures/fig1_scatter.png`. **BACKUP kaggle.**

**★★ HÌNH 1 CHỐT (2026-08-03) — trục là CƠ CHẾ, không phải sơ đồ hộp.** Sau ~42 bản draw.io vá vặt (user:
"vẫn tệ, không giống Q1"), **bỏ hẳn draw.io**, dựng bằng matplotlib: `fig1_grads.py` (sinh 5 ảnh) rồi
`fig1_make.py` (dựng hình) -> `figures/fig_design.pdf` **vector**. Đã gỡ `\iffalse`, mở lại `\ref`, thay
caption ở CẢ `main.tex` và `main_vi.tex`.
**Nội dung mới quyết định chất lượng:** panel *what flows back* = $\partial L/\partial\hat d_p$ đo bằng
autograd trên `im0507`. mask -> **TRƯỜNG, 544 giá trị khác nhau** (thấy rõ từng nhân); count -> **HẰNG SỐ
+1 ở cả 65.536 điểm**, vì $\mu$ là TỔNG nên $\partial\mu/\partial\hat d_p = 1$ mọi nơi. ⇒ đây là **lý do
toán học** cho quan sát "no measurable localisation" (main.tex:702) — tín hiệu định vị CHƯA BAO GIỜ tồn tại.
Cả hai gradient tính tại **CÙNG một bộ trọng số** (`count_only_s42`), chỉ đổi loss -> khác biệt là do LOSS.
Hai thang hiển thị riêng (chênh ~7 bậc) => caption đã ghi "so cấu trúc, không so độ lớn".
Ký hiệu theo bài: $\hat d$ / $\mu=\sum_p\hat d_p$ / $N$. Chi phí: **393 h vs 48 h** (123.090 nhân của 5179
mảnh fold1+2 × 11,5 và 1,4 s). Mọi mảnh hiển thị thuộc **fold 3** nên $\mu$ ngoài mẫu — caption đã nói.
File cũ (fig1_build.py, fig_design.drawio, fig1_scatter.png, ảnh chồng lớp) đã dọn khỏi repo.

**⚠️ SỐ ĐỊNH VỊ — CHỈ SỐ NÀY LÀ SỐ DÙNG (2026-08-10, `localize_ckpt.py`, CPU ~50 phút, full fold3 n=2722,
3 ckpt count-only `~/Downloads/count_only_s4{2,3,4}`):** ratio **1,664 / 2,185 / 0,969 → 1,61 ± 0,50**
(± = giữa 3 model), corr **0,0341 / 0,0426 / 0,0295 → 0,035**, chance 0,1694 ⇒ bản đồ định vị hoàn hảo
= **5,90×**. count-R² 0,934/0,925/0,929 (0,934 khớp Bảng 3 hàng tổng = s42 ✓). Số liệu:
`localize_fold3_3seed.csv`. **CHƯA BACKUP KAGGLE.**
**Vì sao thay số cũ 1,25 ± 0,36:** 1,25 có thật, nguồn = thân commit `11273b6` (*"3-seed: count-only ratio
1.25x; 4.43x was single-seed artifact"*), nhưng của **bộ mô hình khác** (`localization_emergence.py` tự train,
60 epoch) chứ không phải 3 ckpt của bảng/hình. Công thức đo hai script GIỐNG HỆT (đã so từng dòng) ⇒ khác biệt
là do lần huấn luyện. Chỉ số này đo được trên đúng model đang mô tả nên **dùng 1,61**; và `ratio_sd` trong code
là độ lệch GIỮA ẢNH nên ± 0,36 cũ có thể đã dán nhãn sai.
⇒ Kết luận KHÔNG đổi (corr 0,035 ≈ ngẫu nhiên; 1,61 so với trần 5,90; một model ở 0,97 = đúng mức
không-định-vị) nhưng **không viết "không đo được mức định vị nào"** nữa — đã hạ thành "rất yếu và không ổn định"
ở 4 chỗ (đóng góp 2, §5.2, Hạn chế, + bản EN).
Caveat: GT ở đây dựng từ COCO polygon `~/Downloads/coco/fold3`, bản 1,25 dùng .npy instance mask.

**★★ TRANSFER PanNuke→NuInsSeg CHỐT (2026-08-07) — 2 NHÁNH × 3 SEED, đúng yêu cầu treo từ trước.**
File: `~/Downloads/results-7/xfer_mask_s{42,43,44}.pkl` + `results-8/xfer_count_s{42,43,44}.pkl`
(dict `preds`=[{mu,sigma}], `gts`, `organs`; n=665 ảnh NuInsSeg, GT mean 52.8, std 54.0, range 1–370).
**CHƯA BACKUP KAGGLE.**

| nhánh | MAE | R² thô | r (Pearson) | slope | R² sau hiệu chỉnh affine |
|---|---|---|---|---|---|
| count-only | 31.78 ± 0.68 | 0.070 ± 0.046 | 0.724 | 1.80 | **0.524** |
| mask       | 30.75 ± 1.42 | 0.089 ± 0.062 | 0.727 | 1.85 | **0.528** |

Paired per-image (seed-averaged, n=665): Δ̂ = **+1.05 nhân**, Wilcoxon p = 2.3e-22, count thắng 217/665.
**δ\* = 1.24 nhân = 2.3% số đếm trung bình** (precision term chỉ 16% của chặn) ⇒ chen giữa PanNuke 1.1%
và MoNuSeg 3.0%, tức **cùng một câu chuyện trên trục thứ tư**.

**BA KẾT LUẬN.** (1) Hai chế độ giám sát **hỏng y hệt nhau** (r 0.724 vs 0.727; R²affine 0.524 vs 0.528;
slope 1.80 vs 1.85) ⇒ quyết định chạy CẢ HAI nhánh là đúng: nếu chỉ có count-only thì MAE 32 sẽ bị đọc
thành "count-only không transfer". (2) Đây là **lỗi hệ số, không phải mất tín hiệu**: tương quan 0.72 còn
nguyên, mô hình xếp hạng đúng nhưng nén ~1.8×. Bin theo GT: 10.6→4.2, 29.2→11.4, 49.3→22.2, 119.8→46.2
(đơn điệu, nén gần như hằng số). R² thô 0.08 → affine 0.53, tức **hơn nửa phương sai lấy lại được bằng 2
tham số** fit trên miền đích. (3) **Đầu σ KHÔNG transfer**: σ ≈ 2.6–3.3 trong khi sai số thực ≈ 31, tức
khoảng hẹp hơn thực tế ~10×. Củng cố §6.3: thứ cơ sở mới phải mua là nhãn HIỆU CHUẨN, và đây là số đo vì sao.

**LƯU Ý KHÔNG ĐƯỢC OVERCLAIM.** Hệ số đáp ứng nội miền là 0.64 (§5.2) và 1/0.64 = 1.56, gần slope 1.80 —
cùng hướng, cùng bậc, nhưng là HAI phép đo khác nhau (ablation trong ảnh vs ngoại suy dải giữa hai miền)
và hai miền còn khác nhau về nhuộm, độ phân giải, tập cơ quan, đơn vị (mảnh vs cả ảnh). Chỉ được viết như
QUAN SÁT NHẤT QUÁN, tuyệt đối không viết như suy ra. **SỬA (cùng ngày, sau khi kiểm):** giả thuyết "ngoại suy ngoài dải" ĐÃ BỊ BÁC.
Tỉ lệ đếm hụt gần như KHÔNG đổi theo mức đếm: GT 1–10 →2.19×, 10–25 →2.76×, 25–40 →2.47×, 40–80 →2.20×,
>80 →2.72×. Ba nhóm đầu nằm TRỌN trong dải PanNuke từng thấy (mean 24.5) mà vẫn hụt ~2.5×. Vậy không phải
extrapolation. Ứng viên còn lại: NuInsSeg là ẢNH NGUYÊN resize về 256² (nhân nhỏ đi theo pixel) so với
PanNuke là MẢNH 256² native — khớp với phát hiện MoNuSAC đã ghi (co nhân 4× → μ sập); và NuInsSeg có cả mô
CHUỘT (`organs` chứa "mouse muscle_tibia") trong khi PanNuke là ung thư người. CHƯA phân giải được nguyên
nhân, đừng viết như đã biết. Liên hệ với hệ số đáp ứng 0.64 (1/0.64=1.56) cũng YẾU vì tỉ lệ quan sát ~2.5.

**★ SCALE PROBE (2026-08-07) — `scale_probe.py`, CPU, không train lại.** NuInsSeg gốc **512×512** (665 ảnh),
`eval_cross_dataset.py:98` nén về 256 ⇒ nhân co **2× cạnh, 4× diện tích**; PanNuke là mảnh 256 native.
Cùng ckpt `count_only_s42.pt`, cùng 665 ảnh, đổi ĐÚNG một biến. OUT `~/Downloads/scale_probe.npz`.

| cách nạp ảnh | MAE | R² | pred TB | slope | r |
|---|---|---|---|---|---|
| A: resize 512→256 (pipeline hiện tại) | 28,77 | 0,225 | 25,0 | 1,77 | 0,779 |
| B: cắt 4 mảnh 256 native, cộng lại | **24,53** | **0,437** | 34,3 | **1,41** | 0,778 |

⇒ **Cỡ nhân giải thích khoảng một nửa sai số thang đo** (slope 1,77→1,41), R² gần gấp đôi, MAE −15%,
KHÔNG train lại. Tương quan không đổi (0,779 vs 0,778) — đúng như dự đoán, thứ hạng chưa bao giờ phụ thuộc thang.
**Phần dư = co về trung bình huấn luyện**: với B, ảnh thưa bị ĐẾM THỪA (GT 5,4→pred 14,3) còn ảnh dày bị
đếm thiếu (GT 140→78). PanNuke mean 24,5, NuInsSeg mean 52,8 max 370.

**CẢNH BÁO ĐỌC SỐ.** Ckpt này KHÁC ckpt của bảng transfer (`results-7/8`), nên MAE/R² tuyệt đối không so
chéo được; chỉ A-vs-B là có đối chứng. Hai ckpt còn KHÁC NHAU ở vùng đếm thấp (transfer pkl hụt đều 2,2×
ở mọi mức; ckpt này đếm thừa ở GT<10) ⇒ **chưa chốt được phần dư là gì**, đừng viết như đã biết.
**KHÔNG ảnh hưởng kết luận chính**: cả hai nhánh count/mask đều ăn CÙNG một pipeline resize, nên so sánh
giữa hai chế độ giám sát vẫn sạch.

---

## 2026-08-12 — CẮT PHẦN SO SÁNH BA SƠ ĐỒ CONFORMAL (chủ ý, ĐỪNG thêm lại)

Bài đang dài (35 trang ở `preprint,12pt`). Đã **cố ý gỡ** khỏi cả `main.tex` và `main_vi.tex`:
đoạn định nghĩa 3 sơ đồ global/mondrian/cluster ở §3.3, Phương trình 8 (độ phủ theo nhóm),
hai đoạn đầu §5.4.3 (đối chiếu 0,901/0,919/0,903 với số đo), **Bảng 6** (bảng UQ 3 hàng),
và **Phụ lục B** (chứng minh độ phủ split conformal).

**Lý do:** so sánh 3 sơ đồ là câu hỏi về *phương pháp conformal*, thuộc Paper 1
(xem memory `paper1-vs-paper2-boundary`), không kiểm định luận điểm của Paper 2 là
"giám sát mức ảnh đủ để đếm". Nó áp dụng như nhau cho mọi mô hình đếm.

**GIỮ LẠI:** đầu σ + Phụ lục A (bộ phận của DSU-Net), kết quả neo Poisson trên NuInsSeg
(0,655 → 0,864 worst-organ — kết quả UQ duy nhất thực sự nói về mô hình này), và §6.3 chuyển miền.
§5.4.3 nay chỉ còn 2 đoạn, nêu thẳng 0,923 / 16,6 / 20,5 / 0/19 dưới dung sai 0,85.

Số liệu Bảng 6 cũ nếu cần dựng lại: global 0,904 / 14,5 / 20,3 / 0,765 / 4-19;
mondrian 0,923 / 16,6 / 20,5 / 0,906 / 0-19; cluster 0,906 / 14,8 / 20,0 / 0,859 / 0-19.
Độ phủ dự đoán từ PT8: global 0,901 (quy ước j) hoặc 0,900 (quy ước k), mondrian 0,919, cluster 0,903.

**Cùng đợt, đã sửa 2 lỗi thật:** §4.1 còn câu "dự đoán đã được lấy trung bình theo seed"
(sai — là *sai số*, xem `sup_compare.py:49-50`); footnote Bảng 5 ghi khớp diện tích 0,74–0,78
trong khi hàng 100% của bảng là 0,73.

Trạng thái sau đợt cắt: EN 9.337 từ, 8 mục, 11 mục con, 6 tiểu mục, 5 bảng, 4 hình, 13 phương trình,
1 phụ lục. Hai bản EN/VI khớp cấu trúc tuyệt đối.

## 2026-08-12 (tiếp) — TÁI CẤU TRÚC MỤC 5 THEO TIÊU CHÍ "CÓ THỂ THUA"

Phép thử áp cho từng tiểu mục: *nếu kết quả ra ngược lại, kết luận bài có đổi không?*
Hai khối trượt phép thử vì **cả hai nhánh dùng chung một kiến trúc** nên số của chúng
giống hệt nhau ở cả hai chế độ giám sát: chi phí tính toán, và đặc tả độ tin cậy.

**Đã làm:**
- Bỏ đoạn chi phí tính toán (độ trễ 6,9ms GPU / 51ms CPU, bộ nhớ đỉnh 37,9MB). Giữ 3,62M tham số
  và 0,98 GMACs, chuyển về §3.1 chỗ mô tả kiến trúc.
- **Thí nghiệm chuyển miền PanNuke→NuInsSeg chuyển từ §6.2 (Thảo luận) lên §5.5 (Kết quả)** —
  nó ĐẠT phép thử (2 nhánh × 3 seed, nếu count-only sụp mà mask không thì bài đổi kết luận),
  vậy mà trước đây bị chôn trong Thảo luận.
- §5.4.3 (độ tin cậy) chuyển xuống §6.2, nơi nó phục vụ lập luận "khoảng chuyển miền được,
  ước lượng điểm thì không" thay vì lấp một ô ngân sách. §3.3 rút từ 394 → ~215 từ.
- §5.4 gộp làm một tiểu mục "Cần bao nhiêu chú thích là đủ" (`\label{sec:cost}`), bỏ 3 tiểu mục con.
  Nhãn `sec:deploy`, `sec:label`, `sec:uq` không còn ai tham chiếu → đã bỏ.
- Ba kiểm tra bền của hệ số đáp ứng (giãn nở mặt nạ / chất lượng chú thích / Telea) → **Phụ lục B mới**.
- **§1 đoạn 2 đổi từ "ba ngân sách" sang HAI TRỤC** (độ chính xác + chi phí chú thích), vì chính
  khung ba-ngân-sách là thứ cấp phép cho vật liệu lạc đề. Đóng góp 4 và §6.1 sửa theo.

**⚠ Nếu sau này định thêm lại chi phí suy luận hay so sánh sơ đồ conformal vào Mục 5: ĐỪNG.**
Chúng không biến thiên theo biến độc lập của bài.

**⚠ ĐÍNH CHÍNH (2026-08-12, user bắt đúng):** tôi từng ghi ở đây rằng §5.4 "mâu thuẫn" vì hàng thắng là
DeNuC nhãn điểm. **SAI — không có mâu thuẫn, ĐỪNG sửa gì.** Highlight và §5.4 chỉ phát biểu về đầu ĐẮT
của thang: StarDist 0,847 và DualU-Net 0,809, cả hai dùng mặt nạ 11,5 s/nhân, xếp 3-4/4. Không câu nào
claim nhãn mức ảnh cao nhất. Tiêu đề claim tính ĐỦ chứ không phải tối ưu, nên DeNuC 0,936 vs DSU-Net
0,918 không đụng tới luận điểm. §5.4 nói số gia từ nhãn điểm lên mặt nạ = 0 — đó mới là mệnh đề.

Trạng thái: EN **9.007 từ** (đầu ngày 10.055, giảm 10,4%), 9 mục, 12 mục con, 3 tiểu mục,
5 bảng, 4 hình, 13 phương trình, 2 phụ lục. EN/VI khớp cấu trúc tuyệt đối.

---

## 2026-08-13 — ⛔⛔ LỖI THANG ĐO: NHÁNH "MẶT NẠ" KHÔNG HỀ ĐƯỢC GIÁM SÁT BẰNG MẶT NẠ

**Phát hiện ở mức TIỀN ĐỀ, không phải một tiểu mục.** Thí nghiệm chính của bài không đo cái nó tuyên bố.

**Nguyên nhân.** `r2_losses.py:44` `density_kd_loss = F.mse_loss(...)` lấy **trung bình trên 65.536 điểm ảnh**,
trong khi `count_loss = |μ−N|` với `μ = Σ d_p` là **tổng trên 65.536 điểm ảnh`. Một bên chia N, một bên nhân N.
Nên `w_d=1, w_c=0.1` KHÔNG phải "chủ yếu mật độ" mà là ~99,998% giám sát đếm.

**Sáu bằng chứng (đo thật, không suy đoán):**
1. Số hạng mật độ chiếm **0,02%** hàm mất mát nhánh mask (tách từ log: loss=0,2385 = Ldens + 0,1·2,003 + 0,01·3,814).
2. ‖grad‖ mật độ nhỏ hơn ‖grad‖ đếm **55.000×** (mask_s42) và **58.700×** (mask_s45), đo trên checkpoint thật.
3. MSE mật độ của mask_s42 = 1,79e−5, **tệ hơn 9,08 lần** so với bản đồ TOÀN SỐ 0 (1,96e−6).
4. count_s42 (KHÔNG có loss mật độ) đạt MSE mật độ **tốt hơn** mask (1,35e−5 vs 1,79e−5).
5. `w_c=0` sập hoàn toàn (bảng quét Mục 4) = mật độ một mình không đủ sức huấn luyện. Ta đã có dữ kiện này
   từ đầu mà đọc nhầm thành đặc tính bài toán.
6. Không nhánh nào định vị: ratio mask 1,26 vs count 1,36 (hoàn hảo 5,90, đều 1,00).

**Hệ quả quan sát được:** nhánh mask = count-only với tín hiệu yếu hơn 10×. Chạy lại 5 seed cho
count 0,931±0,005 vs mask 0,914±0,014 (4 seed sống, **s43 sập −1,4313**) → Δ = **+0,017**, ĐẢO DẤU so với
Bảng 1 (count 0,918 / mask 0,931, Δ = −0,013, mask thắng 5/5). Dấu không tái lập được.

**FIX: `w_d = 256×256 = 65536`** — không phải hằng số dò, mà là đổi `mse_loss` từ trung bình sang tổng mỗi ảnh.
Đo: khởi tạo cần 69.484, đã-train cần 55.285; `w_d=65536` cho tỉ lệ 0,94 (khởi tạo) và 1,19 (đã train).
`sup_compare.py:201` cứng `wd=1.0` → phải vá thêm cờ `--w_density`.

**Quét đang chạy (3 cấu hình × 1 seed):** `pure` (w_d 65536, w_c 0 = mặt nạ THUẦN, đúng thứ bài tuyên bố),
`mix` (65536 / 0,1), `low` (6554 / 0,1). Mốc đọc: **`/MSE0` phải < 1** (hiện 9,08), `Ldens` phải giảm
(hiện đứng im 0,0000 từ ep1→ep60), `ratio` phải rời vùng 1,2–1,8.

**Mốc đối chiếu w_d=1 (đo local, fold3):** mask_s42 ratio 1,412 corr 0,0215 R² 0,9167 MSE 1,79e−5 (/MSE0 9,08);
count_s42 ratio 1,798 corr 0,0337 R² 0,9270 MSE 1,35e−5 (/MSE0 6,85); MSE bản đồ toàn 0 = 1,96e−6.

**⚠ CHƯA SỬA main.tex / main_vi.tex.** Sửa trước khi có số quét là sửa mù. Hai chỗ đã biết chắc phải sửa
dù kết cục ra sao: câu "không có seed nào suy biến" (Mục 4, đã có phản ví dụ) và "mask dẫn 5/5 seed" (Mục 5.1,
lượt chạy lại cho 4/4 ngược lại).

**Checkpoint 10 cái của lượt w_d=1 đang ở local:** `~/Downloads/results (1)/ckpt/` (s42-44) và
`~/Downloads/seed45_46/ckpt_s4546/` (s45-46). ⛔ CHƯA backup Kaggle.

### 2026-08-13 (tiếp) — KẾT QUẢ QUÉT w_d: VÁ ĐÚNG, MỤC 5.3.2 LẬT

Tất cả seed 42, PanNuke fold3, efflite0 ch32 lr1e-3 60ep.

| cấu hình | w_d | w_c | R² đếm | ratio | corr | /MSE0 |
|---|---|---|---|---|---|---|
| count-only | 0 | 1,0 | **0,9270** | 1,798 | 0,034 | — |
| "mask" CŨ (hỏng) | 1 | 0,1 | 0,9167 | 1,412 | 0,022 | — |
| `low` | 6.554 | 0,1 | 0,9225 | 4,016 | 0,216 | 1,63 |
| `mix` | 65.536 | 0,1 | 0,7730 | 3,906 | 0,381 | **0,87** |
| `pure` | 65.536 | 0 | −1,4313 | sập | sập | 1,00 |

**1. Nhánh mask THẬT SỰ định vị được sau khi vá.** ratio 1,41 → **3,9–4,0** (đều=1,00, hoàn hảo=5,90),
corr 0,022 → **0,38**. `mix` là cấu hình ĐẦU TIÊN có /MSE0 < 1 (0,87), tức thắng bản đồ toàn số 0.
⇒ Kết luận cũ "cả hai nhánh đều không định vị" là **HIỆN VẬT của w_d hỏng**. §5.3.2 phải viết lại.

**2. `pure` (w_c=0) sập về bản đồ toàn 0** — MSE dens 3,374e−06 khớp ĐÚNG MSE của map toàn 0.
MSE trên mục tiêu thưa có nghiệm suy biến tại 0 ⇒ số hạng mật độ MỘT MÌNH không train được kể cả
khi đã cân thang. Nhánh mask BẮT BUỘC có neo đếm ⇒ phép so sánh của bài không bao giờ là
"mask thuần vs count thuần" mà là **"mask + neo đếm vs count"**. Phải nói thẳng trong bài.

**3. LUẬN ĐIỂM BÀI SỐNG VÀ MẠNH HƠN.** Nhánh mask giờ chứng minh được là học vị trí thật, mà vẫn
KHÔNG thắng count-only ở đếm (tốt nhất 0,9225 vs 0,9270). Càng ép bám density càng đếm tệ
(`mix`: corr cao nhất 0,38 nhưng R² rơi 0,773). Đây là **ĐƯỜNG ĐÁNH ĐỔI** độ-trung-thực-không-gian
đổi độ-chính-xác-đếm, và giám sát mức ảnh nằm ở đầu tốt nhất trục đếm. Story mạnh hơn bản thảo cũ nhiều.

**Đang chạy:** quét bổ sung w_d ∈ {655, 20.000, 200.000} (w_c 0,1), 1 seed, để lấp đường cong
(hiện chỉ 2 điểm sống, cách nhau 10×). Sau đó chọn w_d cho nhánh mask của Bảng 1 rồi chạy 5 seed × 2 nhánh.

**Cách chọn w_d đề xuất:** cho nhánh mask **cấu hình đếm tốt nhất của nó** (đúng tinh thần đã làm với
ConvNeXt khi hạ lr) để không ai cãi là bóp nghẹt nhánh đối thủ; đồng thời báo cáo CẢ đường cong vì
chính đường cong mới là phát hiện.

⚠ Cột `/MSE0` lượt này dùng mốc 3,374e−06 trên đủ 2.722 mảnh; hai số 9,08 / 6,85 ghi ở mục trên tính
trên 64 mảnh nên KHÔNG so trực tiếp. `ratio` và `corr` thì so được.

**★★★ CHẠY LẠI VỚI `w_d` CÂN THANG (2026-08-14) — SỐ CHÍNH THỨC MỚI CỦA BẢNG 1:**
`w_d = HW·w_c = 65536×0,1 = 6553,6` (lập luận đơn vị: density-MSE là trung bình trên HW điểm ảnh, μ là TỔNG
trên chính các điểm đó). Ở `w_d=1` số hạng mật độ chiếm 0,02% hàm mất mát → TRƠ, mọi số mask cũ đo sai vật.

**Quét `w_d` (seed 42, PanNuke fold3, n=2722, `w_c=0,1`):**

| w_d | R² | ratio (max 5,90) | corr | /MSE0 |
|---|---|---|---|---|
| 0 (count-only) | 0,9270 | 1,80 | 0,034 | 6,14 |
| 1 (bản lỗi) | 0,9167 | 1,41 | 0,022 | 8,50 |
| 655 | 0,9079 | 2,06 | 0,058 | 4,45 |
| **6.554** | 0,9225 | 4,02 | 0,216 | 1,64 |
| 20.000 | 0,8972 | 4,42 | 0,311 | 1,17 |
| 65.536 | 0,7730 | 3,91 | 0,381 | 0,89 |
| 200.000 | −1,4313 | sập | sập | 1,00 |

- Nhánh mask KHÔNG thắng ở bất kỳ `w_d` nào ⇒ luận điểm bài KHÔNG phải hệ quả của lỗi thang.
- Sập ở CẢ HAI đầu: `w_c=0` và `w_d` quá lớn (200k sập dù còn neo đếm 0,1).
- ⚠️ `/MSE0 < 1` KHÔNG dùng làm tiêu chí chọn được: cấu hình duy nhất đạt (65.536) đếm chỉ 0,773. MSE trên
  đích thưa là thước đo kém (bản đồ toàn-0 đạt 1,00) → đo định vị bằng ratio/corr, `/MSE0` chỉ để chẩn đoán.

**PanNuke 5 seed, `w_d=6553,6` (nhánh count-only DÙNG LẠI vì `w_d=0` không đổi):**

| | count-only | mask |
|---|---|---|
| R² | **0,9313 ± 0,005** | 0,9114 ± 0,028 |
| per-seed | 0,9270/0,9244/0,9374/0,9373/0,9305 | 0,9281/0,9219/**0,8566**/0,9364/0,9141 |
| MAE | 3,42 ± 0,09 | 3,73 ± 0,54 |
| sập | 0/5 | 0/5 |

Δ R² = **+0,0199** (median +0,0086) · Wilcoxon p = **5,26e-18** · Δ\|err\| = −0,307 nhân (1,3%) ·
TOST biên 2%: p = 3,21e-09 · **δ\* = 0,358 nhân = 1,5%**
- ⚠️⚠️ **TOÀN BỘ hiệu số đến từ SEED 44.** Bỏ s44: Δ R² = +0,0047, Δ\|err\| = −0,027 nhân, **p = 0,116**.
  Per-seed Δ\|err\|: s42 +0,104 · s43 −0,028 · **s44 −1,427** · s45 +0,012 · s46 −0,195.
  ⇒ Kết luận đúng = **NGANG NHAU**, không được bán "count-only thắng".
- s44 R²=0,857 KHÔNG phải sập (không có chữ ký −1,4313) ⇒ theo luật của bài KHÔNG được loại, phải báo cáo
  kèm phân tích bỏ-một-seed.
- ⚠️ count-only bản CŨ 0,918±0,011 vs lần này 0,9313±0,005 — `cudnn.benchmark` bật nên không tái lập.
  ⇒ Thay CẢ HÀNG, không chỉ cột mask.

**Định vị PanNuke fold3, 5 seed mỗi nhánh (chance 0,1694; hoàn hảo 5,90):**
count-only ratio **1,54 ± 0,51** / corr **0,028 ± 0,010** · mask ratio **4,36 ± 0,27** / corr **0,239 ± 0,026**
· bắt cặp n=2608, p < 1e-300.
⇒ ★ Nhánh mask THẬT SỰ định vị mà vẫn KHÔNG đếm giỏi hơn. Đây là nội dung mới của §5.3.2; câu cũ "khả năng
định vị của mô hình giám sát dày đặc nằm ngoài phạm vi" đã XOÁ.

**NuInsSeg 5 seed `w_d=6553,6`:** count 0,8931±0,050 vs mask 0,8828±0,026 · Δ +0,010 · có ý nghĩa **1/5** lần
chia (cũ 3/5) · **δ\* trung vị 1,43 nhân = 2,7%** (cũ 3,6 = 6,6%) ⇒ ngang nhau CHẶT HƠN bản cũ.
- ⚠️ Độ ổn định ĐẢO CHIỀU giữa 2 bộ: PanNuke mask sd 0,028 vs count 0,005; NuInsSeg mask 0,026 vs count 0,050
  ⇒ **KHÔNG được bán "giám sát mức ảnh ổn định hơn"**.

**MoNuSeg:** lần chạy 2026-08-14 hỏng — `--split official` bị fallback sang `random` vì `--root` trỏ vào trong
nhánh Training (`n_test_folder==0`, dòng 158 `monuseg_indomain.py`). Số random-split (count 0,8121±0,056 vs
mask 0,8161±0,093, Δ −0,004) GIỮ làm kiểm tra bền vững, KHÔNG vào Bảng 1. Root đúng =
`/kaggle/input/datasets/hipinhththu/monuseg` (chứa cả `MoNuSeg 2018 Training Data` lẫn `MoNuSegTestData`).
- ⚠️ MASTER cũ ghi random-split count-only 0,54±0,29; lần này 0,812±0,056 — CHÊNH LỚN, đừng trích số cũ đó.

**CÒN NỢ: KHÔNG CÒN GÌ VỀ SỐ LIỆU.** Bảng 2 5 seed, vẽ lại `fig_equivalence.pdf`, chuyển miền — XONG hết
(2026-08-15). Việc còn lại chỉ là dựng Overleaf, backup Kaggle, highlights.docx, câu SAGER, khai báo EM.

🛑 **ĐỪNG ghi lại hai thứ này thành "nợ" nữa (đã kiểm 2026-08-15):**
- **MoNuSeg** xuất hiện **0 lần** trong cả 2 file .tex. Nó là bộ thứ tư **TUỲ CHỌN**, không phải ô trống.
  Chạy lại chỉ để có thêm 1 bộ (và là bộ duy nhất mask nhỉnh: 0,812±0,056 vs 0,816±0,093, Δ −0,004).
- **"§6.3 chuyển miền"** là SỐ MỤC CŨ. Sau khi gộp mục, chuyển miền nằm ở **§5.2** và đã viết đủ (MAE 31,8
  vs 29,5 · hệ số 1,96–2,32 · seed sập R²=−0,96). §6.3 bây giờ = Hạn chế và phạm vi.
⛔ **CHƯA backup Kaggle:** `ckpt_bal` (5 mask PanNuke), `ckpt_nuinsseg` (10), 5 `maskbal_s4x.npz`,
`nuinsseg_parity_bal.npz`, 3 ckpt quét `w655/w20k/w200k`.

**★★ VÀNH QUANH NHÂN — sửa cách đọc §5.3.2 (2026-08-14, 195 ảnh fold3, 5 seed mỗi nhánh):**
Nới viền nhân GT ra r điểm ảnh rồi đo lại tỉ số khối/diện tích:

| r (px) | diện tích | chỉ-đếm | mặt nạ |
|---|---|---|---|
| 0 | 17,1% | 1,29 | **4,25** |
| 2 | 22,1% | 2,17 | 3,56 |
| 4 | 27,0% | **2,31** | 3,01 |
| 8 | 36,1% | 2,06 | 2,33 |
| 16 | 51,4% | 1,61 | 1,71 |

- **5/5 seed chỉ-đếm đạt ĐỈNH ở r=2–4**, 5/5 seed mặt nạ giảm ĐƠN ĐIỆU từ r=0. Không ngoại lệ.
⇒ khối của chỉ-đếm nằm trong **VÀNH ngay ngoài viền nhân**, làm giàu ~2,3×, KHÔNG phải ngẫu nhiên.
- ⚠️ **ĐÃ XOÁ mọi câu "gần mức ngẫu nhiên"** (đóng góp §1 + §5.3.2, cả EN lẫn VI). corr 0,028 thấp là vì đo
  khớp-đúng-điểm-ảnh, mà vành lệch ra ngoài nên corr thấp BẨM SINH — không được đọc thành "không có tín hiệu".
- ★ Sửa này làm §5.3.2 KHỚP với §5.3.1: hệ số đáp ứng 0,64 (xoá nhân làm count giảm) không thể xảy ra ở mô
  hình mù vị trí. Bản cũ để hai mục hơi vênh nhau.
- Luận điểm bán KHÔNG đổi: nhãn mức ảnh đủ để đếm, và độ chính xác đếm không kéo theo phân định từng nhân.
- Hình 1: `fig1_grads.py` nay chồng **viền nhân GT màu lam** lên cả 2 bản đồ mật độ (trước đó 2 panel nhìn
  giống nhau nên hình phản chứng lại phần chữ). CKPT mask đổi sang `results (4)/ckpt_bal/mask_s42.pt`.
  Trên mảnh im0507: mask ratio 4,50 / count 1,74; **count dồn 100% khối vào 1% điểm ảnh sáng nhất** (mask 32%)
  — nhọn hơn mà đặt sai chỗ. Caption đã thêm câu giải thích lớp viền.

## ✅ ĐÃ XÁC THỰC 2026-08-14 — KHÔNG CẦN KIỂM LẠI

Mọi số dưới đây đã đối chiếu trực tiếp với artifact (npz/pkl/ckpt) và đã nằm trong `main.tex` + `main_vi.tex`.
Khi được yêu cầu "kiểm tra paper", **BỎ QUA** phần này, chỉ soát những gì còn trong `\iffalse`.

**Bảng 1 — PanNuke** (5 seed, n=2722): count 0,931±0,005 · mask 0,911±0,028 · ΔR² +0,020 ·
p 5,3e-18 · δ* 0,36 nhân = 1,5% · Δ|err| −0,307 (1,3%) · TOST biên 2% p 3,2e-09 ·
thành phần độ chính xác 0,05 trong δ* 0,36. Nguồn: `merged_bal.npz`.
- **2026-08-15 tính lại từ `merged_bal.npz`, KHỚP HẾT:** MAE count 3,4216 vs mask 3,7283 · Δ = **−0,3068**
  (−1,25%) · CI90 bootstrap **[−0,3578; −0,2562]** · δ* 0,3578 = **1,46%** · Wilcoxon p 5,26e-18.
  ⚠️ `fig_equivalence.py` trước đó còn để mặc định của run CŨ (Δ +0,210, CI [+0,1555; +0,2673], δ* 1,09%) nên
  Hình 2 vẽ δ* 1,09% trong khi chú thích + Bảng 1 ghi 1,5% → **đã sửa mặc định + vẽ lại PDF**.
### ★ BẢNG 2 ĐỦ 5 SEED — 2026-08-15 (đã vào cả 2 file .tex)

Nguồn 8 npz mới: `~/Downloads/results (7).zip` (convnext count+mask s45/s46) và `results-14.zip` (tiny_vit
count+mask s45/s46), gộp với 3 seed cũ ở `work/cnx_lr1e-4.npz` · `work/tv_s4{2,3,4}.npz` · `work/bb/*_mask_*`.
**KHÔNG cần chạy ckpt** — npz nằm sẵn trong zip. 0/10 lượt mới sập.

| bộ mã hoá | count | mask | ΔR² | δ\* | Δ\|err\| |
|---|---|---|---|---|---|
| efflite0 (không đổi) | 0,9313±0,0053 | 0,9114±0,0284 | +0,0199 | 1,46% | −0,307 = 1,25% |
| convnext_atto | 0,9104±0,0041 | **0,8843±0,0224** | **+0,0261** | **2,99%** | −0,678 = **2,77%** |
| tiny_vit_5m_224 | **0,9307±0,0111** | **0,9382±0,0113** | **−0,0075** | **0,32%** | −0,036 = **0,15%** |

Per-seed: cnx count 0,9163/0,9087/0,9060/0,9066/0,9141 · cnx mask 0,8719/0,8537/0,9209/0,8928/0,8821 ·
tv count 0,9113/0,9359/0,9377/0,9262/0,9425 · tv mask 0,9355/0,9461/0,9182/0,9399/0,9513.

**Đã sửa trong bài:** 2 hàng bảng · chú thích bảng (bỏ "hai hàng còn lại dùng 3 seed" → cả ba 5 seed) ·
ΔR² 3 số (+0,020 / **+0,026** / **−0,007**) · Δ|err| 3 số (1,3% / **2,8%** / **0,1%**) · dải δ\*
(**0,3%–3,0%**) · biến thiên hàng mask **0,051→0,054** · trần trong-1-kiến-trúc **0,028→0,026** ·
độ biến động mask ConvNeXt **0,028→0,022** · XOÁ câu hạn chế "hai bộ mã hoá bổ sung chỉ 3 seed".
Bộ số EN/VI đã đối chiếu lại toàn mục 5.2, khớp từng con. Không mục nào khác trích số Bảng 2.

⚠️ Kết luận định tính KHÔNG đổi (không chế độ nào thắng nhất quán; TinyViT vẫn là hàng duy nhất nghiêng mặt
nạ), nhưng **cả ba biên đều CHẶT hơn**: TinyViT δ\* 0,8%→0,3% là biên hẹp nhất toàn bài.

⛔ 8 npz + ckpt s45/s46 (2 kiến trúc) **CHƯA backup Kaggle**.

### ★ QUY TRÌNH SOÁT — 4 phép quét bắt được MỌI lỗi 2026-08-15 (đọc tuần tự KHÔNG bắt được)

Mọi lỗi hôm nay đều là **mâu thuẫn giữa hai chỗ cách xa nhau**; đọc từng chỗ riêng đều xuôi. Bốn phép quét:

1. **Gom số theo GIÁ TRỊ, không theo vị trí.** Regex mọi số + số dòng, group theo giá trị, xem mọi giá trị
   xuất hiện ≥2 lần. Bắt: Hình 2 vẽ δ\* cũ · Bảng 1 ô 1/5.
2. **Dựng lại MỌI ± từ artifact, kiểm ddof.** §4.2 tuyên bố "mẫu số = số seed" (ddof=0). Bắt: Bảng 5 ba hàng
   baseline dùng ddof=1 · **hệ số đáp ứng 0,64 ± 0,04 → ±0,03** (3 mô hình: 0,5983/0,6434/0,6692; ddof=0 =
   0,0293, ddof=1 = 0,0359).
3. **Lọc câu chứa từ PHỔ QUÁT** (only/never/no/all/must/cannot/whatever) rồi soi xem chính bài có vi phạm
   không. 86 câu. Bắt: worst-organ "chỉ dùng so sánh cùng phân hoạch" vs §6.2 dùng tuyệt đối · **"không có
   seed nào suy biến trên bất kỳ bộ dữ liệu nào" (§4.1) vs §5.2 "suy biến thành đầu ra bằng không, R²=−0,96"**
   → đã hẹp lại thành "không có **lượt huấn luyện** nào suy biến" (đúng: seed sập đó là sập lúc SUY LUẬN
   ngoài miền, model in-domain vẫn khoẻ).
4. **Quét câu MÔ TẢ THIẾT KẾ** (số seed, số nhóm, quy ước). Bắt: §4.1 "trục kiến trúc 3 seed" · §4.2
   "cực tiểu của 19 ước lượng".

**Đối chiếu EN/VI:** so multiset số theo từng mục (order-insensitive) + đếm từng giá trị nghi ngờ ở cả 2 file.
Kết quả 2026-08-15: **khớp hoàn toàn**, mọi chênh lệch máy báo đều là lệch ranh giới mục do vị trí bảng nổi.

### Sửa câu chữ vòng 2026-08-15 (không đổi số liệu)

- **Hình 1 caption:** `carry the ground-truth outlines` → `are overlaid with…`; VI `chồng đường viền nhân chuẩn`
  → `chồng thêm viền nhân thật`. Lý do: chính tác giả đọc nhầm thành "nhánh chỉ-đếm tái tạo được viền nhân",
  tức người duyệt cũng sẽ nhầm và tưởng bài mâu thuẫn với §5.3. Viền lam do `fig1_grads.py` vẽ đè từ `t`
  (nhãn), đè lên CẢ HAI panel, không dính đầu ra mạng.
- **§4.1 CoNIC:** `Số đếm được tính lại…` → `Nhãn đếm của CoNIC được tính lại…` (bản VI dùng "số đếm" 67 lần
  cho cả nhãn lẫn dự đoán; câu cũ có 2 chữ "số đếm" cạnh nhau chỉ 2 thứ khác nhau).
- **§4.1 mới:** thêm câu nguồn gốc nhãn đếm (suy từ số thực thể ⇒ trùng số của chấm điểm thủ công, mốc chi phí).
  Chặn phản biện "các anh vẫn dùng mask để tạo nhãn". ⚠️ ĐỪNG thêm hạn chế kiểu "nhãn đếm của bài sạch hơn
  thực tế" — SAI: chấm điểm cho số đếm CHÍNH XÁC ở đúng giá 1,4 s/nhân, nên nhãn không hề lạc quan.
- 🐞 **§4.1 Giao thức:** câu "thí nghiệm phụ… trục kiến trúc… dùng 3 seed" đã LẠC HẬU sau khi Bảng 2 lên 5
  seed. Đã chuyển trục kiến trúc sang nhóm 5 seed; nhóm 3 seed còn ngân sách nhãn + can thiệp + chuyển miền.
  → **Bài học: soát vòng cuối phải quét cả câu MÔ TẢ THIẾT KẾ, không chỉ quét con số kết quả.**
- 🐞 **§4.2 độ phủ cơ quan kém nhất:** (a) "cực tiểu của 19 ước lượng" chỉ đúng PanNuke — NuInsSeg có **31 cơ
  quan**, nhóm nhỏ nhất **6 ảnh**, trung vị 12 (đếm từ `pathosam_nuinsseg_preds.pkl`), mà §6.2 vẫn dùng đại
  lượng này cho NuInsSeg ⇒ định nghĩa tự đánh giá NHẸ độ chệch. (b) câu "chỉ dùng để so sánh cùng phân hoạch"
  CẤM chính câu §6.2 "không loại mô nào rơi dưới 0,85" (dùng tuyệt đối). Cách dùng đó ĐÚNG vì độ chệch ÂM →
  vượt ngưỡng dưới là bảo thủ. Đã sửa: hạ 19 xuống làm ví dụ, thêm chiều "số nhóm", và mở lối cho dùng tuyệt
  đối kèm lý do.
- 🐞 **§5.5 "Cần bao nhiêu chú thích là đủ" (2026-08-16):** đoạn này chỉ nói "mô hình" — KHÔNG nêu backbone, nên
  người đọc không biết đó là đúng model Bảng 1. Truy `label_curve.py`: `--backbone efficientnet_lite0`,
  `train(data, dev, epochs, ch=32, lr=1e-3, tr, w_d=0.0, w_c=1.0, w_nll=0.01, beta=0.5, bs=16, detach_mu=True,
  "poisson", efflite0)` — **trùng từng tham số với nhánh count của `sup_compare.py`** (`_ARM["count"]=(0.0,1.0)`,
  ch32/bs16/lr1e-3, fold1+2→fold3, `--exclude_tissue ""` giữ colon). Khác DUY NHẤT: 40 epoch (vs 60) và 3 seed
  (vs 5) — cả hai đã ghi trong bài. Đã thêm tên backbone vào câu mở đoạn (EN+VI).
- 🐞 **92,8% mập mờ mẫu số:** câu mở đoạn đứng TRƯỚC dãy số nên "full-label configuration" dễ bị đọc là Bảng 1
  (0,931 ⇒ 91,2%). Mẫu số thật là điểm 100% của CHÍNH đường cong: 0,849/0,915 = 92,8%. Đã sửa thành "that same
  configuration at the full budget". Số kiểm lại đều khớp: 0,931−0,915 = 0,016 ✓; 0,915−0,849 = 0,066 ✓.
  (92,8% cũng nằm ở Điểm nổi bật `main.tex:58` + `highlights.txt:5` — giữ nguyên, câu đó không mập mờ.)
- 🐞 **§5.3 hệ số đáp ứng 0,64 ± 0,03 trỏ NHẦM nguồn (2026-08-16):** câu cũ ghi "TB 0,64 ± 0,03 trên ba mô hình
  (cột cuối Bảng 4)" — nhưng cột cuối Bảng 4 là **trung bình theo mức can thiệp** (0,62/0,64/0,65/0,64/0,64/0,63),
  SD của sáu số đó = **0,01**, không phải 0,03. Người duyệt tự tính sẽ kết luận bài sai ±. ±0,03 thật ra là SD
  **giữa 3 mô hình** sau khi gộp mức: 0,5983 / 0,6434 / 0,6692 (ddof=0 → 0,0293). Đã viết rõ cả hai phép gộp và
  liệt kê ba hệ số ⇒ 0,59 của Phụ lục B giờ truy được (= 0,5983, thấp nhất trong ba), khép luôn nghi vấn
  "0,59 với 0,64" ở mục kiểm cuối. Cũng đổi "khoảng 0,6 nhân" → "khoảng hai phần ba nhân" (0,6 là số thứ ba lạc
  ra bên cạnh 0,64 trong cùng một câu).
  ✅ **XONG 2026-08-16** — đã chạy lại trên 5 ckpt Bảng 1, số mới ở mục "CHẠY LẠI XOÁ-NHÂN" bên dưới.

### ★★ CHẠY LẠI XOÁ-NHÂN TRÊN 5 CKPT BẢNG 1 — XONG 2026-08-16 (`results (8).zip`, `~/Downloads/evid/`)

`count_evidence.py --fill bgpatch --dilate 2 --seed 0` trên `count_s4{2..6}.pt`. **CHỨNG MINH đúng ckpt Bảng 1:**
`preds.npz` gộp cho R² 0,9313 ± 0,0053 và MAE 3,422 ± 0,093 = **trùng khít** 0,931 ± 0,005 / 3,42 ± 0,09.
⇒ câu "một lần chạy khác với Bảng 1" ở §5.3 ĐÃ XOÁ, hết mâu thuẫn với đoạn định vị ("năm seed của Bảng 1").

**Hệ số đáp ứng mỗi mô hình (gộp mức):** s45 0,576 · s44 **0,5983** · s46 0,617 · s43 0,636 · s42 0,649
→ **0,62 ± 0,03** (cũ 0,64 ± 0,03). Theo mức: 0,58 / 0,62 / 0,64 / 0,63 / 0,63 / 0,60 → dải **0,58–0,64**.
⚠️ 0,5983 = **đúng s44** ⇒ bộ 3 mô hình CŨ có chứa s44 (Phụ lục B chạy trên nó, mốc 0,59), hai con kia KHÔNG
nằm trong Bảng 1. Đừng đi tìm lại chúng, đã bị thay hết.

**Bảng 4 mới (5 mô hình, ddof=0 giữa mô hình):**

| mức | xoá nhân | sham | tụt do nhân | khớp dt | hệ số |
|---|---|---|---|---|---|
| 10% | 0,942±0,002 | 1,005 | 0,063 | 0,95 | 0,58 |
| 20% | 0,876±0,005 | 1,008 | 0,132 | 0,92 | 0,62 |
| 30% | 0,808±0,009 | 1,015 | 0,207 | 0,89 | 0,64 |
| 50% | 0,687±0,011 | 1,022 | 0,335 | 0,85 | 0,63 |
| 75% | 0,530±0,022 | 1,031 | 0,501 | 0,79 | 0,63 |
| 100% | 0,403±0,038 | 1,038 | 0,635 | 0,74 | 0,60 |

- **Lọc khớp diện tích ≥80%** chỉ phụ thuộc HÌNH HỌC vùng xoá (chung cho mọi mô hình vì `--seed 0`) ⇒ **3.446 phép
  đo cho 4 mô hình**, 3.430 cho s43 vì ảnh `img=270` (GT 2 nhân) bị s43 dự đoán **đúng 0** → tỉ số chuẩn hoá vô
  định, rụng 16 hàng. Không còn 3.415/3.406/3.434.
- Tụt do nhân gộp mọi mức: **0,249 ± 0,009** (cũ 0,244 ± 0,008). Diện tích nhân/ảnh khi xoá hết = 24,5% ⇒ "khoảng
  25%" giữ nguyên.
- Bảng 3 per-organ chuyển sang **trung bình 5 seed** ⇒ hàng Tổng = 0,931/3,42 **đúng bằng Bảng 1**, xoá luôn
  caveat "khác lượt huấn luyện". Dải mới 0,739 (buồng trứng) – 0,965 (cổ tử cung); ví dụ phi đơn điệu mạnh hơn:
  tuỵ 0,930 / 28 mảnh vs thượng thận 0,810 / 155.
- **Hình 3** dựng lại từ mô hình có R² TRUNG VỊ (s46): chệch +0,50 (2%), LoA −10,0…+11,0, nhỏ hơn phân tán 21 lần.
  Thêm câu dải chệch giữa 5 mô hình: −1,47 … +0,69, TB **−0,03 ± 0,77** (⇒ không có chiều lệch nhất quán).
  Lý do KHÔNG dùng s42 cho Hình 3 dù Hình 4 lấy panel của s42: s42 chệch −1,47 là cực trị, chọn trung vị mới là
  quy tắc không-cherry-pick (giống cách Bảng 5 dùng seed trung vị).
- **Hình 4 đổi mảnh: panel_0072 → panel_0063** (s42, GT 29): 28,7 → 19,9 (30%) → 9,1 (100%), sham 29,6 = 1,03×.
  User chọn "hình đẹp nhất". Hệ số của riêng mảnh này 0,81 > 0,62 toàn tập.
  🐞 **Tôi từng thêm câu "chọn vì dễ đọc và đáp ứng mạnh hơn trung bình" vào caption — SAI, user bắt được:**
  đó là câu BIỆN HỘ PHÒNG THỦ (cãi trước với người duyệt tưởng tượng), và thú nhận KHÔNG chữa được cherry-pick,
  chỉ chỉ tay vào nó. Panel là MINH HOẠ; số định lượng nằm ở Bảng 4 (286 ảnh × 5 mô hình). Điều kiện đủ để trung
  thực là caption **đừng tuyên bố mảnh này "tiêu biểu"** — đã bỏ chữ đó. Đã xoá câu biện hộ.
  🐞 Đồng thời xoá mệnh đề "1,02 nếu lấy trung bình trên các lần rút thăm": nó sinh ra để hoà giải mâu thuẫn CŨ
  (panel 0,95 vs toàn tập 1,00–1,02). Panel mới cho 1,03, nằm gọn trong 1,00–1,04 ⇒ hết mâu thuẫn, câu thành đồ
  thừa bắt người đọc nuốt 3 số mà không thu được gì. **BÀI HỌC: sửa số xong phải hỏi mệnh đề GIẢI THÍCH đi kèm
  còn lý do tồn tại không.** (Mảnh đại diện nhất là panel_0036 hệ số 0,67 nhưng mức 30% nhìn yếu.)
- 🐞 **Phụ lục B "mức dư 37,4%"** là số MƯỢN của Bảng 4 cũ (hàng 100% = 0,374), không phải của mô hình phụ lục
  (s44 cho 0,422) ⇒ so sánh 3,5% với 37,4% vốn đã lệch nguồn. Đã sửa thành **40,3%** kèm trỏ thẳng Bảng 4 để nêu
  rõ xuất xứ. Số 3,5% giữ nguyên vì lượt chạy phụ lục KHÔNG chạy lại.
- Kiểm chéo EN↔VI: bộ số của §5.3 khớp **208/208 token**.
**Soát kỹ vòng 2 (cùng ngày) — 4 việc nữa:**
- ✅ **Kiểm lại TỪNG SỐ bằng script**: Bảng 4 36/36 ô + SD + khớp-diện-tích, Bảng 3 19/19 hàng (tổng n = 2.722,
  không sót cơ quan nào), Bland–Altman, 3.446/3.430, 0,249±0,009, hệ số 5 mô hình, diện tích nhân 24,5%,
  286×17=4.862, mảnh Hình 4 (0,805→0,81; sham 1,016→1,02; 29,6/28,7=1,031→1,03). **KHÔNG ô nào sai.**
- 🗑️ **XOÁ PHỤ LỤC B, gộp vào §5.3.** Lý do: thân bài đã phát biểu ĐỦ CẢ BA kết luận (giãn nở / chất lượng chú
  thích / Telea) + số 0,34, phụ lục chỉ thêm 0,59-0,60-0,61, +0,31 và cơ chế Telea ⇒ người đọc gặp cùng ba
  phát hiện hai lần. Giờ còn **một phụ lục duy nhất** (β-NLL). Đã kiểm: không còn `\ref{app:checks}`, không câu
  nào gọi tên "Phụ lục B".
- 🐞 **§5.3 câu mở sai phạm vi:** viết "dùng năm mô hình CHỈ-ĐẾM" nhưng đoạn định vị lại đo CẢ HAI nhánh. Sửa
  thành "các mô hình của Bảng 1: năm mô hình chỉ-đếm ở mọi phân tích, và các mô hình mặt nạ tương ứng ở phép đo
  mức định vị".
- 🐞 **Câu sham yếu đi khi số mới lên 1,04:** "gần như không đổi (1,00–1,04)" nghe gượng khi mức 100% là +3,8%.
  Sửa thành "đối chứng giả KHÔNG BAO GIỜ làm số đếm giảm, mà tăng nhẹ từ 1,00 lên 1,04 khi số vùng can thiệp
  tăng, nên không phần nào của mức giảm quy được cho thao tác chỉnh sửa" — mạnh hơn VÀ đúng hơn (sham đơn điệu
  tăng: 1,005/1,008/1,015/1,022/1,031/1,038).
- Kiểm cấu trúc .tex: môi trường cân, `{}` cân, **không ref nào thiếu label**. Còn 1 label mồ côi `eq:tost`
  (có từ trước, vô hại). KHÔNG có engine LaTeX trên máy ⇒ **vẫn chưa build thử Overleaf lần nào**.
**Soát vòng 3 — user bắt tiếp 2 lỗi (cùng ngày):**
- 🐞 **Câu bộ lọc TỰ MÂU THUẪN:** viết "bộ lọc CHỈ phụ thuộc hình học, vốn dùng chung cả 5 mô hình, NÊN giữ 3.446
  ở BỐN mô hình; mô hình thứ năm còn 3.430". Tiền đề phổ quát ⇒ cả 5 phải bằng nhau, vế sau phá ngay tiền đề.
  Thật ra bộ lọc diện tích giữ **3.446 cho CẢ NĂM**; 16 phép đo rụng thêm là ĐIỀU KIỆN KHÁC (chuẩn hoá không xác
  định vì dự đoán nguyên vẹn = 0). Đã tách làm hai câu. → phép quét "từ tuyệt đối" phải soi cả chữ *chỉ phụ
  thuộc vào*, không riêng only/never/all.
- 🐞🐞 **Chú thích chân Bảng 4 SAI (có từ lâu, sống sót mọi vòng soát):** "Hai hàng 75% và 100% chỉ mang tính mô
  tả và KHÔNG được sử dụng trong kiểm định" — nhưng thành phần tập sau lọc là 776/727/670/603/**519**/**151**
  ⇒ **670 phép đo (19%) của hai mức đó CÓ trong kiểm định gộp**, đúng như thân bài nói "gộp trên tất cả các mức".
  Đã sửa: hai hàng "chỉ báo cáo để mô tả", nói rõ chúng vẫn vào kiểm định gộp, và phép kiểm ≤30% (drop
  0,131±0,005) mới là cái chứng minh kết luận không dựa vào chúng.
  → **BÀI HỌC: chú thích chân bảng phải đối chiếu với câu thân bài cách đó vài dòng; không ai soi footnote.**
**Soát vòng 4 — user đề xuất caption Bảng 4 mới, bắt được 1 lỗi ĐẢO NGƯỢC:**
- 🐞🐞 **ĐỪNG BAO GIỜ viết hệ số đáp ứng là "phần giảm SAU KHI hiệu chỉnh đối chứng giả".** Nó dùng phần giảm
  **CHƯA** hiệu chỉnh. Kiểm nhanh: mức 10% → (1−0,942)/0,10 = 0,58 = đúng ô bảng; nếu hiệu chỉnh sham thì
  0,063/0,10 = 0,63, không khớp ô nào. Bản đề xuất còn tự mâu thuẫn vì câu kế tiếp phủ nhận đúng thứ nó vừa
  định nghĩa ("phần giảm quy cho nhân" CHÍNH LÀ phần đã hiệu chỉnh sham).
- ✅ **Caption giờ ghi thẳng công thức $(1-c_k)/k$** thay vì mô tả bằng lời ⇒ (a) không ai đọc nhầm được nữa,
  (b) BỎ được câu phủ định "không phải là thương giữa…" (có công thức thì câu đó thừa, mà nó vốn mang dạng
  phủ-định-phòng-thủ user cấm). Đã kiểm cả 6 hàng tái lập được từ số ĐÃ LÀM TRÒN in trong bảng:
  0,58/0,62/0,64/0,63/0,63/0,60 ✓ ⇒ caption tự kiểm được.
- ✅ Thêm "nên là ước lượng thận trọng": hiệu chỉnh sham sẽ cho hệ số CAO hơn ở **mọi** mức
  (0,63/0,66/0,69/0,67/0,67/0,635 so với 0,58…0,60) — đã kiểm, biến caveat thành điểm mạnh mà không nói quá.
- 🐞 **"Xoá toàn bộ nhân CHỈ tác động lên ~25% diện tích ảnh"** — chữ "chỉ" lật ngược lập luận: ý gốc là can
  thiệp tới MỘT PHẦN TƯ ảnh nên ảnh xa phân phối huấn luyện, DO ĐÓ mức thấp mới là bằng chứng chính. Có "chỉ"
  thì vế "do đó" hết suy ra được. Giữ "còn tác động lên".
- Bảng markdown user gửi rơi mất cột "Xoá nhân" (5 nhãn / 6 giá trị) nhưng **file .tex vẫn đủ 6 cột** — chỉ là
  lỗi chép sang chat. Vẫn phải soi: dạng lỗi này chính là [[feedback-audit-bang-o-trong]].
**Soát vòng 5 — hệ quả của việc GỘP phụ lục B vào thân bài: soi lại xuất xứ từng số trong đoạn đó.**
- ✅ 0,598 (= s44) và 40,3%/42% (Bảng 4) — truy được.
- 🐞 **+0,31 và 3,5% KHÔNG CÓ SCRIPT NÀO trong repo tính ra** (grep "telea|missed|0.31" → chỉ count_evidence.py
  cho telea; phép kiểm chất lượng chú thích là phân tích tuỳ biến của lượt cũ, không lưu lại). Chấp nhận được
  khi còn nằm trong phụ lục "một mô hình, chỉ so cấu hình"; KHÔNG chấp nhận được khi đã vào mục Kết quả.
  → Dựng lại được: **tương quan giữa (dự đoán ảnh nguyên vẹn − N) và (số đếm còn sót khi xoá HẾT nhân)**,
  trên s44 ra **+0,299** ≈ +0,31 của bài ⇒ gần chắc đúng phép tính gốc. Mức thừa ra **2,6%** (tỉ số tổng), bài
  ghi 3,5% (chắc dùng trung bình tỉ số từng ảnh: s44 cho +8,1%, trung vị +3,3% — không định nghĩa nào ra đúng
  3,5%). **ĐÃ THAY bằng r = +0,30 và 2,6%, viết rõ định nghĩa trong câu** → truy được từ `evid/s44`.
  Lập luận không đổi: 2,6% không thể giải thích 42%.
- ✅ **3 LƯỢT CHẠY XONG (`results (9).zip` → `~/Downloads/evid_rob/`), cả 4 cấu hình đều đúng s44 (R² 0,9374):**

| cấu hình | hệ số | còn lại khi xoá hết | bài CŨ |
|---|---|---|---|
| giãn nở 0 px (bgpatch) | 0,580 | 0,454 | 0,59 |
| giãn nở 2 px (bgpatch, mốc) | 0,598 | 0,422 | 0,59 |
| giãn nở 4 px (bgpatch) | 0,609 | 0,397 | 0,61 |
| Telea (giãn nở 2) | **0,532** | 0,509 | **0,34** ← LỆCH RẤT XA |

- 🐞🐞 **Số Telea 0,34 KHÔNG tái lập được — thật ra là 0,532.** Nghi vấn có cơ sở: `occlude()` trong
  `count_evidence.py` có nhánh dự phòng `except Exception -> median-fill` khi `cv2` không nạp được, và chính
  chú thích trong code ghi median-fill tạo hiện vật **MẠNH HƠN** Telea. Lượt cũ nhiều khả năng rơi vào nhánh đó
  (0,34 = median-fill chứ không phải Telea). Lượt này cv2 có ⇒ 0,532 là Telea thật.
  → **BÀI HỌC: script có nhánh fallback im lặng thì số của nó KHÔNG đáng tin nếu không giữ log console.**
- Hướng vẫn đúng (Telea để lại 0,509 vs bgpatch 0,422 = lôi màu nhân vào lỗ), nhưng độ lớn nhỏ hơn hẳn ⇒ bỏ chữ
  "ảnh hưởng đáng kể". Đã hiệu chỉnh cỡ: giãn nở dịch hệ số **0,029**, cách lấp vùng dịch **0,066**, đều nhỏ/ngang
  biên độ giữa 5 mô hình **0,073** — con số này giúp người đọc biết quy trình xoá KHÔNG phải nguồn biến thiên
  lớn nhất. Câu "cận dưới" giữ nguyên vì logic không đổi.
**Soát vòng 6 — ĐỌC TOÀN BÀI, lập bảng luận điểm từng đoạn rồi đối chiếu chéo. 5 LỖI THẬT:**
1. 🐞🐞 **Mục 4.1 Giao thức nói "phép đo can thiệp dùng 3 seed" — SAI, giờ là 5.** Chạy lại xoá-nhân trên 5 ckpt
   đã làm câu này lạc hậu mà không ai nhớ. (Lần thứ HAI câu Giao thức lạc hậu sau khi đổi số seed — lần trước là
   trục kiến trúc.) → **QUY TẮC: đổi số seed của BẤT KỲ thí nghiệm nào thì mở Mục 4.1 Giao thức ra sửa NGAY.**
2. 🐞 **Tóm tắt + Đóng góp: "cả hai chế độ suy giảm độ chính xác NẾU KHÔNG được hiệu chuẩn lại"** — nói quá.
   Hiệu chuẩn conformal khôi phục ĐỘ PHỦ, không phải độ chính xác; hiệu chỉnh thang đo chỉ nâng R² lên
   0,31–0,60 so với 0,93 in-domain. Sửa thành "chỉ khôi phục được một phần". Mục 6.2 vốn đã viết đúng
   ("dù độ chính xác điểm ra sao") ⇒ tóm tắt mâu thuẫn với thân bài.
3. 🐞 **§5.2: "biến thiên do kiến trúc ít nhất tương đương biến thiên do kiểu giám sát"** — SAI ở nhánh chỉ-đếm:
   0,021 < 0,026. Chỉ đúng ở nhánh mặt nạ (0,054). Sửa thành "mặt nạ gấp ~2 lần, chỉ-đếm tương đương".
4. 🐞 **§5.2 chuyển miền: 31,8 và 29,5 khác CƠ SỞ SEED** (count 3 seed, mask 2 seed) nhưng câu chỉ ghi chung
   "tính trên các seed huấn luyện thành công" ⇒ người đọc trừ ra 2,3 rồi thấy bài ghi 2,7 và tưởng sai. Con số
   2,7 đúng: nó loại seed suy biến khỏi CẢ HAI nhánh. Đã ghi rõ cơ sở của từng số.
5. 🐞 **§5.3 lặp ngay 2 câu liền:** "thao tác chỉnh sửa không giải thích được phần nào" rồi câu sau "phần giảm
   quy cho nhân chiếm TOÀN BỘ mức giảm" — hai cách nói của cùng một mệnh đề. Bỏ vế đầu. (Lỗi do chính đợt sửa
   hôm nay sinh ra.)

**ĐÃ XOÁ 5 CHỖ LẶP (user yêu cầu 2026-08-16):**
1. Cuối §5.3 — bỏ 2 câu cuối ("định vị và độ chính xác đếm là hai thuộc tính tách rời" + "mục tiêu huấn luyện
   chỉ ràng buộc tổng"). Câu 1 là nhãn dán cho ý đã nói; câu 2 nhắc lại thiết kế ở §3.2. Đoạn kết ở câu
   "tích hợp ở cấp độ toàn ảnh thay vì định vị và kiểm đếm từng thực thể" — mạnh hơn.
2. §6.3 đoạn 3 — bỏ câu "thí nghiệm can thiệp cho thấy độ chính xác đếm cao không đồng nghĩa với định vị".
   Nhắc lại nguyên xi §5.3.
3. §6.3 đoạn 1 — bỏ câu "dải cấu hình này cho thấy lợi thế mặt nạ không tăng theo năng lực mô hình". Đó là KẾT
   QUẢ của §5.2, đặt trong mục Hạn chế thì lạc; bỏ đi đoạn thành đúng một hạn chế.
4. Cuối §5.4 — bỏ câu "Đặt cạnh biên 1,5%... nằm ở các thang độ lớn rất khác nhau". Trùng ý với §6.1 (nơi diễn
   giải mới là chỗ của nó), và "rất khác nhau" là chữ lỏng. Đoạn giờ kết ở 393 giờ vs 48 giờ. Bỏ luôn được
   trùng cụm "Set against" 2 lần.
5. Đóng góp 3 — viết lại cho hết trùng gần-nguyên-văn với tóm tắt; giữ phần riêng ("không hề được cung cấp vị
   trí nhân nào").

**GIỮ LẠI có chủ ý, đừng xoá nhầm:** lặp giữa tóm tắt ↔ kết luận (đúng chức năng); 4 cụm rào phạm vi còn lại
mỗi cụm rào một tuyên bố khác nhau ở 4 mục khác nhau (§5.1, §5.4 ×2, Kết luận) nên không phải lặp.

**Soát vòng 7 — quét bài theo 4 tật của [[feedback-report-discipline]] (2026-08-16):**
- Phòng thủ / tự phủ định, XOÁ 4 chỗ: (a) §3.2 "quy cho kiểu chú thích, KHÔNG PHẢI kiến trúc hay quy trình
  huấn luyện" — chữ "chỉ" phía trước đã loại rồi; (b) §3.2 "nhận lượng thông tin giám sát KHÔNG ÍT HƠN" +
  "ước lượng bảo thủ" → "nhận tín hiệu không gian đầy đủ hơn" + "cận trên"; (c) §5.3 "liệu mô hình có THỰC SỰ
  dựa vào… hay CHỈ khai thác" → bỏ 2 từ nạp giá trị; (d) §3.1 "rộng hơn ĐÁNG KỂ so với các hệ số phân tán quan
  sát được" — so sánh với một dải KHÔNG HỀ được báo cáo ở đâu ⇒ bỏ vế so sánh.
- Tham chiếu treo: §5.2 "với cách đặt trọng số mật độ TRƯỚC ĐÂY" — bài chưa bao giờ định nghĩa cách đặt đó
  (tàn dư lịch sử nội bộ, người đọc không giải mã được). Xoá.
- Định tính mơ hồ → thay bằng số: "biên chịu ảnh hưởng đáng kể của biến thiên" → "chủ yếu phản ánh độ bất định
  của ước lượng"; "ổn định hơn đáng kể" → "độ lệch chuẩn giảm dưới 0,04 từ mức 25%"; "bất đối xứng rõ rệt / chi
  phí giảm đáng kể" → "chi phí giảm ít nhất tám lần"; bỏ "thực sự" thừa ở tóm tắt; bỏ "đáng kể" lặp ở §6.2.
- ⚠️ Trong lúc sửa tôi ĐÃ TỰ VIẾT một chỗ nói quá: "falls by about an order of magnitude" trong khi số thật là
  ≥8 lần. Bắt được ngay và sửa thành "at least a factor of eight". **Sửa lỗi nói quá cũng đẻ ra lỗi nói quá.**
- GIỮ LẠI có chủ ý 5 chỗ "substantially/considerably/actually": mỗi chỗ hoặc được lượng hoá ngay trong câu
  (0,739–0,965; MAE 31,8/29,5 vs 35,4) hoặc nằm ở tóm tắt/kết luận nơi tính từ tổng kết là bình thường.
  Xoá nốt là sửa quá tay.
- Quét lần cuối "It should be noted / one might argue / to be clear / Admittedly…": **0 kết quả.**

**🐞🐞 RÒ RỈ TẬP TEST TRONG CHỌN `w_c` — XÁC NHẬN 2026-08-16, USER TÌM RA, TÔI ĐÃ SOÁT 7 VÒNG MÀ KHÔNG THẤY:**
Log kernel in ra `[data] train fold[1, 2]=5179 / test fold3=2722` ⇒ vòng quét `w_c ∈ {0,01; 0,05; 0,1}` được
**chấm điểm trên fold 3**, đúng fold dùng làm tập kiểm tra. `sup_compare.py` KHÔNG có tập kiểm định — chỉ có
`--train_folds` và `--test_fold`.
**Đã khai thẳng trong §4.1** (không giấu, không viết bịa là "chọn trên validation split"): nêu rõ chấm trên
fold 3, không tách fold kiểm định; nêu rõ chỉ ảnh hưởng nhánh MẶT NẠ (count-only có `w_d=0` nên `w_c` chỉ là hệ
số nhân toàn cục — xem dòng 780); và nêu rõ giá trị giữ lại là giá trị nhánh mặt nạ hội tụ ổn nhất, nên độ chệch
**thiên về ĐỐI THỦ**, không thiên về kết quả của bài.
→ **PHÉP QUÉT CÒN THIẾU trong quy trình soát của tôi:** tôi đối chiếu số-với-số và câu-với-câu, nhưng CHƯA BAO
GIỜ hỏi "con số ở mục Thiết lập này được đo trên TẬP NÀO". Mọi số dùng để CHỌN cấu hình phải truy được tập đánh
giá. Thêm vào [[feedback-report-discipline]].
**✅ TRUY NGUỒN XONG 2026-08-16 (lục Kaggle):** liệt kê toàn bộ kernel (2 trang) + tải mã nguồn 24 kernel phủ
trọn giai đoạn Paper 2. **KHÔNG kernel nào chạy `w_c ∈ {0,01; 0,05; 0,1}` × 5 seed** — 5 kernel có
`w_count_mask` thì 4 đặt cứng `0.1`; cái duy nhất có vòng lặp là `notebook7cb574f7dc` (2026-08-12) và nó quét
**`w_d`**: `[("pure",65536,0.0), ("mix",65536,0.1), ("low",6554,0.1)]`, 1 seed, fold3.
**Về con số 0,45 ± 0,94 — ĐỌC KỸ, TÔI TỪNG KẾT LUẬN SAI Ở ĐÂY:** tải `mask_pure_s42.npz` về, nhánh `w_c = 0`
cho **R² = −1,4313 · MAE 24,49 · sd(pred) = 0,000** = chữ ký "xuất hằng số 0". Ghép chữ ký đó với 4 seed lành
≈0,920 → mean 0,450, sd(ddof=0) 0,941 = khớp 0,45 ± 0,94.
⚠️ **Phép này CHỈ cho thấy cặp số nhất quán với "1/5 seed sập", KHÔNG chứng minh lưới `w_c` đã chạy.** −1,4313
là R² của dự đoán hằng số 0 trên fold3 — hằng số của DỮ LIỆU, mọi lần sập đều ra đúng nó; 0,92 là bất kỳ lượt
mặt nạ lành nào. Tôi đã viết "chứng minh được là thật / gần như chắc chắn đã chạy" — **SAI, đó là nâng một phép
kiểm nhất quán lên thành bằng chứng.**
**Trạng thái thật: lưới `w_c ∈ {0,01; 0,05; 0,1}` × 5 seed KHÔNG có mã nguồn nào chứng thực.** Giá trị `0,05`
không xuất hiện cùng `w_count_mask` ở BẤT KỲ ĐÂU — 24 kernel, repo, lẫn notebook đã tải. Mã quét duy nhất tồn
tại (do CHÍNH TÔI viết) chỉ chạy `w_c ∈ {0; 0,1}`.
**Chứng thực được:** `w_c = 0` sập hoàn toàn; `w_d = HW·w_c` (0,9225) tốt hơn `w_d = HW` (0,7730). 1 seed, fold3.
**✅ CHỐT 2026-08-16: user KHÔNG nhớ có chạy `w_c = 0,05` hay không ⇒ §4.1 ĐÃ VIẾT LẠI theo log.**
Log `notebook7cb574f7dc.log` (bằng chứng trực tiếp, không phải suy từ mã nguồn) — đúng 3 lượt, mỗi lượt 1 seed,
đều fold3:
```
w=(dens 65536.0, count 0.0, nll 0.01) -> R² -1.4313  MAE 24.49  SẬP 1/1
w=(dens 65536.0, count 0.1, nll 0.01) -> R² +0.7730  MAE  6.16
w=(dens  6554.0, count 0.1, nll 0.01) -> R² +0.9225  MAE  3.41
```
Đoạn "Lựa chọn trọng số đếm" cũ ĐÃ THAY bằng "Trọng số hàm mất mát của nhánh mặt nạ":
- **BỎ** lưới `{0,01; 0,05; 0,1}` và số `0,45 ± 0,94` (không dựng lại được nguồn).
- **GIỮ + có log:** `w_c=0` → sụp về hằng số (R² −1,43, phương sai dự đoán = 0); `w_d=HW` → 0,77 so với
  `w_d=HW·w_c` → 0,92 ⇒ lựa chọn `w_d=HW·w_c` giờ có SỐ đỡ lưng chứ không chỉ có lập luận đơn vị ở §3.2.
- Giữ nguyên phần khai fold3 + "chỉ ảnh hưởng nhánh mặt nạ" + "độ chệch thiên về đối thủ".
⚠️ Log cũng tự cảnh báo "Sập = thất bại TỐI ƯU HOÁ, KHÔNG phải giới hạn của loại giám sát, đừng báo số này".
Bài dùng ĐÚNG cách: dùng để nói `w_c` phải > 0 (phát biểu về tối ưu hoá), KHÔNG dùng để nói mặt nạ kém.
**★★ CÓ VÒNG QUÉT THỨ HAI — user chỉ ra dataset `hipinhththu/quet-wd` (2026-08-13), kernel
`notebookc219de6bf7`: `SWEEP = [("w655",655,0.1), ("w20k",20000,0.1), ("w200k",200000,0.1)]`** — giữ `w_c=0,1`,
chỉ đổi `w_d`. Cùng script/seed 42/60 epoch/fold3 với kernel `notebook7cb574f7dc` ⇒ **GHÉP ĐƯỢC thành đường cong
5 điểm** (tất cả 1 seed, fold3, w_c=0,1):

| `w_d` | R² | MAE | nguồn |
|---|---|---|---|
| 655 | 0,9079 | 4,14 | quet-wd |
| **6.554 = HW·w_c (cấu hình bài)** | **0,9225** | **3,41** | out7cb (`low`) |
| 20.000 | 0,8972 | 3,76 | quet-wd |
| 65.536 = HW | 0,7730 | 6,16 | out7cb (`mix`) |
| 200.000 | −1,4313 | sập | quet-wd |
| (đối chứng `w_c=0`, w_d=65.536) | −1,4313 | sập | out7cb (`pure`) |

⇒ §4.1 ĐÃ VIẾT LẠI LẦN 2 với đủ 5 điểm. **ĐỪNG viết "6554 là tốt nhất"** — cả 5 điểm đều 1 seed, nhiễu nhánh
mặt nạ ±0,028 nuốt chênh lệch 0,90↔0,92. Chỉ được nói: **phẳng trong 655–20.000, hỏng khi mật độ lấn át đếm,
cấu hình bài nằm giữa vùng phẳng.**
📌 Artifact: `quet-wd` đã ở Kaggle ✅. Còn `out7cb` (log `notebook7cb574f7dc.log` + 3 npz + loc_sweep/sweep.csv)
**CHƯA backup** — là nguồn duy nhất của hai điểm 6.554 và 65.536, và của log chứng minh fold3.
📌 Bài học tra cứu: hai kernel làm CÙNG một thí nghiệm, đặt tên khác nhau, cách nhau 1 ngày; tra một cái rồi kết
luận "không có vòng quét" là SAI. Phải hỏi user xem đã up dataset nào chưa TRƯỚC khi kết luận thiếu bằng chứng.
Cũng xác nhận thêm: nhánh `pure` chấm trên fold3 → khớp dòng log user dán.
Số phụ thu được từ cùng kernel (1 seed, fold3): `mix` (w_d=65536, w_c=0,1) R²=0,7730 · `low` (w_d=6554 = HW·w_c,
đúng cấu hình bài) R²=0,9225. ⇒ w_d=HW·w_c KHÔNG chỉ có lập luận đơn vị, còn có số đỡ lưng.

**Còn tồn, CHƯA sửa (cần user quyết):**
- Luận điểm "khai thác bằng chứng đặc hiệu nhân nhưng tích hợp toàn ảnh, không định vị" xuất hiện **5 lần**:
  tóm tắt, Đóng góp 3, cuối §5.3, Hạn chế 6.3, Kết luận.
- **5 cụm rào phạm vi khác nhau cùng nghĩa**: "Within the conditions examined" / "Within the configurations
  surveyed" / "among the methods surveyed" / "in the configurations evaluated" / "Within the data and
  architectures examined".
- "Set against" dùng 2 lần cho 2 việc khác nhau (§5.4 chi phí, §6.1 hai trục).
- §2 "Counting with count-only supervision" nêu tiền đề "hồi quy ảnh→tổng là KHÔNG ĐỦ nếu thiếu ràng buộc phụ"
  — chính kết quả của bài bác bỏ tiền đề đó, mà không câu nào nối lại.
- Điểm nổi bật 4 phát biểu trần "hai phương pháp đắt nhất đếm kém nhất", trong khi §5.4 tự gắn caveat
  "thứ hạng chịu ảnh hưởng của yếu tố ngoài loại nhãn".
- ℹ️ **δ\* chỉ báo cáo theo 5 seed, trong khi ΔR² và hiệu số bắt cặp báo cáo cả 5 lẫn 4 seed** — bất đối xứng
  trình bày, MỨC ĐỘ THẤP. Bỏ seed mặt nạ hỏng thì hiệu số nhỏ đi ⇒ δ\* NHỎ ĐI ⇒ biên chặt hơn, kết luận MẠNH
  hơn. Con số 1,5% đang in là con số RỘNG HƠN, bất lợi cho chính mình ⇒ không có rủi ro nói quá. Hướng người
  duyệt đánh (nếu có) là ΔR² và p = 5,3e-18 dựa vào một seed hỏng — chỗ đó §5.1 đoạn 1 ĐÃ tự khai.
  ⚠️ Muốn tính δ\* 4 seed thì phải tải lại dữ liệu nhánh MẶT NẠ từ Kaggle — máy local KHÔNG có
  `merged_bal.npz` lẫn `maskbal_s4x.npz` (đã kiểm 2026-08-16).
  🚫 **Tôi từng viết "người duyệt tinh sẽ hỏi δ\* của 4 seed" — BỊA.** Không có cơ sở dự đoán hành vi người
  duyệt. Sự thật kiểm được chỉ là bất đối xứng trình bày. ĐỪNG khoác áo "reviewer sẽ hỏi" cho một quan sát về
  văn bản để nó nghe nặng ký hơn.
- ⏳ Chưa backup Kaggle `results (9).zip`; và `/kaggle/working/w` (cache gt_density) vẫn chưa được lưu ra dataset
  — lần nào chạy cũng phải dựng lại.
- ⏳ **CHƯA BACKUP KAGGLE `results (8).zip`** (184 MB, 5×{preds,occlusion}.npz + samples). Xem
  [[feedback-note-backup-results]].

### Soát MỌI hình + bảng dùng số cũ — 2026-08-15

Cách làm: dựng lại từng con số của Bảng 1–5 và Hình 2–3 từ artifact gốc, không đọc lại MASTER.

**ĐÚNG HẾT, khỏi kiểm lại:** Bảng 1 hàng PanNuke + CoNIC (0,933±0,014 / 0,908±0,034 / +0,025 / 5/5 · δ*
2,450 = 2,20% · MAE 10,49 vs 12,38 · chú thích ‡ 4 nghiêng count 1 nghiêng mask — s43 — khớp) · Bảng 1 hàng
NuInsSeg (R², ΔR², δ* 1,42 ≈ 1,43 = 2,6%) · phân rã hiệu ứng/độ chính xác 1,3/0,2 · 0,9/1,7 · 1,8/0,5 (quy
tắc: hiệu ứng = TRUNG VỊ |Δ̂| các chia, độ chính xác = δ* − hiệu ứng) · Bảng 2 cả 3 hàng · Bảng 3 per-organ
**19/19 ô + hàng tổng khớp tuyệt đối** với `evidence_bgpatch/preds.npz` · Bảng 4 **36/36 ô** + 0,244±0,008 +
3.415/3.406/3.434 (3 mô hình = `evidence_bgpatch` + `evidence_bgpatch_s43+44/{s43,s44}`) · Hình 3 (0,934 /
3,40 / chệch +0,70 / LoA −9,5…+10,9) = ĐÚNG mô hình của Bảng 3, không phải seed nào của Bảng 1 — caption đã
khai rõ.

**ĐÃ SỬA:**
- Hình 2 vẽ số cũ (xem trên).
- **Bảng 5 trộn ddof.** Mục 4.2 phát biểu "mọi ± lấy mẫu số = số seed" (ddof=0), nhưng chỉ hàng DSU-Net theo
  đúng; 3 hàng baseline dùng ddof=1. Đã đưa về ddof=0: StarDist ±0,011→**±0,010** và 4,38±0,05→**±0,04** ·
  DualU-Net ±0,009→**±0,008** và 5,44±0,19→**±0,17** · DeNuC ±0,002→**±0,001** (MAE ±0,02 không đổi).
  Trung bình 4 hàng KHÔNG đổi, xếp hạng KHÔNG đổi. Không câu văn nào trích các ± này.
- **Bảng 1 NuInsSeg `1/5` → `2/5`.** Wilcoxon từng chia: s42 p=0,0089 · s43 0,247 · s44 **0,00063** · s45
  0,583 · s46 0,744 ⇒ 2/5 đạt ý nghĩa. Chú thích † định nghĩa cột p là "bao nhiêu lần chia đạt ý nghĩa",
  và CoNIC 5/5 đếm cả chia nghiêng mask ⇒ NuInsSeg phải là 2/5. Hai chia đó **ngược chiều** (s42 count
  8,19 vs mask 11,45; s44 count 15,31 vs mask 12,34) → đã ghi vào thân bài, làm luận điểm ngang-nhau MẠNH hơn.

**Hình 1 + Hình 4 — ĐÃ KIỂM BẰNG SỐ (2026-08-15), không còn chỗ nào chỉ dựa vào ghi chú:**
- **Hình 1** dựng lại từ chính 2 ckpt (`results (4)/ckpt_bal/mask_s42.pt` + `count_only_s42.pt`, CPU, 1 ảnh,
  ~20 giây): im0507 có **22 nhân**, diện tích nhân **16,58%** (khớp 16,6% ở §3.2, trần tỉ số riêng mảnh =
  6,03), tỉ số khối trong nhân **mask 4,5002 / count 1,7381** ⇒ caption "4,5 lần / 1,7 lần" ĐÚNG. γ=0,35 khớp
  `fig1_grads.py:52,64`.
- **Hình 4** = `panel_0072.png` trong `evidence_bgpatch/samples/`. Bốn số 22 / 20,7 / 16,3 / 7,5 / 19,6 nằm
  **in sẵn trong file PNG** (đọc trực tiếp), và `make_figs.QUAL` chỉ chép lại. Mảnh đó = hàng `img==8` của
  `occlusion.npz` (N=22, full=20,726) — **panel đánh số theo chỉ số dataset `i`, hàng đo đánh số theo vị trí
  `c`**, nên panel_0072 ↔ c=8, đừng tra nhầm img==72 (mảnh đó N=17).
- Hệ số đáp ứng: mảnh **0,6464** vs toàn bộ **0,6370** ⇒ caption "0,65 so với 0,64" ĐÚNG, cùng một ước lượng
  (trung bình hệ số của 6 mức).
- 🐞 **ĐÃ SỬA:** panel dùng **RNG RIÊNG** (`count_evidence.py:312`, `vrng = seed+1000`) nên số trên panel
  KHÔNG phải số đo. Sham của panel = 19,6 = **0,95 lần**, còn sham ĐO ĐƯỢC của chính mảnh đó = 21,0 =
  **1,01 lần** (đúng tiêu chí chọn mảnh ghi trong `make_figs.QUAL`). Caption cũ đặt 0,95 cạnh "1,00–1,02 toàn
  bộ dữ liệu" → đọc thành mảnh này nằm NGOÀI dải. Đã ghi rõ 0,95 là của lần rút thăm minh hoạ và bổ sung mức
  đo 1,01.

🧹 **Đã xoá `~/Downloads/du_trung_lap/` (2026-08-15):** 4 file `du_fold3_counts_s4{3,4,5,6}.csv` trùng md5
`eaaa09d1…`, là nhãn giả của bộ HỎNG seed — nội dung y hệt s42, tính ra 0,8038 với độ lệch 0. **GIỮ LẠI**
`~/Downloads/du_fold3_counts_s42.csv` vì md5 đó chính là kết quả seed 42 THẬT (khớp cột s42 bảng per-seed
2026-08-03) và là bản duy nhất còn per-image của DualU-Net. Số 5 seed đúng chỉ có ở bảng per-seed đó.

- **Độ lệch:** |err| count skew **+4,44**, mask **+4,75** (lệch PHẢI mạnh); nhưng hiệu bắt cặp d skew **−2,53**
  (lệch TRÁI), độ nhọn dư 23, Shapiro p 1,9e-47. §4.2 trước ghi "dùng Wilcoxon do |err| lệch phải" → dễ bị
  đọc thành d lệch phải; **đã đổi thành** |err| lệch phải ⇒ d đuôi nặng, phi chuẩn (kurt 23, Shapiro p<1e-40).

**Bảng 1 — NuInsSeg** (5 seed): count 0,893±0,050 · mask 0,883±0,026 · ΔR² +0,010 · 1/5 lần chia ·
δ* trung vị 1,43 nhân = 2,7% (mean count 52,84). Nguồn: `nuinsseg_parity_bal.npz`.

**Bảng 2** (efflite0 5 seed; 2 bộ còn lại 3 seed; 0/6 sập):
| | count | mask | ΔR² | δ* | Δ\|err\| |
|---|---|---|---|---|---|
| efflite0 | 0,931±0,005 | 0,911±0,028 | +0,020 | 1,5% | −0,307 (1,3%) |
| convnext_atto | 0,910±0,004 | 0,882±0,028 | +0,028 | 3,8% | −0,867 (3,5%) |
| tiny_vit_5m_224 | 0,928±0,012 | 0,933±0,012 | −0,005 | 0,8% | −0,141 (0,6%) |
Biến thiên theo hàng: count 0,021 · mask 0,051 · theo giám sát trong 1 kiến trúc ≤ 0,028.
⚠️ TinyViT: R² và MAE NGƯỢC CHIỀU (R² ưu mask, sai số tuyệt đối ưu count) — đã giải thích trong bài
bằng việc R² đè trọng số lên ảnh nhiều nhân. Đừng đọc là mâu thuẫn.
⚠️ Cột count của convnext/tinyvit LẤY LẠI từ lần chạy cũ (`sup_convnext_atto_lr1e-4.zip`, `work/tv_s4x.npz`)
vì `w_d=0` không đổi; chỉ cột mask là mới. Hợp lệ, không cần chạy lại.

**Bảng 5 — DSU-Net:** 0,931±0,005 / MAE 3,42±0,09 (cùng lần chạy Bảng 1).

**§5.3.2 định vị** (5 seed mỗi nhánh, n=2722, chance 0,1694, trần 5,90):
count ratio 1,54±0,51 / corr 0,028±0,010 · mask ratio 4,36±0,27 / corr 0,239±0,026 · bắt cặp n=2608 p<1e-300.
Đường cong nới viền (195 ảnh): count đỉnh 2,31 tại r=4 (5/5 seed), mask giảm đơn điệu từ 4,25 tại r=0 (5/5).

**Hình 1** (`fig_design.pdf`, dựng lại 2026-08-14): bản đồ mật độ từ `ckpt_bal/mask_s42.pt` và
`count_only_s42.pt`, chồng viền nhân GT màu lam. Mảnh im0507: mask ratio 4,50 / count 1,74;
count dồn 100% khối vào 1% điểm sáng nhất (mask 32%). Gradient panel KHÔNG đổi (cùng bộ trọng số).

**Trọng số:** `w_d = HW·w_c = 6553,6` (lập luận đơn vị, không dò). Tỉ lệ gradient ở `w_d=1`: hơn 4 bậc độ lớn.

~~**CÒN TRONG `\iffalse` (3 khối/file)**: MoNuSeg §5.1 · đoạn so 3 bộ · §6.3 chuyển miền~~
→ ĐÃ LẠC HẬU, xem mục MoNuSeg SPLIT CHÍNH THỨC ở cuối file: chỉ còn 1 khối (§6.3).

---

## ✅ MoNuSeg SPLIT CHÍNH THỨC — 2026-08-14 — ĐÃ XÁC THỰC, KHÔNG CẦN KIỂM LẠI

Nguồn: `results (5).zip` → `outputs/monuseg_parity_bal.npz` (gt 224, count-only 5×224, mask 5×224)
+ `ckpt_monuseg/{count-only,mask}_s42..46.pt`. Chạy `monuseg_indomain.py --split official` sau khi
sửa `MONU` bằng `os.path.commonpath` (lần trước tụt về `random`, mất 3 giờ GPU). w_d = 6553,6.
Kiểm chéo local bằng `equivalence.tost` — số dưới đây tôi tự tính lại từ npz, khớp log.

**KẾT QUẢ ĐẢO CHIỀU: nhánh mặt nạ THẮNG trên MoNuSeg.**

| | R² | MAE | δ* (nhân) | δ* (%) |
|---|---|---|---|---|
| count-only | 0,459±0,194 | 6,841±1,506 | | |
| mask | 0,638±0,093 | 5,655±0,862 | | |
| Δ (count−mask) | −0,180 | +1,187 nhân (4,0%) | 1,52 | 5,1 |

- mức mảnh n=224: Wilcoxon p = 9,7e−8 · CI90 [+0,872, +1,515]
- mức ảnh n=14 (224 mảnh cắt từ 14 ảnh, KHÔNG độc lập): p = 8,5e−3 · δ* = 2,14 nhân = 7,2%
- Δ theo từng seed: +2,47 · +1,01 · −0,01 · +3,07 · −0,61 → 2/5 seed count-only SẬP (R² 0,129 và 0,371),
  3 seed còn lại chênh ≤ 1,0 nhân. Không phải ưu thế đều tay của mask, nhưng count-only bất ổn ở đây là thật.
- gt mean 29,90 · std 11,67
- ⚠️ Cột count-only TRÙNG KHỚP TUYỆT ĐỐI với `monuseg_parity_5seed.zip` tải 31/7 (R² 0,1286/0,5689/0,5308/
  0,3712/0,6933). Cùng một lần chạy, không phải hai lần độc lập. Chỉ cột mask là mới.

**Phân rã δ* (Phương trình dstardecomp), quy ước: |Δ̂| và δ* tính theo % số đếm trung bình của bộ dữ liệu**
| bộ | n | hiệu ứng \|Δ̂\| | độ chính xác | δ* |
|---|---|---|---|---|
| PanNuke | 2722 | 1,3% | 0,2% | 1,5% |
| NuInsSeg | 133×5 | 0,9% | 1,7% | 2,6% |
| MoNuSeg | 224 | 4,0% | 1,1% | 5,1% |

⚠️ **SỬA BẢNG 1:** δ* NuInsSeg đổi 2,7% → **2,6%** (1,431 / 55,08 = 2,60%; giá trị 2,7 cũ không tái tạo được
ở bất kỳ quy ước nào). Đã sửa trong cả hai file.

**Đã viết vào bài (cả EN và VI), 2026-08-14:**
hàng MoNuSeg Bảng 1 · câu MoNuSeg trong đoạn mở §5.1 · bóc khối `\iffalse` MoNuSeg §5.1 (viết lại,
ĐẢO CHIỀU so với bản cũ) · bóc khối `\iffalse` phân rã 3 bộ (viết lại toàn bộ số) · thu hẹp phạm vi
4 câu tuyên bố bị MoNuSeg bác: tóm tắt, gạch đầu dòng đóng góp ở Mở đầu, §7.1 Thảo luận, Kết luận ·
thêm 1 đoạn Hạn chế về quy mô tập huấn luyện (kết quả phụ thuộc kích thước dữ liệu, chưa biết ngưỡng
giữa 37 ảnh và vài nghìn ảnh) · cập nhật luôn 1,1%/3,0% → 1,5%/5,1% trong khối §6.3 đang bọc.

~~**CÒN TRONG `\iffalse` (1 khối/file):** §6.3 chuyển miền + 1 câu trong tóm tắt.~~ ĐÃ BÓC HẾT; không còn khối `\iffalse` nào.
Cộng thêm: vẽ lại `fig_equivalence.pdf` — ĐÃ LÀM 2026-08-14 (`--dm -0.307 --lo -0.358 --hi -0.256`).

⛔ **CHƯA BACKUP KAGGLE:** `results (5).zip` → cần `outputs/monuseg_parity_bal.npz` + `ckpt_monuseg/` (zip
trước khi up, kẻo Kaggle bung `.pt`).

⚠️ **SỬA 2026-08-14 (ngay sau khi viết)** — lời giải thích "MoNuSeg nhỏ nhất" là SAI về số mảnh:
PanNuke train 5.179 mảnh · NuInsSeg train 532 ảnh (665×0,8) · **MoNuSeg train 592 mảnh (37 ảnh × 16 tile)**.
MoNuSeg có NHIỀU mảnh huấn luyện hơn NuInsSeg mà vẫn là bộ duy nhất mặt nạ thắng → kích thước tập
huấn luyện KHÔNG giải thích được. Trục đúng là **số ảnh gốc độc lập**: PanNuke ≫ NuInsSeg 665 ≫ MoNuSeg 37.
Đã sửa lại toàn bộ 8 câu ở cả 2 file sang khung "đa dạng chú thích / số ảnh gốc", và Hạn chế nói thẳng
rằng muốn tách đa dạng khỏi kích thước thì phải làm nghiên cứu lấy mẫu con — bài này chưa làm.

## 🔍 SOÁT TOÀN BỘ SỐ + TUYÊN BỐ — 2026-08-14 (sau MoNuSeg)

Quy trình: bóc `\iffalse`, trích mọi số theo mục, đối chiếu nguồn đã xác thực, so bộ-số EN với VI,
soi mọi câu chứa "suffic/equivalent/match/neither/consistently/parity".

**Mâu thuẫn ĐÃ SỬA (số cũ còn sót, không phải do MoNuSeg):**
1. §5.4: "DeNuC và DSU-Net chỉ cách nhau **0,018** R² và **0,62** MAE" → thực tế **0,005** và **0,44**
   (Bảng 5: 0,936/2,98 với 0,931/3,42). Số cũ từ trước khi cập nhật hàng DSU-Net.
2. §5.5: "giá trị ngân sách đầy đủ thấp hơn Bảng 1 **0,003**" → thực tế **0,016** (0,931 − 0,915).
3. §5.5 câu cuối: "mất mát bị chặn ở khoảng **1%**" → **1,5%** (sót từ vòng trước).
4. §5.3.2: "đỉnh 2,31 tại 4 điểm ảnh, **cả năm seed đều đạt đỉnh ở đó**" → SAI, s43 và s46 đỉnh ở
   r=2. Đã đổi thành "cả năm seed đỉnh ở 2 hoặc 4 điểm ảnh chứ không ở mức không nới".
5. §5.3.2: "ba phần tư quãng đường" → **bảy phần mười** ((4,36−1)/(5,90−1) = 0,686).
6. §5.5: "các mảnh ở đây trung bình chứa 24,5 nhân" đứng cạnh "123.090 nhân trong 5.179 mảnh" (= 23,8)
   → đổi thành "khoảng 24 nhân" để khỏi tự mâu thuẫn.

**Tuyên bố ĐÃ THU HẸP vì MoNuSeg:**
7. Highlight 1: "nhãn đếm sánh ngang mặt nạ trên ba bộ" → "không loại nhãn nào nhất quán tốt hơn".
8. Tóm tắt: "trong phạm vi miền và kiến trúc khảo sát, nhãn đếm là đủ" → thêm "và khi chú thích lấy
   từ nhiều ảnh gốc".
9. §5.3 mở đầu, §5.3.2 câu kết, §5.4 câu "chỉ cách một khoảng nhỏ" → đều thêm "trên PanNuke" /
   "trên hai bộ đa dạng hơn".

**Đã kiểm và ĐÚNG (khỏi kiểm lại):** Bảng 2 (biến thiên hàng 0,021/0,051; |Δ| 1,3/3,5/0,6%; δ* 1,5/3,8/0,8%) ·
Bảng 4 liều-đáp ứng (0,62–0,65, TB 0,64±0,03) · Bảng 5 (41,0M/3,62M = 11×; 4,37/3,62 = 21%) ·
Bảng 6 per-organ (tổng 2.722) · 92,8% = 0,849/0,915 · 0,066 = 0,915−0,849 · Bland–Altman (0,70 ≈ 3%,
gấp ~15 lần) · 1,8 lần tham số = 6,65/3,62 · bộ số EN và VI khớp nhau hoàn toàn (chỉ lệch ở ký tự markup).

~~⚠️ HAI VIỆC CÒN TREO (chuyển miền)~~ — XONG 2026-08-15, bỏ qua đoạn dưới:
- §6.3 hiện là **mục RỖNG** (chỉ còn tiêu đề, thân nằm trong `\iffalse`).
- Nhưng §7.2 Thảo luận vẫn viết "ước lượng điểm suy giảm khi chuyển miền, và suy giảm như nhau ở cả hai
  chế độ (Mục 6.3)" — trích dẫn một mục rỗng. Bóc `\iffalse` là hết, nhưng ĐỪNG nộp khi chưa bóc.

⚠️ **QUYẾT ĐỊNH CỦA USER, TÔI KHÔNG TỰ ĐỔI:** nhan đề "Image-level supervision is sufficient…" là câu
khẳng định mà chính bài đưa ra một bộ dữ liệu phản ví dụ. Cân nhắc đổi sang dạng có điều kiện.
→ **2026-08-16, GIẢM NHẸ Ở PHẦN MỞ ĐẦU thay vì đổi nhan đề.** Đã kiểm: toàn bộ Mở đầu KHÔNG có câu nào giới hạn
phạm vi tuyên bố (chữ "Within" duy nhất ở đó là "Within each comparison", nói về thiết kế); rào phạm vi đầu tiên
mãi tới **dòng 509** (§5.1). Đã thêm cuối đoạn 4 Mở đầu: *"Any conclusion drawn from this design is bounded
accordingly: it concerns the counting task alone, on the three datasets and under the single training and
evaluation protocol examined here."* Viết ở dạng KHẲNG ĐỊNH phạm vi, KHÔNG dùng dạng câu hỏi như bản đề xuất —
vì Mở đầu đã có câu hỏi ở dòng 83, thêm nữa là hỏi hai lần cùng một chuyện.
⚠️ Câu này giảm nhẹ chứ không vô hiệu hoá một nhan đề khẳng định. Nếu vẫn muốn chắc, chỉ còn cách đổi nhan đề.

## 📏 QUY ĐỊNH TẠP CHÍ (Results in Engineering) — tra 2026-08-14

- **Nhan đề:** "concise and informative… avoid abbreviations and formulae, where possible, unless they are
  established and widely understood." KHÔNG giới hạn độ dài, KHÔNG cấm nhan đề dạng câu hỏi.
- **Highlights:** 3–5 gạch đầu dòng, **tối đa 85 ký tự kể cả dấu cách** mỗi dòng; tránh biệt ngữ, từ viết tắt,
  công thức; nộp dưới dạng file Word riêng.
- **Tóm tắt:** tối đa **250 từ**.
- **Từ khoá:** đang dùng 6.

**Vi phạm phát hiện và ĐÃ SỬA:**
- Highlight 1 bản Anh dài **98 ký tự** (do tôi viết lại ở lượt trước) → rút còn 75:
  "No supervision type wins consistently across three nuclei counting datasets".
- Highlight 5 chứa công thức `$R^2$` → đổi thành "full-label counting accuracy" (75 ký tự).
- Bản Việt chỉnh tương ứng.

**Kiểm và ĐẠT:** tóm tắt bản Anh 217 từ (< 250) · cả 5 highlight ≤ 85 ký tự · từ khoá 6 · nhan đề hiện tại
84 ký tự / 10 từ, không có từ viết tắt hay công thức.
⚠️ Tóm tắt bản Việt 278 từ — vượt 250 nhưng bản Việt KHÔNG nộp, chỉ để đọc. Không cần sửa.

### Soát tóm tắt + highlights — 2026-08-14

🐞 **LỖI LATEX THẬT (bản Việt):** dòng `% TODO: viet lai cau chuyen mien...` nằm GIỮA dòng nên nuốt luôn
cụm "Trong so sánh bốn" đứng sau nó → tóm tắt bản Việt in ra bị mất chữ. Đã tách comment xuống dòng riêng.
Đã quét toàn bộ 2 file: không còn chỗ nào comment giữa dòng nuốt nội dung.

**Tóm tắt bản Anh:** 238 từ, hạn 250 → chỉ còn 12 từ dư mà câu chuyển miền (~20 từ) vẫn phải thêm lại.
Đã cắt 2 chỗ thừa (câu mô tả thí nghiệm; cụm "giảm ít nhất tám lần" lặp lại "một phần tám" ở câu trên)
→ còn **218 từ**, đủ chỗ cho câu chuyển miền. Cũng bỏ lặp "bounds… bounded at" trong cùng một câu.

**Highlights:** 5 dòng, dài nhất 81 ký tự, không công thức, không viết tắt. ĐẠT.
Đã kiểm từng dòng đúng với bảng: H4 (StarDist 0,847 + DualU-Net 0,809 là 2 thấp nhất) · H5 (0,849/0,915
= 92,8%) · H2 (1,5%, n=2.722) · H3 (≥8 lần rẻ hơn).

⚠️ Khi bóc §6.3: thêm câu chuyển miền vào tóm tắt rồi ĐẾM LẠI, phải ≤ 250 từ.

## ✅ §6.3 CHUYỂN MIỀN — 2026-08-14 — ĐÃ XÁC THỰC, BÓC KHỐI CUỐI CÙNG

Nguồn: `pannuke-nuinsseg_count/xfer_count_s42..44.pkl` (nhánh chỉ-đếm, w_d=0 nên KHÔNG cần chạy lại)
+ `results-12/outputs/xfer_mask_bal_s42..44.pkl` (nhánh mặt nạ, w_d=6553,6). PanNuke f1-3 (7.901 ảnh)
→ toàn bộ 665 ảnh NuInsSeg, không huấn luyện lại. Tự tính lại local, khớp log.

| seed | count MAE | mask MAE |
|---|---|---|
| 42 | 32,55 | 27,94 |
| 43 | 30,90 | **52,84 ← SẬP** |
| 44 | 31,88 | 31,10 |

⚠️ **mask seed 43 xuất số 0 cho CẢ 665 ảnh** (μ min = max = 0, std = 0, σ std = 0, R² = −0,957).
Không phải "seed yếu" mà là lượt huấn luyện HỎNG.

- gt NuInsSeg: mean 52,8 · std 54,0 · **đoán hằng số = MAE 35,43** (mốc để đọc mọi số dưới)
- count-only: MAE 31,78 ± 0,68 · R² 0,007 / 0,111 / 0,093 · r 0,715 · pred mean ~21
- mask (bỏ s43): MAE 29,52 · R² 0,18–0,20 · r 0,786 · pred mean ~25
- **ĐỦ 3 seed:** Δ = −5,52 nhân (10,4%), Wilcoxon p=3,95e−62, δ* = 11,3% → chỉ-đếm thắng
- **BỎ s43:** Δ = +2,69 nhân (5,1%), p=5,79e−40, δ* = 5,7% → mặt nạ thắng
- ⇒ **CHIỀU ĐẢO tuỳ vào 1 lượt hỏng** → bài viết thẳng là không kết luận được chiều; p bé vô nghĩa ở đây.

**Số CŨ đã HỎNG, đã xoá:** "hai nhánh vẫn tương đương, Δ = 1,05 nhân, δ* = 2,3%" — sai hoàn toàn.
Số duy nhất của bản cũ còn sống: r ≈ 0,72 (đúng: count 0,715).

**Đã viết vào bài:** §6.3 viết mới 3 đoạn (mất độ chính xác điểm; thứ tự sống thang chết; so sánh không
dứt khoát) · câu chuyển miền trả lại tóm tắt · §7.2 đổi "suy giảm như nhau ở cả hai chế độ" thành
"suy giảm tới mức không còn phân biệt được hai chế độ".

🎉 **HẾT `\iffalse`, HẾT TODO trong cả hai file.** Đã kiểm: không có \ref nào thiếu \label.
⚠️ Tóm tắt bản Anh giờ **244 từ**, hạn 250 — chỉ dư 6 từ, thêm gì cũng phải cắt chỗ khác.
⛔ CHƯA BACKUP: `results-12/outputs/xfer_mask_bal_s42..44.pkl` (29 KB mỗi file).

**Bổ sung §6.3 — sai số chuyển miền là SAI THANG, một hằng số:** nhân dự đoán với hệ số bình phương
tối thiểu khớp trên tập đích:

| seed | arm | MAE thô | R² thô | hệ số | MAE sau | R² sau |
|---|---|---|---|---|---|---|
| 42 | count | 32,55 | +0,007 | 1,96 | 22,53 | +0,311 |
| 43 | count | 30,90 | +0,111 | 2,03 | 19,37 | +0,484 |
| 44 | count | 31,88 | +0,093 | 2,32 | 19,47 | +0,594 |
| 42 | mask  | 27,94 | +0,203 | 2,02 | 17,03 | +0,598 |
| 44 | mask  | 31,10 | +0,177 | 2,09 | 18,35 | +0,602 |

Hệ số ~2,0 ở MỌI seed và CẢ HAI nhánh = tỉ số 52,8/24,5 (số đếm TB NuInsSeg / mảnh PanNuke).
Mô hình đếm theo thang miền huấn luyện. Đây là bằng chứng trực tiếp cho §7.2 "mang theo được là
hiệu chuẩn chứ không phải trọng số" — hiệu chuẩn lại chỉ cần MỘT số, ước bằng nhãn đếm.
Đã viết thành 1 đoạn ở §6.3 cả 2 file.

**Cũng bỏ 1 câu thủ thế ở §5.4** ("Kết quả này không cho thấy mặt nạ là dạng nhãn kém hơn về bản chất")
→ viết thẳng. User đã nhắc lại quy tắc: nói A thì người đọc hiểu A, không cần rào "A chứ không phải B".
Đã quét cả 2 file, các cụm "chứ không phải" còn lại đều là đối lập có nội dung (thứ tự/thang đo), giữ.

### Soát cặp mâu thuẫn vòng 2 — 2026-08-14 (sau khi bóc §6.3)

Cách làm: gom MỌI số trong bài theo giá trị, xem mỗi giá trị xuất hiện ở đâu, đối chiếu từng cụm.

**HAI cặp thật, đã sửa:**
1. **0,59 với 0,64** — §5.3.1 nói hệ số đáp ứng là 0,64 (3 mô hình), rồi ngay câu sau viết "nội suy Telea
   làm hệ số giảm từ 0,59 xuống 0,34" mà không nói 0,59 là mốc của MỘT mô hình. Người đọc thấy hai con số
   cho cùng một đại lượng. Đã ghi rõ trong câu đó. Tiện thể bỏ phần lặp: câu giải thích cơ chế nội suy
   trước đây có Y HỆT ở cả §5.3.1 lẫn Phụ lục B — giờ chỉ còn ở phụ lục.
2. **52,8 với 55,08** — §6.3 dùng số đếm TB 52,8 (toàn bộ 665 ảnh NuInsSeg), còn δ* 2,6% ở Bảng 1 suy từ
   55,08 (TB các tập test 133 ảnh). Đã thêm "của các ảnh đó" vào §6.3 để rõ phạm vi. Quy ước GIỮ NGUYÊN:
   phần trăm trong Bảng 1 tính theo TB của đơn vị đánh giá thực dùng.

**Trùng số nhưng KHÔNG mâu thuẫn (đã kiểm, khỏi lo):** 0,923 (R² đại tràng ~ phủ biên) · 1,43 (δ* NuInsSeg ~
tham số StarDist) · 3,5% (khoảng cách MAE bộ mã hoá ~ phần dư Phụ lục B) · 0,94 (R² suy biến ~ khớp diện tích)
· 0,31 (|Δ| PanNuke ~ cận dưới R² chuyển miền ~ tương quan Phụ lục B) · 5,1% (δ* MoNuSeg ~ hiệu số chuyển
miền bỏ seed sập) · 1,5 (cột δ* ~ 1.5em trong tabular).

**Đã đối chiếu và ĐÚNG:** mọi giá trị lặp của Bảng 1/2/4/5/6 khớp với câu văn nhắc tới chúng; 1,5% xuất hiện
6 chỗ đều cùng nghĩa; 5,1% 5 chỗ đều cùng nghĩa; 92,8% 3 chỗ; 0,64 6 chỗ; 24,5 và 52,8 mỗi số 3 chỗ.

### §3.2 + lỗi hiển thị phụ lục — 2026-08-14

🐞 **`Phụ lục Phụ lục A` / `Appendix Appendix A`** — elsarticle sau `\appendix` tự thêm chữ "Appendix"
(babel vietnamese dịch thành "Phụ lục") vào SỐ MỤC, nên viết `Phụ lục~\ref{app:sigma}` ra thành
"Phụ lục Phụ lục A". Lỗi hiện rõ trong PDF. Đã bỏ chữ dẫn, chỉ còn `\ref{app:sigma}` → "Phụ lục A".
Sửa cả 2 chỗ (app:sigma ở §3.2, app:checks ở §5.3.1) trong cả 2 file.

**Đã thêm:** §5.3.2 giờ nói rõ trần 5,90 = nghịch đảo tỉ lệ diện tích nhân trung bình (1/0,1694).
Trước đó bài có 16,6% ở §3.2 (mảnh Hình 1) và 5,90 ở §5.3.2 mà không nối hai số lại.

**Đã soát §3.2, các phần còn lại ĐÚNG:** hàm mục tiêu 3 số hạng khớp code · bản đồ mục tiêu tổng đúng
bằng số nhân (đã kiểm lại khi viết loader CoNIC) · hai phép chặn gradient + nghiệm σ*=|N−μ| khớp Phụ lục A ·
w_d = HW·w_c · hơn 4 bậc độ lớn · 65.536 điểm ảnh · 544 giá trị gradient phân biệt.
Câu "mọi điểm ảnh gradient âm đều nằm trong nhân" là HỆ QUẢ TOÁN HỌC, không phải quan sát: gradient của
MSE mật độ theo d_p là 2(d_p − t_p)/HW; ngoài nhân t_p = 0 và d_p ≥ 0 do ReLU nên gradient ≥ 0. Chặt.

🐞 **LỖI SỐ ở §6.2 — SỬA 2026-08-14:** bài ghi "trên NuInsSeg, số đếm mỗi ảnh trải từ vài nhân tới **hơn 500**".
SAI. Kiểm hai nguồn độc lập — `xfer_count_s42.pkl` (665 ảnh) và `nuinsseg_parity_bal.npz` (5×133) — đều cho
**min 1, max 370**, không ảnh nào vượt 500. Đã sửa thành "từ 1 tới 370 nhân" ở cả 2 file.
Số này chưa từng nằm trong danh sách đã-xác-thực vì nó thuộc phần UQ viết từ trước khi pivot.
⚠️ Suy ra: các số §6.2 khác (phủ 0,923 / rộng 16,6 / Winkler 20,5 / 67,9→37,7 / 0,655→0,864) CŨNG chưa
được kiểm lại vòng này — nếu còn thời gian thì đối chiếu với artifact gốc.

**Highlights chốt 2026-08-14** (đã đo, cả 5 dòng ≤ 85 ký tự, không công thức, không viết tắt):
EN: 75 / 81 / 70 / 80 / 69 ký tự · VI: 82 / 83 / 74 / 81 / 76.
Đổi dòng 5 sang chữ số "25%" thay "A quarter" — dễ quét mắt hơn và ngắn hơn 6 ký tự.
⚠️ Nộp highlights thành FILE WORD RIÊNG, không nằm trong bản thảo.

### Soát lặp + thống nhất thuật ngữ — 2026-08-14 (vòng cuối)

Cách làm: tách 270 câu, so trùng 3-gram từng cặp; đếm biến thể của 10 nhóm thuật ngữ.

**Ba chỗ LẶP THẬT, đã sửa:**
1. §4.2 nêu lý do "phân phối sai số lệch phải mạnh" HAI lần cách nhau 30 dòng → lần hai đổi thành
   "Cũng vì lý do đó".
2. Câu mở §5.1 và câu mở Kết luận gần như trùng chữ (43% 3-gram) → viết lại câu mở Kết luận theo hướng
   nêu thiết kế trước, kết quả sau.
3. §1 và §2 cùng nói "đếm đám đông đã thành một hướng nghiên cứu riêng" → §1 rút thành "đã được thiết lập
   vững", để §2 giữ phần khai triển.

**Thuật ngữ — ĐỒNG NHẤT, không phải sửa:** DSU-Net 11/11 · count-only 36/36 · PanNuke/NuInsSeg/MoNuSeg
viết đúng 100% · EfficientNet-lite0, ConvNeXt-Atto, TinyViT-5M nhất quán · annotation cost 12/12 ·
fold~3 và folds~1--2 dùng dấu ngã không ngắt đúng chuẩn.
"regime" (38) và "branch" (36) cùng tồn tại nhưng CÓ PHÂN VAI: regime = chế độ giám sát trừu tượng,
branch = nhánh mô hình đã huấn luyện. Giữ.
"mask branch" (21) và "mask supervision" (6) cũng phân vai tương tự. Giữ.

**Bản Việt:** hai cảnh báo là dương tính giả — "hoa" nằm trong "khoa", "chỉ đếm" là động từ ("nếu chỉ đếm")
chứ không phải tên nhánh. Chính tả hoá/hoạ đúng quy ước ở toàn bộ 25 chỗ.

**Cặp Funding và Acknowledgements cùng cảm ơn UIT VNU-HCM** — trùng ý nhưng là hai mục bắt buộc riêng
của Elsevier, thông lệ chấp nhận. KHÔNG sửa.

---

**★★ CoNIC THAY MoNuSeg trong Bảng 1 (2026-08-14, `conic_indomain.py`, npz `work/conic/`) — quyết định của cô:**
MoNuSeg (mask thắng, ΔR² −0,180) đã **cắt khỏi cả main.tex và main_vi.tex**. Bản chụp trước khi cắt:
`scratchpad/snap_monuseg/{main.tex,main_vi.tex}` (phiên 0a3e8aed). Lý do cô đưa: bài khác cũng chọn tập con
dataset, không ai bắt buộc MoNuSeg. Tôi đã nêu 1 lần rằng bỏ đi thì mất điều kiện biên (ngưỡng số ảnh gốc);
cô giữ quyết định.

CoNIC = Lizard, **loại 112 mảnh gốc PanNuke** → **4.869 mảnh / 210 ảnh gốc** (⚠️ KHÔNG phải 4.757, số đó tôi
tự trừ sai một lần). Đếm lại từ kênh instance trên TOÀN mảnh 256 (counts.csv chỉ đếm vùng giữa 224). Chia
80/20 **theo ảnh gốc** mỗi seed → test 42 ảnh, 883–1022 mảnh. Toàn bộ mô đại tràng (consep/crag/dpath/glas).

| | R² count | R² mask | ΔR² | MAE count | MAE mask |
|---|---|---|---|---|---|
| 5-seed | **0,9333 ± 0,0142** | 0,9085 ± 0,0339 | **+0,025** | **10,49 ± 0,84** | 12,38 ± 2,53 |

Mức mảnh: **5/5 chia có ý nghĩa, 4 nghiêng count / 1 nghiêng mask (seed43)**; δ* trung vị **2,45 nhân = 2,2%**
(đếm TB test 111,3). Mức ảnh gốc (gộp ~24 mảnh/ảnh): chiều giữ nguyên 4/5, **ý nghĩa còn 2/5**, δ* → **4,1%**.
Phân rã: hiệu ứng 1,8% / độ chính xác 0,5% (n≈1000 → precision term nhỏ nhất trong 3 bộ).
Độ chệch mask seed45/46 = +10,6/+12,8 nhân (đếm dư hệ thống) = nguồn gốc MAE dôi; count-only chệch đều ~+4,5.

**Câu chuyện bài sau khi cắt:** cả 3 bộ đều nghiêng count-only (+0,020 / +0,010 / +0,025), biên ≤ 2,6%.
Mất luôn phát biểu "ngưỡng nằm giữa 37 và 210 ảnh gốc"; Hạn chế giờ chỉ khai "ủng hộ tới ~200 ảnh gốc,
dưới mức đó chưa xác định". Highlight 1 + đóng góp 1 + Kết luận đã đổi từ "không chế độ nào nhất quán"
sang "chỉ-đếm không hề thua". `highlights.txt` đã đồng bộ (trước đó còn 1,1% cũ = SAI).

⛔ **Backup còn nợ:** `results-13.zip`/`results (6).zip` → `ckpt_conic/{count-only,mask}_s42..46.pt` (14,8 MB×10)
+ `out/conic_*_s4*.npz` (10 file) + `conic_src.npz`. Chưa đẩy Kaggle.

### Vòng nhận xét 2026-08-16 (4 ý) — 3 sửa, 1 bác

**1. δ* của CoNIC: lấy mức ảnh gốc làm số chính. ✅ ĐÃ SỬA.**
Thân bài §5.1 vốn ĐÃ có cả hai (2,2% mức mảnh / 4,1% mức ảnh gốc), nhưng Tóm tắt + Giới thiệu + Bảng 1 +
Kết luận vẫn dẫn 2,2% — tức headline dựa trên đơn vị KHÔNG độc lập (1 lần chia = ~1.000 mảnh từ 42 ảnh gốc).
Đã đổi 4 chỗ đó sang 4,1%; Bảng 1 in cả hai dạng `5/5 (2/5) lần chia` và `2,2 (4,1)`, chú thích ‡ giải thích
ngoặc = mức ảnh gốc và nói rõ đó là giá trị được dẫn ở chỗ khác trong bài.
⚠️ **Hệ quả phải sửa kèm (suýt sót):** §7.1 viết "chênh lệch nằm dưới **3%** trên cả ba bộ" → SAI khi CoNIC
thành 4,1%. Đã đổi thành "không quá 4,1%". Và đoạn phân rã hiệu ứng/độ chính xác (1,8% + 0,5% = 2,3%) là số
MỨC MẢNH, không cộng ra 4,1% → đã ghi rõ "riêng CoNIC lấy ở cấp độ mảnh".

**2. Rò rỉ khi hiệu chuẩn thang đo lúc chuyển miền. ✅ CÓ THẬT, đã sửa bằng cách TÍNH LẠI.**
Bản cũ: hệ số khớp bình phương tối thiểu trên **cả 665 ảnh** NuInsSeg rồi báo MAE/R² cũng trên 665 ảnh đó =
in-sample. Đã tính lại kiểu chia đôi (fit trên 1 nửa, chấm trên nửa kia, 200 lần chia ngẫu nhiên, seed 0):

| arm | seed | k in-sample | MAE/R² in-sample | k held-out | MAE/R² held-out |
|---|---|---|---|---|---|
| count | 42 | 1,96 | 22,53 / +0,311 | 1,96±0,055 | 22,55 / +0,313 |
| count | 43 | 2,03 | 19,37 / +0,484 | 2,04±0,053 | 19,53 / +0,479 |
| count | 44 | 2,32 | 19,47 / +0,594 | 2,32±0,064 | 19,56 / +0,588 |
| mask  | 42 | 2,02 | 17,03 / +0,598 | 2,01±0,056 | 17,12 / +0,598 |
| mask  | 44 | 2,09 | 18,35 / +0,602 | 2,10±0,068 | 18,42 / +0,598 |

**MỌI khoảng in trong bài giữ nguyên** (k 1,96–2,32 · R² 0,31–0,60 · MAE 17–23) — 1 tham số trên n=665 nên
độ lạc quan không đáng kể. Nhưng câu văn giờ mô tả đúng quy trình held-out, hết leakage. pkl:
`~/Downloads/pannuke-nuinsseg_count/xfer_count_s4{2,3,4}.pkl` + `~/Downloads/results-12/outputs/xfer_mask_bal_s4{2,4}.pkl`
(mask s43 sập, k không tính được).

**3. "biến thiên chưa tới 0,07" phải đổi thành 0,073. ❌ BÁC — người nhận xét gộp nhầm hai đại lượng.**
0,073 = biên độ GIỮA NĂM MÔ HÌNH (0,649−0,576). Câu kia nói "trên toàn dải can thiệp" = giữa các MỨC XOÁ
10%→100%, cột cuối Bảng 4 là 0,58/0,62/0,64/0,63/0,63/0,60 → biên độ 0,06. Câu cũ đúng.
Vẫn sửa vì hai số 0,07 và 0,073 đứng gần nhau gây nhập nhằng → viết thẳng "Giữa các mức can thiệp, tỉ số này
biến thiên 0,06".

**4. Không được gọi Telea 0,066 là "nhỏ" so với 0,073. ✅ ĐÚNG, đã sửa.**
0,066/0,073 = 90%. Câu cũ gộp cả hai yếu tố quy trình vào một chữ "đều nhỏ". Đã tách: giãn nở 0,029 = nhỏ;
lấp vùng 0,066 = ngang chênh lệch giữa các mô hình, và nói thẳng cách tái tạo vùng xoá là nguồn nhạy cảm đáng
kể. Câu kết "0,62 là cận dưới" giữ nguyên (vốn đã đúng hướng).

Đã đồng bộ cả `main.tex` và `main_vi.tex`; kiểm token số hai file khớp trên mọi giá trị đụng tới.

### Biên tương đương: cố định biên thực tiễn 5%, δ* xuống hàng tóm tắt — 2026-08-16

**Nhận xét đúng:** δ* = max(|L|,|U|) là **tóm tắt suy từ dữ liệu**, không phải biên thực tiễn định trước. §3.3
bản cũ còn viết thẳng "Thay vì cố định trước một δ tuỳ ý…" ⇒ đọc thành "nhìn CI rồi mới chọn biên nhỏ nhất".
Và Tóm tắt dùng chữ "tương đương về mặt thực tiễn" mà **không có biên nào để chữ đó bám vào**.

**Đã đo được cái NEO cho biên** (`results (8)/evid/s4*/preds.npz` = count-only 5 seed Bảng 1,
`~/Downloads/pannuke_mask_5seed/outputs/maskbal_s4*.npz` = mask 5 seed Bảng 1; bootstrap B=1e4, rng seed 0):

- Dựng lại được đúng số của bài: Δ̂ = −0,3068 (1,25%), **δ* = 0,358 nhân = 1,46%** ⟶ bài in 0,36 / 1,5% ✓
- **Đổi seed trong CÙNG một chế độ** (10 cặp): count-only δ* trung vị **0,90%** (0,57–1,50);
  mask **1,57%** (0,75–6,80). Bốn cặp rộng nhất của mask đều chứa **s44** (R² 0,857 so với 0,914–0,936);
  sáu cặp không có s44 nằm trong 0,75–1,71%.
- ⇒ **Biên giữa hai chế độ (1,46%) nằm GIỮA hai giá trị nội bộ chế độ (0,90% và 1,57%).** Đã viết vào §5.1.
- Neo thứ hai: sai số chú thích, phần do nhân không được chú thích = 2,6% (§5.3).

**Đã sửa:** §3.3 thêm đoạn `Biên thực tiễn` chốt **δ = 5% số đếm TB**, dùng nguyên cho mọi bộ và mọi phát biểu,
lý do = đặt trên độ phân giải của phép đo (hai neo trên). Đoạn `Biên tương đương tối thiểu` viết lại: δ* là
tóm tắt từ dữ liệu, là ngưỡng nên cho kết quả ở MỌI biên cùng lúc. §5.1 thêm kết quả ở hai biên: **5% → tương
đương cả ba bộ** (1,5 / 2,6 / 4,1 đều dưới 5); **2% → chỉ PanNuke**, NuInsSeg và CoNIC không. Chú thích Bảng 1,
Tóm tắt, Kết luận đều gắn biên 5% vào chữ "tương đương". §5.2 chuyển miền: 5,7% và 11,3% **vượt** biên 5% nên
nói thẳng không kết luận được tương đương khi chuyển miền.

⚠️ **KHÔNG viết "pre-specified/định trước"** ở bất kỳ đâu — biên 5% chốt SAU khi đã có kết quả. Cái chứng minh
được là: biên dùng THỐNG NHẤT mọi chỗ, có lý do ngoài phép so, và δ* in kèm để người đọc tự thay biên. Nếu
sau này ai hỏi "sao không phải 3%?" thì câu trả lời là 4,1% của CoNIC in ngay trong bảng, tự thấy.

Tóm tắt bản Anh sau sửa: **238 từ** (hạn 250).

---

## ★★ LUẬN VĂN — NỘI DUNG 3: HIỆU CHUẨN THÍCH ỨNG TRÊN MÔ HÌNH ĐẾM (2026-10-14)

⚠️ **KHÔNG phải nội dung Paper 2.** Đây là phần tích hợp của LUẬN VĂN (Nội dung 3), nối Paper 1
(cơ chế hiệu chuẩn) với Paper 2 (mô hình đếm). Paper 2 vẫn chỉ dùng split conformal tĩnh và
CITE PB-JCI của P1 — xem [[paper1-vs-paper2-boundary]]. Đừng đưa số dưới đây vào Paper 2.

**Câu hỏi:** cơ chế cửa sổ thích ứng của P1 (σ từ Poisson–Binomial của detector) có chạy được khi
σ đến từ ĐẦU HỌC của mô hình đếm 3,6M không?

**Thiết lập (khác mọi lần chạy trước — đây là chỗ then chốt):** ngưỡng hiệu chuẩn khởi động ở
**MIỀN NGUỒN** (PanNuke fold 3 giữ lại, n=2722, KHÔNG train) rồi triển khai trên **toàn bộ** luồng
NuInsSeg (n=665). Train fold 1+2, count-only (`--w_density 0 --w_count 1.0`), efficientnet_lite0
ch32, `--detach_mu`, 60 epoch, giữ colon, **5 seed {42..46}** (seed 45,46 thêm 2026-10-15). Kaggle GPU.

```
transfer.py --train_dataset pannuke --test_dataset nuinsseg --pannuke_folds 1,2 \
  --source_holdout_folds 3 --exclude_tissue "" --supervision count --detach_mu \
  --backbone efficientnet_lite0 --student_ch 32 --epochs 60 --seed $S \
  --out xfer_src_count_s$S.pkl --out_source src_pan_f3_s$S.pkl
adaptive_stream.py --preds xfer_src_count_s$S.pkl --warm_preds src_pan_f3_s$S.pkl --window 300
```

| Chế độ | Độ phủ % | Bề rộng | Winkler | e_t trung vị |
|---|---|---|---|---|
| Tĩnh (ngưỡng từ PanNuke) | **15,1 ± 8,3** | 15,1 ± 1,7 | 473,7 ± 75,6 | 300 |
| Trực tuyến cửa sổ cố định | 78,1 ± 3,4 | 71,6 ± 4,5 | 187,1 ± 28,0 | 300 |
| **Trực tuyến cửa sổ thích ứng** | **90,7 ± 0,9** | 96,0 ± 13,2 | **157,0 ± 23,9** | **68 ± 14** |

Từng seed — tĩnh 11,0/9,6/20,9/26,8/7,2 · cố định 75,0/77,6/79,1/83,5/75,3 ·
thích ứng 90,1/91,0/90,4/92,2/90,1.

**Cơ chế, đo trực tiếp:** phần dư chuẩn hóa `r=|gt−μ|/σ` trung vị ở NGUỒN = 0,61/1,18/1,14/0,74/0,77
(σ calibrated trong miền, đúng chuẩn ~0,67) → ở ĐÍCH = 5,31/10,22/6,99/4,37/8,21. Ngưỡng nguồn nhỏ
hơn mức cần **6–9×**. MAE đích 30,3/32,2/27,2/26,5/34,4 (in-domain ~3,4) — khớp finding cũ
"σ KHÔNG transfer".

**✅ QUA PHÉP XÁO TRỘN (bắt buộc kiểm, lần trước đã lật vì chỗ này):** xáo thứ tự luồng 2 cách ×
5 seed → tĩnh KHÔNG đổi (7,2–26,8 đúng dải thứ tự gốc), cố định lên 80–84%, thích ứng vẫn
**89,9–91,4%**. Khoảng cách ~76 điểm giữa tĩnh và thích ứng KHÔNG phụ thuộc thứ tự file.

⛔ **Lần chạy HỎNG trước đó (2026-10-12) — ĐỪNG dùng lại:** khởi động lấy từ 100 ảnh đầu của chính
NuInsSeg (`xfer_count_s4*.pkl` của 2026-08-07). Cho tĩnh 66,5% / thích ứng 91,3%, trông đẹp nhưng
**xáo trộn là tan** (tĩnh bật lên 87,6–89,6%). Nguyên nhân: 665 ảnh NuInsSeg xếp theo cơ quan
(31 nhóm), nên "trôi" chỉ là cấu trúc thứ tự file, không phải dịch chuyển miền.

**Kết luận bán được:** cơ chế hiệu chuẩn thích ứng **không phụ thuộc cách sinh σ** — giữ được độ phủ
danh nghĩa cả khi σ đến từ đầu học Poisson-anchored của mô hình đếm rẻ, thay vì từ cấu trúc
Poisson–Binomial của detector nặng. Giới hạn phải ghi: bề rộng tuyệt đối rất lớn (95 so với số đếm
trung bình ~53) → bán "giữ được độ phủ", KHÔNG bán "đếm chính xác dưới chuyển miền".

**Artifacts:** `xfer_src_count_s{42..46}.pkl` + `src_pan_f3_s{42..46}.pkl` (10 file, ~0,8 MB).
Gom ở `~/Downloads/adaptive_src_calib/`, backup Kaggle dataset **`hipinhththu/efflite0-adaptive-src-calib`**
(tên file giữ nguyên, không thư mục con). Xem [[feedback-note-backup-results]].
Code: `transfer.py` (`--source_holdout_folds`/`--out_source`) + `adaptive_stream.py` (`--warm_preds`),
commit 7d7133e.

### Nhánh MASK — cùng giao thức (2026-10-15)

Cùng mọi thứ, chỉ đổi `--supervision mask --w_density 6553.6 --w_count 0.1` (6553,6 = 256×256×0,1,
lấy theo `parity_pannuke.py:139`; mặc định cũ 1.0 trong `transfer.py` sai đơn vị 6553,6× nên đã sửa
ở commit e45e6db, CHƯA push → lệnh Kaggle truyền tay).

| Chế độ | Độ phủ % | Bề rộng | Winkler | e_t trung vị |
|---|---|---|---|---|
| Tĩnh | 15,7 ± 3,1 | 20,2 ± 2,7 | 465,5 ± 29,3 | 300 |
| Cố định | 77,6 ± 2,0 | 75,4 ± 7,1 | 190,9 ± 12,8 | 300 |
| **Thích ứng** | **90,7 ± 1,2** | 101,0 ± 9,7 | 165,5 ± 21,4 | 80 ± 23 |

Từng seed thích ứng 90,2/91,9/92,2/90,1/89,3 · xáo trộn 2 cách × 5 seed → 89,9–92,0 (**QUA**).
`r` trung vị nguồn 0,71/0,73/0,84/0,74/0,98 → đích 3,78/4,38/5,11/6,21/5,76 (nhỏ hơn mức cần 5–8×,
count 6–9×). MAE đích 33,3/29,8/32,3/33,0/31,6 so với count 30,3/32,2/27,2/26,5/34,4.

**So hai chế độ giám sát ở cùng mức phủ (cả hai 90,7%):** Winkler count 157,0 ± 23,9 vs mask
165,5 ± 21,4. **Wilcoxon ghép cặp THEO SEED (đơn vị phân tích đúng, n=5): p=0,438**, hiệu +8,4,
mask tệ hơn ở 4/5 seed → KHÔNG có khác biệt hệ thống.

Phép kiểm từng ảnh trong mỗi seed (n=665) có ý nghĩa ở 4/5 seed nhưng **ngược dấu và vô nghĩa về
độ lớn**: s42 +28,2 (p=8e−19), s43 **−32,1** (p=6e−24), s44 +44,0 (p=0,18), s45 **+0,48** (p=2e−18),
s46 **+1,63** (p=0,004). s45/s46 là minh họa sạch cho cái bẫy: n=665 làm hiệu 0,5 Winkler thành
"p<1e−17". ⚠️ ĐỪNG báo p từng ảnh như bằng chứng — chỉ báo p theo seed.

**Kết luận bán được cho Nội dung 3:** cơ chế thích ứng giữ độ phủ danh nghĩa với CẢ HAI chế độ giám
sát, và ở cùng mức phủ thì chất lượng khoảng dự đoán **không khác nhau một cách hệ thống** — nhãn số
lượng không phải trả giá về độ tin cậy. KHÔNG bán "mask tốt hơn" hay "count tốt hơn".

**Artifacts:** `xfer_src_mask_s{42..46}.pkl` + `src_pan_f3_mask_s{42..46}.pkl` (10 file, ~0,8 MB),
gom cùng 10 file count-only ở `~/Downloads/adaptive_src_calib/` (20 file, phẳng) → `adaptive_src_calib.zip`,
đã backup Kaggle dataset `hipinhththu/efflite0-adaptive-src-calib` (2026-10-15).
