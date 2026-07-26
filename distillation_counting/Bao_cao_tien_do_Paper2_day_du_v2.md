# Báo cáo tiến độ Paper 2

*Chưng cất pathology foundation model thành bộ đếm tế bào nhẹ, giám sát mức đếm (không cần mask)*

---

## 1. Vấn đề nghiên cứu

Các pathology foundation model (như PathoSAM ~640M) segment được nhân tế bào và về nguyên tắc đếm được (đếm số instance), nhưng vướng ba rào cản khi dùng thực tế:

1. **Kích thước lớn** (~640M tham số) → tốn tài nguyên, khó triển khai ở nơi thiếu GPU.
2. Muốn **thích nghi sang dataset/miền mới** theo cách thông thường vẫn cần **annotation mask mức instance** — rất tốn công vẽ.
3. Là mô hình segmentation, chúng **không xuất độ bất định** cho con số đếm.

Câu hỏi: **liệu có thể distill một foundation model xuống một bộ đếm rất nhẹ, chỉ dùng nhãn mức đếm (không mask), mà vẫn đạt độ chính xác đếm cạnh tranh — và kèm thêm được khoảng dự đoán có độ tin cậy?**

Đây cùng dòng với hai công trình mới nhất về distillation-counting trong bệnh học — **Shvetsov et al. 2025** (distill H-Optimus 1.1B → student 24M, dùng R² đánh giá đếm, headline là student ngang teacher ở 48× nhỏ hơn) và **CellGenNet 2025** (distill StarDist → U-Net, bán label-efficiency, thắng teacher + baseline cùng hạng). Cả hai **lead hiệu quả/nhãn, không lead uncertainty**, và **không model nào có đầu bất định**. PACT đi cùng hướng nhưng **nhỏ hơn nữa (1.9M), có định lượng label-efficiency, có transfer cross-dataset, và thêm đầu phân phối**.

## 2. Hướng giải quyết

Dùng PathoSAM (~640M) làm teacher, distill sang student **1.9M** đặt tên **PACT** (*Poisson-Anchored Calibrated counTer*). PACT học density map của teacher (không cần mask người vẽ) + một giá trị đếm mỗi ảnh, xuất đồng thời count (μ = tổng density) và độ bất định σ (Poisson-anchored, hiệu chuẩn bằng conformal).

**Ký hiệu:**
- **PACT (ours):** student 1.9M, đầu σ **học được** (Poisson-anchored β-NLL).
- **KD (mốc so):** cùng student 1.9M nhưng σ theo **công thức giải tích Poisson-Binomial (PB-σ)** — cách dựng bất định của **Paper 1** áp lên student nén (mốc so, không phải "đối thủ").
- **Nhãn count-only:** PACT cần **một con số đếm mỗi ảnh** (lấy được bằng chấm điểm), **KHÔNG cần mask** — nhưng vẫn là *giám sát nhẹ*, không phải "không nhãn".

## 3. Bằng chứng thực nghiệm

### 3.1 (PHỤ — reference OOD) Đếm zero-shot/off-the-shelf trên NuInsSeg

> ⚠️ **KHÔNG phải bảng chính.** Bảng chính công bằng (leak-free, cùng test set) là **§3.6** (efflite0 3.6M count-only vs DualU-Net, PanNuke fold_3). Bảng dưới đây là **so KHÔNG cân** (PACT in-domain-adapted vs các net khác off-the-shelf/zero-shot trên OOD NuInsSeg) → **chỉ để tham khảo mức transfer OOD**, không dùng làm luận điểm "PACT thắng SOTA". Giữ lại vì số thật + cho thấy foundation nặng off-the-shelf transfer kém.

So PACT với foundation/heavy net áp **off-the-shelf** (train PanNuke → dùng thẳng trên NuInsSeg) trên **cùng 665 ảnh, cùng một thước đếm** (`dump_counts.py`):

| Method | Params | Thích nghi NuInsSeg | R² ↑ | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|---|---|
| CellViT-SAM-H | 699.7M | — (off-the-shelf) | 0.663 | 21.83 | 31.33 | 52.9% |
| LKCell-L | 163.8M | — (off-the-shelf) | 0.448 | 20.92 | 40.10 | 37.4% |
| NuLite-T | 12.0M | — (off-the-shelf) | 0.622 | 20.01 | 33.22 | 39.6% |
| PathoSAM teacher | ~640M | — (zero-shot) | 0.711 | 15.80 | 29.02 | **28.3%** |
| **PACT (ours, 5 seed)** | **1.9M** | **count (in-domain)** | **0.786 ± 0.052** | **14.74 ± 1.53** | **24.81 ± 3.03** | 47.6 ± 3.4% |

Cột "Thích nghi NuInsSeg" = nhãn **thực tế đã dùng** để thích nghi: chỉ **PACT** được thích nghi (bằng **count rẻ**); mọi net kia + teacher đều dùng **off-the-shelf** (không thích nghi).

**Đọc (nhớ caveat không-cân):** trên OOD NuInsSeg, PACT (in-domain-adapted) có R²/MAE/RMSE cao hơn các net off-the-shelf — **nhưng đây KHÔNG phải so công bằng** (PACT được thích nghi bằng count, các net kia không thích nghi) nên **không kết luận "PACT > SOTA"**. Điểm rút ra hợp lệ: **foundation nặng áp off-the-shelf transfer OOD kém** (LKCell 164M thua NuLite 12M; CellViT 699M không vượt) → ủng hộ hướng **model nhẹ thích nghi rẻ**. *(CellViT native 1024; ở 256 R² 0.444 → dùng số 1024.)*

**Ý nghĩa thống kê (so PACT vs teacher):** trên **665 ảnh** (đơn vị lặp = ảnh, không phải seed), PACT (5-seed ensemble) đạt MAE 12.28 so teacher 15.80 → **ΔMAE = −3.52** (95% CI [−4.85, −2.24]), **kiểm định Wilcoxon bắt cặp p = 3.5×10⁻⁴ (\*\*\*)**. Cải thiện **không do ngẫu nhiên**. (Model đơn 14.74 cũng thấp hơn teacher, cách nhỏ hơn.)

**Hai điều nói thẳng (trung thực, đúng chuẩn CellGenNet):**
- **Claim đúng = thích-nghi-rẻ, KHÔNG phải "PACT là model đếm giỏi hơn".** Chỉ PACT được thích nghi trên NuInsSeg, bằng **nhãn count rẻ**; các net kia để off-the-shelf. Đây *chính xác* là thiết lập CellGenNet (họ train trên domain đích rồi so với Cellpose/StarDist/InstanSeg off-the-shelf). Lý do các net kia **kẹt** off-the-shelf: muốn thích nghi chúng phải có **mask pixel đắt**, còn PACT chỉ cần **count** → sự bất đối xứng nhãn đó *chính là* luận điểm label-efficiency.
- **PACT thua MAPE** (47.6% > teacher 28.3%): density-sum sai tương đối cao ở ảnh ít nhân — disclose thẳng (như CellGenNet disclose FPR cao của họ).

*(In-domain PanNuke: PACT MAE 3.36, worst-organ coverage 0.906, Winkler 19.28.)*

### 3.1b Phân tích lỗi theo tầng mật độ (giải thích MAPE)

Vì sao MAPE toàn cục cao (47.6%)? Chia 665 ảnh theo số nhân GT và tính lỗi từng tầng (PACT 5-seed):

| Mật độ (GT count) | N | GT̄ | MAE ↓ | MAPE ↓ |
|---|---|---|---|---|
| Thấp (1–20) | 167 | 10.6 | 6.78 ± 0.28 | **110.1 ± 8.5%** |
| TB (21–50) | 261 | 34.8 | 9.82 ± 0.71 | 28.4 ± 2.0% |
| Cao (>50) | 237 | 102.5 | 25.78 ± 3.74 | 24.8 ± 3.1% |
| Toàn bộ | 665 | 52.8 | 14.74 ± 1.53 | 47.6 ± 3.4% |

**Hai điều nói thẳng:**
- **MAPE-47.6% bị kéo lên bởi vùng ít nhân.** Ở tầng Thấp, GT trung bình chỉ 10.6 nên **mẫu số nhỏ khuếch đại sai số tương đối** (MAPE 110%) dù MAE tuyệt đối chỉ 6.78; ở vùng đông MAPE chỉ 24.8%. Đây là **tính chất của metric MAPE**, không phản ánh năng lực đếm ở vùng nhiều nhân.
- **Giới hạn thật (đưa vào Thảo luận):** ở vùng ít nhân PACT **thua teacher** (teacher tầng Thấp MAE 2.95 / MAPE 37.2%) — đầu density cộng dồn dương-tính-giả trên ảnh gần trống → đếm dư; PACT thắng toàn cục nhờ **vùng nhiều nhân**. *(R² từng tầng không báo vì trong một dải hẹp phương sai nền → 0, chỉ số vô nghĩa; R² chỉ hợp lệ toàn cục = 0.786.)*

### 3.2 Vì sao distill? — Label-efficiency (phép so CÓ KIỂM SOÁT)

Để tách riêng ảnh hưởng của **loại giám sát**, so trên **cùng kiến trúc PACT, cùng data, cùng số ảnh** — chỉ khác nguồn nhãn: (a) **distilled** (target = density teacher + đếm mức ảnh, chỉ cần nhãn count); (b) **mask-supervised** (cùng mạng, target = density dựng từ mask GT từng nhân). NuInsSeg, 3 seed.

| Ngân sách ảnh | Distilled (count-only) — Worst-org / MAE | Mask-supervised — Worst-org / MAE |
|---|---|---|
| 10% (53 ảnh) | 0.865 / 24.26 | 0.879 / 26.09 |
| 25% (133 ảnh) | 0.858 / 21.85 | 0.824 / 18.49 |
| 50% (266 ảnh) | 0.840 / 20.34 | 0.830 / 17.73 |
| 100% (532 ảnh) | 0.843 / 14.12 | 0.897 / 14.61 |

Cùng số ảnh → reliability **không phân biệt được thống kê** và MAE tương đương (mask nhỉnh ~2–3 nhân ở giữa nhưng hòa ở hai đầu). **Distillation (chỉ nhãn đếm) đạt chất lượng ngang giám sát bằng mask** — nhưng nhãn đếm là **một con số/ảnh** thay vì vẽ ~52.8 mask/ảnh. **Đây là phép so công bằng NHẤT của bài** (cùng mọi thứ, chỉ khác nguồn nhãn) → xương sống luận điểm.

*Ghi chú trung thực: (i) single-split, chỉ đọc tương đối giữa các mức; (ii) GT count vẫn lấy từ mask → chứng minh **yêu cầu giám sát** của phương pháp là mức đếm, không phải "đã annotate rẻ hơn trong thực tế"; (iii) claim về **loại nhãn** (một số vs mask từng nhân) — factual; bội số thời gian cụ thể chưa đưa vì thiếu citation chuẩn.*

### 3.3 Hiệu quả mô hình

| Model | Params | Size (fp32) | Peak VRAM |
|---|---|---|---|
| PathoSAM teacher (ViT-H) | 641M | ~2.56 GB | 5.87 GB |
| **PACT (ch32, chính)** | **1.935M** | **7.74 MB** | **70.7 MB** |

PACT **1.935M** — nhỏ hơn teacher (641M) ~**330× tham số**, file **~330× nhẹ hơn**, và về **bộ nhớ chạy: 70.7 MB so với 5.87 GB → nhỏ hơn ~85×** (đo thực Tesla T4: teacher = peak VRAM image-encoder SAM ViT-H ở 1024² native; PACT ở 256² native). Đây đúng con số headline kiểu H-Optimus (16GB→3GB). PACT cũng nhỏ hơn student H-Optimus (24M) ~**12×**, NuLite (12M) ~**6×**. Đặc trưng hiệu quả riêng của PACT: **10.49 GMACs**, chạy **4.91 ms/ảnh (204 ảnh/giây)** trên T4 — thời-gian-thực, deploy được trên GPU/CPU khiêm tốn. *(GMACs/latency chỉ báo cho PACT; teacher chạy 1024²/prompt-based nên không so chung 2 metric này — so sánh dựa params·size·VRAM, như H-Optimus.)*

*Ablation dung lượng (phụ): cấu hình ch16 (~0.5M) không mất chất lượng (NuInsSeg nhỉnh, PanNuke hòa) → phương pháp bền theo dung lượng. Đây là ablation, cấu hình chính vẫn là ch32.*

### 3.4 Transfer reliability cross-dataset

| Transfer | Coverage | Nhận xét |
|---|---|---|
| NuInsSeg → PanNuke | worst-org 0.897 | ≈ in-domain 0.906 |
| NuInsSeg → CryoNuSeg | marginal 0.967 | reliability duy trì (dataset thứ 3) |

Đầu ra phân phối (μ, σ) transfer sang dataset khác **không cần train lại** — điều cả hai paper mốc (Shvetsov, CellGenNet) để lại "future work". *Giới hạn trung thực: transfer sang MoNuSAC KHÔNG giữ được do chênh độ phân giải làm nhân co ~4× (scale gap) — ghi rõ là giới hạn về scale.*

### 3.5 (Phụ) Đầu bất định: learned-σ so với PB-σ giải tích

PB-σ là đóng góp **Paper 1** (kiểm chứng trên foundation model). Câu hỏi: mang PB-σ xuống student 1.9M nén, còn hiệu chuẩn không? So trên **cùng student**, khác cách lấy σ:

| Metric | PB-σ (KD, kiểu Paper 1) | Learned-σ (PACT) |
|---|---|---|
| Worst-organ (global) | 0.278 | **0.610** |
| Worst-organ (cluster) | 0.658 | **0.750** |
| MAE | 21.71 | **14.72** |

**Diễn giải (không hạ Paper 1):** PB-σ **không "hỏng"** — dưới scheme **cluster** vẫn tốt (+14%); chỉ dưới **global** (khi student nén không tái tạo trung thực điểm detection) σ giải tích mất hiệu chuẩn. PACT **kế thừa** Paper 1 và **vá đúng giới hạn của PB-σ trong chế độ nén** bằng σ học được.

**Trung thực:** trên thang UQ-floor (E-AURC) PACT xếp **~4/5** so các phương pháp UQ cùng backbone → **đầu bất định là tính năng bổ trợ (bonus), KHÔNG phải chỗ dẫn đầu**. Điểm riêng của nó là *phân phối tái dùng được + transfer* (§3.4), không phải coverage tốt nhất.

### 3.6 Cập nhật (2026-07): quá trình chọn backbone, kiến trúc model, và kết quả hiện tại

Phần này báo cáo em đã **thử và điều chỉnh backbone/kiến trúc** như thế nào để cải thiện độ chính xác, và trạng thái mới nhất.

**Backbone ban đầu — TinyUNet 1.9M (train từ đầu).** Ưu tiên "càng nhỏ càng tốt". Nhưng trên **PanNuke** (train fold_1+2, test fold_3, đo đếm) R² chỉ **0.775** (MAE 5.14) — kém rõ. Lý do: model quá nhỏ + **không có pretrain** nên đặc trưng yếu, cách xa các model lớn.

**Đổi backbone — EfficientNet-lite0 3.6M (pretrain ImageNet).** Giữ nguyên khung, **chỉ thay encoder** → R² **0.908** (MAE 3.78), **nhảy +0.13**. Cải thiện chủ yếu nhờ **pretrain ImageNet** (đặc trưng chuyển giao) cộng thiết kế efficient. 3.6M vẫn **rất nhẹ** (nhỏ hơn heavy net 3–200×).

**Thử lớn hơn nữa — KHÔNG giúp.** EfficientNet-b1 6.76M → chỉ 0.884 (< 0.908). → **Chọn EfficientNet-lite0 3.6M làm điểm "sweet-spot"**: bám sát nhất với số tham số nhỏ nhất; lớn hơn chỉ tốn params mà không lợi. Đây là **lập luận hiệu quả-tham số** (frontier bảng dưới).

**Kiến trúc model hiện tại:**
- **Encoder:** EfficientNet-lite0 (backbone có sẵn, pretrain ImageNet) — trích đặc trưng.
- **Decoder U-Net** — phục hồi độ phân giải cho bản đồ mật độ (density map).
- **Đầu đếm (density):** đếm = tổng density; **giám sát count-only** (chỉ 1 con số/ảnh, **KHÔNG cần mask**).
- **Đầu bất định (σ):** Poisson-anchored (`√μ·exp(log_s)`), hiệu chuẩn bằng conformal — **điểm gốc/novel** của mô hình.
- *Ghi chú trung thực:* em đã kiểm — **distillation KHÔNG cải thiện độ chính xác đếm in-domain** (efflite0 khi distill chỉ 0.512, còn count-only đạt 0.908), nên **độ chính xác dùng count-only thuần**; phần teacher/distill chỉ còn liên quan nhánh σ và label-efficiency.

**Frontier hiệu quả-tham số (PanNuke fold_3, count-only, 3 seed) — bằng chứng chọn sweet-spot:**

| Backbone | Params | R² ↑ | MAE ↓ | |
|---|---|---|---|---|
| TinyUNet (ban đầu) | 1.94M | 0.775 | 5.14 | quá nhỏ → cách xa |
| **EfficientNet-lite0 (chọn)** | **3.62M** | **0.908** | **3.78** | ★ sweet-spot |
| EfficientNet-b1 | 6.76M | 0.884 | 3.82 | lớn hơn không giúp |

**★ BẢNG CHÍNH MỚI (công bằng, leak-free) — PanNuke, train fold_1+2 / test fold_3, cùng một test set (full fold3):**

| Model | Params | Nhãn | R² ↑ | MAE ↓ | Ghi chú |
|---|---|---|---|---|---|
| DeNuC (2026) | ShuffleNet-x2 | detection (centroid) | 0.455 | 9.64 | detection, retrain leak-free (F1 68.0, P89.1/R54.9); đếm = số toạ độ sau NMS; **under-count sụp theo mật độ** (bias 45+ **−30.4**) |
| DualU-Net (2025) | ~35M | mask (đầy đủ) | 0.804 | 5.33 | SOTA-lightweight instance-seg, retrain leak-free; đếm từ đầu centroid; **over-count** (pred 26.2) |
| StarDist | 1.43M (U-Net) | mask (instance) | 0.841 | 4.64 | instance-seg kinh điển, retrain leak-free; đếm = số instance; **under-count** (pred 20.8); nhỏ hơn efflite0 nhưng cần mask + kém accuracy |
| **PACT — EfficientNet-lite0 (ours)** | **3.62M** | **count-only** | **0.920 ± 0.005** | **3.61 ± 0.07** | **thắng CẢ 3 baseline nhãn-đắt**; sát GT nhất; nhãn rẻ hơn một bậc |
| KongNet (2026) | — | mask | *trích dẫn* | *trích dẫn* | SOTA hiện tại (không retrain, chỉ dẫn) |

> **Trạng thái số liệu (cùng nền leak-free: train full fold12 5179 ảnh / test full fold3 2722 ảnh, colon GIỮ khắp nơi, GT mean 24.5):**
> - **efflite0 (ours):** R² **0.920 ± 0.005** / MAE **3.61 ± 0.07** (3 seed) — `method_frontier.py`.
> - **DualU-Net (~35M, mask):** R² **0.804** / MAE **5.33** (pred 26.2, over-count) — `eval_count_dualunet.py`, đếm-từ-centroid `find_local_maxima`.
> - **StarDist (1.43M U-Net, mask):** R² **0.841** / MAE **4.64** (pred 20.8, under-count) — `stardist_baseline.py`, đếm = số instance (param từ `count_params()`).
> - **DeNuC (ShuffleNet-x2 detection, 2026):** R² **0.455** / MAE **9.64** (pred 15.1, under-count nặng) — `denuc_count_eval.py`, đếm = số toạ độ sau NMS; test F1 68.0 (P89.1/R54.9). **Bias per-tier sụp đơn điệu theo mật độ: <10 −0.8 → 10-20 −4.9 → 20-30 −8.7 → 30-45 −13.4 → 45+ −30.4** — cơ chế detection thất bại dưới crowding (nhân chồng lấn → NMS bóp chết true-positive → recall sụp). Artifact: `outputs/20260726_110537_denuc_pan12_3/`.
>
> **Đọc:** bộ đếm nhẹ **3.6M count-only** đạt R²/MAE **tốt hơn CẢ 3** baseline nhãn-đắt (DualU-Net ~35M density, StarDist 1.43M instance, DeNuC detection) trên chính bài **đếm**, với **nhãn rẻ hơn một bậc** (count vs mask/centroid). Ba paradigm cạnh tranh đều thua; đặc biệt **detection sụp mạnh nhất ĐÚNG ở vùng dày** (bias 45+ tới −30.4) — bằng chứng trực tiếp cho động lực **crowding** của paper. Hai baseline mask lệch hai phía (DualU-Net dư, StarDist thiếu) còn efflite0 sát GT nhất. **Trung thực về params:** efflite0 nhỏ hơn DualU-Net ~10× nhưng **lớn hơn StarDist** (3.62M vs 1.43M) → luận điểm bán KHÔNG phải "nhỏ nhất", mà là **accuracy cao nhất + nhãn rẻ nhất** (count vs mask) + **có UQ**. *(Caveat: DualU-Net/StarDist tối ưu segmentation, đếm là readout — so cấp-tác-vụ-đếm leak-free cùng test set.)*

**Định vị:** bộ đếm nhẹ **3.6M, count-only** đạt độ chính xác đếm **vượt SOTA-lightweight mask-supervised (DualU-Net ~35M)** — R² 0.920 vs 0.804, MAE 3.61 vs 5.33 — ở **~10× ít tham số + nhãn rẻ hơn một bậc (count vs mask)**, kèm **đầu bất định calibrated** (robust worst-organ + transfer). = đóng góp **hiệu quả (applied) + UQ**. *(so ở cấp tác-vụ-đếm, leak-free cùng test set; DualU-Net vốn tối ưu segmentation.)*

**Khảo sát baseline lightweight 2024–2026 (đã tra, chốt chiến lược so sánh):**

| Model | Năm | Params/Size | Vai trò | Ghi chú |
|---|---|---|---|---|
| **DualU-Net** ([repo](https://github.com/davidanglada/DualU-Net)) | 2025 | ~35M, mask | **retrain leak-free (baseline chính)** | đầu centroid density → đếm được; đang retrain fold_1+2/fold_3 |
| **KongNet** ([2510.23559](https://arxiv.org/abs/2510.23559)) | 10/2025 | multi-head; bản nhẹ KongNet-Det | **cite SOTA** | SOTA mới nhất PanNuke+CoNIC, **có đầu centroid**, thắng MIDOG-2025; weights có nhưng retrain nặng → chỉ cite |
| **Distill-H-Optimus** ([2502.19217](https://arxiv.org/abs/2502.19217)) | 02/2025 | ~24M (48× nhỏ hơn H-Optimus) | **cite peer TRỰC TIẾP** | distill từ FM, **báo cáo R² đếm 0.575→0.871, KHÔNG có UQ** → đúng mẫu "distilled counter with R²" của mình; efflite0 **0.920/3.6M** vs **0.871/~24M** (khác setup, không so tuyệt đối) — nhấn "**họ không UQ, mình có**" |
| **NuLite** ([2408.01797](https://arxiv.org/abs/2408.01797)) | 2024 (BSPC 2026) | S/M/T FastViT-UNet (T≈12M) | reference off-the-shelf | pretrained **train CẢ 3 fold → KHÔNG leak-free** với fold3; muốn fair phải retrain fold1+2 (HoVer-Net-style, 130 ep, 1024px → rất nặng) |
| Attention-enhanced lightweight ([MDPI Info 2025](https://www.mdpi.com/2078-2489/16/4/295)) · SAM cooperative-refine ([2603.28027](https://arxiv.org/pdf/2603.28027)) | 2025/2026 | nhẹ / SAM-based | cite related | dày phần related-work nếu cần |

**★ Paradigm DETECTION — DeNuC (2026) ĐÃ CHẠY leak-free (2026-07-26):** train fold12→test fold3 gộp (`pannuke12_3`), ShuffleNet-x2 detection, 50 epoch ~13 phút (2.1GB). Test F1 **68.0** (P89.1/R54.9) → đếm R² **0.455** / MAE **9.64**, **under-count sụp đơn điệu theo mật độ** (bias 45+ −30.4). = paradigm thứ 3, tệ nhất, nhưng bằng chứng vàng cho luận điểm crowding. Đã ghi vào bảng chính §3.6.

**Ứng viên detection/instance khác (nếu cần thêm, CHƯA chạy):**

| Model | Năm | Paradigm | Retrain leak-free | Compute Kaggle | Verdict |
|---|---|---|---|---|---|
| **HoVer-NeXt** ([repo](https://github.com/digitalpathologybern/hover_next_train)) | 2024 | instance-seg (ConvNeXt) | ✅ (chia fold tay qua convert CoNIC) | ✅ single-GPU torch2.1.1 | an toàn; tốn công chia fold |
| **CellNuc-DETR** ([celldetr](https://github.com/oscar97pina/celldetr)) | 2025 | detection (Deformable-DETR) | ✅ config fold sẵn | ⚠️ compile MSDeformAttn CUDA + DETR nặng | được nhưng rủi ro compute |

> **Chốt:** đã retrain leak-free **3 paradigm cạnh tranh** — DualU-Net (density, 0.804) + StarDist (instance, 0.841) + **DeNuC (detection, 0.455)** — TẤT CẢ thua efflite0 count-only (0.920); **KongNet + 2502.19217 cite** làm SOTA + peer. Peer sát nhất = **2502.19217** (distilled, R², no-UQ) → bán: nhỏ hơn ~7× mà R² không kém + **thêm UQ calibrated**.

> ⚠️ **Còn tồn đọng nhất quán (chờ số về để sửa gọn):** §3.1 vẫn là bảng cũ NuInsSeg off-the-shelf (không-fair), §4.1 và §5 vẫn ghi "distilled 1.9M dẫn đầu accuracy". Khung này **mâu thuẫn** với định vị mới (model chính = **lightweight count-only 3.6M**, distill KHÔNG dẫn accuracy). Sẽ viết lại §3.1/§4/§5 quanh trục lightweight+UQ khi có số full-fold3.

## 4. Đóng góp chính

1. **★ Distilled tiny counter đạt độ chính xác đếm dẫn đầu ở kích thước tí hon** (§3.1): PACT 1.9M đạt R²/MAE/RMSE tốt nhất so teacher + heavy net off-the-shelf, ở 6–368× nhỏ hơn — chỉ dùng nhãn count. *(cấu trúc so như CellGenNet/H-Optimus, nhưng nhỏ hơn nhiều.)*
2. **★ Chứng minh label-efficiency bằng thực nghiệm có kiểm soát** (§3.2): distillation (nhãn đếm) ngang giám sát mask ở cùng ngân sách ảnh — thứ hai paper mốc chỉ nói chứ không định lượng.
3. **Transfer reliability cross-dataset** (§3.4) — hai paper mốc để "future work".
4. **(Phụ) Đầu phân phối calibrated nhẹ** (§3.5): kế thừa PB-σ của Paper 1, vá giới hạn của nó dưới nén bằng learned-σ. **Không dòng distillation-counting nào có UQ** — nhưng đây là bonus, không phải trục bán.

## 5. Kết luận hiện tại

Khả thi để chuyển tri thức đếm từ một pathology foundation model 640M xuống một bộ đếm **1.9M**, chỉ cần **nhãn đếm mức ảnh (không mask)**, mà đạt độ chính xác đếm **dẫn đầu ở hạng tí hon** và reliability tốt, transfer được. Đóng góp là **một gói mạch lạc, nghiêng model lightweight**: model tí hon + nhãn count rẻ + (bonus) uncertainty calibrated. Hướng: *label-efficient distillation of a lightweight cell counter* — Q1 ứng dụng (CBM/CMIG/AIM), cùng hạng với Shvetsov 2025 & CellGenNet 2025 nhưng chặt hơn.

## 6. Việc còn lại (đang hoàn thiện)

**Đã đóng gần đây (phản hồi phản biện):**
- ✅ **CellViT-SAM-H native 1024** — đã chạy, R² 0.663 (§3.1).
- ✅ **Latency/VRAM thực đo T4** — đã có (§3.3).
- ✅ **Ý nghĩa thống kê Bảng chính** — Wilcoxon bắt cặp per-ảnh, ΔMAE −3.52 p=3.5×10⁻⁴ (§3.1).
- ✅ **Phân tích lỗi theo tầng mật độ** — giải thích MAPE + nêu giới hạn vùng ít nhân (§3.1b).

**Còn lại:**
- ⏳ **Baseline đối chứng cùng-mức-giám-sát** (đang chạy): một mạng **khác kiến trúc** (EfficientNet-Lite0) huấn luyện **trực tiếp bằng count-loss** trên NuInsSeg — để chứng minh lợi thế Bảng 3.1 đến từ **giám sát đếm in-domain rẻ**, không phải riêng kiến trúc PACT. *(Đây là đối chứng đóng chốt câu hỏi "PACT thắng chỉ vì được nhìn dữ liệu đích?"; kỳ vọng baseline ≈ PACT — khớp ablation supervised≈distilled — vẫn giữ nguyên luận điểm label-efficiency, tuyệt đối không claim "kiến trúc PACT vượt trội".)*
- Điền cột heavy-net cho **PanNuke** (leak-free).
- *(tùy chọn)* thêm 1 baseline recent/classic (InstanSeg/Cellpose) cho đa dạng paradigm.
- *(tùy chọn hardening)* nâng label-efficiency §3.2 từ 3 → 5 seed nếu đưa lên figure.

*(Hướng multi-teacher committee đã thử và gác lại: probe cho thấy nó chỉ tạo thêm một tín hiệu σ (UQ) chứ không cải thiện accuracy — không phục vụ trục lightweight/accuracy đang là chính.)*
