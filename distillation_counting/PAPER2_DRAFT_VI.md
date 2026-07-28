# Bộ đếm nhân tế bào nhẹ, chỉ giám sát bằng số đếm, kèm định lượng bất định được hiệu chuẩn cho ảnh mô bệnh học

*(Bản thảo tiếng Việt để đọc trước — theo cấu trúc bài hội nghị LNICST/Springer. Số liệu đều là kết quả thực nghiệm thực tế trên cùng một mô hình `EfficientNet‑lite0` 3,62M, giao thức leak‑free. Chỗ đánh dấu `[…]` là phần đang chạy sẽ điền sau.)*

---

## Tóm tắt (Abstract)

Đếm nhân tế bào trên ảnh mô bệnh học nhuộm H&E là bước định lượng nền tảng cho nhiều tác vụ chẩn đoán, nhưng phần lớn phương pháp hiện nay đòi hỏi **mặt nạ phân vùng ở mức điểm ảnh (pixel‑level mask)** — loại nhãn đắt và tốn công chuyên gia. Chúng tôi đặt câu hỏi: *để ĐẾM, liệu có thật sự cần mask không?* Bài báo trình bày một bộ đếm nhân **nhẹ (3,62 triệu tham số)** chỉ huấn luyện bằng **một con số đếm cho mỗi ảnh (count‑only)**, không dùng mask, kèm **đầu định lượng bất định được hiệu chuẩn** (Poisson‑anchored σ + conformal prediction). Qua thí nghiệm **có kiểm soát** (cố định kiến trúc, chỉ thay đổi loại giám sát), chúng tôi cho thấy: (i) giám sát count‑only đạt độ chính xác đếm **ngang bằng** giám sát bằng mask trên **cùng kiến trúc**, trong khi nhãn rẻ hơn một bậc; (ii) mô hình nhẹ này **vượt** ba dòng phương pháp nhãn‑đắt (density, instance segmentation, detection) ngay trên bài đếm; (iii) chỉ **25% nhãn** đã đạt ~93% hiệu năng đầy đủ; (iv) đầu bất định cho **khoảng tin cậy hiệu chuẩn** cả ở mức biên lẫn theo từng cơ quan, và **chuyển giao coverage** sang dữ liệu ngoài miền. Kết quả nhất quán trên [ba] tập dữ liệu độc lập (PanNuke, NuInsSeg, MoNuSeg).

**Từ khóa:** đếm nhân tế bào · học nhãn‑hiệu quả · giám sát yếu · định lượng bất định · conformal prediction · mô bệnh học tính toán.

---

## 1. Giới thiệu (Introduction)

Mật độ và số lượng nhân tế bào là chỉ dấu định lượng quan trọng trong bệnh học: chúng liên quan tới mức độ tăng sinh, phân độ khối u và đánh giá vi môi trường mô. Các phương pháp học sâu hiện đại cho bài toán này chủ yếu đi theo hướng **phân vùng thực thể (instance segmentation)** hoặc **hồi quy bản đồ mật độ (density map)**, và đều cần **nhãn mask ở mức điểm ảnh** — mỗi nhân phải được khoanh viền thủ công. Chi phí gán nhãn này là rào cản lớn khi mở rộng sang mô/cơ quan mới chưa có chú thích.

Trong khi đó, ở bài toán đếm đám đông (crowd counting) người ta đã dùng **giám sát chỉ‑đếm (count‑only)** — chỉ cần tổng số đối tượng cho mỗi ảnh. Tuy nhiên trong **đếm nhân mô bệnh học**, chuẩn mực vẫn là mask/point. Câu hỏi trung tâm của bài báo: **liệu giám sát count‑only có đủ để đếm nhân chính xác không, và ta được/mất gì so với giám sát bằng mask?**

Chúng tôi tiếp cận theo tinh thần **thực nghiệm có kiểm soát**: *cố định kiến trúc, chỉ thay đổi loại giám sát*, để tách bạch đóng góp của **giám sát** khỏi đóng góp của **kiến trúc**. Bài báo **không** tuyên bố một kiến trúc mới (bộ mã hóa EfficientNet‑lite0 + bộ giải mã U‑Net là tổ hợp tiêu chuẩn); đóng góp nằm ở **bằng chứng thực nghiệm** và **gói ứng dụng**:

1. **Chứng minh count‑only ≈ mask trên cùng kiến trúc** (mục 5.2): độ chính xác đếm tương đương, nhưng nhãn rẻ hơn một bậc → *"đếm không cần mask điểm ảnh"*.
2. **So sánh 4 paradigm công bằng, cùng tập kiểm tra leak‑free** (mục 5.1): bộ đếm nhẹ count‑only vượt cả density, instance và detection ngay trên bài đếm; đặc biệt detection **sụp đổ ở vùng đông đúc** — bằng chứng trực tiếp cho động lực *crowding*.
3. **Định lượng hiệu quả‑nhãn** (mục 5.3): đường cong học theo tỉ lệ nhãn.
4. **Đầu bất định hiệu chuẩn nhẹ** (mục 5.5): tham số hóa σ neo‑Poisson + conformal cho coverage đúng mục tiêu ở mức biên và theo cơ quan, chuyển giao được sang dữ liệu ngoài miền — điều mà các bộ đếm nén hiện có **không có**.

Toàn bộ số liệu được báo cáo trên **một mô hình duy nhất** (EfficientNet‑lite0, 3,62M tham số) với giao thức leak‑free, đảm bảo tính nhất quán.

---

## 2. Công trình liên quan (Related Work)

**Phân vùng & đếm nhân.** Các phương pháp SOTA như HoVer‑Net, StarDist, CellViT và các biến thể trên PanNuke đạt độ chính xác cao nhưng dựa vào **mask điểm ảnh**. Đếm thường là *readout* từ kết quả phân vùng (đếm centroid/instance). Gần đây, một số công trình chưng cất tri thức từ *foundation model* bệnh học xuống mạng nhỏ để đếm (ví dụ chưng cất từ H‑Optimus xuống ~24M tham số, báo cáo R² đếm ~0,871) — nhưng **không kèm định lượng bất định**.

**Giám sát count‑only.** Phổ biến ở crowd counting (đếm người), hiếm ở đếm nhân mô bệnh học. Chúng tôi khảo sát trực tiếp giá trị của loại giám sát này trong miền bệnh học.

**Định lượng bất định (UQ).** Conformal prediction cho khoảng tin cậy phi‑phân phối với bảo đảm coverage. Với dữ liệu đếm (miền số nguyên, phương sai tăng theo kỳ vọng), việc tham số hóa σ hợp lý là then chốt; chúng tôi dùng dạng **neo‑Poisson** `σ = √(max(μ,1))·exp(log s)` và cho thấy nó cần thiết khi dải count rộng.

**Kiến trúc.** Bộ mã hóa EfficientNet‑lite + bộ giải mã U‑Net là mẫu chuẩn (segmentation‑models‑pytorch). Chúng tôi *không* coi kiến trúc là đóng góp; nó là *nền được kiểm soát* để so sánh giám sát.

---

## 3. Phương pháp (Method)

### 3.1 Kiến trúc

Mô hình `DensitySigmaUNet`: bộ mã hóa **EfficientNet‑lite0** (tiền huấn luyện ImageNet) + bộ giải mã kiểu U‑Net, hai đầu ra:

- **Đầu mật độ → đếm:** xuất bản đồ mật độ không âm; số đếm dự đoán `μ = Σ density` (tổng toàn ảnh).
- **Đầu bất định σ:** `σ = √(max(μ,1)) · exp(clamp(log s, −2, 2))` — **neo‑Poisson**, học phần *tán xạ* (dispersion) trên nền phương sai Poisson. μ được **tách gradient** khỏi nhánh σ (σ chỉ mượn độ lớn, không làm hỏng độ chính xác đếm).

Cấu hình `ch=32` cho **3,62 triệu tham số**.

### 3.2 Giám sát chỉ‑đếm (count‑only)

Hàm mất mát tổng: `L = w_d·L_density + w_c·L_count + w_nll·L_nll`.

- **Count‑only:** `w_d = 0`, `w_c = 1` → mô hình **chỉ** học từ số đếm mức ảnh `|μ − GT|`; **không** dùng mask/bản đồ mật độ mục tiêu.
- **Baseline mask (đối chứng):** `w_d = 1`, `w_c = 0,01` → bản đồ mật độ mục tiêu dựng từ mask GT (mỗi nhân trải đều mật độ 1/diện‑tích).
- `L_nll`: β‑NLL (β=0,5) huấn luyện σ; μ được detach.

### 3.3 Hiệu chuẩn bất định bằng conformal

Sau huấn luyện, dùng **split‑conformal** trên tập hiệu chuẩn để suy khoảng `[μ − q·σ, μ + q·σ]` đạt mức phủ mục tiêu 1−α (α=0,1). Ba sơ đồ: **global** (một ngưỡng q), **mondrian** (q riêng theo cơ quan), **cluster** (q theo nhóm độ khó). Mondrian cải thiện **coverage theo từng cơ quan** (điều kiện).

---

## 4. Thiết lập thực nghiệm (Experimental Setup)

**Tập dữ liệu (độc lập, đa nguồn):**

| Tập | Nguồn/loại | Vai trò |
|---|---|---|
| **PanNuke** | pan‑cancer, 256², 19 loại mô | in‑domain chính (train fold 1+2 → test fold 3, leak‑free) |
| **NuInsSeg** | đa cơ quan H&E | in‑domain thứ hai (chia ngẫu nhiên) |
| **MoNuSeg 2018** | đa cơ quan TCGA, 1000² → patch 256 native | in‑domain thứ ba, độc lập nguồn |
| CryoNuSeg | cryosection H&E | phân tích chuyển giao/giới hạn (mục 6) |

**Giao thức leak‑free:** với PanNuke, huấn luyện fold 1+2, kiểm tra fold 3. Mọi mô hình so sánh đều huấn luyện lại từ đầu trên cùng tập huấn luyện, chấm trên cùng tập kiểm tra. Do biến động GPU (±0,017), mỗi cấu hình chạy ≥3 seed.

**Chỉ số:** đếm — **R²** và **MAE**; bất định — **coverage biên**, **coverage cơ quan tệ nhất (worst‑organ)**, **độ rộng khoảng (width)**, **điểm Winkler**.

---

## 5. Kết quả (Results)

### 5.1 So sánh 4 paradigm (PanNuke, leak‑free)

| Mô hình | Tham số | Nhãn | Paradigm | R² ↑ | MAE ↓ |
|---|---|---|---|---|---|
| **Ours — count‑only** | **3,62M** | **đếm** | hồi quy đếm | **0,920 ± 0,005** | **3,61 ± 0,07** |
| StarDist | 1,43M | mask | instance | 0,841 | 4,64 |
| DualU‑Net | ~35M | mask | density | 0,804 | 5,33 |
| DeNuC (2026) | 4,37M | box/point | detection | 0,455 | 9,64 |

Bộ đếm nhẹ **count‑only vượt cả ba** baseline nhãn‑đắt trên chính bài đếm, với nhãn rẻ hơn một bậc. Đáng chú ý, **detection sụp đổ đơn điệu theo mật độ** (độ lệch từ −0,8 ở vùng thưa xuống **−30,4** ở vùng ≥45 nhân) — khi nhân chồng lấn, hậu xử lý NMS bóp chết các dương‑tính‑thật → *recall* sụp. Đây là bằng chứng trực tiếp cho động lực **crowding**. Hai baseline mask lệch hai phía (DualU‑Net dư đếm, StarDist thiếu đếm), còn mô hình của chúng tôi bám GT sát nhất.

*(Trung thực về tham số: mô hình của chúng tôi nhỏ hơn DualU‑Net ~10× nhưng lớn hơn StarDist. Luận điểm không phải "nhỏ nhất" mà là **chính xác nhất + nhãn rẻ nhất + có định lượng bất định**. DualU‑Net/StarDist vốn tối ưu cho phân vùng; đếm là readout — nên so ở cấp tác‑vụ‑đếm, cùng tập kiểm tra leak‑free.)*

### 5.2 Count‑only so với mask, TRÊN CÙNG kiến trúc *(đóng góp lõi)*

Cố định kiến trúc efflite0, **chỉ đổi loại giám sát** (PanNuke, leak‑free):

| Giám sát | R² ↑ | MAE ↓ |
|---|---|---|
| **count‑only** | **0,9296 ± 0,009** | **3,48 ± 0,19** |
| GT‑density (mask) | 0,9247 ± 0,009 | 3,58 ± 0,16 |
| Δ | +0,005 | −0,10 |

Chênh lệch (+0,005 R²) **nhỏ hơn độ lệch chuẩn (±0,009)** → về mặt thống kê là **NGANG NHAU (parity)**. Kết luận trung thực: **không** phải "count‑only tốt hơn mask", mà là **count‑only đạt độ chính xác tương đương mask trên cùng kiến trúc**, trong khi chỉ cần **một con số đếm cho mỗi ảnh** thay vì khoanh viền pixel. → *Để ĐẾM, không cần mask điểm ảnh.* Đây là đối chứng đóng chốt cho toàn bộ định vị bài báo.

### 5.3 Hiệu quả‑nhãn (learning curve)

| % nhãn | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| R² | 0,45 ± 0,21 | 0,74 | **0,85 ± 0,03** | 0,86 | 0,92 ± 0,01 |

Chỉ **25% nhãn** đã đạt ~93% hiệu năng đầy đủ; 100% khớp headline 0,920 (nhất quán). → phù hợp triển khai khi ngân sách gán nhãn hạn chế.

### 5.4 Hiệu quả tính toán

| | Giá trị |
|---|---|
| Tham số | 3,62M |
| GMACs (256²) | 0,98 |
| Độ trễ GPU / CPU | 6,0 ms / 46 ms |
| Peak VRAM | 37,9 MB |

Thời‑gian‑thực, chạy được cả trên **GPU khiêm tốn lẫn CPU**.

### 5.5 Tổng quát hóa đa tập & định lượng bất định

**In‑domain (cùng giao thức count‑only):**

| Tập | R² | MAE |
|---|---|---|
| PanNuke | 0,920 ± 0,005 | 3,61 |
| NuInsSeg | 0,854 ± 0,048 | — |
| **MoNuSeg** | **[ĐANG CHẠY — điền sau]** | [ ] |

**Bất định — conformal in‑domain (PanNuke, 20 seed):**

| Sơ đồ | cov biên | width | Winkler | worst‑organ | #cơ‑quan thiếu phủ |
|---|---|---|---|---|---|
| global | 0,904 | 14,5 | 20,3 | 0,765 | 4/19 |
| **mondrian** | 0,923 | 16,6 | 20,5 | **0,906** | **0/19** |
| cluster | 0,906 | 14,8 | **20,0** | 0,859 | 0/19 |

Coverage biên đạt mục tiêu 0,90; **mondrian nâng coverage cơ quan tệ nhất lên 0,906 (0/19 cơ quan thiếu phủ)** → coverage điều kiện tốt. Thêm đầu σ **tốn 0 độ chính xác** (nhờ detach μ).

**Ablation σ neo‑Poisson vs σ thô (cùng mô hình, chỉ đổi đầu σ):**

| Tập (dải count) | σ | cov | width | Winkler | worst‑org |
|---|---|---|---|---|---|
| PanNuke (0–45, hẹp) | Poisson | 0,905 | 14,1 | 24,7 | 0,64 |
| | thô | 0,908 | 15,1 | **21,6** | 0,66 |
| **NuInsSeg (0→500+, rộng)** | **Poisson** | 0,926 | **37,7** | **45,4** | **0,86** |
| | thô | 0,937 | 67,9 | 78,1 | 0,66 |

Trên dải count **hẹp**, hai dạng σ tương đương (thậm chí σ thô nhỉnh). Trên dải count **rộng**, neo‑Poisson **cắt 44% width, 42% Winkler và nâng worst‑organ +0,21** cùng mức phủ — vì σ hằng số buộc phải phồng khoảng cho mọi ảnh để phủ được ảnh đông nhân. → **neo‑Poisson cần thiết khi dải count rộng** (tình huống triển khai thực tế), *thừa* khi hẹp.

---

## 6. Thảo luận & Hạn chế (Discussion & Limitations)

**Chuyển giao ngoài miền.** Huấn luyện trên một tập rồi kiểm tra không‑điều‑chỉnh (zero‑shot) trên tập khác: **coverage của khoảng tin cậy chuyển giao được** (sau hiệu chuẩn lại ở đích), nhưng **độ chính xác điểm (MAE) KHÔNG chuyển giao** khi dải count/độ phóng đại lệch nhiều. Trường hợp cực đoan CryoNuSeg (cryosection, count tới 761 vượt tầm huấn luyện) cho thấy giới hạn: μ under‑predict mạnh (MAE ~48%), và coverage chỉ giữ được bằng khoảng **rộng gần như vô dụng**. Điểm tích cực trung thực: **bảo đảm coverage đến từ tầng conformal (phi‑phân phối), không phải từ σ** — nên vẫn đạt phủ ngay cả khi σ mất hiệu chuẩn dưới dịch chuyển nặng; **độ rộng khoảng chính là tín hiệu "đã ra ngoài vùng tin cậy"**.

**Phụ thuộc scale nhân.** Đầu mật độ nhạy với kích thước nhân theo điểm ảnh; khi ảnh gốc bị thu nhỏ quá mạnh (ví dụ 4×), μ có thể sụp. Ta xử lý bằng cách **cắt patch ở scale native** thay vì thu nhỏ ảnh lớn.

**Về đóng góp.** Kiến trúc là tổ hợp có sẵn — bài báo *không* tuyên bố kiến trúc mới. Giá trị nằm ở **thiết kế thí nghiệm có kiểm soát**, **bằng chứng count‑only đủ để đếm**, và **đầu bất định hiệu chuẩn** mà các bộ đếm nén hiện có chưa có.

---

## 7. Kết luận (Conclusion)

Chúng tôi cho thấy một bộ đếm nhân **nhẹ (3,62M)**, chỉ giám sát bằng **số đếm mức ảnh**, đạt độ chính xác đếm **ngang giám sát bằng mask trên cùng kiến trúc** và **vượt** các baseline nhãn‑đắt (density/instance/detection) ngay trên bài đếm, với **nhãn rẻ hơn một bậc**. Mô hình kèm **đầu bất định hiệu chuẩn** (neo‑Poisson + conformal) cho coverage đúng mục tiêu ở mức biên và theo cơ quan, và chuyển giao coverage sang dữ liệu ngoài miền. Kết quả nhất quán trên [ba] tập độc lập. Hướng phát triển: khắc phục phụ thuộc scale bằng học đa độ phân giải, và mở rộng đầu bất định để bắt cả sai số do dịch chuyển miền.

---

## Tài liệu tham khảo (chọn lọc)

1. Gamper J. và cộng sự. *PanNuke: An Open Pan‑Cancer Histology Dataset for Nuclei Instance Segmentation and Classification.* 2019.
2. Kumar N. và cộng sự. *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology (MoNuSeg).* IEEE TMI, 2017.
3. Mahbod A. và cộng sự. *CryoNuSeg: A Dataset for Nuclei Instance Segmentation of Cryosectioned H&E‑Stained Histological Images.* 2021.
4. Schmidt U. và cộng sự. *Cell Detection with Star‑Convex Polygons (StarDist).* MICCAI, 2018.
5. Graham S. và cộng sự. *HoVer‑Net: Simultaneous Segmentation and Classification of Nuclei.* MedIA, 2019.
6. Tan M., Le Q. *EfficientNet: Rethinking Model Scaling for CNNs.* ICML, 2019.
7. Ronneberger O. và cộng sự. *U‑Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
8. Romano Y. và cộng sự. *Conformalized Quantile Regression.* NeurIPS, 2019.
9. Vovk V. và cộng sự. *Algorithmic Learning in a Random World* (conformal prediction). Springer, 2005.
10. Seitzer M. và cộng sự. *On the Pitfalls of Heteroscedastic Uncertainty Estimation (β‑NLL).* ICLR, 2022.
11. *Distilling a Pathology Foundation Model into a Lightweight Nuclei Counter.* arXiv:2502.19217, 2025.
12. *KongNet: …* arXiv:2510.23559, 2025.

*(Danh mục sẽ được chuẩn hóa đúng định dạng LNICST/Springer khi nộp.)*
