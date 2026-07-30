# Upload lên Overleaf

Folder này **tự chứa đủ** (đã có sẵn template `sn-jnl.cls` + `sn-mathphys-num.bst`).
Chỉ cần **upload cả folder** → compile ngay, không phải đi lấy template ở đâu khác.

**Các bước:**
1. Overleaf → New Project → **Upload Project** → chọn toàn bộ folder này (hoặc nén .zip rồi upload).
2. Compiler = **pdfLaTeX** (đã cấu hình sẵn cho tiếng Việt: babel + T5 fontenc).
3. Recompile.

**File:**
- `main.tex` — bản thảo (Springer Nature `sn-jnl`, tiếng Việt).
- `references.bib` — tài liệu tham khảo.
- `sn-jnl.cls`, `sn-mathphys-num.bst` — template Springer (đã kèm sẵn).
- **`figures/`** — 3 hình PDF (`fig_agreement`, `fig_qualitative`, `fig_doseresp`). **PHẢI upload cả thư mục
  này**, kẻo Overleaf báo `File 'fig_agreement.pdf' not found`. Dựng lại bằng
  `python ../make_figs.py --lang en --preds ... --occ ... --panel ...` (xem đầu file đó).

**Nếu dấu tiếng Việt lỗi:** đổi Compiler sang **XeLaTeX**, rồi trong `main.tex` thay
`inputenc`/`fontenc`/`babel` bằng `fontspec` + `polyglossia` (ghi chú sẵn ở đầu file).

**Còn thiếu (điền sau):** các ô `[ĐANG CHẠY]` — dòng PanNuke + giá trị $p$ ở bảng parity, và hệ số đáp ứng
3 hạt giống ở Mục 4.3.

**Nếu compile lỗi:** dán nguyên văn thông báo lỗi (dòng `! LaTeX Error: ...` trong log). Hai nguyên nhân
thường gặp: (1) thiếu thư mục `figures/`; (2) Overleaf giữ cache cũ sau khi sửa nhiều — bấm mũi tên cạnh
Recompile → **Recompile from scratch**.
