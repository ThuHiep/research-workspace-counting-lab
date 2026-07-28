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

**Nếu dấu tiếng Việt lỗi:** đổi Compiler sang **XeLaTeX**, rồi trong `main.tex` thay
`inputenc`/`fontenc`/`babel` bằng `fontspec` + `polyglossia` (ghi chú sẵn ở đầu file).

**Còn thiếu (điền sau):** Bảng in-domain — ô **MoNuSeg** đang là `[ĐANG CHẠY]`, thay bằng số khi có kết quả.
