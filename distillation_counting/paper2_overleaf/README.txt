HƯỚNG DẪN BIÊN DỊCH TRÊN OVERLEAF
==================================

File: main.tex  (định dạng Springer LNCS / LNICST — dùng cho EAI AICON)

CÁCH 1 (khuyến nghị — để có sẵn llncs.cls):
  1. Vào Overleaf → New Project → Templates → tìm "Springer LNCS" (hoặc
     "Lecture Notes in Computer Science") → Open as Template.
     (Template này đi kèm file lớp llncs.cls + splncs04.bst.)
  2. Trong project đó, mở main.tex có sẵn, XÓA hết nội dung, DÁN toàn bộ
     nội dung main.tex này vào.
  3. Menu → Compiler: để "pdfLaTeX". Nhấn Recompile.

CÁCH 2 (upload trực tiếp):
  - Upload main.tex này lên project mới. Nếu báo lỗi "llncs.cls not found":
    tải llncs.cls từ trang Springer (bộ "LNCS LaTeX package") rồi upload cùng.

TIẾNG VIỆT:
  - Mặc định file dùng pdfLaTeX + fontenc[T5] + babel[vietnamese] → hiển thị
    dấu tiếng Việt tốt trong đa số trường hợp.
  - NẾU dấu bị lỗi/thiếu: đổi Compiler sang "XeLaTeX", rồi trong main.tex:
      * XÓA 3 dòng: \usepackage[T5]{fontenc}, \usepackage[utf8]{inputenc},
        \usepackage[vietnamese]{babel}
      * BỎ ghi chú (xóa dấu %) 3 dòng: fontspec, polyglossia, setmainlanguage.

CÒN THIẾU (điền khi có):
  - Tên tác giả + đơn vị + email (khối \author, \institute).
  - Tiêu đề: hiện đặt tạm, có thể rút gọn.
  - Bảng 4 (MoNuSeg): ô R^2 đang là [ĐANG CHẠY] — thay bằng số khi có kết quả.
  - Danh mục tài liệu: đang ở dạng thebibliography thủ công (không cần chạy
    BibTeX). Khi nộp chính thức nên chuyển sang .bib + splncs04.bst.
