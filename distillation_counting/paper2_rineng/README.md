# Results in Engineering (Elsevier) - Special Issue 335088

- `main.tex`    - BAN NOP (tieng Anh), pdfLaTeX
- `main_vi.tex` - ban tieng Viet DE DOC, khong nop. Moi so lieu/nhan/bang/hinh giu y nguyen -> doi chieu duoc.

Upload len Overleaf: ca hai .tex + `references.bib` + `figures/`.
`elsarticle.cls` va `elsarticle-num.bst` co san tren Overleaf, khong can kem theo.
Doi file chinh trong Menu > Main document de build ban con lai.
Neu dau tieng Viet loi: doi main_vi.tex sang XeLaTeX (huong dan o dau file).

## Hinh 1 - PHAI XUAT TRUOC KHI BUILD

`figures/fig_design.drawio` la NGUON. Bai dang \includegraphics{fig_design.pdf}, file PDF do CHUA co.
Mo drawio > File > Export as > PDF, bo chon "Crop", luu de len `figures/fig_design.pdf`.

Sinh lai anh nhung trong so do (neu doi patch):
    python ../fig1_patches.py && python ../fig1_embed.py
`fig1_embed.py` idempotent (da kiem: chay 3 lan ra cung md5), chay lai bao nhieu lan cung duoc.
NHUNG neu ban tu sua bo cuc trong drawio thi dung chay lai - no se ghi de cac o mkim/ctim/he.

Ban Springer sn-jnl GIU nguyen o `../paper2_overleaf/` - dung neu RINENG desk-reject.
