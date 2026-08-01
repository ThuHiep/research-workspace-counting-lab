#!/usr/bin/env python3
"""fig1_build.py - sinh fig_design.drawio: hai hang chong doc, doi chieu theo COT.

Theo quy uoc hinh cua bai bao (Qu 2020 arXiv:2007.05448; TransCrowd arXiv:2104.09116):
hop chu nhat + mui ten co huong, nhan (a)/(b) chu thuong dam o goc trai, chu thich mang
phan giai thich chu khong nhoi chu vao hinh.

Nguyen tac mau: phan DUNG CHUNG ve trung tinh (xam/den) va giong het o ca hai hang;
mau CHI danh dau cho khac biet = mui ten giam sat. Mau khong dung de trang tri.
  python fig1_patches.py && python fig1_build.py
"""
import base64
from pathlib import Path

F = Path("paper2_rineng/figures")
b64 = lambda p: base64.b64encode((F / p).read_bytes()).decode()
MASK, PLAIN, DENS = b64("fig1_mask.png"), b64("fig1_plain.png"), b64("fig1_density.png")

INK, MUT, NET = "#222222", "#8A8A8A", "#E4E4E4"
ORG, BLU = "#C4561F", "#1F5FA8"          # dam hon ban truoc -> in den trang van doc duoc
C = []


def box(i, v, x, y, w, h, stroke=INK, fill="#FFFFFF", fs=17, bold=0, dash=0, fc=INK, sh="", sw=1):
    C.append(f'<mxCell id="{i}" value="{v}" style="{sh or "rounded=0"};html=1;fillColor={fill};'
             f'strokeColor={stroke};strokeWidth={sw};dashed={dash};fontSize={fs};fontStyle={bold};'
             f'fontColor={fc};align=center;verticalAlign=middle;whiteSpace=wrap;" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def txt(i, v, x, y, w, h, fs=15, bold=0, fc=INK, al="center"):
    C.append(f'<mxCell id="{i}" value="{v}" style="text;html=1;fontSize={fs};fontStyle={bold};'
             f'fontColor={fc};align={al};verticalAlign=middle;whiteSpace=wrap;" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def img(i, data, x, y, w, h, border=MUT):
    C.append(f'<mxCell id="{i}" value="" style="shape=image;html=1;imageAspect=0;imageBorder={border};'
             f'image=data:image/png,{data};" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(i, s, t, ex, ey, nx, ny, col=INK, lbl="", dash=0, sw=1):
    C.append(f'<mxCell id="{i}" value="{lbl}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
             f'strokeWidth={sw};strokeColor={col};dashed={dash};fontSize=15;fontStyle=1;fontColor={col};'
             f'labelBackgroundColor=#FFFFFF;exitX={ex};exitY={ey};exitDx=0;exitDy=0;'
             f'entryX={nx};entryY={ny};entryDx=0;entryDy=0;" edge="1" parent="1" '
             f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/></mxCell>')


# cot dung chung cho CA HAI hang. Duong di xuoi ve NHO va NHAT (giong het nhau, khong phai
# cho khac biet); vung giam sat ve TO va DAM (day moi la cho khac biet).
XI, XN, XM, XS, XU, XA = 90, 230, 440, 590, 690, 250
S, SM = 84, 84


def row(g, Y, hdr, mask_sup):
    txt(f"{g}h", hdr, 20, Y, 460, 28, fs=19, bold=1, al="left")

    # --- duong di xuoi: nho, nhat, giong het o ca hai hang ---
    img(f"{g}im", PLAIN, XI, Y + 46, S, S)
    for k, (dx, h) in enumerate(zip((0, 26, 52, 78, 104), (S, 64, 46, 64, S))):
        box(f"{g}e{k}", "", XN + dx, Y + 88 - h // 2, 20, h, MUT, NET)
    box(f"{g}pm", "", XM, Y + 46, SM, SM, MUT)
    for j in range(1, 4):
        box(f"{g}gv{j}", "", XM + j * SM // 4, Y + 46, 1, SM, "#E4E4E4", "#E4E4E4")
        box(f"{g}gh{j}", "", XM, Y + 46 + j * SM // 4, SM, 1, "#E4E4E4", "#E4E4E4")
    box(f"{g}sg", "&#8721;", XS, Y + 68, 40, 40, MUT, "#FFFFFF", 17, 1, sh="ellipse")
    box(f"{g}mu", "&#956;", XU, Y + 65, 110, 46, MUT, "#FFFFFF", 18, 1)
    # nhan cua map/count phai nam TREN: duong giam sat di xuong ngay duoi hai o do
    txt(f"{g}c0", "input patch", XI - 18, Y + 138, 120, 20, fs=14, fc=MUT)
    txt(f"{g}c1", "encoder + decoder", XN - 20, Y + 138, 190, 20, fs=14, fc=MUT)
    txt(f"{g}c2", "density map 256&#215;256", XM - 48, Y + 20, 180, 20, fs=14, fc=MUT)
    txt(f"{g}c3", "predicted count", XU - 20, Y + 20, 150, 20, fs=14, fc=MUT)
    for a, b in (("im", "e0"), ("e4", "pm"), ("pm", "sg"), ("sg", "mu")):
        edge(f"{g}f{a}{b}", f"{g}{a}", f"{g}{b}", 1, .5, 0, .5, MUT)

    # --- vung giam sat: TO, DAM. Hang (b) ve MO HAN = nhanh vang mat ---
    col = ORG if mask_sup else BLU
    off = MUT if mask_sup else "#D0D0D0"          # mau nhanh density o hang (b): mo han
    dc = ORG if mask_sup else "#D0D0D0"
    d = 0 if mask_sup else 1
    box(f"{g}Ld", "L&lt;sub&gt;density&lt;/sub&gt;&#10;&lt;b&gt;" +
        ("compares 65,536 values" if mask_sup else "switched off, w&lt;sub&gt;d&lt;/sub&gt; = 0") + "&lt;/b&gt;",
        XM - 55, Y + 178, 194, 56, dc, "#FFFFFF", 15, 0, d, fc=dc)
    box(f"{g}Lc", "L&lt;sub&gt;count&lt;/sub&gt;&#10;&lt;b&gt;compares 1 value&lt;/b&gt;",
        XU - 42, Y + 178, 194, 56, col, "#FFFFFF", 15, 0, 0, fc=col)
    edge(f"{g}p1", f"{g}pm", f"{g}Ld", .5, 1, .5, 0, dc, "", d, 3 if mask_sup else 1)
    edge(f"{g}p2", f"{g}mu", f"{g}Lc", .5, 1, .5, 0, col, "", 0, 3)

    if mask_sup:
        img(f"{g}an", MASK, XA, Y + 252, S, S, ORG)
        edge(f"{g}a0", f"{g}an", f"{g}tg", 1, .5, 0, .5, ORG)
        img(f"{g}tg", DENS, XM, Y + 252, S, S, ORG)
        txt(f"{g}lt", "expert outlines every nucleus&#160;&#8594;&#160;target map&#10;"
                      "&lt;b&gt;11.5 s&lt;/b&gt; per nucleus", XA - 60, Y + 344, 320, 40)
    else:
        box(f"{g}tg", "no target", XM, Y + 252, S, S, "#D0D0D0", "#FFFFFF", 14, 0, 1, fc="#C0C0C0")
    edge(f"{g}s0", f"{g}tg", f"{g}Ld", .5, 0, .5, 1, dc, "", d, 3 if mask_sup else 1)
    box(f"{g}N", "N = 22", XU, Y + 256, 110, 46, col, "#FFFFFF", 17, 1)
    txt(f"{g}l2", "one number per image&#10;&lt;b&gt;&#8804; 1.4 s&lt;/b&gt; per nucleus",
        XU - 45, Y + 344, 200, 40)
    edge(f"{g}s1", f"{g}N", f"{g}Lc", .5, 0, .5, 1, col, "", 0, 3)


row("a", 30, "(a)&#160;&#160;mask supervision", True)
row("b", 460, "(b)&#160;&#160;count-only supervision", False)
txt("note", "the forward path (grey) is identical in both rows: same architecture (3.62M parameters), same "
            "training data, schedule, test set and seeds", 20, 880, 900, 22, fs=15, fc=MUT, al="left")

xml = ('<mxfile host="app.diagrams.net"><diagram id="fig1" name="Figure 1">'
       '<mxGraphModel dx="1200" dy="900" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
       'arrows="1" fold="1" page="1" pageScale="1" pageWidth="940" pageHeight="925" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(C) +
       '</root></mxGraphModel></diagram></mxfile>')
assert "&#8212;" not in xml and "\u2014" not in xml
(F / "fig_design.drawio").write_text(xml)
print(f"fig_design.drawio {len(xml)/1e6:.2f} MB | {len(C)} cell")
