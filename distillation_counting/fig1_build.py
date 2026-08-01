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


# cot dung chung cho CA HAI hang -> mat doc doc xuong la so sanh duoc
XI, XA, XN, XM, XS, XU = 170, 430, 370, 600, 770, 870
S = 110


def row(g, Y, hdr, mask_sup):
    txt(f"{g}h", hdr, 20, Y, 460, 28, fs=19, bold=1, al="left")

    # --- duong di xuoi: VE Y HET nhau o ca hai hang, mau trung tinh ---
    img(f"{g}im", PLAIN, XI, Y + 44, S, S)
    for k, (dx, h) in enumerate(zip((0, 32, 64, 96, 128), (S, 82, 60, 82, S))):
        box(f"{g}e{k}", "", XN + dx, Y + 99 - h // 2, 26, h, MUT, NET)
    box(f"{g}pm", "", XM, Y + 44, S, S, INK)
    box(f"{g}sg", "&#8721;", XS, Y + 76, 46, 46, INK, "#FFFFFF", 19, 1, sh="ellipse")
    box(f"{g}mu", "&#956;", XU, Y + 72, 140, 54, INK, "#FFFFFF", 19, 1)

    txt(f"{g}li", "input patch", XI - 15, Y + 160, 140, 22, fc=MUT)
    txt(f"{g}ln", "encoder + decoder", XN - 20, Y + 160, 200, 22, fc=MUT)
    txt(f"{g}lm", "predicted density map", XM - 45, Y + 16, 200, 22, fc=MUT)
    txt(f"{g}lu", "predicted count", XU, Y + 16, 140, 22, fc=MUT)

    edge(f"{g}f0", f"{g}im", f"{g}e0", 1, .5, 0, .5)
    edge(f"{g}f1", f"{g}e4", f"{g}pm", 1, .5, 0, .5)
    edge(f"{g}f2", f"{g}pm", f"{g}sg", 1, .5, 0, .5)
    edge(f"{g}f3", f"{g}sg", f"{g}mu", 1, .5, 0, .5)

    # --- nguon giam sat: DAY moi la cho hai hang khac nhau, va CHI cho nay co mau ---
    col = ORG if mask_sup else BLU
    if mask_sup:
        img(f"{g}an", MASK, XA, Y + 215, S, S, ORG)
        txt(f"{g}la", "expert outlines every nucleus&#10;&lt;b&gt;11.5 s&lt;/b&gt; per nucleus",
            XA - 40, Y + 332, 190, 40)
        img(f"{g}tg", DENS, XM, Y + 215, S, S, ORG)
        txt(f"{g}lt", "target density map", XM - 40, Y + 332, 190, 22)
        edge(f"{g}a0", f"{g}an", f"{g}tg", 1, .5, 0, .5, ORG)
        edge(f"{g}s0", f"{g}tg", f"{g}pm", .5, 0, .5, 1, ORG, "w&lt;sub&gt;d&lt;/sub&gt; = 1", sw=2)
    else:
        box(f"{g}tg", "no pixel-level target", XM, Y + 215, S, S, MUT, "#FFFFFF", 15, 0, dash=1, fc=MUT)
        edge(f"{g}s0", f"{g}tg", f"{g}pm", .5, 0, .5, 1, MUT, "w&lt;sub&gt;d&lt;/sub&gt; = 0", dash=1)
    box(f"{g}N", "N = 22", XU, Y + 243, 140, 54, col, "#FFFFFF", 18, 1)
    txt(f"{g}l2", ("image count, also used" if mask_sup else
                   "image count, &lt;b&gt;the only label&lt;/b&gt;&#10;&#8804; &lt;b&gt;1.4 s&lt;/b&gt; per nucleus"),
        XU - 25, Y + 332, 190, 40)
    edge(f"{g}s1", f"{g}N", f"{g}mu", .5, 0, .5, 1, col,
         "w&lt;sub&gt;c&lt;/sub&gt; = " + ("0.1" if mask_sup else "1"), sw=2)


row("a", 30, "(a)&#160;&#160;mask supervision", True)
row("b", 440, "(b)&#160;&#160;count-only supervision", False)
txt("note", "grey elements are identical in both rows: same architecture (3.62M parameters), same training "
            "data, 60-epoch schedule, test set and 5 seeds", 20, 836, 1040, 22, fs=15, fc=MUT, al="left")

xml = ('<mxfile host="app.diagrams.net"><diagram id="fig1" name="Figure 1">'
       '<mxGraphModel dx="1200" dy="900" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
       'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1080" pageHeight="880" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(C) +
       '</root></mxGraphModel></diagram></mxfile>')
assert "&#8212;" not in xml and "—" not in xml, "ban thao khong dung gach ngang dai"
(F / "fig_design.drawio").write_text(xml)
print(f"fig_design.drawio {len(xml)/1e6:.2f} MB | {len(C)} cell")
