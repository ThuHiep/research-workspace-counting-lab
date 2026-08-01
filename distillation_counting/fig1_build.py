#!/usr/bin/env python3
"""fig1_build.py - sinh fig_design.drawio: SO DO LUONG, hai bang doi chieu.

Cung mot duong di: anh -> mang -> ban do mat do -> tong -> mu.
Khac nhau duy nhat = TIN HIEU GIAM SAT BAM VAO DAU:
  A  ban do bi giam sat tung diem anh (w_d=1) VA so dem (w_c=0.1)
  B  nhanh ban do BI CAT (w_d=0), chi con so dem (w_c=1)
Cat hay khong cat la MUI TEN, khong phai cau chu.
  python fig1_patches.py && python fig1_build.py
"""
import base64
from pathlib import Path

F = Path("paper2_rineng/figures")
b64 = lambda p: base64.b64encode((F / p).read_bytes()).decode()
MASK, PLAIN, DENS = b64("fig1_mask.png"), b64("fig1_plain.png"), b64("fig1_density.png")

ORG, ORGF = "#EB6834", "#FDF0EA"
BLU, BLUF = "#2A78D6", "#EAF2FC"
PUR, GRN = "#7B5EA7", "#1BAF7A"
INK, MUT = "#333333", "#9A9A9A"
C = []


def box(i, v, x, y, w, h, stroke=INK, fill="#FFFFFF", fs=19, bold=0, dash=0, r=1, fc=INK, sh=""):
    C.append(f'<mxCell id="{i}" value="{v}" style="{sh or f"rounded={r};arcSize=8"};html=1;fillColor={fill};'
             f'strokeColor={stroke};strokeWidth=2;dashed={dash};fontSize={fs};fontStyle={bold};'
             f'fontColor={fc};align=center;verticalAlign=middle;whiteSpace=wrap;" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def txt(i, v, x, y, w, h, fs=17, bold=0, fc=INK):
    C.append(f'<mxCell id="{i}" value="{v}" style="text;html=1;fontSize={fs};fontStyle={bold};'
             f'fontColor={fc};align=center;verticalAlign=middle;whiteSpace=wrap;" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def img(i, data, x, y, w, h, border):
    C.append(f'<mxCell id="{i}" value="" style="shape=image;html=1;imageAspect=0;imageBorder={border};'
             f'image=data:image/png,{data};" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(i, s, t, ex, ey, nx, ny, col=INK, lbl="", dash=0, w=2):
    C.append(f'<mxCell id="{i}" value="{lbl}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
             f'strokeWidth={w};strokeColor={col};dashed={dash};fontSize=17;fontStyle=1;fontColor={col};'
             f'labelBackgroundColor=#FFFFFF;exitX={ex};exitY={ey};exitDx=0;exitDy=0;'
             f'entryX={nx};entryY={ny};entryDx=0;entryDy=0;" edge="1" parent="1" '
             f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/></mxCell>')


def panel(g, X, col, colf, title, supervised):
    """Mot bang. supervised=True -> nhanh ban do CON; False -> BI CAT."""
    box(f"{g}P", "", X, 60, 660, 500, col, colf)
    box(f"{g}T", title, X + 20, 78, 620, 44, col, "#FFFFFF", 22, 1, fc=col)

    # duong di xuoi, GIONG HET o ca hai bang
    img(f"{g}img", PLAIN, X + 26, 190, 84, 84, col)
    for k, (dx, h) in enumerate(zip((150, 178, 206, 234, 262), (84, 62, 44, 62, 84))):
        box(f"{g}e{k}", "", X + dx, 232 - h // 2, 22, h, PUR, "#D8CBEA", r=0)
    box(f"{g}map", "", X + 320, 190, 84, 84, col, "#FFFFFF")
    box(f"{g}sig", "&#8721;", X + 432, 210, 44, 44, INK, "#FFFFFF", 22, 1, sh="ellipse")
    box(f"{g}mu", "&#956;", X + 506, 208, 124, 48, INK, "#FFFFFF", 22, 1)

    txt(f"{g}Li", "input", X + 18, 280, 100, 22, fc=MUT)
    txt(f"{g}Ln", "EfficientNet-lite0 + U-Net", X + 122, 280, 220, 22, fc=MUT)
    txt(f"{g}Lm", "density map", X + 316, 162, 92, 22, fc=MUT)
    txt(f"{g}Lu", "predicted count", X + 496, 176, 144, 22, fc=MUT)

    # nguon giam sat nam DUOI, ban NGUOC len duong di
    if supervised:
        # chuoi chu thich: chuyen gia khoanh vien -> ban do muc tieu. Chi phi 11.5 s nam o khau khoanh.
        img(f"{g}ann", MASK, X + 170, 400, 84, 84, col)
        txt(f"{g}La", "expert outlines 22 nuclei&#10;&lt;b&gt;11.5 s&lt;/b&gt; per nucleus",
            X + 118, 492, 188, 44)
        img(f"{g}tgt", DENS, X + 320, 400, 84, 84, col)
        txt(f"{g}Lt", "target density map", X + 310, 492, 104, 22)
        edge(f"{g}a0", f"{g}ann", f"{g}tgt", 1, 0.5, 0, 0.5, col)
    else:
        box(f"{g}tgt", "no target", X + 320, 400, 84, 84, MUT, "#FFFFFF", 17, 0, dash=1, fc=MUT)
        txt(f"{g}Lt", "nothing to compare&#10;the map against", X + 268, 492, 188, 44, fc=MUT)
    box(f"{g}N", "N = 22", X + 506, 418, 124, 48, col, "#FFFFFF", 20, 1)
    txt(f"{g}Ld", "one number per image&#10;&lt;b&gt;&#8804; 1.4 s&lt;/b&gt; per nucleus",
        X + 496, 492, 144, 44)

    edge(f"{g}f0", f"{g}img", f"{g}e0", 1, 0.5, 0, 0.5, INK)
    edge(f"{g}f1", f"{g}e4", f"{g}map", 1, 0.5, 0, 0.5, INK)
    edge(f"{g}f2", f"{g}map", f"{g}sig", 1, 0.5, 0, 0.5, INK)
    edge(f"{g}f3", f"{g}sig", f"{g}mu", 1, 0.5, 0, 0.5, INK)
    # hai mui ten giam sat: DAY moi la cho hai bang khac nhau
    if supervised:
        edge(f"{g}s0", f"{g}tgt", f"{g}map", 0.5, 0, 0.5, 1, col, "w&lt;sub&gt;d&lt;/sub&gt; = 1", w=3)
    else:
        edge(f"{g}s0", f"{g}tgt", f"{g}map", 0.5, 0, 0.5, 1, MUT, "w&lt;sub&gt;d&lt;/sub&gt; = 0", dash=1)
        box(f"{g}cut", "&#215;", X + 343, 284, 38, 38, MUT, "#FFFFFF", 24, 1, fc=MUT, sh="ellipse")
    edge(f"{g}s1", f"{g}N", f"{g}mu", 0.5, 0, 0.5, 1, col,
         "w&lt;sub&gt;c&lt;/sub&gt; = " + ("0.1" if supervised else "1"), w=3)


panel("a", 20, ORG, ORGF, "A &#183; TRAINED ON PIXEL MASKS", True)
panel("b", 784, BLU, BLUF, "B &#183; TRAINED ON ONE COUNT", False)
box("cheap", "&#8805; 8&#215;&#10;cheaper&#10;to label", 690, 400, 88, 84, GRN, "#FFFFFF", 16, 1, fc=GRN)
txt("note", "both panels: identical architecture (3.62M), training data, schedule, test set and seeds",
    20, 578, 1424, 26, fs=18, fc=PUR)

xml = ('<mxfile host="app.diagrams.net"><diagram id="fig1" name="Figure 1">'
       '<mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
       'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1470" pageHeight="620" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(C) +
       '</root></mxGraphModel></diagram></mxfile>')
assert "&#8212;" not in xml and "—" not in xml, "ban thao khong dung gach ngang dai"
(F / "fig_design.drawio").write_text(xml)
print(f"fig_design.drawio {len(xml)/1e6:.2f} MB | {len(C)} cell")
