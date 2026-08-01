#!/usr/bin/env python3
"""fig1_build.py - sinh TOAN BO fig_design.drawio: hai luong song song, doi chieu tung buoc.

Trai = quy trinh MASK, phai = quy trinh COUNT-ONLY. Cung hang = cung buoc, nen doc ngang
la thay ngay khac nhau o dau. Anh deu la anh THAT (fig1_patches.py).
  python fig1_patches.py && python fig1_build.py
"""
import base64
from pathlib import Path

F = Path("paper2_rineng/figures")
b64 = lambda p: base64.b64encode((F / p).read_bytes()).decode()
MASK, PLAIN, DENS = b64("fig1_mask.png"), b64("fig1_plain.png"), b64("fig1_density.png")

ORG, ORGF = "#EB6834", "#FDF0EA"
BLU, BLUF = "#2A78D6", "#EAF2FC"
PUR, PURF = "#7B5EA7", "#EDE7F6"
GRN, GRNF = "#1BAF7A", "#EDF8F3"
INK = "#333333"
C = []


def box(i, v, x, y, w, h, stroke=INK, fill="#FFFFFF", fs=20, bold=0, dash=0, r=1, fc=INK):
    C.append(f'<mxCell id="{i}" value="{v}" style="rounded={r};arcSize=8;html=1;fillColor={fill};'
             f'strokeColor={stroke};strokeWidth=2;dashed={dash};fontSize={fs};fontStyle={bold};'
             f'fontColor={fc};align=center;verticalAlign=middle;whiteSpace=wrap;" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def txt(i, v, x, y, w, h, fs=20, bold=0, fc=INK, al="center"):
    C.append(f'<mxCell id="{i}" value="{v}" style="text;html=1;fontSize={fs};fontStyle={bold};'
             f'fontColor={fc};align={al};verticalAlign=middle;whiteSpace=wrap;" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def img(i, data, x, y, w, h, border):
    C.append(f'<mxCell id="{i}" value="" style="shape=image;html=1;imageAspect=0;imageBorder={border};'
             f'image=data:image/png,{data};" vertex="1" parent="1">'
             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(i, s, t, ex, ey, nx, ny, col=INK, lbl=""):
    C.append(f'<mxCell id="{i}" value="{lbl}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
             f'strokeWidth=2;strokeColor={col};fontSize=17;fontColor={col};labelBackgroundColor=#FFFFFF;'
             f'exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={nx};entryY={ny};entryDx=0;entryDy=0;" '
             f'edge="1" parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/>'
             f'</mxCell>')


# ---- anh dau vao dung chung ----
img("inimg", PLAIN, 655, 30, 150, 150, PUR)
txt("inlbl", "&lt;b&gt;the same H&amp;E patch&lt;/b&gt; (PanNuke fold 3), annotated two ways",
    480, 188, 500, 30, fs=20)

# ---- khung 2 luong ----
box("laneA", "", 40, 250, 640, 500, ORG, ORGF, dash=0)
box("laneB", "", 780, 250, 640, 500, BLU, BLUF, dash=0)
box("hA", "TRACK A &#183; PIXEL-LEVEL MASKS", 60, 268, 600, 46, ORG, "#FFFFFF", 23, 1, fc=ORG)
box("hB", "TRACK B &#183; ONE IMAGE-LEVEL COUNT", 800, 268, 600, 46, BLU, "#FFFFFF", 23, 1, fc=BLU)

# hang 1: chuyen gia lam gi
img("aAnn", MASK, 60, 330, 140, 140, ORG)
txt("aAnnT", "&lt;b&gt;1. what the expert does&lt;/b&gt;&#10;outline every one of the 22 nuclei"
             "&#10;&lt;b&gt;11.5 s&lt;/b&gt; of expert time per nucleus", 216, 330, 444, 140, al="left")
img("bAnn", PLAIN, 800, 330, 140, 140, BLU)
box("bNum", "22", 878, 408, 62, 62, "#FFFFFF", BLU, 30, 1, fc="#FFFFFF")
txt("bAnnT", "&lt;b&gt;1. what the expert does&lt;/b&gt;&#10;write down a single number"
             "&#10;at most &lt;b&gt;1.4 s&lt;/b&gt; per nucleus", 956, 330, 444, 140, al="left")
box("cheap", "&#8805; 8&#215;&#10;cheaper", 660, 365, 140, 70, GRN, "#FFFFFF", 21, 1, fc=GRN)

# hang 2: tin hieu huan luyen thu duoc
img("aTgt", DENS, 60, 490, 140, 140, ORG)
txt("aTgtT", "&lt;b&gt;2. training target&lt;/b&gt;&#10;a density map built from the masks"
             "&#10;it integrates to 22", 216, 490, 444, 140, al="left")
box("bTgt", "none", 800, 490, 140, 140, BLU, "#FFFFFF", 26, 1, dash=1, fc=BLU)   # tranh ky tu la
txt("bTgtT", "&lt;b&gt;2. training target&lt;/b&gt;&#10;&lt;b&gt;no pixel-level target at all&lt;/b&gt;"
             "&#10;only the number 22 is available", 956, 490, 444, 140, al="left")

# hang 3: bien duy nhat
box("aW", "&lt;b&gt;3. loss&lt;/b&gt;&#160;&#160; (w&lt;sub&gt;d&lt;/sub&gt;, w&lt;sub&gt;c&lt;/sub&gt;)"
          " = (1, 0.1)", 60, 650, 600, 76, ORG, "#FFFFFF", 22)
box("bW", "&lt;b&gt;3. loss&lt;/b&gt;&#160;&#160; (w&lt;sub&gt;d&lt;/sub&gt;, w&lt;sub&gt;c&lt;/sub&gt;)"
          " = (0, 1)", 800, 650, 600, 76, BLU, "#FFFFFF", 22)

# ---- dai dung chung ----
box("shared", "&lt;b&gt;IDENTICAL IN BOTH TRACKS&lt;/b&gt;&#10;EfficientNet-lite0 encoder + U-Net decoder "
              "&#183; 3.62M params &#183; same training data&#10;same 60-epoch schedule &#183; same test set "
              "&#183; same 5 seeds &#183; both predict a map whose sum is &#956;",
    40, 786, 1380, 116, PUR, PURF, 20, fc=PUR)

# ---- ket qua ----
box("rA", "R&#178; = &lt;b&gt;0.931 &#177; 0.006&lt;/b&gt;", 40, 942, 640, 84, ORG, ORGF, 26, fc=INK)
box("rB", "R&#178; = &lt;b&gt;0.918 &#177; 0.011&lt;/b&gt;", 780, 942, 640, 84, BLU, BLUF, 26, fc=INK)

box("vd", "", 40, 1066, 1380, 150, GRN, GRNF)
# dung &lt;sup&gt; chu KHONG dung ky tu so mu U+207B: phong hay thieu glyph -> ra o vuong
txt("vd1", "the gap is &lt;b&gt;REAL&lt;/b&gt;&#10;&#10;&#916;R&#178; = &#8722;0.013&#10;"
           "Wilcoxon p = 1.1&#215;10&lt;sup&gt;&#8722;13&lt;/sup&gt;", 70, 1076, 620, 130, fs=21)
txt("vd2", "and it is &lt;b&gt;BOUNDED&lt;/b&gt;&#10;&#10;TOST &#948;* = 0.27 nuclei&#10;"
           "= 1.1&#37; of the mean count", 770, 1076, 620, 130, fs=21)
C.append('<mxCell id="vsep" value="" style="html=1;fillColor=%s;strokeColor=%s;" vertex="1" parent="1">'
         '<mxGeometry x="729" y="1096" width="2" height="90" as="geometry"/></mxCell>' % (GRN, GRN))
txt("foot", "PanNuke fold 3 &#183; n = 2722 patches &#183; 5 seeds per arm", 40, 1230, 1380, 30, fs=19)

# ---- mui ten ----
edge("e1", "inimg", "laneA", 0, 0.5, 0.5, 0, PUR)
edge("e2", "inimg", "laneB", 1, 0.5, 0.5, 0, PUR)
edge("e3", "aW", "shared", 0.5, 1, 0.25, 0, ORG)
edge("e4", "bW", "shared", 0.5, 1, 0.75, 0, BLU)
edge("e5", "shared", "rA", 0.25, 1, 0.5, 0, ORG)
edge("e6", "shared", "rB", 0.75, 1, 0.5, 0, BLU)
edge("e7", "rA", "vd", 0.5, 1, 0.25, 0, ORG)
edge("e8", "rB", "vd", 0.5, 1, 0.75, 0, BLU)

xml = ('<mxfile host="app.diagrams.net"><diagram id="fig1" name="Figure 1">'
       '<mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
       'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1460" pageHeight="1290" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(C) +
       '</root></mxGraphModel></diagram></mxfile>')
assert "&#8212;" not in xml and "—" not in xml, "ban thao khong dung gach ngang dai"
(F / "fig_design.drawio").write_text(xml)
print(f"fig_design.drawio {len(xml)/1e6:.2f} MB | {len(C)} cell")
