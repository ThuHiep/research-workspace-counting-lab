#!/usr/bin/env python3
"""fig1_embed.py - nhung anh THAT vao fig_design.drawio (Hinh 1).

Sua 3 cho lam so do bi chung chung:
  mkim  hinh chu nhat den + 14 ellipse gia  -> anh THAT phu 22 duong vien nhan
  ctim  o trang chi co chu "22"             -> DUNG anh do, khong ve gi, + the "22"
  he    mot patch KHAC                      -> dung LUON anh do
Ba o cung mot anh = "mot anh, hai kieu nhan" moi doc ra duoc.
Chay lai duoc nhieu lan (idempotent).
  python fig1_embed.py
"""
import base64, re, sys
from pathlib import Path

F = Path("paper2_rineng/figures")
DW = F / "fig_design.drawio"
b64 = lambda p: base64.b64encode((F / p).read_bytes()).decode()
MASK, PLAIN, DENS = b64("fig1_mask.png"), b64("fig1_plain.png"), b64("fig1_density.png")
IMST = "shape=image;html=1;imageAspect=0;imageBorder={};image=data:image/png,{};"
IMG_CELL = ('<mxCell id="{}" value="" style="' + IMST + '" vertex="1" parent="1">'
            '<mxGeometry x="{}" y="{}" width="{}" height="{}" as="geometry"/></mxCell>')

t = DW.read_text()

# 14 ellipse gia -> bo han
t, n = re.subn(r'<mxCell id="nu\d+".*?</mxCell>\s*', "", t, flags=re.S)

def put(t, cid, style, geo):
    new = f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">' \
          f'<mxGeometry {geo} as="geometry"/></mxCell>'
    t, k = re.subn(r'<mxCell id="%s".*?</mxCell>' % cid, lambda _: new, t, flags=re.S)
    assert k == 1, f"khong thay cell {cid}"
    return t

t = put(t, "mkim", IMST.format("#EB6834", MASK), 'x="150" y="126" width="160" height="160"')
t = put(t, "ctim", IMST.format("#2A78D6", PLAIN), 'x="880" y="126" width="160" height="160"')
t = put(t, "he",   IMST.format("#7B5EA7", PLAIN), 'x="110" y="392" width="150" height="150"')

# chu thich canh lai theo anh cao 160
for cid in ("mkc", "ctc"):
    t = re.sub(r'(<mxCell id="%s".*?<mxGeometry x="\d+" )y="126" width="218" height="120"' % cid,
               r'\1y="126" width="218" height="160"', t, flags=re.S)

# the "22" = TOAN BO nhan cua anh ben phai
if 'id="ctnum"' not in t:
    badge = ('<mxCell id="ctnum" value="22" style="rounded=1;arcSize=20;html=1;fillColor=#2A78D6;'
             'strokeColor=#FFFFFF;strokeWidth=3;fontSize=44;fontStyle=1;fontColor=#FFFFFF;" '
             'vertex="1" parent="1"><mxGeometry x="978" y="222" width="86" height="64" as="geometry"/></mxCell>')
    t = t.replace('<mxCell id="ctc"', badge + '<mxCell id="ctc"', 1)

# --- dai dau ra: cho thay w_d=0 KHONG giam sat ban do, chi rang buoc TONG ---
D = 110
if 'id="hd"' not in t:
    # day MOI cell nam duoi sh, khong liet ke tay -> khong the sot cell nao
    t = re.sub(r'(<mxGeometry x="\d+" y=")(\d+)"',
               lambda m: m.group(1) + (str(int(m.group(2)) + D) if int(m.group(2)) >= 600 else m.group(2)) + '"', t)
    t = re.sub(r'(<mxCell id="sh".*?height=")182', r'\g<1>292', t, flags=re.S)
    t = re.sub(r'(<mxCell id="shb".*?height=")110', r'\g<1>60', t, flags=re.S)
    box = ('<mxCell id="{}" value="{}" style="rounded=1;html=1;fillColor=#FFFFFF;strokeColor=#7B5EA7;'
           'strokeWidth=2;fontSize=21;fontColor=#333333;align=center;verticalAlign=middle;" '
           'vertex="1" parent="1"><mxGeometry x="{}" y="{}" width="{}" height="{}" as="geometry"/></mxCell>')
    # 2 dong chu dai -> 1 dong ngan (khoi tim truoc day toan chu)
    t = re.sub(r'(<mxCell id="shb" value=")[^"]*"',
               r'\g<1>EfficientNet-lite0 encoder + U-Net decoder &#183; 3.62M params &#183; '
               r'same data, schedule, test set and 5 seeds"', t)
    strip = (
        # ANH THAT thay cho chu "DENSITY MAP": ban do muc tieu, tong = 22
        IMG_CELL.format("hdimg", "#7B5EA7", DENS, 350, 516, 120, 120)
        + box.format("hd", "w&lt;sub&gt;d&lt;/sub&gt; = 0 &#8594;&#10;map &lt;b&gt;never&lt;/b&gt; "
                           "supervised&#10;pixel by pixel", 484, 516, 186, 120)
        + box.format("hmu", "&#956; = &lt;b&gt;sum&lt;/b&gt; of the map = 22", 690, 516, 380, 52)
        + box.format("hsg", "&#963; = &#8730;max(&#956;,1) &#215; exp(log s)", 690, 584, 380, 52))
    edge = ('<mxCell id="{}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=2;'
            'strokeColor=#7B5EA7;exitX={};exitY={};exitDx=0;exitDy=0;entryX={};entryY={};entryDx=0;entryDy=0;" '
            'edge="1" parent="1" source="{}" target="{}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    strip += edge.format("e_d1", 1, 0.5, 0, 0.5, "hd", "hmu") + edge.format("e_d2", 0.5, 1, 0.5, 0, "hmu", "hsg")
    t = t.replace('<mxCell id="sw"', strip + '<mxCell id="sw"', 1)

# ban thao khong dung gach ngang dai o BAT KY dau -> hinh phai theo
t = t.replace("same network &#8212; 3.62M", "same network, 3.62M").replace("&#8212;", "&#183;")
assert "&#8212;" not in t and "—" not in t

DW.write_text(t)
print(f"bo {n} ellipse gia | drawio {DW.stat().st_size/1e6:.2f} MB")
