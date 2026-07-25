#!/usr/bin/env python3
"""haam_module.py — HAAM (của user) ADAPT cho density counting = D-HAAM.

Giữ LÕI ý tưởng gốc của user: chạy CA + SA SONG SONG, trộn bằng CỔNG α HỌC ĐƯỢC (M = α·CA + (1-α)·SA),
+ residual. Sửa 3 chỗ cho hợp bài ĐẾM (density = Σ, không phải classification):

  (1) BỎ `M / M.max()`  — gốc lấy max TOÀN tensor (kể cả batch) -> ghép các ảnh + đổi scale density
      theo activation lớn nhất -> HỎNG magnitude (đếm phụ thuộc độ lớn density) + bất ổn. Residual + sigmoid
      đã đủ bound. Với counting KHÔNG được rescale feature bằng thống kê toàn cục.
  (2) fusion_mlp 2C->1 (gốc) -> 2C->C/r->1  — bớt nghẽn thông tin.
  (3) SA nâng thành ĐA-TỈ-LỆ (dilated 1/2/4)  — nhân đổi mật độ/kích thước (saturation vùng dày, đo được);
      SA đa-scale cho module "chú ý đúng tỉ lệ" theo vùng -> nhắm đúng failure. (ms=False = SA gốc 1 scale.)

Giữ SiLU (lựa chọn của user) + residual M+x (không phá tín hiệu density).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hid = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hid, 1, bias=False)
        self.fc2 = nn.Conv2d(hid, channels, 1, bias=False)

    def forward(self, x):
        a = self.fc1(F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1))
        a = self.fc2(F.silu(a))
        return torch.sigmoid(a) * x


class SpatialAttention(nn.Module):
    """SA gốc (1 scale)."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def _stat(self, x):
        return torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)

    def forward(self, x):
        return torch.sigmoid(self.conv(self._stat(x))) * x


class SpatialAttentionMS(nn.Module):
    """SA ĐA-TỈ-LỆ: dilated 1/2/4 -> scale/density aware (nhắm saturation vùng dày)."""
    def __init__(self, kernel_size=3, dilations=(1, 2, 4)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size, padding=d * (kernel_size // 2), dilation=d) for d in dilations])

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        a = sum(c(s) for c in self.convs) / len(self.convs)
        return torch.sigmoid(a) * x


class CBAM(nn.Module):
    """CBAM chuẩn (bản của user): CAM (Linear) -> SAM (k=7) nối tiếp + residual. Làm mốc so với D-HAAM."""
    def __init__(self, channels, r=16):
        super().__init__()
        hid = max(channels // r, 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hid), nn.ReLU(inplace=True), nn.Linear(hid, channels))
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def _cam(self, x):
        b, c, _, _ = x.size()
        mx = self.mlp(F.adaptive_max_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        av = self.mlp(F.adaptive_avg_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        return torch.sigmoid(mx + av) * x

    def _sam(self, x):
        s = torch.cat([x.max(1, keepdim=True).values, x.mean(1, keepdim=True)], 1)
        return torch.sigmoid(self.conv(s)) * x

    def forward(self, x):
        return self._sam(self._cam(x)) + x


class DHAAM(nn.Module):
    """Density-HAAM: CA + SA(song song) trộn cổng α học được + residual. ms=True -> SA đa-tỉ-lệ."""
    def __init__(self, channels, ms=True, reduction=8):
        super().__init__()
        hid = max(channels // reduction, 4)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttentionMS() if ms else SpatialAttention()
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, hid, 1), nn.SiLU(),
            nn.Conv2d(hid, 1, 1), nn.Sigmoid())

    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(x)
        alpha = self.fusion(torch.cat([ca, sa], dim=1))     # (B,1,H,W) cổng per-pixel
        M = alpha * ca + (1 - alpha) * sa
        return M + x                                         # residual, KHÔNG M.max
