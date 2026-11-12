#!/usr/bin/env python3
"""'상윤과 승민의 크리스마스' — 20초 16비트풍 픽셀 아트 애니메이션 렌더러.

216x384 저해상도 캔버스에 모든 것을 코드로 그리고(도트), 15비트 컬러 + 오더드 디더링을
거쳐 정수배(5x) 확대로 1080x1920 세로 영상을 만든다.

사용법:
  python3 make_video.py --preview 2,6,9      # 해당 초의 프레임을 PNG 시트로 저장
  python3 make_video.py --out video.mp4 --audio music.wav
"""
import argparse
import math
import os
import subprocess
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 216, 384
SCALE = 5
FPS = 30
DUR = 20.0
NF = int(FPS * DUR)
GY = 316                 # 지면(눈밭) 높이
WW = 484                 # 월드 가로 길이
TREE_CX = 328
FEET_Y = 342
HERE = os.path.dirname(os.path.abspath(__file__))

SEUNGMIN = 'girl'        # 트리를 발견하는 '승민' 캐릭터 ('girl' 또는 'boy')
SANGYOON = 'boy' if SEUNGMIN == 'girl' else 'girl'

# 타임라인 (음악 96BPM: 1박 = 0.625초)
T_STOP = 8.75            # 승민이 트리 발견 (14박)
T_GO = 9.6               # 트리로 달려감
T_LIGHT0 = 10.9375       # 전구 점등 시작 (17.5박)
T_STAR = 12.5            # 별 점등 (20박)
T_ARRIVE = 12.4
T_WALK0 = 3.6            # 걷기 시작 기준 시각
T_FRONT = 15.0           # 돌아서서 정면
T_TITLE1 = 15.3125       # 제목 1행 (24.5박)
T_TITLE2 = 15.9375       # 제목 2행 (25.5박)


# ───────────────────────── 기본 유틸 ─────────────────────────
def C(h, a=255):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def F3(h):
    c = C(h)
    return np.array(c[:3], np.float32)


def hsh(x, y=0, s=0):
    n = (int(x) * 374761393 + int(y) * 668265263 + int(s) * 2147483647) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n ^= n >> 16
    return (n & 0xFFFF) / 65536.0


def clamp(v, a=0.0, b=1.0):
    return a if v < a else b if v > b else v


def smooth(v):
    v = clamp(v)
    return v * v * (3 - 2 * v)


def ease_out_back(p):
    p = clamp(p)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
BAYER_FULL = np.tile(BAYER, (H // 4, W // 4))


def new(w, h):
    return Image.new('RGBA', (w, h), (0, 0, 0, 0))


def npa(img):
    return np.array(img).astype(np.float32)


def blit(dst, src, x, y, mask_out=None):
    """알파 합성. dst: HxWx3 float, src: hxwx4 float."""
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4] / 255.0
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a
    if mask_out is not None:
        mask_out[y0:y1, x0:x1] = np.maximum(mask_out[y0:y1, x0:x1], a[..., 0])


def glow(L, cx, cy, r, col, k, ry=None):
    """빛 버퍼 L(HxWx3)에 부드러운 원형/타원형 빛을 더한다."""
    ry = ry or r
    x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, L.shape[1]))
    y0, y1 = int(max(cy - ry, 0)), int(min(cy + ry + 1, L.shape[0]))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt(((xx - cx) / r) ** 2 + ((yy - cy) / ry) ** 2)
    f = np.clip(1 - d, 0, 1) ** 2 * k
    L[y0:y1, x0:x1] += f[..., None] * np.asarray(col, np.float32)


WARM = (1.0, 0.80, 0.46)
GOLD = (1.0, 0.86, 0.50)


# ───────────────────────── 하늘 · 원경 ─────────────────────────
def make_sky():
    stops = [(0.0, (8, 10, 38)), (0.28, (17, 22, 70)), (0.52, (33, 38, 104)), (0.70, (58, 54, 134)),
             (0.85, (96, 78, 152)), (1.0, (140, 104, 168))]
    NB = 22
    lut = []
    for b in range(NB + 1):
        v = b / NB
        for i in range(len(stops) - 1):
            if stops[i][0] <= v <= stops[i + 1][0]:
                u = (v - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
                lut.append([stops[i][1][k] + (stops[i + 1][1][k] - stops[i][1][k]) * u for k in range(3)])
                break
    lut = np.array(lut, np.float32)
    yy = np.arange(H)[:, None] / 300.0
    b = np.floor(np.clip(yy, 0, 1) * NB + BAYER_FULL - 0.5).astype(int)
    return lut[np.clip(b, 0, NB)]


def make_stars():
    stars = []
    for i in range(110):
        x = int(hsh(i, 1, 3) * W)
        y = int(hsh(i, 2, 3) ** 1.4 * 235)
        br = 0.35 + 0.65 * hsh(i, 3, 3)
        stars.append((x, y, br, 1.5 + 3 * hsh(i, 4, 3), 6.28 * hsh(i, 5, 3), hsh(i, 6, 3) > 0.88))
    return stars


def make_moon():
    r = 14
    im = new(2 * r + 1, 2 * r + 1)
    px = im.load()
    for y in range(2 * r + 1):
        for x in range(2 * r + 1):
            d = math.hypot(x - r, y - r)
            if d <= r + 0.2:
                c = '#fff3c9'
                if math.hypot(x - r + 5, y - r + 4) > r + 1.5:
                    c = '#efd698'
                elif math.hypot(x - r + 3, y - r + 2) > r - 0.5:
                    c = '#f9e6b0'
                px[x, y] = C(c)
    for (cx, cy, cr) in ((9, 10, 2.2), (17, 15, 3.0), (12, 20, 1.6), (19, 7, 1.4)):
        for y in range(2 * r + 1):
            for x in range(2 * r + 1):
                if math.hypot(x - cx, y - cy) <= cr and px[x, y][3]:
                    px[x, y] = C('#ecd391')
    return npa(im)


def make_clouds():
    w = 360
    im = new(w, H)
    d = ImageDraw.Draw(im)
    defs = [(30, 60, 1.0), (150, 38, 0.8), (250, 84, 1.2), (330, 52, 0.7), (95, 150, 0.9), (215, 176, 1.0)]
    for (cx, cy, s) in defs:
        body, hi = C('#2a3078'), C('#3c4494')
        if cy > 120:
            body, hi = C('#3d3f8a'), C('#5152a2')
        blobs = [(-22, 0, 12, 4), (-8, -3, 12, 6), (8, -1, 14, 5), (24, 1, 10, 3)]
        for (bx, by, rx, ry) in blobs:
            d.ellipse([cx + (bx - rx) * s, cy + (by - ry) * s, cx + (bx + rx) * s, cy + (by + ry) * s], fill=hi)
        for (bx, by, rx, ry) in blobs:
            d.ellipse([cx + (bx - rx) * s, cy + (by - ry) * s + 2, cx + (bx + rx) * s, cy + (by + ry) * s], fill=body)
        d.rectangle([cx - 34 * s, cy + 3 * s, cx + 34 * s, cy + 8 * s], fill=(0, 0, 0, 0))
    return npa(im)


def make_mountains():
    w = 280
    arr = np.zeros((H, w, 4), np.uint8)
    peaks = [(8, 186, 1.0), (58, 158, 0.85), (112, 178, 1.15), (160, 150, 0.8), (214, 180, 1.05), (262, 162, 0.9)]
    rock_l, rock_d = C('#333c88'), C('#272e6c')
    snow_l, snow_d = C('#929de0'), C('#626fbc')
    for x in range(w):
        best = None
        for (pxk, pyk, sl) in peaks:
            y = pyk + abs(x - pxk) * sl
            if best is None or y < best[0]:
                best = (y, pxk, pyk)
        y0 = int(best[0] + (hsh(x // 2, 1, 7) - 0.5) * 3)
        for y in range(y0, 310):
            depth = y - best[2]
            lit = x < best[1] + depth * 0.28 + (hsh(y // 3, 3, 5) - 0.5) * 3
            snowline = 15 + 7 * hsh(x // 3, 2, 9) + (0 if lit else -5)
            if depth < snowline:
                arr[y, x] = snow_l if lit else snow_d
            else:
                arr[y, x] = rock_l if lit else rock_d
                if lit and hsh(x // 3, y // 3, 4) > 0.95:
                    arr[y, x] = rock_d
    return arr.astype(np.float32)


def make_forest():
    w = 310
    im = new(w, H)
    d = ImageDraw.Draw(im)
    rows = [(284, '#202a66', '#3d4b96', 26, 48, 1), (302, '#141b48', '#2c397c', 32, 58, 2)]
    for row, (base, col, snow, hmin, hmax, seed) in enumerate(rows):
        x = -12.0
        while x < w + 12:
            h = hmin + hsh(int(x), row, seed) * (hmax - hmin)
            tw = h * 0.40
            for k in range(4):
                top = base - h + k * h * 0.20
                bot = top + h * 0.44
                half = tw * (0.34 + 0.22 * k)
                d.polygon([(x, top), (x + half, bot), (x - half, bot)], fill=C(col))
                d.line([(x - 1, top + 1), (x - half + 1, bot - 1)], fill=C(snow))
            x += 8 + hsh(int(x), row, seed + 5) * 9
        d.rectangle([0, base, w, H], fill=C(col))
    return npa(im)


def make_hills():
    w = 330
    im = new(w, H)
    px = im.load()
    surf = []
    for x in range(w):
        y0 = int(293 + 6 * math.sin(x / 37.0 + 1) + 3 * math.sin(x / 15.0))
        surf.append(y0)
        for y in range(y0, H):
            dy = y - y0
            c = '#b3bef2' if dy < 2 else '#8793d8'
            if dy >= 2 and (dy + (x + y) % 2 * 3) > 9 + 4 * BAYER[y % 4, x % 4]:
                c = '#727ec6'
            px[x, y] = C(c)
    d = ImageDraw.Draw(im)
    # 멀리 보이는 작은 집들과 교회
    for i, hx in enumerate([14, 52, 96, 150, 188, 236, 282, 316]):
        by = surf[min(hx + 5, w - 1)] + 1
        if i == 3:      # 교회
            d.rectangle([hx, by - 13, hx + 11, by], fill=C('#33305f'))
            d.polygon([(hx - 1, by - 13), (hx + 5.5, by - 20), (hx + 12, by - 13)], fill=C('#c9d3fa'))
            d.rectangle([hx + 3, by - 30, hx + 8, by - 16], fill=C('#33305f'))
            d.polygon([(hx + 2, by - 30), (hx + 5.5, by - 39), (hx + 9, by - 30)], fill=C('#c9d3fa'))
            d.rectangle([hx + 5, by - 27, hx + 6, by - 24], fill=C('#ffd56b'))
            d.rectangle([hx + 2, by - 8, hx + 3, by - 4], fill=C('#ffd56b'))
            d.rectangle([hx + 8, by - 8, hx + 9, by - 4], fill=C('#ffd56b'))
            continue
        ww = 11 + int(hsh(i, 1, 21) * 4)
        hh = 7 + int(hsh(i, 2, 21) * 3)
        d.rectangle([hx, by - hh, hx + ww, by], fill=C('#34305f'))
        d.polygon([(hx - 2, by - hh), (hx + ww / 2, by - hh - 7), (hx + ww + 2, by - hh)], fill=C('#c5cff8'))
        d.line([(hx - 2, by - hh), (hx + ww + 2, by - hh)], fill=C('#8f9be0'))
        d.rectangle([hx + 2, by - hh + 2, hx + 3, by - hh + 4], fill=C('#ffd56b'))
        if ww > 12:
            d.rectangle([hx + ww - 4, by - hh + 2, hx + ww - 3, by - hh + 4], fill=C('#ffb955'))
    for i in range(16):   # 작은 침엽수
        tx = int(hsh(i, 5, 22) * w)
        by = surf[tx] + 2
        hh = 9 + int(hsh(i, 6, 22) * 8)
        d.polygon([(tx, by - hh), (tx + 4, by), (tx - 4, by)], fill=C('#1c2658'))
        d.line([(tx - 1, by - hh + 2), (tx - 3, by - 2)], fill=C('#7d8ad2'))
    return npa(im)


# ───────────────────────── 마을 (메인 레이어, 알베도) ─────────────────────────
class World:
    """메인 레이어: 알베도 이미지 + 발광 마스크 + 정적 조명 + 동적 요소 목록."""

    def __init__(self):
        self.img = new(WW, H)
        self.emis = Image.new('L', (WW, H), 0)
        self.d = ImageDraw.Draw(self.img)
        self.de = ImageDraw.Draw(self.emis)
        self.L = np.zeros((H, WW, 3), np.float32)
        self.bulbs = []       # (x, y, colorRGB, phase)
        self.smokes = []      # (x, y)

    def em_rect(self, box, fill):
        self.d.rectangle(box, fill=fill)
        self.de.rectangle(box, fill=255)


BULB_COLS = [(255, 217, 90), (255, 96, 96), (110, 215, 255), (255, 217, 90), (255, 140, 210), (150, 255, 130)]


def draw_ground(wd):
    px = wd.img.load()
    for x in range(WW):
        top = GY + int(round(1.4 * math.sin(x / 23.0) + 0.9 * math.sin(x / 9.0 + 2)))
        p_top = 331 + int(round(1.6 * math.sin(x / 31.0)))
        p_bot = 350 + int(round(1.6 * math.sin(x / 27.0 + 1)))
        for y in range(top, H):
            dy = y - top
            n = hsh(x, y, 31)
            if dy < 2:
                c = '#ffffff'
            elif y < p_top - 1:
                c = '#e3ebff'
                if n > 0.93:
                    c = '#cfdaf9'
            elif y == p_top - 1:
                c = '#c3cff5'
            elif y <= p_bot:
                c = '#cdd8f8'
                if hsh(x // 2, y, 32) > 0.80:
                    c = '#bcc9f2'
                if hsh(x // 3, y // 2, 33) > 0.9:
                    c = '#dde6fd'
            elif y == p_bot + 1:
                c = '#ffffff'
            else:
                v = (y - p_bot) / float(H - p_bot)
                c = '#e6edff'
                if v + (BAYER[y % 4, x % 4] - 0.5) * 0.5 > 0.55:
                    c = '#cbd6f7'
                if v + (BAYER[y % 4, x % 4] - 0.5) * 0.4 > 0.9:
                    c = '#b2c0ee'
                if n > 0.965:
                    c = '#ffffff'
            px[x, y] = C(c)
    d = wd.d
    # 눈 둔덕
    for i in range(14):
        mx = int(hsh(i, 1, 41) * WW)
        my = 360 + int(hsh(i, 2, 41) * 18)
        rw = 8 + int(hsh(i, 3, 41) * 10)
        d.ellipse([mx - rw, my - 2, mx + rw, my + 4], fill=C('#f4f8ff'))
        d.arc([mx - rw, my - 2, mx + rw, my + 4], 10, 170, fill=C('#bccaf3'))


def draw_bush(wd, x, y, w, h, seed=0, lights=False):
    """눈 덮인 덤불 (화면 아래쪽 전경)."""
    d = wd.d
    px = wd.img.load()
    blobs = [(-0.32, 0.15, 0.36, 0.5), (0.0, -0.05, 0.42, 0.62), (0.32, 0.18, 0.36, 0.48)]
    for yy in range(int(y - h), int(y + 3)):
        for xx in range(int(x - w / 2 - 2), int(x + w / 2 + 3)):
            if not (0 <= xx < WW and 0 <= yy < H):
                continue
            best = None
            for (bx, by, rx, ry) in blobs:
                cx_, cy_ = x + bx * w, y - h * 0.45 + by * h
                dd = ((xx - cx_) / (rx * w)) ** 2 + ((yy - cy_) / (ry * h)) ** 2
                if dd <= 1 and (best is None or dd < best[0]):
                    best = (dd, (yy - (cy_ - ry * h)) / (2 * ry * h))
            if best is None:
                continue
            v = best[1]
            n = hsh(xx, yy, seed)
            if v < 0.30 + 0.12 * n:
                c = '#ffffff' if v < 0.2 else '#e3ebff'
            elif v < 0.38 + 0.1 * n:
                c = '#b9c8f5'
            else:
                c = '#1f7048' if n > 0.35 else '#175a3c'
                if n > 0.88:
                    c = '#2f8f5a'
            px[xx, yy] = C(c)
    for i in range(int(w / 5)):
        bx = int(x + (hsh(i, 1, seed) - 0.5) * w * 0.8)
        by = int(y - h * (0.12 + 0.3 * hsh(i, 2, seed)))
        if lights:
            col = BULB_COLS[i % len(BULB_COLS)]
            wd.bulbs.append((bx, by, col, hsh(i, 3, seed) * 6.28))
        else:
            d.point([(bx, by)], fill=C('#ff5a5a'))
            d.point([(bx + 1, by)], fill=C('#c83040'))
    d.ellipse([x - w / 2 - 3, y - 1, x + w / 2 + 3, y + 4], fill=C('#f4f8ff'))


def draw_front_fence(wd):
    """화면 맨 아래의 눈 덮인 울타리와 전구 줄 (전경)."""
    d = wd.d
    px = wd.img.load()
    for (ry, th) in ((365, 3), (375, 3)):
        d.rectangle([0, ry, WW, ry + th - 1], fill=C('#8f6248'))
        d.line([(0, ry + th - 1), (WW, ry + th - 1)], fill=C('#6a4532'))
        for xx in range(WW):
            px[xx, ry - 1] = C('#ffffff')
            if hsh(xx // 3, ry, 70) > 0.55:
                px[xx, ry - 2] = C('#ffffff')
            if hsh(xx // 2, ry, 71) > 0.8:
                px[xx, ry] = C('#e3ebff')
    posts = list(range(10, WW, 26))
    for k, x in enumerate(posts):
        d.rectangle([x, 357, x + 4, H - 1], fill=C('#8f6248'))
        d.line([(x, 357), (x, H - 1)], fill=C('#ae7c5c'))
        d.line([(x + 4, 357), (x + 4, H - 1)], fill=C('#6a4532'))
        d.rectangle([x - 1, 354, x + 5, 356], fill=C('#ffffff'))
        d.rectangle([x, 353, x + 4, 353], fill=C('#ffffff'))
        d.line([(x - 1, 357), (x + 5, 357)], fill=C('#b9c8f5'))
        if k + 1 < len(posts):                      # 기둥 사이에 늘어진 전구 줄
            x0, x1 = x + 4, posts[k + 1]
            n = x1 - x0
            for i in range(n + 1):
                u = i / float(n)
                xx = x0 + i
                yy = int(round(359 + 4.5 * 4 * u * (1 - u)))
                px[xx, yy] = C('#2a2748')
                if i % 5 == 3:
                    col = BULB_COLS[(i // 5 + k) % len(BULB_COLS)]
                    wd.bulbs.append((xx, yy + 1, col, hsh(xx, 12, k) * 6.28))
                    glow(wd.L, xx, yy + 1, 7, [c / 255.0 for c in col], 0.10)


def draw_window(wd, x, y, w, h, frame='#5c3626', curtain='#f08f6e', cat=False, glow_k=0.55):
    d = wd.d
    d.rectangle([x - 1, y - 1, x + w, y + h], fill=C(frame))
    for yy in range(h):
        v = yy / max(1, h - 1)
        if v < 0.35:
            c = '#fff6bd'
        elif v < 0.7:
            c = '#ffdc6e'
        else:
            c = '#ffb347'
        wd.em_rect([x, y + yy, x + w - 1, y + yy], C(c))
    # 커튼
    d.polygon([(x, y), (x + 2, y), (x, y + 4)], fill=C(curtain))
    d.polygon([(x + w - 1, y), (x + w - 3, y), (x + w - 1, y + 4)], fill=C(curtain))
    # 창살
    d.line([(x + w // 2, y), (x + w // 2, y + h - 1)], fill=C(frame))
    d.line([(x, y + h // 2 - 1), (x + w - 1, y + h // 2 - 1)], fill=C(frame))
    if cat:      # 창가의 고양이 실루엣
        cx = x + w // 2 + 2
        d.rectangle([cx, y + h - 5, cx + 3, y + h - 1], fill=C('#3a2434'))
        d.rectangle([cx, y + h - 7, cx + 2, y + h - 5], fill=C('#3a2434'))
        d.point([(cx, y + h - 8), (cx + 2, y + h - 8)], fill=C('#3a2434'))
        d.point([(cx + 4, y + h - 2), (cx + 5, y + h - 3)], fill=C('#3a2434'))
    # 창턱 + 눈
    d.rectangle([x - 2, y + h + 1, x + w + 1, y + h + 2], fill=C('#7c4b35'))
    d.rectangle([x - 2, y + h, x + w + 1, y + h], fill=C('#ffffff'))
    d.point([(x - 1, y + h - 1), (x + w, y + h - 1), (x + 2, y + h - 1)], fill=C('#ffffff'))
    glow(wd.L, x + w / 2, y + h / 2, 26, WARM, glow_k * 1.5)
    glow(wd.L, x + w / 2, GY + 10, 22, WARM, 0.55, ry=9)


def draw_door(wd, x, w, h, col='#7a4630', col_d='#5a3222', wreath=True):
    d = wd.d
    base = GY + 1
    y = base - h
    d.rectangle([x - 1, y - 1, x + w, base], fill=C('#3e2318'))
    d.rectangle([x, y, x + w - 1, base], fill=C(col))
    d.rectangle([x + 2, y + 2, x + w - 3, y + h // 2 - 2], fill=C(col_d))
    d.rectangle([x + 2, y + h // 2 + 1, x + w - 3, base - 3], fill=C(col_d))
    d.point([(x + w - 3, y + h // 2)], fill=C('#ffd76a'))
    if wreath:
        cx, cy = x + w // 2, y + 6
        d.ellipse([cx - 4, cy - 4, cx + 3, cy + 3], outline=C('#2f9455'), width=2)
        d.point([(cx - 4, cy - 1), (cx + 2, cy + 3), (cx + 3, cy - 2)], fill=C('#1f6e3d'))
        d.point([(cx - 2, cy - 4), (cx + 2, cy - 3), (cx - 3, cy + 2)], fill=C('#ff5a5a'))
        d.rectangle([cx - 1, cy + 3, cx, cy + 4], fill=C('#e04848'))
    # 계단의 눈
    d.rectangle([x - 3, base - 1, x + w + 2, base + 1], fill=C('#ffffff'))


def roof_snow_col(i, th):
    if i == 0:
        return C('#ffffff')
    if i >= th - 1:
        return C('#b9c8f5')
    return C('#eef3ff')


def draw_house(wd, x, w, wall_h, roof_h, pal, style, layout, chimney=None, seed=0):
    d = wd.d
    px = wd.img.load()
    base = GY + 2
    top = base - wall_h
    cx = x + (w - 1) / 2.0
    wall, wall_d, wall_l = C(pal['wall']), C(pal['wall_d']), C(pal['wall_l'])
    # 굴뚝 (지붕 뒤)
    if chimney is not None:
        chx, chh = x + chimney, 16
        if style == 'gable':
            ytop = top - roof_h * (1 - abs(chx + 4 - cx) / (w / 2 + 5)) - chh + 6
        else:
            ytop = top - roof_h - chh + 8
        ytop = int(ytop)
        d.rectangle([chx, ytop, chx + 8, ytop + chh + 6], fill=C('#8d4d3e'))
        for yy in range(ytop + 2, ytop + chh + 6, 3):
            d.line([(chx, yy), (chx + 8, yy)], fill=C('#6b382f'))
        d.rectangle([chx + 6, ytop, chx + 8, ytop + chh + 6], fill=C('#6b382f'))
        d.rectangle([chx - 1, ytop - 2, chx + 9, ytop], fill=C('#ffffff'))
        d.rectangle([chx, ytop - 3, chx + 7, ytop - 3], fill=C('#ffffff'))
        d.line([(chx - 1, ytop + 1), (chx + 9, ytop + 1)], fill=C('#b9c8f5'))
        wd.smokes.append((chx + 4, ytop - 5))
    # 벽
    d.rectangle([x, top, x + w - 1, base], fill=wall)
    tex = pal.get('tex', 'plank')
    for yy in range(top, base + 1):
        for xx in range(x, x + w):
            if tex == 'plank':
                if (yy - top) % 5 == 4:
                    px[xx, yy] = wall_d
                elif (yy - top) % 5 == 0 and hsh(xx // 3, yy, seed) > 0.6:
                    px[xx, yy] = wall_l
            elif tex == 'brick':
                r = (yy - top) // 4
                if (yy - top) % 4 == 3 or (xx - x + (r % 2) * 4) % 8 == 0:
                    px[xx, yy] = wall_d
                elif hsh(xx // 8 + r * 7, r, seed) > 0.8:
                    px[xx, yy] = wall_l
            else:   # plaster
                if hsh(xx // 2, yy // 2, seed) > 0.9:
                    px[xx, yy] = wall_l
    d.rectangle([x + w - 3, top, x + w - 1, base], fill=wall_d)
    d.rectangle([x, top, x + w - 1, top + 2], fill=wall_d)
    if tex == 'plaster':      # 목조 뼈대
        beam = C(pal['beam'])
        d.rectangle([x, top, x + 2, base], fill=beam)
        d.rectangle([x + w - 3, top, x + w - 1, base], fill=beam)
        d.rectangle([x, top + wall_h // 2 - 4, x + w - 1, top + wall_h // 2 - 3], fill=beam)
    # 지붕
    o = 5
    roof, roof_d = C(pal['roof']), C(pal['roof_d'])
    lights_y = {}
    if style == 'gable':
        d.polygon([(x, top), (cx, top - roof_h + 5), (x + w - 1, top)], fill=wall)
        if tex == 'plaster':
            beam = C(pal['beam'])
            d.line([(cx, top - roof_h + 6), (cx, top)], fill=beam, width=2)
            d.line([(x, top), (x + w - 1, top)], fill=beam, width=2)
        else:
            for yy in range(top - roof_h + 6, top, 5):
                hw = (w / 2) * (yy - (top - roof_h + 5)) / float(roof_h - 5)
                d.line([(cx - hw + 2, yy), (cx + hw - 2, yy)], fill=wall_d)
        half = w / 2.0 + o
        for xx in range(int(x - o), int(x + w + o)):
            u = abs(xx - cx) / half
            yr = top + 3 - (roof_h + 3) * (1 - u)
            yr = int(round(yr))
            th = 4 + int(hsh(xx // 3, 1, seed + 3) * 2.2)
            for i in range(4):
                if 0 <= yr + i < H:
                    px[xx, yr + i] = roof if i < 3 else roof_d
            for i in range(th):
                if 0 <= yr - th + i:
                    px[xx, yr - th + i] = roof_snow_col(i, th)
            lights_y[xx] = yr + 5
            if hsh(xx, 7, seed) > 0.80 and u > 0.25:           # 고드름
                ln = 2 + int(hsh(xx, 8, seed) * 3)
                for i in range(ln):
                    px[xx, yr + 4 + i] = C('#dbe6ff') if i < ln - 1 else C('#a9bdf5')
        for ex in (x - o - 1, x + w + o - 1):                  # 처마 끝 눈덩이
            d.ellipse([ex - 2, top - 2, ex + 3, top + 3], fill=C('#ffffff'))
            d.point([(ex, top + 3), (ex + 1, top + 3), (ex + 2, top + 2)], fill=C('#b9c8f5'))
    else:
        rt = top - roof_h
        inset = 9
        for yy in range(rt, top + 3):
            v = (yy - rt) / float(top + 2 - rt)
            xl = int(round(x + inset - (inset + o) * v))
            xr = int(round(x + w - 1 - inset + (inset + o) * v))
            for xx in range(xl, xr + 1):
                c = roof
                if (yy - rt) % 4 == 3:
                    c = roof_d
                elif (xx + ((yy - rt) // 4) * 3) % 6 == 0:
                    c = roof_d
                px[xx, yy] = c
                sdepth = 6 + 5 * (0.5 + 0.5 * math.sin(xx * 0.45 + seed)) + 2 * hsh(xx // 2, 3, seed)
                edge_boost = 3 if (xx - xl < 3 or xr - xx < 3) else 0
                if yy - rt < sdepth + edge_boost:
                    i = yy - rt
                    px[xx, yy] = C('#ffffff') if i < 2 else (C('#b9c8f5') if i >= sdepth + edge_boost - 1 else C('#eef3ff'))
        d.rectangle([x + inset - 1, rt - 2, x + w - inset, rt - 1], fill=C('#ffffff'))
        ey = top + 3
        for xx in range(int(x - o), int(x + w + o)):
            px[xx, ey] = roof_d
            px[xx, ey - 1] = C('#ffffff')
            if hsh(xx // 2, 5, seed) > 0.5:
                px[xx, ey - 2] = C('#ffffff')
            lights_y[xx] = ey + 3
            if hsh(xx, 7, seed) > 0.72:
                ln = 2 + int(hsh(xx, 8, seed) * 4)
                for i in range(ln):
                    px[xx, ey + 1 + i] = C('#dbe6ff') if i < ln - 1 else C('#a9bdf5')
    # 창문/문
    for item in layout:
        if item[0] == 'win':
            _, rx, ry, ww, hh = item[:5]
            draw_window(wd, x + rx, top + ry, ww, hh, cat=(len(item) > 5 and item[5] == 'cat'),
                        frame=pal.get('frame', '#5c3626'), curtain=pal.get('curtain', '#f08f6e'))
        elif item[0] == 'door':
            _, rx, ww, hh = item[:4]
            draw_door(wd, x + rx, ww, hh, col=pal.get('door', '#7a4630'), col_d=pal.get('door_d', '#5a3222'))
            lx, ly = x + rx + ww + 3, base - hh + 3              # 현관등
            d.rectangle([lx, ly - 1, lx + 2, ly - 1], fill=C('#2a2440'))
            wd.em_rect([lx, ly, lx + 2, ly + 3], C('#fff0a8'))
            glow(wd.L, lx + 1, ly + 2, 20, WARM, 0.7)
        elif item[0] == 'round':
            _, rx, ry = item[:3]
            ax, ay = x + rx, top + ry
            d.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=C(pal.get('frame', '#5c3626')))
            wd.d.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=C('#ffdc6e'))
            wd.de.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=255)
            d.line([(ax, ay - 3), (ax, ay + 3)], fill=C(pal.get('frame', '#5c3626')))
            d.line([(ax - 3, ay), (ax + 3, ay)], fill=C(pal.get('frame', '#5c3626')))
            glow(wd.L, ax, ay, 12, WARM, 0.4)
    # 처마 전구
    k = 0
    for xx in sorted(lights_y):
        if (xx - int(x)) % 6 == 2:
            col = BULB_COLS[(k + seed) % len(BULB_COLS)]
            wd.bulbs.append((xx, lights_y[xx], col, hsh(xx, 9, seed) * 6.28))
            glow(wd.L, xx, lights_y[xx], 7, [c / 255.0 for c in col], 0.10)
            k += 1
    # 벽 아래 쌓인 눈
    for xx in range(x - 2, x + w + 2):
        hgt = 2 + int(2 * hsh(xx // 3, 4, seed + 9))
        for i in range(hgt):
            px[xx, base - i] = C('#ffffff') if i == hgt - 1 else C('#e3ebff')


def draw_lamp(wd, x, tall=48):
    d = wd.d
    base = GY + 3
    top = base - tall
    d.rectangle([x - 3, base - 3, x + 3, base], fill=C('#232044'))
    d.rectangle([x - 2, base - 6, x + 2, base - 3], fill=C('#232044'))
    d.rectangle([x - 1, top + 9, x + 1, base - 6], fill=C('#232044'))
    d.line([(x - 1, top + 10), (x - 1, base - 7)], fill=C('#4a4780'))
    d.rectangle([x - 2, top + 20, x + 2, top + 21], fill=C('#232044'))
    # 등갓
    d.polygon([(x - 4, top + 9), (x + 4, top + 9), (x + 5, top - 1), (x - 5, top - 1)], fill=C('#232044'))
    wd.em_rect([x - 3, top + 1, x + 3, top + 7], C('#ffe9a0'))
    wd.em_rect([x - 1, top + 2, x + 1, top + 6], C('#fffbe0'))
    d.line([(x - 4, top), (x - 4, top + 8)], fill=C('#232044'))
    d.line([(x + 4, top), (x + 4, top + 8)], fill=C('#232044'))
    d.polygon([(x - 6, top - 1), (x + 6, top - 1), (x, top - 6)], fill=C('#232044'))
    d.polygon([(x - 6, top - 3), (x + 6, top - 3), (x, top - 8)], fill=C('#ffffff'))
    d.line([(x - 5, top - 2), (x + 5, top - 2)], fill=C('#b9c8f5'))
    # 리본
    d.rectangle([x - 4, top + 15, x + 4, top + 17], fill=C('#e04848'))
    d.rectangle([x - 1, top + 14, x + 1, top + 18], fill=C('#ff7a6e'))
    d.point([(x - 4, top + 18), (x + 4, top + 18)], fill=C('#b3312f'))
    d.rectangle([x - 4, base - 1, x + 4, base + 1], fill=C('#ffffff'))
    glow(wd.L, x, top + 4, 52, WARM, 1.05)
    glow(wd.L, x, GY + 16, 46, WARM, 0.85, ry=17)


def draw_pine(wd, x, h, baubles=False, seed=0):
    d = wd.d
    base = GY + 3
    d.rectangle([x - 1, base - 4, x + 1, base], fill=C('#5b3a2b'))
    n = 4
    for k in range(n):
        top = base - h + k * h * 0.19
        bot = top + h * 0.40
        half = h * 0.40 * (0.36 + 0.22 * k)
        d.polygon([(x, top), (x + half, bot), (x - half, bot)], fill=C('#1f7a4f'))
        d.polygon([(x, top + 2), (x + half - 1, bot), (x + 1, bot)], fill=C('#17603f'))
        d.line([(x - half + 1, bot), (x + half - 1, bot)], fill=C('#114a32'))
        # 눈
        d.polygon([(x, top - 1), (x + half * 0.5, top + (bot - top) * 0.55), (x - half * 0.55, top + (bot - top) * 0.6)],
                  fill=C('#ffffff'))
        d.line([(x - half * 0.5, top + (bot - top) * 0.6), (x + half * 0.45, top + (bot - top) * 0.56)], fill=C('#c3d1f8'))
    if baubles:
        for i in range(5):
            bx = x + int((hsh(i, 1, seed) - 0.5) * h * 0.45)
            by = base - int(h * (0.15 + 0.5 * hsh(i, 2, seed)))
            col = BULB_COLS[i % len(BULB_COLS)]
            wd.bulbs.append((bx, by, col, hsh(i, 3, seed) * 6.28))
    d.ellipse([x - 7, base - 1, x + 7, base + 3], fill=C('#ffffff'))


def draw_fence(wd, x0, x1):
    d = wd.d
    base = GY + 3
    d.rectangle([x0, base - 9, x1, base - 8], fill=C('#8a5e45'))
    d.rectangle([x0, base - 4, x1, base - 3], fill=C('#6f4835'))
    d.line([(x0, base - 10), (x1, base - 10)], fill=C('#ffffff'))
    for xx in range(x0, x1 + 1, 8):
        d.rectangle([xx, base - 12, xx + 2, base], fill=C('#8a5e45'))
        d.line([(xx + 2, base - 11), (xx + 2, base)], fill=C('#6f4835'))
        d.rectangle([xx - 1, base - 14, xx + 3, base - 13], fill=C('#ffffff'))
        d.line([(xx, base - 12), (xx + 2, base - 12)], fill=C('#c3d1f8'))


def draw_snowman(wd, x):
    d = wd.d
    base = GY + 6
    sh, wh = C('#c3d1f8'), C('#ffffff')
    for (cy, r) in ((base - 7, 8), (base - 19, 6), (base - 29, 5)):
        d.ellipse([x - r, cy - r, x + r, cy + r], fill=sh)
        d.ellipse([x - r, cy - r, x + r - 2, cy + r - 2], fill=wh)
    hy = base - 29
    d.point([(x - 2, hy - 1), (x + 2, hy - 1)], fill=C('#22203a'))
    d.rectangle([x, hy + 1, x + 3, hy + 1], fill=C('#ff9a3d'))
    d.point([(x - 2, hy + 3), (x - 1, hy + 4), (x, hy + 4), (x + 1, hy + 3)], fill=C('#22203a'))
    d.rectangle([x - 5, hy + 5, x + 5, hy + 7], fill=C('#e04848'))
    d.rectangle([x + 2, hy + 7, x + 4, hy + 12], fill=C('#e04848'))
    d.line([(x - 5, hy + 7), (x + 1, hy + 7)], fill=C('#b3312f'))
    d.point([(x, base - 20), (x, base - 16), (x, base - 9)], fill=C('#22203a'))
    d.rectangle([x - 5, hy - 6, x + 5, hy - 5], fill=C('#2b2848'))       # 모자
    d.rectangle([x - 3, hy - 12, x + 3, hy - 6], fill=C('#2b2848'))
    d.rectangle([x - 3, hy - 8, x + 3, hy - 7], fill=C('#e04848'))
    d.rectangle([x - 3, hy - 13, x + 2, hy - 13], fill=wh)
    d.line([(x - 6, base - 21), (x - 12, base - 26)], fill=C('#6b4532'))   # 나뭇가지 팔
    d.line([(x + 6, base - 21), (x + 12, base - 25)], fill=C('#6b4532'))
    d.point([(x - 12, base - 28), (x + 13, base - 27)], fill=C('#6b4532'))


def draw_wire(wd, x0, y0, x1, y1, sag, seed):
    """집과 가로등 사이에 늘어진 전구 줄."""
    px = wd.img.load()
    n = int(abs(x1 - x0))
    k = 0
    for i in range(n + 1):
        u = i / float(n)
        xx = int(round(x0 + (x1 - x0) * u))
        yy = int(round(y0 + (y1 - y0) * u + sag * 4 * u * (1 - u)))
        px[xx, yy] = C('#2a2748')
        if i % 6 == 3:
            col = BULB_COLS[(k + seed) % len(BULB_COLS)]
            wd.bulbs.append((xx, yy + 1, col, hsh(xx, 11, seed) * 6.28))
            glow(wd.L, xx, yy + 1, 7, [c / 255.0 for c in col], 0.10)
            k += 1


def build_world():
    wd = World()
    draw_ground(wd)
    draw_front_fence(wd)
    # 뒤쪽 소품
    draw_pine(wd, 68, 30, seed=3)
    draw_fence(wd, 146, 176)
    draw_pine(wd, 150, 26, baubles=True, seed=5)
    draw_pine(wd, 236, 34, seed=8)
    # 집 A: 크림색 벽 + 빨간 지붕
    draw_house(wd, 4, 58, 46, 30,
               dict(wall='#f1dfc0', wall_d='#d2b894', wall_l='#fff1d8', roof='#b8473f', roof_d='#8a2f31', tex='plaster',
                    beam='#7a4a34', door='#3f7f5a', door_d='#2c5f43'),
               'gable', [('win', 8, 14, 12, 14), ('door', 30, 12, 24), ('round', 29, -9)], chimney=40, seed=1)
    # 집 B: 2층 파란 판자집
    draw_house(wd, 80, 66, 66, 22,
               dict(wall='#7f96c8', wall_d='#6079ad', wall_l='#9db1dc', roof='#4a4470', roof_d='#383358', tex='plank',
                    frame='#f3ead8', curtain='#e86a6a', door='#b5533f', door_d='#8f3c2f'),
               'side', [('win', 8, 10, 13, 15), ('win', 44, 10, 13, 15, 'cat'), ('win', 8, 40, 13, 16), ('door', 40, 13, 25)],
               chimney=10, seed=2)
    # 집 C: 벽돌집
    draw_house(wd, 178, 52, 44, 34,
               dict(wall='#b8654e', wall_d='#93483c', wall_l='#cf8062', roof='#5a4a6e', roof_d='#42365a', tex='brick',
                    frame='#f3ead8', curtain='#7fbf8f', door='#3f6fa8', door_d='#2c5285'),
               'gable', [('win', 30, 14, 13, 14), ('door', 8, 12, 24), ('round', 26, -11)], chimney=6, seed=3)
    # 집 D (트리 오른쪽)
    draw_house(wd, 418, 54, 50, 28,
               dict(wall='#e9d6b8', wall_d='#c9b08c', wall_l='#fff1d8', roof='#3f7a66', roof_d='#2c5a4c', tex='plaster',
                    beam='#6e4632', door='#a8483f', door_d='#80332f'),
               'gable', [('win', 6, 14, 12, 14), ('round', 27, -9)], seed=4)
    # 앞쪽 소품
    draw_lamp(wd, 71)
    draw_snowman(wd, 163)
    draw_lamp(wd, 244, tall=52)
    draw_lamp(wd, 411, tall=52)
    draw_wire(wd, 62, 272, 71, 272, 1, 1)
    draw_wire(wd, 72, 272, 82, 258, 2, 2)
    draw_wire(wd, 146, 262, 178, 276, 5, 3)
    draw_wire(wd, 230, 276, 244, 270, 2, 4)
    return wd


# ───────────────────────── 거대한 크리스마스 트리 ─────────────────────────
TREE_X0, TREE_Y0, TREE_W, TREE_H = TREE_CX - 90, 92, 180, 240
T_BOT = [142, 169, 196, 223, 250, 276, 301]
T_HW = [20, 31, 41, 51, 60, 68, 75]
T_TOP = [108] + [T_BOT[i - 1] - 17 for i in range(1, 7)]
T_THW = [0.5] + [T_HW[i - 1] * 0.45 for i in range(1, 7)]
STAR_Y = 103


def tier_edge(i, dx):
    lobe = (dx + 5 * i + 100) % 10
    return T_BOT[i] - 6.0 * (dx / float(T_HW[i])) ** 2 - abs(lobe - 5) * 0.9


def make_tree():
    im = new(TREE_W, TREE_H)
    px = im.load()
    d = ImageDraw.Draw(im)
    ox, oy = TREE_X0, TREE_Y0
    cx = TREE_CX - ox
    greens = [C('#0f4534'), C('#176040'), C('#24804c'), C('#3ba25a'), C('#63c06e')]
    # 눈 둔덕 + 줄기
    d.ellipse([cx - 40, 316 - oy, cx + 40, 330 - oy], fill=C('#f4f8ff'))
    d.rectangle([cx - 6, 292 - oy, cx + 5, 320 - oy], fill=C('#6a4532'))
    d.rectangle([cx + 2, 292 - oy, cx + 5, 320 - oy], fill=C('#4e3024'))
    d.rectangle([cx - 6, 292 - oy, cx - 5, 320 - oy], fill=C('#8a5e45'))
    snow_caps = []
    balls = []
    for i in reversed(range(7)):
        prev_bot = T_BOT[i - 1] if i > 0 else T_TOP[0]
        for y in range(T_TOP[i], T_BOT[i] + 1):
            v = (y - T_TOP[i]) / float(T_BOT[i] - T_TOP[i])
            half = T_THW[i] + (T_HW[i] - T_THW[i]) * v ** 0.9
            for dx in range(-int(half) - 2, int(half) + 3):
                if abs(dx) > half + (hsh(dx, y, i) - 0.5) * 2.2:
                    continue
                if y > tier_edge(i, dx):
                    continue
                sv = clamp((y - (prev_bot - 8)) / float(T_BOT[i] - prev_bot + 8))
                s = 0.10 + 0.80 * sv + (hsh(dx, y, i + 11) - 0.5) * 0.36
                if int(abs(dx) * 0.8 + y + i * 3) % 5 == 0:
                    s -= 0.26
                s -= 0.16 * (dx / float(T_HW[i]))
                idx = int(clamp(s, 0, 0.999) * 4)
                if s > 1.02:
                    idx = 4
                px[cx + dx, y - oy] = greens[idx]
                # 가지 바깥쪽에 얹힌 눈
                if abs(dx) > half - 2.2 and hsh(dx // 2, y // 2, i + 40) > 0.45:
                    px[cx + dx, y - oy] = C('#f4f8ff')
        # 가지 끝 (눈 / 방울)
        for dx in range(-T_HW[i] + 3, T_HW[i] - 2):
            if (dx + 5 * i + 100) % 10 == 5:
                ey = tier_edge(i, dx)
                r = hsh(dx, i, 77)
                if r > 0.42:
                    snow_caps.append((dx, int(ey) - 1, i))
                elif r > 0.06:
                    balls.append((dx, int(ey) + 2, int(hsh(dx, i, 78) * 4)))
        # 눈은 해당 단을 그린 직후 얹는다 (윗단이 가릴 수 있게)
        for (dx, ey, ti) in [c for c in snow_caps if c[2] == i]:
            x, y = cx + dx, ey - oy
            for (ax, ay, c) in ((-4, 0, '#ffffff'), (-3, -1, '#ffffff'), (-2, -1, '#ffffff'), (-1, -2, '#ffffff'),
                                (0, -2, '#ffffff'), (1, -2, '#ffffff'), (2, -1, '#ffffff'), (3, -1, '#ffffff'),
                                (4, 0, '#ffffff'), (-3, 0, '#ffffff'), (-2, 0, '#f4f8ff'), (-1, -1, '#ffffff'),
                                (0, -1, '#ffffff'), (1, -1, '#ffffff'), (2, 0, '#f4f8ff'), (3, 0, '#ffffff'),
                                (-1, 0, '#f4f8ff'), (0, 0, '#f4f8ff'), (1, 0, '#f4f8ff'),
                                (-3, 1, '#b9c8f5'), (-2, 1, '#b9c8f5'), (-1, 1, '#c9d6fa'), (0, 1, '#c9d6fa'),
                                (1, 1, '#b9c8f5'), (2, 1, '#b9c8f5'), (3, 1, '#b9c8f5')):
                if 0 <= x + ax < TREE_W and 0 <= y + ay < TREE_H:
                    px[x + ax, y + ay] = C(c)
    # 꼭대기 눈
    for (ax, ay) in ((0, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)):
        px[cx + ax, T_TOP[0] - oy + ay] = C('#ffffff')
    # 방울 장식
    ball_cols = [('#e84a4a', '#a82a3a', '#ffb0a0'), ('#f7c33f', '#c48a1e', '#fff3b0'),
                 ('#4f97ea', '#2c62b5', '#bfe0ff'), ('#f08ad0', '#b8559a', '#ffd6f0')]
    for (dx, by, ci) in balls:
        base, dark, hi = ball_cols[ci]
        x, y = cx + dx, by - oy
        if y + 5 >= TREE_H:
            continue
        px[x, y - 1] = C('#d9c08a')
        for ay in range(5):
            for ax in range(-2, 3):
                if abs(ax) == 2 and ay in (0, 4):
                    continue
                c = base
                if ax >= 1 and ay >= 3 or ax == 2 or ay == 4:
                    c = dark
                px[x + ax, y + ay] = C(c)
        px[x - 1, y + 1] = C(hi)
    # 선물 상자
    gifts = [(-52, 10, 8, '#b565d9', '#ffe680'), (-39, 15, 12, '#e04848', '#ffd95a'), (-22, 11, 9, '#3f7fd9', '#ffffff'),
             (11, 12, 10, '#3aa35c', '#ff6a5a'), (25, 16, 13, '#f2b63a', '#e04848'), (43, 10, 8, '#e86aa6', '#ffffff')]
    for (gx, gw, gh, col, rib) in gifts:
        x0, y1 = cx + gx, 325 - oy
        y0 = y1 - gh
        cc = C(col)
        dk = tuple(int(v * 0.72) for v in cc[:3]) + (255,)
        d.rectangle([x0, y0, x0 + gw - 1, y1], fill=cc)
        d.rectangle([x0 + gw - 3, y0, x0 + gw - 1, y1], fill=dk)
        d.line([(x0, y1), (x0 + gw - 1, y1)], fill=dk)
        mx = x0 + gw // 2 - 1
        d.rectangle([mx, y0, mx + 1, y1], fill=C(rib))
        d.rectangle([x0, y0 + gh // 2 - 1, x0 + gw - 1, y0 + gh // 2], fill=C(rib))
        d.rectangle([mx - 2, y0 - 2, mx - 1, y0 - 1], fill=C(rib))
        d.rectangle([mx + 2, y0 - 2, mx + 3, y0 - 1], fill=C(rib))
        d.point([(x0, y0 - 1), (x0 + 1, y0 - 1), (x0 + gw - 2, y0 - 1)], fill=C('#ffffff'))
    # 전구 줄: 아래에서 위로 감아 올라가는 나선
    bulbs = []
    path = []
    seg = []
    for n_i, i in enumerate(range(6, 0, -1)):
        hw = T_HW[i] * 0.84
        n = max(3, int(2 * hw / 6.4))
        sgn = 1 if n_i % 2 == 0 else -1
        for k in range(n + 1):
            u = k / float(n)
            dx = sgn * (-hw + 2 * hw * u)
            y = (T_BOT[i] - 6) - 9 * u + 3.5 * math.sin(math.pi * u) - 5.5 * (dx / float(T_HW[i])) ** 2
            path.append((dx, y))
            seg.append(i)
    for (dx, y) in ((-6, 133), (1, 128), (6, 122), (-2, 117)):
        path.append((dx, y))
        seg.append(0)
    for j in range(len(path) - 1):          # 줄
        (ax, ay), (bx, by) = path[j], path[j + 1]
        if seg[j] == seg[j + 1]:
            d.line([(cx + ax, ay - oy), (cx + bx, by - oy)], fill=C('#b8923c'))
    for j, (dx, y) in enumerate(path):
        col = BULB_COLS[j % len(BULB_COLS)]
        bulbs.append((TREE_CX + int(round(dx)), int(round(y)), col, hsh(j, 5, 99) * 6.28, j / float(len(path) - 1)))
    return npa(im), bulbs


# ───────────────────────── 캐릭터 ─────────────────────────
PAL = {
    'boy': dict(skin='#ffd9b5', skin_d='#f0b690', hair='#4a2f25', hair_d='#38221b', hat='#2e9468', hat_d='#1f6e4c',
                brim='#f6eeda', brim_d='#d9cbb0', pom='#f6eeda', coat='#4063b8', coat_d='#2c468c', coat_l='#5f83da',
                trim=None, scarf='#e84f48', scarf_d='#b53432', leg='#3c4474', leg_d='#2b3058', boot='#7a4e36',
                boot_d='#573524', mit='#e84f48', long=False),
    'girl': dict(skin='#ffdfc6', skin_d='#f3bd9f', hair='#7d432a', hair_d='#5f2f1f', hat='#fff6e6', hat_d='#e3d2bb',
                 brim='#ffb5c6', brim_d='#ea8fa6', pom='#ff8fad', coat='#e64c52', coat_d='#b4303e', coat_l='#fa7670',
                 trim='#fff8ee', scarf='#ffd651', scarf_d='#dba228', leg='#553a60', leg_d='#3f2a49', boot='#f8f0e0',
                 boot_d='#cfc0a8', mit='#fff8ee', long=True),
}
OUTLINE = C('#1d1433')
EYE = '#2a1b33'


def outline(im, col=OUTLINE):
    a = np.array(im)
    m = a[..., 3] > 0
    nb = np.zeros_like(m)
    nb[1:] |= m[:-1]
    nb[:-1] |= m[1:]
    nb[:, 1:] |= m[:, :-1]
    nb[:, :-1] |= m[:, 1:]
    a[nb & ~m] = col
    return a


@lru_cache(maxsize=None)
def char_sprite(who, view, step=-1, expr='smile', arm='down', link=None, bob=0):
    """48x56 스프라이트. 기준점: (24, 51) = 발 중앙 바닥."""
    p = PAL[who]
    im = new(48, 56)
    px = im.load()

    def R(x0, y0, x1, y1, c):
        c = C(c)
        for y in range(int(y0), int(y1) + 1):
            for x in range(int(x0), int(x1) + 1):
                if 0 <= x < 48 and 0 <= y < 56:
                    px[x, y] = c

    def P(x, y, c):
        R(x, y, x, y, c)

    def LN(x0, y0, x1, y1, c, w=3):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2 + 1)
        for i in range(n + 1):
            fx, fy = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n
            R(round(fx) - w // 2, round(fy) - w // 2, round(fx) - w // 2 + w - 1, round(fy) - w // 2 + w - 1, c)

    def ELL(cx, cy, rx, ry, c, cond=None):
        for y in range(int(cy - ry) - 1, int(cy + ry) + 2):
            for x in range(int(cx - rx) - 1, int(cx + rx) + 2):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0 and (cond is None or cond(x, y)):
                    P(x, y, c)

    tall = 1 if who == 'boy' else 0
    walk_bob = [1, 0, 1, 0][step] if step >= 0 else 0
    uy = walk_bob + bob - tall
    girl = p['long']

    def hat(side=False):
        ELL(24, 16 + uy, 10, 7.6, p['hat'], cond=lambda x, y: y <= 15 + uy)
        ELL(24, 16 + uy, 10, 7.6, p['hat_d'], cond=lambda x, y: y <= 15 + uy and (x >= 30 or (x >= 28 and y >= 13 + uy)))
        for x in (19, 24):
            P(x, 10 + uy, p['hat_d'])
            P(x, 12 + uy, p['hat_d'])
        R(15, 15 + uy, 33, 17 + uy, p['brim'])
        P(14, 16 + uy, p['brim'])
        P(34, 16 + uy, p['brim'])
        for x in range(16, 34, 2):
            R(x, 16 + uy, x, 17 + uy, p['brim_d'])
        pcx = 23 if side else 24
        ELL(pcx, 6.5 + uy, 3.2, 3.2, p['pom'])
        P(pcx + 2, 8 + uy, p['brim_d'])
        P(pcx + 1, 9 + uy, p['brim_d'])
        P(pcx - 1, 5 + uy, '#ffffff')

    if view == 'side':
        nd, fd, nl, fl = {0: (3, -3, 0, 0), 1: (0, 0, 0, 2), 2: (-3, 3, 0, 0), 3: (0, 0, 2, 0)}.get(step, (-1, 2, 0, 0))
        lx = 22
        R(lx + fd, 44 + uy, lx + fd + 3, 48 - fl, p['leg_d'])
        R(lx + fd, 49 - fl, lx + fd + 4, 51 - fl, p['boot_d'])
        R(lx + nd, 44 + uy, lx + nd + 3, 48 - nl, p['leg'])
        R(lx + nd, 49 - nl, lx + nd + 4, 51 - nl, p['boot'])
        R(lx + nd, 51 - nl, lx + nd + 4, 51 - nl, p['boot_d'])
        if link == 'b':     # 뒤로 뻗은 팔 (몸 뒤)
            LN(22, 35 + uy, 15, 39, p['coat_d'])
            R(12, 38, 14, 41, p['mit'])
        bl, br = (17, 31) if girl else (19, 29)
        for y in range(33, 46):
            v = (y - 33) / 12.0
            vv = v ** 1.6 if girl else v
            l, r = round(20 - (20 - bl) * vv), round(28 + (br - 28) * vv)
            R(l, y + uy, r, y + uy, p['coat'])
            R(l, y + uy, l + 1, y + uy, p['coat_d'])
            P(r, y + uy, p['coat_l'])
        if p['trim']:
            R(bl, 44 + uy, br, 45 + uy, p['trim'])
            R(bl, 45 + uy, bl + 2, 45 + uy, '#d9cdbd')
        else:
            R(bl, 45 + uy, br, 45 + uy, p['coat_d'])
        if girl:   # 긴 머리
            R(15, 24 + uy, 19, 36 + uy, p['hair'])
            R(16, 37 + uy, 18, 37 + uy, p['hair'])
            R(17, 26 + uy, 17, 35 + uy, p['hair_d'])
        # 목도리
        flt = [0, 1, 0, -1][step] if step >= 0 else 0
        R(13, 31 + uy + flt, 19, 33 + uy + flt, p['scarf'])
        R(13, 31 + uy + flt, 14, 33 + uy + flt, p['scarf_d'])
        R(19, 30 + uy, 30, 33 + uy, p['scarf'])
        R(19, 33 + uy, 30, 33 + uy, p['scarf_d'])
        # 머리
        ELL(24, 22 + uy, 9.3, 8.4, p['skin'])
        ELL(24, 22 + uy, 9.3, 8.4, p['hair'], cond=lambda x, y: x <= 20)
        ELL(24, 22 + uy, 9.3, 8.4, p['skin_d'], cond=lambda x, y: y >= 29 + uy and x > 20)
        for x in range(21, 30):
            P(x, 18 + uy, p['hair'])
        for x in (21, 22, 24, 25, 27):
            P(x, 19 + uy, p['hair'])
        P(21, 20 + uy, p['hair'])
        # 얼굴
        if expr == 'happy':
            for (ex, ey) in ((26, 22), (27, 21), (28, 21), (29, 22)):
                P(ex, ey + uy, EYE)
        elif expr == 'blink':
            R(27, 22 + uy, 29, 22 + uy, EYE)
        else:
            R(27, 20 + uy, 28, 23 + uy, EYE)
            P(27, 20 + uy, '#ffffff')
        R(28, 25 + uy, 30, 25 + uy, '#ff9f94')
        if expr in ('open', 'happy'):
            R(30, 26 + uy, 31, 27 + uy, '#9a3040')
        else:
            P(31, 26 + uy, '#b04a4a')
        hat(side=True)
        # 앞쪽 팔
        if link == 'f':
            LN(26, 35 + uy, 33, 39, p['coat_l'])
            R(33, 38, 35, 41, p['mit'])
        elif arm == 'point':
            LN(26, 35 + uy, 34, 31 + uy, p['coat_l'])
            R(35, 29 + uy, 37, 32 + uy, p['mit'])
        elif arm == 'up':
            LN(26, 35 + uy, 31, 27 + uy, p['coat_l'])
            R(31, 23 + uy, 34, 26 + uy, p['mit'])
        elif arm is not None:
            sw = [-3, 0, 3, 0][step] if step >= 0 else 0
            LN(24, 35 + uy, 24 + sw, 40 + uy, p['coat_l'])
            R(23 + sw, 41 + uy, 25 + sw, 43 + uy, p['mit'])
    else:
        front = view == 'front'
        # 다리
        for (lx, shade) in ((20, False), (25, True)):
            R(lx, 44 + uy, lx + 3, 48, p['leg_d'] if shade else p['leg'])
            bx0 = lx - 1 if lx == 20 else lx
            R(bx0, 49, bx0 + 4, 51, p['boot'])
            R(bx0, 51, bx0 + 4, 51, p['boot_d'])
        bl, br = (15, 33) if girl else (17, 31)
        for y in range(33, 46):
            v = (y - 33) / 12.0
            vv = v ** 1.6 if girl else v
            l, r = round(18 - (18 - bl) * vv), round(30 + (br - 30) * vv)
            R(l, y + uy, r, y + uy, p['coat'])
            R(r - 1, y + uy, r, y + uy, p['coat_d'])
            P(l, y + uy, p['coat_l'])
        if front:
            R(24, 33 + uy, 24, 45 + uy, p['trim'] or p['coat_d'])
            for by in (37, 41):
                P(26, by + uy, '#ffd76a')
        if p['trim']:
            R(bl, 44 + uy, br, 45 + uy, p['trim'])
            R(br - 2, 45 + uy, br, 45 + uy, '#d9cdbd')
        else:
            R(bl, 45 + uy, br, 45 + uy, p['coat_d'])
        # 목도리
        R(18, 30 + uy, 30, 33 + uy, p['scarf'])
        R(18, 33 + uy, 30, 33 + uy, p['scarf_d'])
        if front:
            R(26, 33 + uy, 29, 38 + uy, p['scarf'])
            R(26, 38 + uy, 29, 38 + uy, p['scarf_d'])
            R(29, 34 + uy, 29, 38 + uy, p['scarf_d'])
        else:
            R(19, 33 + uy, 22, 39 + uy, p['scarf'])
            R(19, 39 + uy, 22, 39 + uy, p['scarf_d'])
        # 머리
        if front:
            ELL(24, 22 + uy, 9.3, 8.4, p['skin'])
            ELL(24, 22 + uy, 9.3, 8.4, p['skin_d'], cond=lambda x, y: y >= 29 + uy)
            R(16, 18 + uy, 32, 18 + uy, p['hair'])
            for x in (16, 17, 19, 20, 22, 26, 27, 29, 31, 32):
                P(x, 19 + uy, p['hair'])
            for x in (15, 16, 32, 33):
                R(x, 19 + uy, x, 23 + uy, p['hair'])
            if girl:
                R(13, 20 + uy, 16, 36 + uy, p['hair'])
                R(32, 20 + uy, 35, 36 + uy, p['hair'])
                R(14, 37 + uy, 15, 37 + uy, p['hair'])
                R(33, 37 + uy, 34, 37 + uy, p['hair'])
                R(14, 24 + uy, 14, 35 + uy, p['hair_d'])
                R(34, 24 + uy, 34, 35 + uy, p['hair_d'])
            # 얼굴
            if expr == 'happy':
                for (ex, ey) in ((18, 22), (19, 21), (20, 21), (21, 22), (27, 22), (28, 21), (29, 21), (30, 22)):
                    P(ex, ey + uy, EYE)
            elif expr == 'blink':
                R(19, 22 + uy, 21, 22 + uy, EYE)
                R(27, 22 + uy, 29, 22 + uy, EYE)
            else:
                R(19, 20 + uy, 20, 23 + uy, EYE)
                R(28, 20 + uy, 29, 23 + uy, EYE)
                P(19, 20 + uy, '#ffffff')
                P(28, 20 + uy, '#ffffff')
            R(17, 25 + uy, 18, 25 + uy, '#ff9f94')
            R(30, 25 + uy, 31, 25 + uy, '#ff9f94')
            if expr in ('open', 'happy'):
                R(23, 26 + uy, 25, 27 + uy, '#9a3040')
                P(24, 27 + uy, '#ff8f9a')
            else:
                P(23, 26 + uy, '#b04a4a')
                P(24, 27 + uy, '#b04a4a')
                P(25, 26 + uy, '#b04a4a')
        else:
            ELL(24, 22 + uy, 9.3, 8.4, p['hair'])
            for x in (18, 22, 26, 30):
                R(x, 20 + uy, x, 27 + uy, p['hair_d'])
            if girl:
                R(15, 22 + uy, 33, 36 + uy, p['hair'])
                R(16, 37 + uy, 32, 37 + uy, p['hair'])
                for x in (18, 22, 26, 30):
                    R(x, 20 + uy, x, 36 + uy, p['hair_d'])
                R(18, 30 + uy, 30, 31 + uy, p['scarf'])
        hat()
        # 팔: lside=화면 왼쪽 팔, rside=화면 오른쪽 팔
        la, ra = arm if isinstance(arm, tuple) else (arm, arm)
        if link == 'r':
            ra = 'link'
        if link == 'l':
            la = 'link'
        for side_, mode in ((-1, la), (1, ra)):
            sx = 24 + side_ * 8        # 어깨 x (16 / 32)
            if mode == 'link':
                LN(sx, 35 + uy, 24 + side_ * 11, 40, p['coat_l'])
                R(24 + side_ * 11 - 1, 40, 24 + side_ * 11 + 1, 43, p['mit'])
            elif mode == 'up':
                LN(sx, 35 + uy, 24 + side_ * 12, 28 + uy, p['coat_l'])
                R(24 + side_ * 12 - 1, 24 + uy, 24 + side_ * 12 + 2, 27 + uy, p['mit'])
            elif mode in ('wave0', 'wave1'):
                ex = 24 + side_ * (13 if mode == 'wave0' else 11)
                ey = 28 if mode == 'wave0' else 26
                LN(sx, 35 + uy, ex, ey + uy, p['coat_l'])
                R(ex - 1, ey - 4 + uy, ex + 2, ey - 1 + uy, p['mit'])
            else:
                R(sx - 1, 34 + uy, sx + 1, 41 + uy, p['coat_l'])
                R(sx - 1 + (0 if side_ < 0 else 0), 34 + uy, sx - 1 if side_ > 0 else sx + 1, 41 + uy, p['coat_d'])
                R(sx - 1, 34 + uy, sx + 1, 41 + uy, p['coat_l']) if False else None
                R(sx - 1, 42 + uy, sx + 1, 44 + uy, p['mit'])
    return outline(im).astype(np.float32)


def make_bubble():
    im = new(13, 16)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([1, 1, 11, 11], radius=3, fill=C('#ffffff'))
    d.polygon([(4, 11), (7, 11), (5, 14)], fill=C('#ffffff'))
    d.rectangle([6, 3, 7, 7], fill=C('#e8404a'))
    d.rectangle([6, 9, 7, 10], fill=C('#e8404a'))
    return outline(im).astype(np.float32)


# ───────────────────────── 제목 ─────────────────────────
def make_title():
    font = ImageFont.truetype(os.path.join(HERE, 'assets', 'Galmuri11-Bold.ttf'), 12)
    lines = [('상윤과 승민의', 30, T_TITLE1), ('크리스마스', 58, T_TITLE2)]
    glyphs = []
    S = 2
    grad = ['#fffdf0', '#fff7d0', '#ffefad', '#ffe58a', '#ffd966', '#ffcb4a', '#ffbb38', '#fda92e', '#f59427', '#ee8322']
    for li, (text, y0, t0) in enumerate(lines):
        total = sum(font.getlength(ch) for ch in text) * S
        x = (W - total) / 2.0
        j = 0
        for ch in text:
            adv = font.getlength(ch)
            if ch != ' ':
                m = Image.new('L', (16, 16), 0)
                dd = ImageDraw.Draw(m)
                dd.fontmode = '1'
                dd.text((1, 1), ch, font=font, fill=255)
                m = np.array(m.resize((32, 32), Image.NEAREST)) > 0
                big = np.zeros((40, 40), bool)
                big[4:36, 4:36] = m
                out = big.copy()
                for _ in range(2):                         # 2px 외곽선 (8방향 팽창)
                    o2 = out.copy()
                    o2[1:] |= out[:-1]
                    o2[:-1] |= out[1:]
                    o2[:, 1:] |= out[:, :-1]
                    o2[:, :-1] |= out[:, 1:]
                    o2[1:, 1:] |= out[:-1, :-1]
                    o2[:-1, :-1] |= out[1:, 1:]
                    o2[1:, :-1] |= out[:-1, 1:]
                    o2[:-1, 1:] |= out[1:, :-1]
                    out = o2
                g = np.zeros((42, 40, 4), np.float32)
                sh = np.zeros((42, 40), bool)
                sh[2:42] = out                              # 그림자 (아래로 2px)
                g[sh] = (20, 10, 38, 210)
                g[:40][out] = (58, 22, 44, 255)
                ys, xs = np.nonzero(big)
                rows = ys - ys.min()
                span = max(1, rows.max())
                for yy, xx, r in zip(ys, xs, rows):
                    c = C(grad[min(len(grad) - 1, int(r / float(span + 1) * len(grad)))])
                    g[yy, xx] = c
                glyphs.append(dict(img=g, fill=big, x=int(round(x)) - 4 - 2, y=y0 - 4 - 2, t=t0 + j * 0.05, line=li))
                j += 1
            x += adv * S
    return glyphs


# ───────────────────────── 타임라인 ─────────────────────────
def build_cam():
    keys = [(-1.0, -8.0), (0.0, 0.0), (4.6, 37.0), (6.0, 47.0), (8.75, 117.0), (9.6, 122.0), (11.5, 220.0), (30.0, 220.0)]
    ts = np.arange(-1.0, DUR + 1.0, 1.0 / 120)
    v = np.interp(ts, [k[0] for k in keys], [k[1] for k in keys])
    n = 72                                  # 0.6초 박스 필터 → 부드러운 가감속
    v = np.convolve(np.pad(v, n, mode='edge'), np.ones(n) / n, mode='same')[n:-n]
    return ts, v


CAM_T, CAM_V = build_cam()


def cam_x(t):
    return float(np.clip(np.interp(t, CAM_T, CAM_V), 0, WW - W))


def pair_x(t):
    if t < T_STOP:
        return 36.0 * (t - T_WALK0)
    x_stop = 36.0 * (T_STOP - T_WALK0)
    if t < T_GO:
        return x_stop
    if t < T_ARRIVE:
        return x_stop + (TREE_CX - x_stop) * (t - T_GO) / (T_ARRIVE - T_GO)
    return float(TREE_CX)


def hop(t, t0, dur=0.36, h=7):
    p = (t - t0) / dur
    if 0 <= p <= 1:
        return -h * math.sin(math.pi * p)
    return 0.0


# ───────────────────────── 렌더러 ─────────────────────────
class Renderer:
    def __init__(self):
        self.sky = make_sky()
        self.stars = make_stars()
        self.moon = make_moon()
        self.clouds = make_clouds()
        self.mount = make_mountains()
        self.forest = make_forest()
        self.hills = make_hills()
        wd = build_world()
        self.main = npa(wd.img)
        self.emis = np.array(wd.emis).astype(np.float32) / 255.0
        self.Lstatic = wd.L
        self.bulbs = wd.bulbs
        self.smokes = wd.smokes
        self.tree, self.tbulbs = make_tree()
        self.bubble = make_bubble()
        self.title = make_title()
        # 눈송이: (x0, y0, 속도, 흔들림 폭, 주파수, 위상, 크기)
        self.snow = []
        for layer, (n, v0, v1, par) in enumerate(((80, 13, 20, 0.3), (75, 24, 34, 0.7), (42, 40, 56, 1.2))):
            fl = []
            for i in range(n):
                fl.append((hsh(i, 1, 60 + layer) * (W + 16), hsh(i, 2, 60 + layer) * (H + 16),
                           v0 + hsh(i, 3, 60 + layer) * (v1 - v0), 2 + hsh(i, 4, 60 + layer) * (3 + 3 * layer),
                           0.6 + hsh(i, 5, 60 + layer) * 1.4, hsh(i, 6, 60 + layer) * 6.28, hsh(i, 7, 60 + layer)))
            self.snow.append((fl, par))
        # 발자국
        self.steps = []
        k = 0
        t = T_WALK0
        while t < T_ARRIVE:
            if not (T_STOP <= t < T_GO):
                self.steps.append((t, pair_x(t) - 12 + (k % 2) * 2, FEET_Y + (k % 2) * 2))
                self.steps.append((t + 0.06, pair_x(t) + 12 + (k % 2) * 2, FEET_Y + 1 - (k % 2) * 2))
            t += 0.25 if t < T_STOP else 0.18
            k += 1

    # ---- 눈 ----
    def draw_snow(self, f, t, cam, layer, sat=None):
        fl, par = self.snow[layer]
        for (x0, y0, v, amp, fr, ph, sz) in fl:
            x = (x0 + amp * math.sin(fr * t + ph) - 5.0 * t * (0.5 + layer * 0.4) - cam * par) % (W + 16) - 8
            y = (y0 + v * t) % (H + 16) - 8
            xi, yi = int(x), int(y)
            if layer == 0:
                if 0 <= xi < W and 0 <= yi < H:
                    f[yi, xi] = f[yi, xi] * 0.35 + np.array((176, 186, 235), np.float32) * 0.65
                continue
            col = np.array((232, 238, 255), np.float32) if layer == 1 else np.array((255, 255, 255), np.float32)
            if sat is not None and 0 <= xi < W and 0 <= yi < H:
                s = sat[yi, xi]
                col = col * (1 - 0.5 * s) + np.array((255, 236, 190), np.float32) * 0.5 * s
            pts = [(0, 0, 1.0)]
            if layer == 1 and sz > 0.6:
                pts += [(1, 0, 0.9)]
            if layer == 2:
                if sz > 0.5:
                    pts += [(1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0), (-1, 0, 0.45), (2, 1, 0.45), (0, -1, 0.45), (1, 2, 0.45)]
                else:
                    pts += [(1, 0, 0.5), (-1, 0, 0.5), (0, 1, 0.5), (0, -1, 0.5)]
            for (ax, ay, a) in pts:
                xx, yy = xi + ax, yi + ay
                if 0 <= xx < W and 0 <= yy < H:
                    f[yy, xx] = f[yy, xx] * (1 - a) + col * a

    # ---- 캐릭터 ----
    def draw_chars(self, alb, cmask, t, cam):
        if t < 3.0:
            return
        px_ = pair_x(t)
        sx = int(round(px_)) - int(round(cam))
        if sx < -40:
            return
        A, B = SANGYOON, SEUNGMIN          # A: 화면 왼쪽(뒤따름), B: 화면 오른쪽(앞장섬)
        ya = yb = 0.0
        if t < T_STOP:
            st = int((t - T_WALK0) * 8) % 4
            sa = char_sprite(A, 'side', st, 'smile', None, 'f')
            sb = char_sprite(B, 'side', (st + 2) % 4, 'smile', 'swing', 'b')
            gap = 12
        elif t < T_GO:
            yb = hop(t, T_STOP, 0.34, 8) + hop(t, 9.3, 0.26, 4)
            ya = hop(t, 9.02, 0.3, 5) + hop(t, 9.36, 0.24, 4)
            eb = 'open'
            ea = 'smile' if t < 8.98 else 'open'
            sb = char_sprite(B, 'side', -1, eb, 'point' if t > 9.0 else 'up', 'b')
            sa = char_sprite(A, 'side', -1, ea, None, 'f')
            gap = 12
        elif t < T_ARRIVE:
            st = int((t - T_GO) * 11) % 4
            sa = char_sprite(A, 'side', st, 'happy', None, 'f')
            sb = char_sprite(B, 'side', (st + 2) % 4, 'happy', 'swing', 'b')
            gap = 12
        elif t < T_FRONT:
            hb = hop(t, T_STAR + 0.02, 0.36, 8) + hop(t, T_STAR + 0.46, 0.36, 7) + hop(t, 13.7, 0.3, 4)
            ha = hop(t, T_STAR + 0.10, 0.36, 8) + hop(t, T_STAR + 0.54, 0.36, 7) + hop(t, 13.85, 0.3, 4)
            ya, yb = ha, hb
            bob = 1 if (t * 1.3) % 1.0 < 0.5 and t > 14.0 else 0
            sa = char_sprite(A, 'back', -1, 'smile', ('up' if ha < -1 else 'down', 'down'), 'r', bob)
            sb = char_sprite(B, 'back', -1, 'smile', ('down', 'up' if hb < -1 else 'down'), 'l', bob)
            gap = 12.5
        else:
            ya = hop(t, T_FRONT, 0.34, 7) + hop(t, 17.5, 0.3, 4)
            yb = hop(t, T_FRONT + 0.06, 0.34, 7) + hop(t, 17.56, 0.3, 4)
            blink = (16.9 < t < 17.02) or (18.7 < t < 18.82)
            ea = 'blink' if blink else ('happy' if (t < 15.5 or 17.5 < t < 18.2) else 'smile')
            eb = 'blink' if blink else ('happy' if (t < 15.5 or 17.5 < t < 18.3) else 'smile')
            wv = 'wave0' if int(t * 5) % 2 == 0 else 'wave1'
            arm_a = (wv if (16.6 < t < 17.9 or t > 18.9) else 'down', 'down')
            arm_b = ('down', wv if (15.4 < t < 16.5 or 18.0 < t < 19.4) else 'down')
            bob = 1 if (t * 1.2) % 1.0 < 0.5 else 0
            sa = char_sprite(A, 'front', -1, ea, arm_a, 'r', bob if ya == 0 else 0)
            sb = char_sprite(B, 'front', -1, eb, arm_b, 'l', bob if yb == 0 else 0)
            gap = 12.5
        xa, xb = sx - int(gap), sx + int(math.ceil(gap))
        # 바닥 그림자
        for (xc, yo) in ((xa, ya), (xb, yb)):
            w = 8 - int(min(3, -yo / 3))
            x0, x1 = max(0, xc - w), min(W, xc + w + 1)
            if x0 < x1:
                alb[FEET_Y:FEET_Y + 2, x0:x1] *= np.array((0.74, 0.78, 0.90), np.float32)
        blit(alb, sa, xa - 24, FEET_Y - 51 + int(round(ya)), cmask)
        blit(alb, sb, xb - 24, FEET_Y - 51 + int(round(yb)), cmask)
        self.char_pos = (xa, xb, int(round(ya)), int(round(yb)))

    def frame(self, t):
        cam = cam_x(t)
        ci = int(round(cam))
        f = self.sky.copy()
        # 별
        for (x, y, br, sp, ph, big) in self.stars:
            tw = 0.55 + 0.45 * math.sin(t * sp + ph)
            a = br * tw
            f[y, x] = f[y, x] * (1 - a) + np.array((255, 250, 225), np.float32) * a
            if big and tw > 0.6:
                for (ax, ay) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if 0 <= x + ax < W and 0 <= y + ay < H:
                        f[y + ay, x + ax] = f[y + ay, x + ax] * 0.6 + np.array((200, 205, 255), np.float32) * 0.4
        # 별똥별
        if 2.3 < t < 3.0:
            u = (t - 2.3) / 0.7
            for k in range(12):
                xx, yy = int(176 - 78 * u + k * 1.6), int(26 + 36 * u - k * 0.74)
                if 0 <= xx < W and 0 <= yy < H:
                    a = (1 - k / 12.0) * math.sin(math.pi * u) ** 0.5
                    f[yy, xx] = f[yy, xx] * (1 - a) + np.array((255, 252, 225), np.float32) * a
        # 달 + 구름
        mx, my = int(round(52 - cam * 0.05)), 116
        halo = np.zeros((H, W, 3), np.float32)
        glow(halo, mx, my, 44, (0.75, 0.78, 1.0), 0.30)
        glow(halo, mx, my, 22, (1.0, 0.95, 0.75), 0.25)
        f += halo * 255
        blit(f, self.moon, mx - 14, my - 14)
        cofs = int(cam * 0.04 + t * 1.6) % 360
        cl = np.concatenate([self.clouds[:, cofs:], self.clouds[:, :cofs]], axis=1)[:, :W]
        blit(f, cl, 0, 0)
        # 원경
        blit(f, self.mount[:, int(cam * 0.08):int(cam * 0.08) + W], 0, 0)
        self.draw_snow(f, t, cam, 0)
        blit(f, self.forest[:, int(cam * 0.2):int(cam * 0.2) + W], 0, 0)
        blit(f, self.hills[:, int(cam * 0.45):int(cam * 0.45) + W], 0, 0)

        # ── 메인 레이어: 알베도 합성 ──
        main = self.main[:, ci:ci + W]
        alb = main[..., :3].copy()
        alpha = main[..., 3:4] / 255.0
        amask = np.zeros((H, W), np.float32)
        em = self.emis[:, ci:ci + W].copy()
        L = self.Lstatic[:, ci:ci + W].copy()
        L *= 1.0 + 0.03 * math.sin(t * 7.0)
        # 발자국
        for (ts, wx, wy) in self.steps:
            if ts <= t:
                x = int(round(wx)) - ci
                if 0 <= x < W - 2:
                    alb[int(wy), x:x + 3] = (170, 184, 232)
                    alb[int(wy) + 1, x:x + 3] = (238, 243, 255)
        # 트리
        p_light = clamp((t - T_LIGHT0) / (T_STAR - T_LIGHT0))
        star_on = t >= T_STAR
        tmask = np.zeros((H, W), np.float32)
        blit(alb, self.tree, TREE_X0 - ci, TREE_Y0, tmask)
        amask = np.maximum(amask, tmask)
        if p_light > 0:
            y_th = 325 - p_light * 232
            rows = np.clip((np.arange(H) - y_th) / 22.0, 0, 1)[:, None]
            pulse = 1.0 + (0.05 * math.sin(t * 5.0) if star_on else 0)
            rows = rows * (np.arange(H) < 304)[:, None]      # 잎 부분만 스스로 빛남
            L += (tmask * rows * pulse)[..., None] * np.array((0.46, 0.44, 0.27), np.float32)
            kbig = (0.34 + 0.26 * (1 if star_on else 0)) * smooth(p_light)
            glow(L, TREE_CX - ci, 232, 150, GOLD, kbig * pulse, ry=170)
            glow(L, TREE_CX - ci, 344, 190, GOLD, (0.85 + 0.45 * star_on) * smooth(p_light), ry=56)
            halo_k = (0.22 + 0.20 * star_on) * smooth(p_light) * pulse
            halo = np.zeros((H, W, 3), np.float32)
            glow(halo, TREE_CX - ci, 205, 175, (1.0, 0.72, 0.36), halo_k, ry=200)
            f += halo * 255                                  # 트리 뒤 하늘로 번지는 금빛
        # 캐릭터
        cmask = np.zeros((H, W), np.float32)
        self.char_pos = None
        self.draw_chars(alb, cmask, t, cam)
        L += cmask[..., None] * np.array((0.34, 0.31, 0.26), np.float32)
        amask = np.maximum(amask, cmask)
        # 트리 전구 → 빛
        on_bulbs = []
        for (bx, by, col, ph, s) in self.tbulbs:
            if p_light > 0 and s <= p_light * 1.02:
                age = t - (T_LIGHT0 + s * (T_STAR - T_LIGHT0))
                b = 0.70 + 0.30 * math.sin(t * 5.0 + ph)
                if age < 0.18:
                    b = 1.6
                on_bulbs.append((bx - ci, by, col, b, ph))
                glow(L, bx - ci, by, 8, [c / 255.0 for c in col], 0.42 * b)
        if star_on:
            sp = 1.0 + 0.12 * math.sin(t * 6.0)
            glow(L, TREE_CX - ci, STAR_Y, 34, (1.0, 0.9, 0.55), 0.9 * sp)

        # ── 조명 적용 ──
        amb = np.array((0.50, 0.57, 0.88), np.float32)
        s = np.clip(L.max(axis=2), 0, 1)
        light = amb[None, None, :] * (1 - 0.88 * s[..., None]) + L * 1.08
        # 캐릭터는 색이 날아가지 않게 조명 색을 절반쯤 중화
        lc = light * 0.5 + light.mean(axis=2, keepdims=True) * 0.55
        light = light * (1 - cmask[..., None]) + np.minimum(lc, 1.12) * cmask[..., None]
        light = np.minimum(light, 1.25)
        light = light * (1 - em[..., None]) + em[..., None]
        lit = np.clip(alb * light, 0, 255)
        a_main = np.maximum(alpha[..., 0], amask)[..., None]
        f = f * (1 - a_main) + lit * a_main
        f += L * 255 * 0.15                                  # 공기 중 번짐

        # 눈밭의 반짝임
        for j in range(60):
            wx = int(hsh(j, 1, 95) * WW)
            wy = 322 + int(hsh(j, 2, 95) * 60)
            x = wx - ci
            if 0 <= x < W and (t * 0.7 + hsh(j, 3, 95) * 3) % 3.0 < 0.22 and cmask[wy, x] == 0:
                f[wy, x] = (255, 255, 255)
                for (ax, ay) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if 0 <= x + ax < W and wy + ay < H:
                        f[wy + ay, x + ax] = f[wy + ay, x + ax] * 0.5 + 127

        # ── 발광 요소 (조명 이후) ──
        # 굴뚝 연기
        for si, (wx, wy) in enumerate(self.smokes):
            for k in range(6):
                age = (t * 0.55 + k / 6.0 + si * 0.37) % 1.0
                x = wx - ci + 5 * math.sin(age * 5 + k + si) + age * 8
                y = wy - age * 26
                r = 1.5 + age * 3.5
                a = 0.5 * (1 - age) ** 1.3
                x0, x1, y0, y1 = int(x - r), int(x + r) + 1, int(y - r), int(y + r) + 1
                for yy in range(max(0, y0), min(H, y1)):
                    for xx in range(max(0, x0), min(W, x1)):
                        if (xx - x) ** 2 + (yy - y) ** 2 <= r * r:
                            f[yy, xx] = f[yy, xx] * (1 - a) + np.array((196, 200, 236), np.float32) * a
        # 처마/줄 전구
        for (bx, by, col, ph) in self.bulbs:
            x = bx - ci
            if 0 <= x < W:
                b = 0.5 + 0.5 * math.sin(t * 3.2 + ph)
                c = np.array(col, np.float32)
                if b > 0.35:
                    f[by, x] = c * 0.6 + 255 * 0.4
                    if b > 0.8:
                        for (ax, ay) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            if 0 <= x + ax < W:
                                f[by + ay, x + ax] = f[by + ay, x + ax] * 0.55 + c * 0.45
                else:
                    f[by, x] = c * 0.45
        # 트리 전구
        for (bx, by, col, ph, s_) in self.tbulbs:
            x = bx - ci
            if not (1 <= x < W - 2):
                continue
            c = np.array(col, np.float32)
            lit_b = p_light > 0 and s_ <= p_light * 1.02
            if not lit_b:
                f[by:by + 2, x:x + 2] = c * 0.22 + np.array((18, 24, 48), np.float32)
                continue
            b = 0.70 + 0.30 * math.sin(t * 5.0 + ph)
            age = t - (T_LIGHT0 + s_ * (T_STAR - T_LIGHT0))
            core = c * 0.35 + 255 * 0.65
            f[by:by + 2, x:x + 2] = core
            ring = ((-1, 0), (-1, 1), (2, 0), (2, 1), (0, -1), (1, -1), (0, 2), (1, 2))
            for (ax, ay) in ring:
                xx, yy = x + ax, by + ay
                if 0 <= xx < W and 0 <= yy < H:
                    k = 0.75 * b
                    f[yy, xx] = f[yy, xx] * (1 - k) + c * k
            tw = (t * 0.9 + ph) % 2.2
            if age < 0.18 or tw < 0.16:                        # 반짝 십자 섬광
                for d_ in range(1, 4):
                    for (ax, ay) in ((d_ + 1, 0), (-d_, 0), (0, -d_), (0, d_ + 1)):
                        xx, yy = x + ax, by + ay
                        if 0 <= xx < W and 0 <= yy < H:
                            k = 0.9 - 0.25 * d_
                            f[yy, xx] = f[yy, xx] * (1 - k) + 255 * k
        # 트리 꼭대기 별
        self.draw_star(f, TREE_CX - ci, STAR_Y, t, star_on)

        # 느낌표 말풍선
        if T_STOP <= t < 9.55 and self.char_pos:
            xa, xb, ya, yb = self.char_pos
            pop = clamp((t - T_STOP) / 0.12)
            by = FEET_Y - 51 - 12 + yb - int(4 * ease_out_back(pop)) + 6
            blit(f, self.bubble, xb - 6 + 2, by)

        # 전경 눈
        sat = np.clip(L.max(axis=2), 0, 1)
        self.draw_snow(f, t, cam, 1, sat)
        self.draw_sparkles(f, t, ci)
        self.draw_snow(f, t, cam, 2, sat)

        # 별 점등 순간의 섬광
        if star_on and t < T_STAR + 0.4:
            k = (1 - (t - T_STAR) / 0.4) ** 2
            f = f + np.array((110, 88, 44), np.float32) * k

        self.draw_title(f, t)

        # 페이드 인 (8단계)
        if t < 0.9:
            k = math.floor(smooth(t / 0.9) * 8) / 8.0
            f *= k
        return self.quantize(f)

    def draw_star(self, f, x, y, t, on):
        im = new(21, 21)
        d = ImageDraw.Draw(im)
        pts = []
        for i in range(10):
            r = 8.6 if i % 2 == 0 else 3.8
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((10 + r * math.cos(a), 10 + r * math.sin(a)))
        if on:
            d.polygon(pts, fill=C('#ffe36a'), outline=C('#e09a22'))
            d.ellipse([7, 7, 13, 13], fill=C('#fffbe0'))
            d.ellipse([8, 8, 11, 11], fill=C('#ffffff'))
        else:
            d.polygon(pts, fill=C('#5e5a66'), outline=C('#2c2a44'))
            d.point([(9, 8), (10, 8)], fill=C('#7c788a'))
        blit(f, npa(im), x - 10, y - 10)
        if on:
            ln = 13 + 4 * math.sin(t * 6.0) + max(0, 14 * (1 - (t - T_STAR) / 0.5))
            for d_ in range(9, int(9 + ln)):
                k = max(0.0, 1 - (d_ - 9) / ln) * 0.95
                for (ax, ay) in ((d_, 0), (-d_, 0), (0, -d_), (0, d_)):
                    xx, yy = x + ax, y + ay
                    if 0 <= xx < W and 0 <= yy < H:
                        f[yy, xx] = f[yy, xx] * (1 - k) + np.array((255, 248, 200), np.float32) * k
            for d_ in range(6, int(6 + ln * 0.45)):
                k = max(0.0, 1 - (d_ - 6) / (ln * 0.45)) * 0.7
                for (ax, ay) in ((d_, d_), (-d_, d_), (d_, -d_), (-d_, -d_)):
                    xx, yy = x + ax, y + ay
                    if 0 <= xx < W and 0 <= yy < H:
                        f[yy, xx] = f[yy, xx] * (1 - k) + np.array((255, 232, 150), np.float32) * k

    def spark(self, f, x, y, size, col, k=1.0):
        col = np.array(col, np.float32)
        pts = [(0, 0, 1.0)]
        if size >= 2:
            pts += [(1, 0, .8), (-1, 0, .8), (0, 1, .8), (0, -1, .8)]
        if size >= 3:
            pts += [(2, 0, .45), (-2, 0, .45), (0, 2, .45), (0, -2, .45)]
        for (ax, ay, a) in pts:
            xx, yy = int(x) + ax, int(y) + ay
            if 0 <= xx < W and 0 <= yy < H:
                a *= k
                c = np.array((255, 255, 255), np.float32) if (ax, ay) == (0, 0) and size >= 2 else col
                f[yy, xx] = f[yy, xx] * (1 - a) + c * a

    def draw_sparkles(self, f, t, ci):
        if t < T_STAR:
            return
        cx = TREE_CX - ci
        dt = t - T_STAR
        # 별에서 터져 나오는 금가루
        if dt < 1.6:
            for i in range(34):
                a = hsh(i, 1, 80) * 6.28
                sp = 26 + hsh(i, 2, 80) * 60
                x = cx + math.cos(a) * sp * dt * (1 - 0.25 * dt)
                y = STAR_Y + math.sin(a) * sp * dt * (1 - 0.25 * dt) + 34 * dt * dt
                life = 0.8 + hsh(i, 3, 80) * 0.8
                if dt < life:
                    k = 1 - dt / life
                    self.spark(f, x, y, 2 if k > 0.5 else 1, (255, 226, 120), min(1.0, k * 1.5))
        # 트리 주변에서 깜빡이는 반짝임
        fade = clamp(dt / 0.6)
        for j in range(46):
            per = 0.9 + hsh(j, 1, 81) * 1.1
            cyc = (t + hsh(j, 2, 81) * per) / per
            n, fr = int(cyc), cyc - int(cyc)
            if fr > 0.5:
                continue
            v = hsh(j, n, 82)
            y = 104 + v * 215
            hw = 12 + (y - 104) * 0.36
            x = cx + (hsh(j, n, 83) * 2 - 1) * hw
            size = [1, 2, 3, 2, 1][min(4, int(fr / 0.5 * 5))]
            col = (255, 240, 170) if j % 3 else (255, 255, 255)
            self.spark(f, x, y, size, col, fade)
        # 위로 떠오르는 따뜻한 빛 알갱이
        for j in range(30):
            sp = 7 + hsh(j, 1, 84) * 9
            y0 = hsh(j, 2, 84) * 260
            y = 345 - ((y0 + sp * dt) % 260)
            x = cx + (hsh(j, 3, 84) * 2 - 1) * 96 + 5 * math.sin(t * 1.3 + j)
            k = fade * (0.45 + 0.4 * math.sin(t * 3 + j * 1.7)) * clamp((345 - y) / 30) * clamp((y - 85) / 40)
            if k > 0:
                self.spark(f, x, y, 1, (255, 222, 140), k)

    def draw_title(self, f, t):
        if t < T_TITLE1:
            return
        for gi, g in enumerate(self.title):
            p = (t - g['t']) / 0.36
            if p <= 0:
                continue
            e = ease_out_back(min(1.0, p))
            yo = int(round(-14 * (1 - e)))
            img = g['img']
            if p < 1:
                img = img.copy()
                img[..., 3] *= clamp(p * 3)
            # 광택이 훑고 지나가는 효과
            for (ts, te) in ((17.25, 17.95), (18.95, 19.65)):
                if ts < t < te:
                    xb = -20 + (t - ts) / (te - ts) * (W + 40)
                    ys, xs = np.nonzero(g['fill'])
                    sel = np.abs((xs + g['x']) + (ys + g['y'] - 55) * 0.5 - xb) < 5
                    if sel.any():
                        img = img.copy()
                        img[ys[sel], xs[sel], :3] = (255, 255, 255)
            blit(f, img, g['x'], g['y'] + yo)
            if 0.85 < p < 1.6:                               # 착지 반짝임
                k = 1 - abs(p - 1.2) / 0.4
                self.spark(f, g['x'] + 30, g['y'] + 8, 3, (255, 246, 190), clamp(k))
        # 제목 주변 반짝임
        if t > T_TITLE2 + 0.5:
            for j in range(9):
                per = 1.1 + hsh(j, 1, 90) * 0.9
                cyc = (t + hsh(j, 2, 90) * per) / per
                n, fr = int(cyc), cyc - int(cyc)
                if fr > 0.45:
                    continue
                x = 22 + hsh(j, n, 91) * (W - 44)
                y = 24 + hsh(j, n, 92) * 62
                size = [1, 2, 3, 2, 1][min(4, int(fr / 0.45 * 5))]
                self.spark(f, x, y, size, (255, 244, 190))

    @staticmethod
    def quantize(f):
        """SNES풍 15비트 컬러 + 4x4 오더드 디더링."""
        v = np.clip(f, 0, 255) / 255.0 * 31.0
        q = np.floor(v + (BAYER_FULL[..., None] - 0.5) * 0.9 + 0.5)
        return np.clip(q * (255.0 / 31.0), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', help='쉼표로 구분한 초 목록')
    ap.add_argument('--out', default='video.mp4')
    ap.add_argument('--audio')
    ap.add_argument('--sprites', action='store_true')
    args = ap.parse_args()
    r = Renderer()
    if args.sprites:
        sheet = Image.new('RGBA', (48 * 8, 56 * 2), (90, 100, 160, 255))
        for row, who in enumerate(('boy', 'girl')):
            specs = [('side', 0, 'smile', 'swing', None), ('side', 1, 'smile', 'swing', 'b'), ('side', 2, 'happy', None, 'f'),
                     ('side', -1, 'open', 'point', 'b'), ('back', -1, 'smile', ('up', 'down'), 'r'),
                     ('front', -1, 'smile', 'down', 'r'), ('front', -1, 'happy', ('wave0', 'down'), 'l'),
                     ('front', -1, 'blink', ('down', 'wave1'), None)]
            for i, (v, st, ex, arm, ln) in enumerate(specs):
                s = Image.fromarray(char_sprite(who, v, st, ex, arm, ln).astype(np.uint8))
                sheet.alpha_composite(s, (i * 48, row * 56))
        sheet.resize((sheet.width * 4, sheet.height * 4), Image.NEAREST).save(args.out)
        return
    if args.preview:
        ts = [float(x) for x in args.preview.split(',')]
        k = 2 if len(ts) > 2 else 3
        sheet = Image.new('RGB', (len(ts) * (W * k + 4), H * k), (0, 0, 0))
        for i, t in enumerate(ts):
            im = Image.fromarray(r.frame(t)).resize((W * k, H * k), Image.NEAREST)
            sheet.paste(im, (i * (W * k + 4), 0))
        sheet.save(args.out)
        return
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-']
    if args.audio:
        cmd += ['-i', args.audio]
    cmd += ['-vf', f'scale={W * SCALE}:{H * SCALE}:flags=neighbor:out_color_matrix=bt709,format=yuv420p',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-colorspace', 'bt709', '-color_primaries', 'bt709',
            '-color_trc', 'bt709', '-r', str(FPS)]
    if args.audio:
        cmd += ['-c:a', 'aac', '-b:a', '192k', '-t', str(DUR)]
    cmd += ['-movflags', '+faststart', args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NF):
        proc.stdin.write(r.frame(i / FPS).tobytes())
        if i % 60 == 0:
            print(f'frame {i}/{NF}', flush=True)
    proc.stdin.close()
    proc.wait()
    print('done', args.out)


if __name__ == '__main__':
    main()
