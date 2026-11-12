#!/usr/bin/env python3
"""'상윤과 승민의 크리스마스' — 오리지널 오르골 + 벨 음악 (20초, 96BPM, 8마디).

순수 numpy 합성. 출력: music.wav (44.1kHz stereo 16bit)
"""
import sys
import wave
import numpy as np

SR = 44100
DUR = 20.0
BPM = 96.0
BEAT = 60.0 / BPM            # 0.625s
N = int(SR * DUR)
TRANSPOSE = 2                # C장조로 적고 D장조로 연주
rng = np.random.default_rng(1225)

NOTE_IDX = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def freq(name):
    n = NOTE_IDX[name[0]]
    octv = int(name[-1])
    if len(name) == 3:
        n += 1 if name[1] == '#' else -1
    midi = 12 * (octv + 1) + n + TRANSPOSE
    return 440.0 * 2 ** ((midi - 69) / 12)


def add(buf, start_s, sig, pan=0.0, gain=1.0):
    """sig(mono)를 등전력 패닝으로 스테레오 버퍼에 더한다."""
    i0 = int(start_s * SR)
    if i0 >= N or i0 < 0:
        return
    n = min(len(sig), N - i0)
    a = (pan + 1) * np.pi / 4
    buf[i0:i0 + n, 0] += sig[:n] * np.cos(a) * gain
    buf[i0:i0 + n, 1] += sig[:n] * np.sin(a) * gain


def music_box(f, dur=2.2, vel=1.0):
    """오르골 빗살: 기음 + 약한 배음 + 비조화 부분음(6.27f) + 짧은 타격음."""
    t = np.arange(int(dur * SR)) / SR
    tau = 0.9 * (880.0 / f) ** 0.35
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    y += 0.30 * np.sin(2 * np.pi * 2 * f * t + 0.3) * np.exp(-t / (tau * 0.45))
    y += 0.10 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / (tau * 0.25))
    if 6.27 * f < 15000:
        y += 0.22 * np.sin(2 * np.pi * 6.27 * f * t) * np.exp(-t / 0.045)
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.0025) * 0.06
    y += click
    y *= np.minimum(1.0, t / 0.0015)
    return y * vel


def glock(f, dur=3.0, vel=1.0):
    """글로켄슈필/차임: 자유단 막대의 비조화 부분음."""
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for ratio, amp, tau in ((1.0, 1.0, 1.3), (2.756, 0.45, 0.55), (5.404, 0.22, 0.22), (8.933, 0.10, 0.10)):
        if ratio * f < 16000:
            y += amp * np.sin(2 * np.pi * ratio * f * t) * np.exp(-t / tau)
    y *= np.minimum(1.0, t / 0.001)
    return y * vel


def sleigh(vel=1.0, dur=0.28):
    """썰매 방울: 여러 개의 높은 금속성 공명 + 잘게 흔들리는 엔벨로프."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for _ in range(9):
        f = rng.uniform(3200, 9500)
        ph = rng.uniform(0, 6.28)
        y += np.sin(2 * np.pi * f * t + ph) * rng.uniform(0.4, 1.0)
    env = np.zeros(n)
    for k in range(3):                       # 짤랑짤랑 2~3번 흔들림
        t0 = k * rng.uniform(0.022, 0.034)
        tt = np.clip(t - t0, 0, None)
        env += (t >= t0) * np.exp(-tt / 0.045) * (1.0 - 0.25 * k)
    noise = rng.standard_normal(n)
    noise = noise - np.concatenate(([0], noise[:-1]))      # 하이패스 느낌
    y = (y / 9.0 + 0.10 * noise) * env
    return y * vel


def pad(f, dur, vel=1.0):
    """포근한 패드: 디튠된 사인 + 느린 어택/릴리즈."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for det in (-0.004, 0.0, 0.004):
        y += np.sin(2 * np.pi * f * (1 + det) * t + rng.uniform(0, 6.28))
        y += 0.25 * np.sin(2 * np.pi * 2 * f * (1 + det) * t + rng.uniform(0, 6.28))
    env = np.minimum(1.0, t / 0.5) * np.minimum(1.0, (dur - t) / 0.6)
    return y / 3.0 * env * vel


dry = np.zeros((N, 2))
wet_send = np.zeros((N, 2))


def play(inst, beat, name, vel=1.0, pan=0.0, send=0.35, **kw):
    t = beat * BEAT + rng.uniform(-0.004, 0.004)
    sig = inst(freq(name), vel=vel * rng.uniform(0.92, 1.05), **kw)
    add(dry, max(0.0, t), sig, pan)
    add(wet_send, max(0.0, t), sig, pan, gain=send)


# ───────────────────────── 멜로디 (오르골) ─────────────────────────
melody = [
    # 1마디 C — 고요한 마을
    (0.0, 'E5', .55), (0.5, 'G5', .55), (1.0, 'C6', .7), (2.0, 'E6', .7), (2.5, 'D6', .6), (3.0, 'C6', .65), (3.5, 'G5', .55),
    # 2마디 F–G
    (4.0, 'A5', .6), (4.5, 'C6', .6), (5.0, 'F6', .75), (6.0, 'E6', .7), (6.5, 'D6', .65), (7.0, 'B5', .6), (7.5, 'D6', .65),
    # 3마디 C–Am — 손잡고 걷는 테마 (통통 튀는 리듬)
    (8.0, 'E6', .8), (8.75, 'E6', .55), (9.0, 'G6', .8), (9.5, 'E6', .7), (10.0, 'C6', .8), (10.75, 'C6', .55), (11.0, 'E6', .75), (11.5, 'A5', .7),
    # 4마디 Dm–G — 14박에 "!" (트리 발견)
    (12.0, 'D6', .8), (12.75, 'D6', .55), (13.0, 'F6', .8), (13.5, 'D6', .7), (15.0, 'D6', .7), (15.5, 'G6', .8),
    # 5마디 F–G — 불빛이 타고 올라가는 상행
    (16.0, 'A5', .75), (16.5, 'C6', .75), (17.0, 'F6', .8),
    (17.5, 'A5', .6), (17.75, 'C6', .62), (18.0, 'D6', .66), (18.25, 'E6', .7), (18.5, 'G6', .74),
    (18.75, 'A6', .78), (19.0, 'B6', .82), (19.25, 'D7', .86), (19.5, 'B6', .88), (19.75, 'D7', .92),
    # 6마디 C–Am — 별 점등!
    (20.0, 'C7', 1.0), (21.0, 'G6', .8), (21.5, 'E6', .75), (22.0, 'A6', .9), (23.0, 'G6', .8), (23.5, 'E6', .75),
    # 7마디 F–G — 제목 등장
    (24.0, 'F6', .8), (24.5, 'A6', .85), (25.0, 'C7', .9), (25.5, 'A6', .85), (26.0, 'G6', .8), (26.5, 'B6', .8), (27.0, 'D7', .85), (27.5, 'B6', .75),
    # 8마디 C — 마무리
    (28.0, 'C7', .95), (29.0, 'G6', .7), (29.5, 'E6', .65),
    (30.0, 'C6', .6), (30.25, 'E6', .62), (30.5, 'G6', .66), (30.75, 'C7', .7), (31.0, 'E7', .8),
]
for b, n, v in melody:
    play(music_box, b, n, vel=0.50 * v, pan=0.12, send=0.40)

# ───────────────────────── 반주 아르페지오 (낮은 오르골) ─────────────────────────
ARP = {
    'C': ['C4', 'G4', 'E5', 'G4'], 'F': ['F4', 'C5', 'A4', 'C5'], 'G': ['G4', 'D5', 'B4', 'D5'],
    'Am': ['A3', 'E4', 'C5', 'E4'], 'Dm': ['D4', 'A4', 'F4', 'A4'],
}
chords = [  # (시작 박, 코드) — 반 마디 단위
    (0, 'C'), (2, 'C'), (4, 'F'), (6, 'G'),
    (8, 'C'), (10, 'Am'), (12, 'Dm'), (14, 'G'),
    (16, 'F'), (18, 'G'), (20, 'C'), (22, 'Am'),
    (24, 'F'), (26, 'G'), (28, 'C'),
]
for b0, ch in chords:
    notes = ARP[ch]
    if b0 < 8:                      # 인트로는 성기게
        play(music_box, b0, notes[0], vel=0.26, pan=-0.25)
        play(music_box, b0 + 1.0, notes[2], vel=0.20, pan=-0.25)
        play(music_box, b0 + 1.5, notes[1], vel=0.16, pan=-0.25)
    elif b0 == 28:                  # 마지막 마디: 한 번만 펼치고 울림
        for k, nn in enumerate(notes):
            play(music_box, b0 + k * 0.5, nn, vel=0.24, pan=-0.25)
    else:
        for k in range(4):
            play(music_box, b0 + k * 0.5, notes[k], vel=0.24 if k == 0 else 0.18, pan=-0.25)

# ───────────────────────── 패드 + 베이스 ─────────────────────────
PAD = {
    'C': ['C3', 'G3', 'E4'], 'F': ['F3', 'C4', 'A4'], 'G': ['G3', 'D4', 'B3'],
    'Am': ['A3', 'E4', 'C4'], 'Dm': ['D3', 'A3', 'F4'],
}
for i, (b0, ch) in enumerate(chords):
    b1 = chords[i + 1][0] if i + 1 < len(chords) else 32
    d = (b1 - b0) * BEAT + 0.5
    lvl = 0.035 if b0 < 8 else 0.05
    for nn in PAD[ch]:
        sig = pad(freq(nn), d, vel=lvl)
        add(dry, b0 * BEAT, sig, pan=rng.uniform(-0.4, 0.4))
        add(wet_send, b0 * BEAT, sig, gain=0.5)

# ───────────────────────── 썰매 방울 ─────────────────────────
for b in np.arange(0, 32, 0.5):
    if b < 8:                                   # 인트로: 1, 3박만 아주 살짝
        v = 0.05 if b % 2 == 0 else 0.0
    elif b < 16:                                # 걷기: 4분음표 + 약한 엇박
        v = 0.11 if b % 1 == 0 else 0.045
    elif b < 20:                                # 점등 빌드업: 점점 크게
        v = 0.07 + 0.09 * (b - 16) / 4
    elif b < 28:
        v = 0.12 if b % 1 == 0 else 0.05
    elif b < 30:
        v = 0.08 if b % 1 == 0 else 0.0
    else:
        v = 0.0
    if v > 0:
        sig = sleigh(vel=v)
        t = b * BEAT
        add(dry, t, sig, pan=rng.uniform(-0.5, 0.5))
        add(wet_send, t, sig, gain=0.25)
for b in np.arange(18.0, 20.0, 0.25):           # 빌드업 끝 16분음표 롤
    sig = sleigh(vel=0.04 + 0.05 * (b - 18) / 2, dur=0.15)
    add(dry, b * BEAT, sig, pan=rng.uniform(-0.6, 0.6))

# ───────────────────────── 효과성 벨 ─────────────────────────
# "!" 트리 발견 (14박 = 8.75초)
play(glock, 14.0, 'G6', vel=0.22, pan=0.3, dur=1.2)
play(glock, 14.25, 'C7', vel=0.26, pan=0.3, dur=1.5)
# 별 점등 (20박 = 12.5초): 차임 화음 + 벨트리
for nn, v in (('C5', .30), ('G5', .22), ('C6', .30), ('E6', .22), ('G6', .18), ('C7', .2)):
    play(glock, 20.0, nn, vel=v, pan=rng.uniform(-0.3, 0.3), send=0.6, dur=4.0)
shimmer = ['C7', 'D7', 'E7', 'G7', 'A7', 'C8', 'E7', 'G7', 'D7', 'A7', 'C8', 'G7', 'E7', 'C8', 'D8', 'E8']
for k, nn in enumerate(shimmer):
    play(glock, 20.0 + 0.09 * k + 0.05, nn, vel=0.07 * (1 - k / 22), pan=rng.uniform(-0.8, 0.8), send=0.7, dur=1.2)
# 반짝임 (점등 후 드문드문)
for b, nn in ((21.5, 'E7'), (22.5, 'G7'), (23.25, 'C8'), (23.75, 'A7')):
    play(glock, b, nn, vel=0.06, pan=rng.uniform(-0.7, 0.7), send=0.7, dur=1.0)
# 제목 등장 (24.5박, 25.5박)
play(glock, 24.5, 'A6', vel=0.16, pan=-0.2, send=0.5, dur=2.0)
play(glock, 24.5, 'C7', vel=0.14, pan=-0.2, send=0.5, dur=2.0)
play(glock, 25.5, 'C7', vel=0.16, pan=0.2, send=0.5, dur=2.0)
play(glock, 25.5, 'F7', vel=0.13, pan=0.2, send=0.5, dur=2.0)
# 마지막 화음 (28박) + 끝 반짝 (31박)
for nn, v in (('C5', .22), ('E5', .16), ('G5', .16), ('C6', .2)):
    play(glock, 28.0, nn, vel=v, pan=rng.uniform(-0.3, 0.3), send=0.6, dur=4.0)
for k, nn in enumerate(['G7', 'C8', 'E8']):
    play(glock, 31.0 + 0.08 * k, nn, vel=0.07, pan=rng.uniform(-0.6, 0.6), send=0.7, dur=1.5)

# ───────────────────────── 리버브 (지수 감쇠 노이즈 IR, FFT 컨볼루션) ─────────────────────────
ir_len = int(2.2 * SR)
t_ir = np.arange(ir_len) / SR
wet = np.zeros((N, 2))
for c in range(2):
    ir = rng.standard_normal(ir_len) * np.exp(-t_ir / 0.55)
    ir[: int(0.012 * SR)] = 0                     # 프리딜레이
    spec = np.fft.rfft(ir)                        # 어둡고 부드러운 잔향을 위한 로우패스
    fr = np.fft.rfftfreq(ir_len, 1 / SR)
    spec *= 1.0 / (1.0 + (fr / 5000.0) ** 2)
    spec *= (fr / 250.0) ** 2 / (1.0 + (fr / 250.0) ** 2)
    ir = np.fft.irfft(spec, ir_len)
    ir /= np.sqrt(np.sum(ir ** 2))
    L = N + ir_len
    nfft = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(wet_send[:, c], nfft) * np.fft.rfft(ir, nfft), nfft)
    wet[:, c] = y[:N]

mix = dry + 0.55 * wet

# ───────────────────────── 마스터 ─────────────────────────
t = np.arange(N) / SR
fade = np.minimum(1.0, t / 0.02) * np.clip((DUR - t) / 0.9, 0, 1) ** 1.5
mix *= fade[:, None]
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.6) / np.tanh(1.6)           # 부드러운 리미팅
mix *= 0.55                                       # 약 -14 LUFS

out = sys.argv[1] if len(sys.argv) > 1 else 'music.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print('wrote', out, 'samples', N, 'peak', float(np.max(np.abs(mix))))
