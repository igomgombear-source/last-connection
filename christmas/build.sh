#!/usr/bin/env bash
# 음악 합성 → 영상 렌더링 → MP4 합치기 (python3 + numpy + Pillow + ffmpeg 필요)
set -euo pipefail
cd "$(dirname "$0")"
tmp="$(mktemp -d)"
python3 make_music.py "$tmp/music.wav"
python3 make_video.py --audio "$tmp/music.wav" --out sangyoon_seungmin_christmas.mp4
rm -rf "$tmp"
