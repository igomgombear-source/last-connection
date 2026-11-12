# 상윤과 승민의 크리스마스

20초 분량의 16비트풍 픽셀 아트 크리스마스 애니메이션 (세로 9:16, 1080x1920, 30fps, 오르골·벨 음악 포함).

- 결과물: `sangyoon_seungmin_christmas.mp4`
- 다시 만들기: `./build.sh`
- `make_video.py` — 216x384 캔버스에 모든 그림을 코드로 그리고 5배 정수 확대
- `make_music.py` — 오리지널 오르골/벨 음악 합성 (96BPM, 8마디)
- `assets/Galmuri11-Bold.ttf` — 제목용 픽셀 폰트 (Galmuri, SIL OFL 1.1 — `assets/Galmuri-LICENSE.txt`)

`make_video.py` 상단의 `SEUNGMIN = 'girl'` 을 `'boy'` 로 바꾸면 트리를 발견하는 캐릭터가 바뀝니다.
