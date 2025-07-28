from __future__ import annotations
"""
Seven_Mecro_control_R3.py · 2025‑07‑13  R8
────────────────────────────────────────────────────────
1) 자동 분기  (쫄작완료 ↔ 쫄작스타터)
2) 단계별 탭 시퀀스 + 템플릿 다중 매칭 지원
3) --debug 옵션으로 실시간 점수·좌표 로그 출력
4) **NEW** 2단계 이후 동일 이미지가 10회 연속 감지되면 복구 좌표(1730,173) 탭 후
   시퀀스를 처음부터 재시작 (루프 교착 해소)
"""

# ────────────────────────── 기본 설정
import argparse
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, date
import cv2
import numpy as np
import psutil
from ppadb.client import Client
import json

BS_INSTALL = Path(r"C:/Program Files/BlueStacks_nxt")
TEMPL_DIR = Path(__file__).parent / "templates"


@dataclass
class GameConfig:
    """실행 중 변경 가능한 설정 값 모음."""

    debug: bool = True
    skill_log: bool = True
    match_th: float = 0.75
    scan_th: float = 0.70
    loop_delay: float = 0.30
    report_interval: int = 10
    stuck_repeat: int = 10
    stuck_coord: Tuple[int, int] = (1809, 56)
    default_cooldown: float = 6.0
    max_dungeon_runs: int = 2


CFG = GameConfig()

ADB_CANDS = [
    Path(r"C:/Android/platform-tools/adb.exe"),
    Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe",
]

_growth_state = {"date": date.today().isoformat(), "count": 0}

# ────────────────────────── 좌표·템플릿 매핑
# 공통 스킬 바 위치(게임 해상도 1971×1131 기준)
ROW1_Y, ROW2_Y = 917, 999
COL_X = [1058, 1256, 1448, 1646, 1846]

# 요일‑별 던전 이름 매핑 (월=0 … 일=6)
DUNGEON_SCHEDULE = {
    0: "불",         # 월
    1: "물",         # 화
    2: "땅",         # 수
    3: "빛",         # 목
    4: "암흑",       # 금
    5: "골드",       # 토
    6: "골드",       # 일
}

# 각 던전에 대응하는 템플릿 파일명
ELEMENT_TPL = {
    "불":   "불의 원소 던전.png",
    "물":   "물의 원소 던전.png",
    "땅":   "땅의 원소 던전.png",
    "빛":   "빛의 원소 던전.png",
    "암흑": "암흑의 원소 던전.png",
    "골드": "골드 던전.png",
}

# 몬스터 스킬 인식용 템플릿 (element -> 몬스터 스킬 이미지 파일명)
MONSTER_SKILL_TPL = {
    "불":   "불_푸키_스킬.png",
    "물":   "물_푸키_스킬.png",
    "땅":   "땅_푸키_스킬.png",
    "빛":   "빛_푸키_스킬.png",
    "암흑": "암흑_푸키_스킬.png",
    "골드": "골드_푸키_스킬.png",
}

# 스킬 위치 매핑 (좌표는 실제 환경에 맞게 조정)
SKILL_POSITIONS = {
    "불":   [(200, 917), (260, 950), (320, 950)],
    "물":   [(350, 950), (410, 950), (470, 950)],
    "땅":   [(1058, 917), (1448, 917), (1846, 917), (1646, 917), (1246, 917), (1846, 917)],
    "빛":   [(650, 950), (710, 950), (770, 950)],
    "암흑": [(800, 950), (860, 950), (920, 950)],
    "골드": [(950, 950), (1010, 950), (1070, 950)],
}

# 1열 스킬 위치 (917) 2열 스킬 위치 (999)
# 1열 스킬 위치 왼쪽부터 1.(1058), 2.(1256), 3.(1448),4.(1646),5.(1846)

# 스킬 준비 템플릿 매핑 (element -> 리스트 of 준비 템플릿 파일명)
SKILL_SEQUENCE = {
    "불":   ["스킬1_ready.png", "스킬2_ready.png", "스킬3_ready.png"],
    "물":   ["스킬4_ready.png", "스킬5_ready.png", "스킬6_ready.png"],
    "땅":   ["에반스킬1-1.png", "레이첼스킬2-1.png", "제이븐스킬1-1.png", "유리스킬2-1.png", "유이스킬2-1.png", "제이븐스킬1-1.png"],
    "빛":   ["조커스킬2-1.png", "에반스킬1-1.png", "클로에스킬2-1.png", "유이스킬2-1.png"],
    "암흑": ["스킬11_ready.png", "스킬12_ready.png"],
    "골드": []
}

SKILL_COOLDOWN = {
    "에반스킬1-1.png": 114.0,
    "에반스킬1-2.png": 114.0,
    "레이첼스킬2-1.png" : 66.0,
    "제이븐스킬1-1.png" : 104.0,
    "유리스킬2-1.png" : 72.0,
    "유이스킬2-1.png" : 124.0,
    "클로에스킬2-1.png": 60.0,
    "조커스킬2-1.png": 60.0,
    
    # 나머지는 DEFAULT_COOLDOWN 사용
}


# 스킬 준비 상태 감지를 위한 템플릿 (원소별 또는 공통 준비 아이콘)
SKILL_READY_TPL = "스킬준비.png"
ROI_RADIUS = 40  # 매칭 ROI 반경(px) – 기존 22 → 40 로 확대
CLEAR_TPL_NAME = "성장던전클리어.png"
# ────────────────────────── ADB & BlueStacks

def find_adb() -> str:
    adb = shutil.which("adb")
    if adb:
        return adb
    for p in ADB_CANDS:
        if p.is_file():
            return str(p)
    raise FileNotFoundError("adb.exe not found (PATH 확인)")

ADB_EXE = find_adb()
HD_ADB  = BS_INSTALL / "HD-Adb.exe"

def detect_ports() -> List[int]:
    """BlueStacks가 리슨 중인 127.0.0.1:포트 목록을 추출"""
    ports: set[int] = set()
    if HD_ADB.is_file():
        try:
            out = subprocess.check_output([str(HD_ADB), "devices"], text=True)
            ports |= {int(m.group(1)) for m in re.finditer(r"emulator-(\d+)\s+device", out)}
        except subprocess.SubprocessError:
            pass
    for c in psutil.net_connections(kind="tcp"):
        if c.status == psutil.CONN_LISTEN and c.laddr:
            try:
                if "hd-player" in psutil.Process(c.pid).name().lower():
                    ports.add(c.laddr.port)
            except Exception:
                pass
    ports.add(5555)
    return sorted(ports)

class BlueStacks:
    def __init__(self):
        self.client = Client("127.0.0.1", 5037)
        subprocess.run([ADB_EXE, "start-server"], capture_output=True)
        self.dev = None
        self.connect()

    def connect(self):
        for p in detect_ports():
            subprocess.run([ADB_EXE, "connect", f"127.0.0.1:{p}"], capture_output=True)
            try:
                self.dev = self.client.device(f"127.0.0.1:{p}")
                if self.dev:
                    print("[ADB]", self.dev.serial, "연결")
                    return
            except Exception:
                pass
        raise RuntimeError("ADB 연결 실패")

    def ensure(self):
        """연결 확인 및 재연결"""
        try:
            self.dev.shell("echo 0")
        except Exception:
            self.connect()
        return self.dev

    def screenshot(self) -> np.ndarray:
        """현재 화면을 OpenCV 이미지(Numpy array)로 반환"""
        dev = self.ensure()
        raw = dev.screencap()  # PNG 바이너리
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img

    def tap(self, pos: Tuple[int, int]) -> None:
        """지정 좌표에 탭 입력"""
        x, y = pos
        dev = self.ensure()
        dev.shell(f"input tap {x} {y}")
        time.sleep(0.3)
        
# ────────────────────────── 이미지 로드 & 매칭

def load_tpl(name: str) -> np.ndarray:
    """이미지를 캐시에 담아 읽기"""
    if name not in TPL:
        path = TEMPL_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        TPL[name] = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return TPL[name]

_TPL_CACHE: Dict[str, np.ndarray] = {}

def get_tpl(name: str) -> np.ndarray:
    if name not in _TPL_CACHE:
        _TPL_CACHE[name] = load_tpl(name)
    return _TPL_CACHE[name]

def match(screen: np.ndarray, tpl: np.ndarray,
          *, roi_x0: int = 0, roi_y0: int = 0,
          roi_x1: int | None = None, roi_y1: int | None = None,
          scales=(0.8, 0.9, 1.0, 1.1, 1.2)) -> Tuple[float, Tuple[int, int] | None]:
    h, w = screen.shape[:2]
    x0, x1 = roi_x0, roi_x1 or w
    y0, y1 = roi_y0, roi_y1 or h

    roi = screen[y0:y1, x0:x1]
    edgeR = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 50, 150)
    edgeT0 = cv2.Canny(cv2.cvtColor(tpl,  cv2.COLOR_BGR2GRAY), 50, 150)

    best_sc, best_coord = 0.0, None
    for s in scales:
        eT = cv2.resize(edgeT0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if eT.shape[0] > edgeR.shape[0] or eT.shape[1] > edgeR.shape[1]:
            continue
        _, sc, _, loc = cv2.minMaxLoc(cv2.matchTemplate(edgeR, eT, cv2.TM_CCOEFF_NORMED))
        if sc > best_sc:
            cx = loc[0] + eT.shape[1] // 2 + x0
            cy = loc[1] + eT.shape[0] // 2 + y0
            best_sc, best_coord = sc, (cx, cy)
    return best_sc, best_coord


# 일일 입장 횟수 체크 및 증가 함수
def check_and_increment_dungeon_count():
    today = datetime.today().date()
    count = dungeon_run_counts.get(today, 0)
    if count >= CFG.max_dungeon_runs:
        raise RuntimeError(f"하루 최대 {CFG.max_dungeon_runs}회 입장 초과")
    dungeon_run_counts[today] = count + 1

# ────────────────────────── 전역 성장던전 카운터

def _reset_growth_counter():
    today = date.today().isoformat()
    if _growth_state["date"] != today:
        _growth_state["date"] = today
        _growth_state["count"] = 0
        
def can_enter_growth() -> bool:
    _reset_growth_counter()
    return _growth_state["count"] < CFG.max_dungeon_runs

def register_growth_run():
    _reset_growth_counter()
    _growth_state["count"] += 1
    if CFG.debug:
        print(
            f"[LIMIT] 성장던전 실행 횟수 {_growth_state['count']}/{CFG.max_dungeon_runs} (오늘)"
        )
        
# ────────────────────────── 공통: 스테이지 클리어 대기 ────────────────

def wait_for_stage_clear(bs: BlueStacks, timeout: float = 120) -> bool:
    """클리어 템플릿 감지까지 대기. 성공 시 True."""
    clear_tpl = get_tpl(CLEAR_TPL_NAME)
    t0 = time.time()
    while time.time() - t0 < timeout:
        frame = bs.screenshot()
        if clear_tpl is not None and match(frame, clear_tpl)[0] >= CFG.match_th:
            
            if CFG.debug:
                print("[INFO] 스테이지 클리어 감지")
            return True
        time.sleep(0.5)
    return False

def use_skill_loop(bs: BlueStacks, element: str):
    if not SKILL_SEQUENCE[element]:
        return

    tpl_list = [(name, get_tpl(name)) for name in SKILL_SEQUENCE[element]]
    last_tap = {name: 0.0 for name, _ in tpl_list}
    clear_tpl = get_tpl(CLEAR_TPL_NAME)
    ROI_Y0 = 850  # 스킬 바 시작 Y

    while True:
        frame = bs.screenshot()
        if match(frame, clear_tpl)[0] >= CFG.match_th:
            if CFG.debug:
                print("[INFO] 성장 던전 클리어 감지, 스킬 루프 종료")
            return

        roi = frame[ROI_Y0:, :]
        now = time.time()

        for name, tpl in tpl_list:
            cooldown = SKILL_COOLDOWN.get(name, CFG.default_cooldown)
            if now - last_tap[name] < cooldown:
                continue
            sc, pt = match(roi, tpl)
            if sc >= CFG.match_th:
                bs.tap((pt[0], pt[1] + ROI_Y0))
                last_tap[name] = now
                if CFG.skill_log:
                    print(f"[SKILL] {name} fired score={sc:.2f} cd={cooldown}s")
                time.sleep(0.15)

        time.sleep(0.10)


# ────────────────────────── 시퀀스 정의

SEQS: Dict[str, List[Dict]] = {
    # ───────────── 화면 전환 2 ─────────────
    "화면전환2": [
        {"name": "화면1 인식", "tpl": "화면1_의뢰판.png", "tap": False},
        {"name": "오른쪽→왼쪽 드래그", "action": "DRAG",
         "from": (1050, 160), "to": (450, 160)},
        {"name": "화면2 확인", "tpl": "화면2.png", "tap": False},
    ],

    # ───────────── 펫 성장 ─────────────
    "펫 성장": [
        {"name": "성장 버튼",  "tpl": "성장_이벤트.png", "tap": True, "fail_max": 3, "roi": (0, None)},
        {"name": "이벤트 표시", "tpl": ["펫 성장1.png", "이벤트표시1.png"],
         "tap": True, "all": True, "roi": (0, None)},                         # 두 이미지 모두 필요
        {"name": "일괄 부화",  "tpl": "일괄부화.png",  "tap": True, "roi": (0, None)},
        {"name": "1초 대기",   "action": "WAIT", "sec": 1.0},
        {"name": "일괄 등록",  "tpl": "일괄등록.png", "tap": True, "roi": (0, None)},
        {"name": "1초 대기",   "action": "WAIT", "sec": 1.0, "roi": (0, None)},
        {"name": "ESC",        "action": "ESC", "roi": (0, None)},
    ],

    # ───────────── 쫄작 스타터 ─────────────
    "쫄작 스타터": [
        {"name": "쫄작 스타터", "tpl": "쫄작스타터.png", "tap": True, "roi": (0, None)},
        {"name": "설정 확인",   "tpl": "쫄작설정.png",   "tap": True, "roi": (0, None), "fail_max": 10,},
        {"name": "1초 대기",   "action": "WAIT", "sec": 1.0, "roi": (0, None)},
        {"name": "모험 반복",   "tpl": "쫄작시작.png",  "tap": True, "roi": (0, None), "fail_max": 15,},
        {"name": "1초 대기",   "action": "WAIT", "sec": 5.0, "roi": (0, None)},
        {"name": "최소화",     "tpl": "화면최소화.png", "tap": True, "roi": (0, None), "fail_max": 15,},
    ],

    # ───────────── 쫄작 재시작 ─────────────
    "쫄작 재시작": [
        {"name": "쫄작 완료", "tpl": [
            "쫄작완료.png", "쫄작완료1.png", "쫄작완료2.png",
            "쫄작완료3.png", "쫄작완료4.png"
        ], "tap": True, "roi": (0, None), "fail_max": 3,},
        {"name": "1초 대기",   "action": "WAIT", "sec": 1.0, "roi": (0, None)},
        {"name": "재시작 버튼", "tpl": "쫄작재시작.png", "tap": True, "roi": (0, None), "fail_max": 10,},
        {"name": "1초 대기",   "action": "WAIT", "sec": 5.0, "roi": (0, None)},
        {"name": "최소화",     "tpl": "화면최소화.png", "tap": True, "roi": (0, None), "fail_max": 15,},
    ],
}

# 템플릿 캐시 (lazy)
TPL: Dict[str, np.ndarray] = {}

# 일별 입장 횟수 기록 (date -> count)
dungeon_run_counts = {}

# 분기용 (전체 화면 ROI)
# # ───────────── 분기용 매핑 ─────────────
# BRANCH_MAP: list[dict] = [
#     {"tpl": "화면1_의뢰판.png", "seq": "화면전환2", "skip": 1, "tap": False},  # 클릭 금지
#     # {"tpl": "성장_이벤트.png",  "seq": "펫 성장",   "skip": 1, "tap": True},
#     {"tpl": "쫄작완료.png",     "seq": "쫄작 재시작", "skip": 1, "tap": True},
#     {"tpl": "쫄작스타터.png",   "seq": "쫄작 스타터", "skip": 1, "tap": True},
# ]
# ────────────────────────── BRANCH_MAP (리스트형) – 생략: R8 상태 유지 ──
BRANCH_MAP: List[Dict] = [
    {"tpl": "화면1_의뢰판.png", "seq": "화면전환2", "skip": 1, "tap": False},
    # {"tpl": "성장던전.png", "seq": "성장던전 자동 사냥", "skip": 1, "tap": True},
    {"tpl": "쫄작완료.png", "seq": "쫄작 재시작", "skip": 1, "tap": True},
    {"tpl": "쫄작스타터.png", "seq": "쫄작 스타터", "skip": 1, "tap": True},
]
BLOCK_TPLS = ["쫄작완료.png", "쫄작스타터.png"]  # 성장던전 시 분기 차단용

# "성장던전 자동 사장" 시퀀스 정의 및 추가
SEQS.update({
    "성장던전 자동 사냥": [
        # 0) 일일 입장 횟수 체크
        {
            "name": "입장 횟수 체크",
            "action": "FUNC",
            "func": lambda bs: check_and_increment_dungeon_count(),
            "sec": 1.0,
            "skip": 1
        },
        # 1) 오늘 요일 맞춤 던전 버튼 탭
        {
            "name": "던전 버튼",
            "tpl": ELEMENT_TPL[DUNGEON_SCHEDULE[datetime.today().weekday()]],
            "tap": True,
            "fail_max": 5,
            "sec": 1.0,
            "skip": 1
        },
        # 2) 던전 시작 버튼 탭
        {
            "name": "던전 입장 버튼",
            "tpl": "성장던전입장.png",
            "fail_max": 5,
            "tap": True,
            "sec": 1.0
        },
        # 3) 던전 시작 버튼 탭
        {
            "name": "던전 시작 버튼",
            "tpl": "성장던전시작.png",
            "fail_max": 5,
            "tap": True,
            "sec": 1.0
        },
        # 4) 스킬 사용: 요일별 스킬 시퀀스 호출
        {
            "name": "스킬 사용",
            "action": "FUNC",
            "func": lambda bs: use_skill_loop(
                bs,
                DUNGEON_SCHEDULE[datetime.today().weekday()]
            ),
            "fail_max": 10,
            "tap": True,
            "sec": 1.0,
            "fail_max": 5,
        },
        # 5) 스테이지 클리어 대기
        {
            "name": "스테이지 클리어 대기",
            "action": "FUNC",
            "func": wait_for_stage_clear,
            "sec": 1.0,
            "fail_max": 5,
        },
        # 6) 던전 반복 대기
        {
            "name": "던전 반복 대기",
            "action": "esc",
            "seconds": 0.5,
            "tap": True, 
            "roi": (0, None), 
            "fail_max": 10,
            "sec": 1.0
        },
        # 7) 화면 전환
        {
            "name": "대기",
            "action": "esc",
            "seconds": 1.0,
            "tap": True, 
            "roi": (0, None), 
            "fail_max": 10,
            "sec": 1.0,
            "fail_max": 5,
        },
    ]
})

for item in BRANCH_MAP:
    load_tpl(item["tpl"])

# ────────────────────────── STUCK DETECTION (2단계 이후 전용)
def enter_daily_dungeon(bs):
    element = DUNGEON_SCHEDULE[datetime.today().weekday()]
    tpl_name = ELEMENT_TPL[element]
    tpl = load_tpl(tpl_name)

    for n in range(2):                 # 두 번 입장
        # ① 던전 버튼 탐지 → 탭
        frame = bs.screenshot()
        sc, pt = match(frame, tpl)
        if sc < CFG.match_th:
            raise RuntimeError(f"{tpl_name} 탐지 실패(점수 {sc:.2f})")
        bs.tap(pt)
        time.sleep(0.4)

        # ② '시작' 버튼(예: start.png) 탭
        start_tpl = load_tpl("start.png")
        for _ in range(20):
            frame = bs.screenshot()
            sc, pt = match(frame, start_tpl)
            if sc >= CFG.match_th:
                bs.tap(pt)
                break
            time.sleep(0.2)

        # ③ 전투 화면 진입 → 요일‑별 스킬 사용
        time.sleep(1.5)                # 로딩 대기(필요 시 조정)
        use_skill_sequence(bs, element)

        # ④ 클리어 대기
        wait_for_stage_clear(bs)
        time.sleep(0.5)

def use_skill_sequence(bs, element):
    """전투 중 요일‑별 스킬 순서 실행"""
    seq = SKILL_SEQUENCE.get(element, [])
    if not seq:
        return  # 스킬 없음

    for step in seq:
        if isinstance(step, tuple):       # 고정 좌표 탭
            bs.tap(step)
        else:                             # 템플릿 매칭 후 탭
            tpl = load_tpl(step)
            for _ in range(10):           # 최대 10프레임 시도
                frame = bs.screenshot()
                sc, pt = match(frame, tpl)
                if sc >= CFG.match_th:
                    bs.tap(pt)
                    break
                time.sleep(0.1)
        time.sleep(0.2)                   # 스킬 간 짧은 딜레이

def is_stuck(dev, tpl_names: List[str], repeat: int = CFG.stuck_repeat) -> bool:
    """동일 템플릿이 repeat회 연속 감지되면 True"""
    for cnt in range(repeat):
        time.sleep(CFG.loop_delay)
        img_bgr = cv2.imdecode(np.frombuffer(dev.screencap(), np.uint8), cv2.IMREAD_COLOR)
        best_sc = 0.0
        for fn in tpl_names:
            sc, _ = match(img_bgr, load_tpl(fn), roi_y0=0, scales=(1.0,))
            best_sc = max(best_sc, sc)
        if best_sc < CFG.match_th:
            if CFG.debug:
                print(f"    ↪ STUCK check break @{cnt+1}/{repeat} ({best_sc:.2f})        ")
            return False  # 정상 전환
    return True  # repeat 모두 통과 → 교착

# 헬퍼 함수 ────────────────────────────────────────
def _is_true(val) -> bool:
    """bool, 숫자, 문자열(True/False/Yes/1)을 모두 불형으로 안전 변환."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes")
    return False

# ────────────────────────── 시퀀스 실행

def run_seq(bs: "BlueStacks", seq_name: str, start_idx: int = 0) -> bool:
    """시퀀스 실행: 성공 True / 실패 False 반환"""
    dev   = bs.ensure()
    steps = SEQS[seq_name][start_idx:]

    print(f"\n[SEQ] {seq_name} ▶")

    for idx, step in enumerate(steps, start=start_idx):
        name = step.get("name", f"STEP{idx}")

        # ── 특수 액션 처리 ────────────────────────────────
        if "action" in step:
            act = step["action"].upper()
            if act == "WAIT":
                time.sleep(float(step.get("sec", 1.0)))
            elif act == "ESC":
                dev.shell("input keyevent 111")
                time.sleep(0.3)
            elif act == "DRAG":
                (x1, y1), (x2, y2) = step["from"], step["to"]
                duration = int(step.get("duration", 300))
                dev.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")
                time.sleep(0.6)
            elif act == "FUNC":
                # step["func"]에 담긴 함수를 실행
                step["func"](bs)
                # 함수 실행 후 바로 다음 스텝으로 넘어감
                continue
            else:
                print(f"  ⚠ 미지원 action {act}")
            continue

        # ── 템플릿 매칭 단계 ──────────────────────────────
        tpl_raw      = step["tpl"]
        tpl_list     = [tpl_raw] if isinstance(tpl_raw, str) else list(tpl_raw)
        tap          = _is_true(step.get("tap", False))
        require_all  = _is_true(step.get("all", False))
        fail_max     = int(step.get("fail_max", 0))      # 0 = 무제한
        retries      = 0
        y0, y1       = step.get("roi", (0, None))

        while True:
            frame = cv2.imdecode(
                np.frombuffer(dev.screencap(), np.uint8), cv2.IMREAD_COLOR)

            scores, coords = [], []
            for fn in tpl_list:
                s, c = match(frame, load_tpl(fn), roi_y0=y0, roi_y1=y1)
                scores.append(s)
                coords.append(c)

            ok = all(s >= CFG.match_th for s in scores) if require_all \
                 else any(s >= CFG.match_th for s in scores)

            if ok:
                if tap:
                    target = coords[0] if require_all else coords[int(np.argmax(scores))]
                    dev.shell(f"input tap {target[0]} {target[1]}")
                    time.sleep(1)
                break  # 다음 step으로
            else:
                retries += 1
                if fail_max and retries >= fail_max:
                    print(f"  ✕ {name}: {retries}회 실패 → 시퀀스 중단")
                    dev.shell("input keyevent 111")
                    time.sleep(1)
                    return False            # 시퀀스 실패 신고
                time.sleep(CFG.loop_delay)

    print(f"[SEQ] {seq_name} 완료\n")
    return True

# ────────────────────────── 분기 + 즉시 탭

def decide_sequence(bs: BlueStacks):
    """BRANCH_MAP 평가 → (seq, skip, coord).
    조건 불충족 시 (None, 0, (0, 0)) 반환 – R3와 동일한 인터페이스 유지
    """
    frame = bs.screenshot()

    for item in BRANCH_MAP:
        # ① 메인 템플릿 일치율 확인
        tpl_main = load_tpl(item["tpl"])
        sc, pt = match(frame, tpl_main, scales=(1.0,))
        if sc < CFG.scan_th:
            continue

        seq  = item["seq"]
        skip = item.get("skip", 0)

        # ② 성장던전 분기 – 하루 횟수·차단 템플릿 검사
        if seq == "성장던전 자동 사냥":
            if not can_enter_growth():
                if CFG.debug:
                    print("[LIMIT] 성장던전 하루 제한 초과 – 건너뜀")
                continue
            # 차단 아이콘(쫄작완료·쫄작스타터) 존재 시 진입 금지
            if any(match(frame, load_tpl(bt))[0] >= CFG.scan_th for bt in BLOCK_TPLS):
                if CFG.debug:
                    print("[BRANCH] 성장던전 차단 – 차단 템플릿 감지")
                continue

        # ③ 조건 통과 → 필요 시 탭, 결과 반환
        if item.get("tap", True):
            bs.tap(pt)
        return seq, skip, pt

    # ④ 모든 항목 미충족 → None 반환
    return None, 0, (0, 0)


# ────────────────────────── 종합 리포트

def debug_report(img: np.ndarray, *, roi_y0: int = 0):
    print("\n[REPORT] 템플릿 매칭 (score ≥ 0.2)")
    for name in sorted(TPL):
        tpl = TPL[name]
        sc, coord = match(img, tpl, roi_y0=roi_y0, scales=(1.0,))
        if sc < 0.2:
            continue
        print(f"  {name:<15} → {sc:.3f} @ {coord}")
    print("[END REPORT]\n")


def parse_args() -> argparse.Namespace:
    """CLI 인수 파싱"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="디버그 로그 출력")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=CFG.max_dungeon_runs,
        help="하루 최대 던전 입장 횟수",
    )
    return parser.parse_args()

# ────────────────────────── MAIN 루프

def main() -> None:
    args = parse_args()
    CFG.debug = args.debug
    CFG.max_dungeon_runs = args.max_runs

    bs = BlueStacks()          # 블루스택 ADB 연결 래퍼
    print("[INFO] Seven_Mecro 시작")

    while True:
        # 1) 현재 화면으로부터 실행할 시퀀스 결정
        seq_name, skip, _ = decide_sequence(bs)
        if seq_name is None:
            time.sleep(CFG.loop_delay)      # 아무 조건도 못 찾음 → 재시도
            continue

        # 2) 시퀀스 실행
        completed = run_seq(bs, seq_name, start_idx=skip)
        if not completed:
            print(f"[WARN] {seq_name} 실패, 다음 분기 탐색으로 넘어감")
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT]")
