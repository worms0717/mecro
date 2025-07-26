from __future__ import annotations
"""Simplified macro controller for Seven Knights.

This module provides a basic macro runner that interacts with BlueStacks
via ADB.  Images are matched with OpenCV and actions are described as
sequences of steps.  The code has been cleaned up for easier
maintenance and future extension.
"""

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import json
import os
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
import psutil
from ppadb.client import Client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BS_INSTALL = Path(r"C:/Program Files/BlueStacks_nxt")
TEMPL_DIR = Path(__file__).parent / "templates"

DEBUG = True
MATCH_TH = 0.75
SCAN_TH = 0.70
LOOP_DELAY = 0.30
REPORT_INTERVAL = 10

STUCK_REPEAT = 10
STUCK_COORD = (1809, 56)

DEFAULT_COOLDOWN = 6.0
MAX_DUNGEON_RUNS = 2

ADB_CANDS = [
    Path(r"C:/Android/platform-tools/adb.exe"),
    Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe",
]

# ---------------------------------------------------------------------------
# Template handling
# ---------------------------------------------------------------------------
_tpl_cache: Dict[str, np.ndarray] = {}


def load_tpl(name: str) -> np.ndarray:
    """Return a cached template image."""
    if name not in _tpl_cache:
        path = TEMPL_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        _tpl_cache[name] = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return _tpl_cache[name]


def match(
    screen: np.ndarray,
    tpl: np.ndarray,
    *,
    roi_x0: int = 0,
    roi_y0: int = 0,
    roi_x1: Optional[int] = None,
    roi_y1: Optional[int] = None,
    scales: Iterable[float] = (0.8, 0.9, 1.0, 1.1, 1.2),
) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Return best match score and location."""
    h, w = screen.shape[:2]
    x0, x1 = roi_x0, roi_x1 or w
    y0, y1 = roi_y0, roi_y1 or h

    roi = screen[y0:y1, x0:x1]
    edge_r = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 50, 150)
    edge_t0 = cv2.Canny(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), 50, 150)

    best_sc, best_coord = 0.0, None
    for s in scales:
        edge_t = cv2.resize(edge_t0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if edge_t.shape[0] > edge_r.shape[0] or edge_t.shape[1] > edge_r.shape[1]:
            continue
        _, sc, _, loc = cv2.minMaxLoc(cv2.matchTemplate(edge_r, edge_t, cv2.TM_CCOEFF_NORMED))
        if sc > best_sc:
            cx = loc[0] + edge_t.shape[1] // 2 + x0
            cy = loc[1] + edge_t.shape[0] // 2 + y0
            best_sc, best_coord = sc, (cx, cy)
    return best_sc, best_coord


# ---------------------------------------------------------------------------
# BlueStacks wrapper
# ---------------------------------------------------------------------------

def _find_adb() -> str:
    adb = shutil.which("adb")
    if adb:
        return adb
    for cand in ADB_CANDS:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError("adb.exe not found")


ADB_EXE = _find_adb()
HD_ADB = BS_INSTALL / "HD-Adb.exe"


def _detect_ports() -> List[int]:
    ports: set[int] = set()
    if HD_ADB.is_file():
        try:
            out = subprocess.check_output([str(HD_ADB), "devices"], text=True)
            ports |= {int(m.group(1)) for m in re.finditer(r"emulator-(\d+)\s+device", out)}
        except subprocess.SubprocessError:
            pass
    for conn in psutil.net_connections(kind="tcp"):
        if conn.status == psutil.CONN_LISTEN and conn.laddr:
            try:
                if "hd-player" in psutil.Process(conn.pid).name().lower():
                    ports.add(conn.laddr.port)
            except Exception:
                pass
    ports.add(5555)
    return sorted(ports)


class BlueStacks:
    """Simple ADB client wrapper."""

    def __init__(self) -> None:
        self.client = Client("127.0.0.1", 5037)
        subprocess.run([ADB_EXE, "start-server"], capture_output=True)
        self.dev = None
        self.connect()

    # Public API -------------------------------------------------------------
    def screenshot(self) -> np.ndarray:
        raw = self.ensure().screencap()
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    def tap(self, pos: Tuple[int, int]) -> None:
        x, y = pos
        self.ensure().shell(f"input tap {x} {y}")
        time.sleep(0.3)

    # Internal ---------------------------------------------------------------
    def ensure(self):
        try:
            self.dev.shell("echo 0")
        except Exception:
            self.connect()
        return self.dev

    def connect(self) -> None:
        for port in _detect_ports():
            subprocess.run([ADB_EXE, "connect", f"127.0.0.1:{port}"], capture_output=True)
            try:
                self.dev = self.client.device(f"127.0.0.1:{port}")
                if self.dev:
                    if DEBUG:
                        print("[ADB]", self.dev.serial, "connected")
                    return
            except Exception:
                pass
        raise RuntimeError("ADB connection failed")


# ---------------------------------------------------------------------------
# Gameplay helpers
# ---------------------------------------------------------------------------
_element_schedule = {
    0: "불",
    1: "물",
    2: "땅",
    3: "빛",
    4: "암흑",
    5: "골드",
    6: "골드",
}

_element_tpl = {
    "불": "불의 원소 던전.png",
    "물": "물의 원소 던전.png",
    "땅": "땅의 원소 던전.png",
    "빛": "빛의 원소 던전.png",
    "암흑": "암흑의 원소 던전.png",
    "골드": "골드 던전.png",
}

_skill_sequence = {
    "불": ["스킬1_ready.png", "스킬2_ready.png", "스킬3_ready.png"],
    "물": ["스킬4_ready.png", "스킬5_ready.png", "스킬6_ready.png"],
    "땅": ["에반스킬1-1.png", "레이첼스킬2-1.png", "제이븐스킬1-1.png", "유리스킬2-1.png", "유이스킬2-1.png", "제이븐스킬1-1.png"],
    "빛": ["조커스킬2-1.png", "에반스킬1-1.png", "클로에스킬2-1.png", "유이스킬2-1.png"],
    "암흑": ["스킬11_ready.png", "스킬12_ready.png"],
    "골드": [],
}

_skill_cooldown = {
    "에반스킬1-1.png": 114.0,
    "에반스킬1-2.png": 114.0,
    "레이첼스킬2-1.png": 66.0,
    "제이븐스킬1-1.png": 104.0,
    "유리스킬2-1.png": 72.0,
    "유이스킬2-1.png": 124.0,
    "클로에스킬2-1.png": 60.0,
    "조커스킬2-1.png": 60.0,
}

CLEAR_TPL_NAME = "성장던전클리어.png"


def wait_for_stage_clear(bs: BlueStacks, timeout: float = 120) -> bool:
    clear_tpl = load_tpl(CLEAR_TPL_NAME)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if match(bs.screenshot(), clear_tpl)[0] >= MATCH_TH:
            if DEBUG:
                print("[INFO] stage clear detected")
            return True
        time.sleep(0.5)
    return False


def use_skill_loop(bs: BlueStacks, element: str) -> None:
    seq = _skill_sequence.get(element)
    if not seq:
        return

    tpl_list = [(name, load_tpl(name)) for name in seq]
    last_tap = {name: 0.0 for name, _ in tpl_list}
    clear_tpl = load_tpl(CLEAR_TPL_NAME)
    roi_y0 = 850

    while True:
        frame = bs.screenshot()
        if match(frame, clear_tpl)[0] >= MATCH_TH:
            if DEBUG:
                print("[INFO] dungeon clear detected, stopping skills")
            return

        roi = frame[roi_y0:]
        now = time.time()

        for name, tpl in tpl_list:
            cooldown = _skill_cooldown.get(name, DEFAULT_COOLDOWN)
            if now - last_tap[name] < cooldown:
                continue
            sc, pt = match(roi, tpl)
            if sc >= MATCH_TH:
                bs.tap((pt[0], pt[1] + roi_y0))
                last_tap[name] = now
                if DEBUG:
                    print(f"[SKILL] {name} fired score={sc:.2f} cd={cooldown}s")
                time.sleep(0.15)

        time.sleep(0.10)


# ---------------------------------------------------------------------------
# Daily run counter
# ---------------------------------------------------------------------------
_growth_state = {"date": date.today().isoformat(), "count": 0}


def _reset_growth_counter() -> None:
    today = date.today().isoformat()
    if _growth_state["date"] != today:
        _growth_state["date"] = today
        _growth_state["count"] = 0


def can_enter_growth() -> bool:
    _reset_growth_counter()
    return _growth_state["count"] < MAX_DUNGEON_RUNS


def register_growth_run() -> None:
    _reset_growth_counter()
    _growth_state["count"] += 1
    if DEBUG:
        print(f"[LIMIT] runs today: {_growth_state['count']}/{MAX_DUNGEON_RUNS}")


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

@dataclass
class Step:
    name: str
    tpl: Optional[Iterable[str]] = None
    action: Optional[str] = None
    tap: bool = False
    roi: Tuple[int, Optional[int]] = (0, None)
    sec: float = 1.0
    fail_max: int = 0
    all: bool = False
    func: Optional[Callable[[BlueStacks], None]] = None


SEQS: Dict[str, List[Step]] = {
    "화면전환2": [
        Step("화면1 인식", tpl=["화면1_의뢰판.png"]),
        Step(
            "오른쪽→왼쪽 드래그",
            action="DRAG",
            func=lambda bs: bs.ensure().shell("input swipe 1050 160 450 160 300"),
        ),
        Step("화면2 확인", tpl=["화면2.png"]),
    ],
    "펫 성장": [
        Step("성장 버튼", tpl=["성장_이벤트.png"], tap=True, fail_max=3),
        Step("이벤트 표시", tpl=["펫 성장1.png", "이벤트표시1.png"], tap=True, all=True),
        Step("일괄 부화", tpl=["일괄부화.png"], tap=True),
        Step("1초 대기", action="WAIT", sec=1.0),
        Step("일괄 등록", tpl=["일괄등록.png"], tap=True),
        Step("1초 대기", action="WAIT", sec=1.0),
        Step("ESC", action="ESC"),
    ],
    "쫄작 스타터": [
        Step("쫄작 스타터", tpl=["쫄작스타터.png"], tap=True),
        Step("설정 확인", tpl=["쫄작설정.png"], tap=True, fail_max=10),
        Step("1초 대기", action="WAIT", sec=1.0),
        Step("모험 반복", tpl=["쫄작시작.png"], tap=True, fail_max=15),
        Step("1초 대기", action="WAIT", sec=5.0),
        Step("최소화", tpl=["화면최소화.png"], tap=True, fail_max=15),
    ],
    "쫄작 재시작": [
        Step(
            "쫄작 완료",
            tpl=["쫄작완료.png", "쫄작완료1.png", "쫄작완료2.png", "쫄작완료3.png", "쫄작완료4.png"],
            tap=True,
            fail_max=3,
        ),
        Step("1초 대기", action="WAIT", sec=1.0),
        Step("재시작 버튼", tpl=["쫄작재시작.png"], tap=True, fail_max=10),
        Step("1초 대기", action="WAIT", sec=5.0),
        Step("최소화", tpl=["화면최소화.png"], tap=True, fail_max=15),
    ],
}


# Growth dungeon sequence
SEQS.update(
    {
        "성장던전 자동 사냥": [
            Step(
                "입장 횟수 체크",
                action="FUNC",
                func=lambda bs: register_growth_run() if can_enter_growth() else (_ for _ in ()).throw(RuntimeError("limit")),
            ),
            Step(
                "던전 버튼",
                tpl=[_element_tpl[_element_schedule[datetime.today().weekday()]]],
                tap=True,
                fail_max=5,
            ),
            Step("던전 입장 버튼", tpl=["성장던전입장.png"], tap=True, fail_max=5),
            Step("던전 시작 버튼", tpl=["성장던전시작.png"], tap=True, fail_max=5),
            Step(
                "스킬 사용",
                action="FUNC",
                func=lambda bs: use_skill_loop(bs, _element_schedule[datetime.today().weekday()]),
            ),
            Step(
                "스테이지 클리어 대기",
                action="FUNC",
                func=wait_for_stage_clear,
            ),
            Step("던전 반복 대기", action="ESC"),
            Step("대기", action="ESC", sec=1.0),
        ]
    }
)


# Preload templates for branch detection
BRANCH_MAP: List[Dict[str, object]] = [
    {"tpl": "화면1_의뢰판.png", "seq": "화면전환2", "skip": 1, "tap": False},
    {"tpl": "쫄작완료.png", "seq": "쫄작 재시작", "skip": 1, "tap": True},
    {"tpl": "쫄작스타터.png", "seq": "쫄작 스타터", "skip": 1, "tap": True},
]

for item in BRANCH_MAP:
    load_tpl(item["tpl"])

BLOCK_TPLS = ["쫄작완료.png", "쫄작스타터.png"]


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------

def _is_true(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes")
    return False


def run_seq(bs: BlueStacks, seq_name: str, start_idx: int = 0) -> bool:
    dev = bs.ensure()
    steps = SEQS[seq_name][start_idx:]

    print(f"\n[SEQ] {seq_name} ▶")

    for idx, step in enumerate(steps, start=start_idx):
        if step.action:
            act = step.action.upper()
            if act == "WAIT":
                time.sleep(step.sec)
                continue
            if act == "ESC":
                dev.shell("input keyevent 111")
                time.sleep(0.3)
                continue
            if act == "DRAG" and step.func:
                step.func(bs)
                time.sleep(0.6)
                continue
            if act == "FUNC" and step.func:
                step.func(bs)
                continue
            print(f"  ⚠ unsupported action {act}")
            continue

        tpl_list = list(step.tpl or [])
        tap = _is_true(step.tap)
        require_all = _is_true(step.all)
        fail_max = step.fail_max
        retries = 0
        y0, y1 = step.roi

        while True:
            frame = cv2.imdecode(np.frombuffer(dev.screencap(), np.uint8), cv2.IMREAD_COLOR)
            scores, coords = [], []
            for fn in tpl_list:
                s, c = match(frame, load_tpl(fn), roi_y0=y0, roi_y1=y1)
                scores.append(s)
                coords.append(c)
            ok = all(s >= MATCH_TH for s in scores) if require_all else any(s >= MATCH_TH for s in scores)
            if ok:
                if tap:
                    target = coords[0] if require_all else coords[int(np.argmax(scores))]
                    dev.shell(f"input tap {target[0]} {target[1]}")
                    time.sleep(step.sec)
                break
            retries += 1
            if fail_max and retries >= fail_max:
                print(f"  ✕ {step.name}: {retries} failures")
                dev.shell("input keyevent 111")
                time.sleep(1)
                return False
            time.sleep(LOOP_DELAY)

    print(f"[SEQ] {seq_name} 완료\n")
    return True


def decide_sequence(bs: BlueStacks) -> Tuple[Optional[str], int, Tuple[int, int]]:
    frame = bs.screenshot()

    for item in BRANCH_MAP:
        sc, pt = match(frame, load_tpl(item["tpl"]), scales=(1.0,))
        if sc < SCAN_TH:
            continue

        seq = item["seq"]
        skip = item.get("skip", 0)
        if seq == "성장던전 자동 사냥":
            if not can_enter_growth():
                if DEBUG:
                    print("[LIMIT] growth dungeon limit reached")
                continue
            if any(match(frame, load_tpl(bt))[0] >= SCAN_TH for bt in BLOCK_TPLS):
                if DEBUG:
                    print("[BRANCH] growth dungeon blocked")
                continue
        if item.get("tap", True):
            bs.tap(pt)
        return seq, int(skip), pt

    return None, 0, (0, 0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    bs = BlueStacks()
    print("[INFO] Seven Mecro started")

    while True:
        seq_name, skip, _ = decide_sequence(bs)
        if seq_name is None:
            time.sleep(LOOP_DELAY)
            continue

        if not run_seq(bs, seq_name, start_idx=skip):
            print(f"[WARN] {seq_name} failed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT]")
