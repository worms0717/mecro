#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlueStacks ROI Captor v1.1
 ├─ F4 또는 Alt(좌·우) 키를 누르면 1회 캡처 세션 시작
 ├─ 그 외 키를 누르면 프로그램 즉시 종료
 ├─ --set-origin 옵션 시
 │     ① 첫 클릭 → 게임 (0,0)   ② 두 번째 클릭 → 게임 우하단
 │     → off_x/off_y/sx/sy 네 값 모두 자동 산출·저장
 ├─ --dx / --dy  : 오프셋 미세 보정(px)
 ├─ --recalib    : 템플릿 매칭 자동 보정 강제 재실행
필수 라이브러리: pynput ppadb opencv-python numpy pywin32
"""

import argparse, datetime as dt, json, os, re, sys, threading, time
from pathlib import Path
import cv2, numpy as np
from pynput import keyboard, mouse
from ppadb.client import Client as AdbClient
import win32con, win32gui, win32ui

# ──────────────────────── ADB 헬퍼 ────────────────────────
def adb_device(port=5037, serial=None):
    devs = AdbClient("127.0.0.1", port).devices()
    if not devs:
        raise RuntimeError("ADB 디바이스가 없습니다.")
    if serial:
        dev = next((d for d in devs if d.serial == serial), None)
        if not dev:
            raise RuntimeError(f"디바이스 '{serial}' 를 찾을 수 없습니다.")
        return dev
    return devs[0]

def emu_size(dev):
    wm = dev.wm_size()
    return (int(wm.width), int(wm.height)) if hasattr(wm, "width") else tuple(map(int, wm.split("x")))

def screencap(dev):
    for fn in (lambda: dev.screencap(),
               lambda: dev.shell("screencap -p", decode=False)):
        try:
            t0 = time.time()
            data = fn()
            return data, int((time.time() - t0) * 1000)
        except Exception:
            pass
    # 레거시: 파일 경유
    tmp = "/sdcard/__scr.png"
    dev.shell(f"screencap -p {tmp}")
    local = Path("_scr.png")
    dev.pull(tmp, str(local))
    data = local.read_bytes()
    local.unlink(missing_ok=True)
    return data, 0

# ─────────────── BlueStacks 창 탐색 / 캡처 ────────────────
def hwnd_bluestacks(pattern="BlueStacks"):
    pat = re.compile(pattern, re.I)
    found = []

    def cb(h, _):
        if not win32gui.IsWindowVisible(h) or win32gui.GetWindow(h, win32con.GW_OWNER):
            return True
        try:
            if pat.search(win32gui.GetWindowText(h)):
                found.append(h)
        except Exception:
            pass
        return True

    win32gui.EnumWindows(cb, None)
    return found[0] if found else None

def grab_client(hwnd):
    l, t, r, b = win32gui.GetClientRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetWindowDC(hwnd)
    mdc = win32ui.CreateDCFromHandle(hdc)
    sdc = mdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap(); bmp.CreateCompatibleBitmap(mdc, w, h)
    sdc.SelectObject(bmp)
    sdc.BitBlt((0, 0), (w, h), mdc, (0, 0), win32con.SRCCOPY)
    buf = bmp.GetBitmapBits(True)
    img = np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3]
    # 해제
    win32gui.DeleteObject(bmp.GetHandle())
    sdc.DeleteDC(); mdc.DeleteDC(); win32gui.ReleaseDC(hwnd, hdc)
    return img

# ───────────── 템플릿 매칭 자동 보정 ──────────────
def calibrate(dev, hwnd, cache_path, force=False, thr_score=0.32):
    cache = Path(cache_path)
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    print("[CALIB] 시작…")
    win_g = cv2.cvtColor(grab_client(hwnd), cv2.COLOR_BGR2GRAY)
    emu_png, _ = screencap(dev)
    emu_g = cv2.cvtColor(cv2.imdecode(np.frombuffer(emu_png, np.uint8), 1), cv2.COLOR_BGR2GRAY)
    ex, ey = emu_size(dev)

    best = None
    for rot in (0, 90):
        eg = cv2.rotate(emu_g, cv2.ROTATE_90_CLOCKWISE) if rot == 90 else emu_g
        for scale in np.arange(0.3, 2.05, 0.05):
            tpl = cv2.resize(eg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th, tw = tpl.shape
            if th > win_g.shape[0] or tw > win_g.shape[1]:
                continue
            res = cv2.matchTemplate(win_g, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)
            if best is None or val > best[0]:
                best = (val, loc, rot, tw, th)
                if val >= 0.95:
                    break
        if best and best[0] >= 0.95:
            break

    if not best or best[0] < thr_score:
        raise RuntimeError(f"보정 실패(score {best[0]:.3f} < {thr_score})")

    val, (x, y), rot, tw, th = best
    sx, sy = ex / tw, ey / th
    data = dict(off_x=x, off_y=y, sx=sx, sy=sy, rot=rot,
                w_emu=ex, h_emu=ey, calib_done=False)
    cache.write_text(json.dumps(data, indent=2))
    print(f"[CALIB] 완료 off=({x},{y}) scale=({sx:.4f},{sy:.4f}) rot={rot}° score={val:.3f}")
    return data

# ─────────────────────── ROI 캡처 클래스 ───────────────────────
class ROICaptor:
    def __init__(self, dev, hwnd, cal, out_dir, args, cal_file):
        self.dev, self.hwnd, self.cal = dev, hwnd, cal
        self.out_dir = Path(out_dir); self.out_dir.mkdir(exist_ok=True)
        self.args, self.cal_file = args, cal_file
        self.clicks = []
        self.calib_pts = []   # 두 점 저장

    # ------------ 좌표 변환 (screen → game) ------------
    def to_game(self, xs, ys):
        c = self.cal
        if c["rot"] == 0:
            return (xs - c["off_x"]) * c["sx"], (ys - c["off_y"]) * c["sy"]
        # 90° 회전
        return (ys - c["off_y"]) * c["sx"], (c["off_x"] + c["w_emu"] * c["sx"] - xs) * c["sy"]

    # ------------ 클릭 이벤트 ------------
    def on_click(self, x, y, button, pressed):
        if not pressed:
            return

        # --- 두 지점 보정 모드 ---
        if self.args.set_origin and not self.cal["calib_done"]:
            self.calib_pts.append((x, y))
            print(f"[CALIB-CLICK] #{len(self.calib_pts)} = {x},{y}")

            if len(self.calib_pts) == 2:
                (x1, y1), (x2, y2) = self.calib_pts
                emu_w, emu_h = self.cal["w_emu"], self.cal["h_emu"]
                # 오프셋
                self.cal["off_x"], self.cal["off_y"] = x1, y1
                # 스케일
                self.cal["sx"] = emu_w / max(1, (x2 - x1))
                self.cal["sy"] = emu_h / max(1, (y2 - y1))
                self.cal["calib_done"] = True
                Path(self.cal_file).write_text(json.dumps(self.cal, indent=2))
                print(f"[CALIB-SAVE] off=({x1},{y1})  "
                      f"sx={self.cal['sx']:.5f}  sy={self.cal['sy']:.5f}")
                self.calib_pts.clear()
                return False  # 보정 세션 종료
            return True       # 두 번째 점 대기
        # --- ROI 캡처 흐름 ---
        self.clicks.append((x, y))
        gx, gy = self.to_game(x, y)
        print(f"[CLICK] screen=({x},{y}) → game≈({gx:.1f},{gy:.1f})")
        if len(self.clicks) == 2:
            self.save_roi()
            return False
        return True

    # ------------ ROI 저장 ------------
    def save_roi(self):
        (sx, sy), (ex, ey) = self.clicks
        gx1, gy1 = self.to_game(sx, sy); gx2, gy2 = self.to_game(ex, ey)
        l, r = sorted(map(int, [gx1, gx2])); t, b = sorted(map(int, [gy1, gy2]))
        png, ms = screencap(self.dev)
        img = cv2.imdecode(np.frombuffer(png, np.uint8), 1)
        h, w = img.shape[:2]
        l, r = max(0, l), min(r, w); t, b = max(0, t), min(b, h)
        if r - l < 2 or b - t < 2:
            print("[WARN] ROI가 작아 저장하지 않음"); return
        roi = img[t:b, l:r]
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fn = f"roi_{r-l}x{b-t}_{ts}.png"; cv2.imwrite(str(self.out_dir / fn), roi)
        print(f"[ROI] {l},{t}-{r},{b}  {r-l}x{b-t}px 저장={fn} ({ms} ms)")

    # ------------ 두-클릭 세션 ------------
    def capture_session(self):
        self.clicks.clear()
        print("[INFO] 두 지점을 클릭하세요…")
        with mouse.Listener(on_click=self.on_click) as ml:
            ml.join()

# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="BlueStacks")
    ap.add_argument("--port", type=int, default=5037)
    ap.add_argument("--serial")
    ap.add_argument("--out", default="snaps")
    ap.add_argument("--recalib", action="store_true")
    ap.add_argument("--set-origin", action="store_true",
                    help="좌상단·우하단 두 지점 클릭으로 오프셋·스케일 재보정")
    ap.add_argument("--dx", type=float, default=0, help="off_x 미세 보정(px)")
    ap.add_argument("--dy", type=float, default=0, help="off_y 미세 보정(px)")
    args = ap.parse_args()

    # ① BlueStacks 창 찾기
    hwnd = hwnd_bluestacks(args.title)
    if not hwnd:
        sys.exit("❌ BlueStacks 창을 찾을 수 없습니다.")

    # ② ADB 디바이스
    dev = adb_device(args.port, args.serial)

    # ③ 보정 파일 로드 / 템플릿 매칭
    cal_file = f"calib_{args.serial or 'default'}.json"
    cal = calibrate(dev, hwnd, cal_file, force=args.recalib)

    # ③-a --set-origin 이면 보정 플래그 강제 초기화
    if args.set_origin:
        cal["calib_done"] = False

    # ④ dx/dy 미세 보정
    cal["off_x"] += args.dx
    cal["off_y"] += args.dy
    if args.set_origin or args.dx or args.dy:
        Path(cal_file).write_text(json.dumps(cal, indent=2))

    # ⑤ ROI 캡처 객체
    captor = ROICaptor(dev, hwnd, cal, args.out, args, cal_file)

    # ⑥ 키 리스너: F4 또는 Alt → 세션 / 기타 키 → 종료
    def on_press(key):
        if key in (keyboard.Key.f4, keyboard.Key.alt_l, keyboard.Key.alt_r):
            threading.Thread(target=captor.capture_session, daemon=True).start()
        else:
            print(f"[EXIT] {key} 입력 – 프로그램 종료")
            os._exit(0)

    print("▶ F4 또는 Alt 키 → 캡처 세션 | 다른 키 입력 시 종료")
    with keyboard.Listener(on_press=on_press) as kl:
        kl.join()


if __name__ == "__main__":
    main()
