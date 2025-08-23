
import argparse
import math
import os
import re
import sys
import time
from collections import deque
import datetime
import logging
import threading
import pygame
import psutil

# ----------------------------
# Optional GPU metrics via Windows Performance Counters
# ----------------------------
WMI_CLIENT = None
if sys.platform == "win32":
    try:
        import wmi  # type: ignore
    except Exception:
        logging.exception("wmi import failed")
        wmi = None
else:
    wmi = None

# ----------------------------
# Optional PDH access (pywin32)
# ----------------------------
if sys.platform == "win32":
    try:
        import win32pdh  # type: ignore
    except Exception:
        logging.exception("win32pdh import failed")
        win32pdh = None
else:
    win32pdh = None

# ----------------------------
# Optional Windows Event Log (pywin32)
# ----------------------------
WIN32_EVT_OK = False
if sys.platform == "win32":
    try:
        import win32evtlog  # type: ignore
        WIN32_EVT_OK = True
    except Exception:
        WIN32_EVT_OK = False

logging.basicConfig(level=logging.INFO)

# ----------------------------
# Display config (portrait)
# ----------------------------
WIDTH, HEIGHT = 480, 800   # portrait: 480 wide x 800 tall
FPS = 30

# Throttle CPU sampling to calm the gauge needle
CPU_SAMPLE_SEC = 0.5
# Throttle disk usage sampling (expensive system calls)
DISK_SAMPLE_SEC = 5.0
MEM_SAMPLE_SEC = 0.5
NET_SAMPLE_SEC = 0.25

# ----------------------------
# Pip-Boy vibe (colors & style)
# ----------------------------
GREEN = (0, 255, 70)
DIM_GREEN = (0, 120, 30)
DARK = (0, 10, 0)
BLACK = (0, 0, 0)

BG_COLOR = BLACK
SCANLINE_COLOR = (0, 35, 0)
SCANLINE_STEP = 4

FONT_NAME = "Courier New"  # try monospace; fallback handled below
FONT_SIZE = 20
FONT_SIZE_SMALL = 16
FONT_SIZE_LARGE = 26

# ----------------------------
# Layout geometry
# ----------------------------
TOP_BAND_H = 270
GAUGE_RADIUS = 85
GAUGE_ANGLE_MIN = math.radians(210)   # -150°
GAUGE_ANGLE_MAX = math.radians(330)   # +150°
GAUGE_SWEEP = GAUGE_ANGLE_MAX - GAUGE_ANGLE_MIN

MID_BAND_Y = TOP_BAND_H + 10
# Shrink mid band to make room for ticker under the net graph
MID_BAND_H = 360

BAR_H = 28
BAR_GAP = 22
BAR_LEFT = 24
BAR_RIGHT = WIDTH - 24

# Ticker geometry (bottom strip)
TICKER_H = 40
TICKER_Y = HEIGHT - TICKER_H

# Net graph geometry (now above ticker)
BOT_BAND_H = HEIGHT - (TOP_BAND_H + MID_BAND_H) - TICKER_H
GRAPH_Y = HEIGHT - TICKER_H - BOT_BAND_H + 6
GRAPH_H = BOT_BAND_H - 18
GRAPH_POINTS = WIDTH  # one point per x pixel

# ----------------------------
# Logo sizing and placement
# ----------------------------
LOGO_START_MAX = 400        # instructions: max width/height of start screen logo
LOGO_MAIN_MAX = 250         # instructions: max width/height of in-app logo
LOGO_START_Y_OFFSET = -140  # instructions: vertical offset from screen center for start logo
LOGO_MAIN_X_OFFSET = 0      # instructions: horizontal offset for in-app logo
LOGO_MAIN_Y_OFFSET = -35    # instructions: vertical offset from graph for in-app logo

# ----------------------------
# Alerts / thresholds
# ----------------------------
CPU_HOT_PCT = 95.0
GPU_HOT_PCT = 95.0
RAM_HOT_PCT = 90.0

# ----------------------------
# Utility
# ----------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def pct(a, b):
    b = max(1e-9, b)
    return 100.0 * (a / b)

def human_gb(x):
    return f"{x:.1f} GB"

_SCANLINE_SURF = None
_TEXT_CACHE: dict[tuple[int, str, tuple[int, int, int]], pygame.Surface] = {}


def _ensure_wmi_client():
    global WMI_CLIENT
    if WMI_CLIENT is not None or sys.platform != "win32" or wmi is None:
        return WMI_CLIENT
    try:
        WMI_CLIENT = wmi.WMI(namespace=r"root\CIMV2")
    except Exception:
        logging.exception("WMI init failed")
        WMI_CLIENT = None
    return WMI_CLIENT


def _gpu_3d_util_windows() -> float | None:
    _ensure_wmi_client()
    if WMI_CLIENT is None:
        return None
    try:
        if not hasattr(WMI_CLIENT, "Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine"):
            return None
        engines = WMI_CLIENT.Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine()
        if not engines:
            return None
        total = 0.0
        found_3d = False
        for e in engines:
            name = getattr(e, "Name", "") or ""
            u = float(getattr(e, "UtilizationPercentage", 0) or 0)
            if "engtype_3D" in name:
                total += u
                found_3d = True
        if not found_3d:
            # fallback: sum all engines so we still get a value
            total = sum(float(getattr(e, "UtilizationPercentage", 0) or 0) for e in engines)
        return clamp(total, 0.0, 100.0)
    except Exception:
        logging.exception("GPU perf counter query failed")
        return None


def _gpu_3d_util_windows_pdh() -> float | None:
    """Query total 3D engine utilization using PDH.

    Opens a query for ``GPU Engine(*engtype_3D)\\Utilization Percentage``
    counters, sums the results, and returns the utilization percentage
    clamped to ``0-100``.  ``None`` is returned on any failure.
    """
    if sys.platform != "win32" or win32pdh is None:
        return None
    query = counter = None
    try:
        query = win32pdh.OpenQuery()
        counter = win32pdh.AddCounter(
            query,
            r"\\GPU Engine(*engtype_3D)\\Utilization Percentage",
            0,
        )
        # Prime the query and then read a sample
        win32pdh.CollectQueryData(query)
        win32pdh.CollectQueryData(query)
        _, vals = win32pdh.GetFormattedCounterArray(counter, win32pdh.PDH_FMT_DOUBLE)
        total = sum(v.get("value", 0.0) for v in vals)
        return clamp(total, 0.0, 100.0)
    except Exception:
        logging.exception("GPU PDH query failed")
        return None
    finally:
        if counter is not None:
            try:
                win32pdh.RemoveCounter(counter)
            except Exception:
                pass
        if query is not None:
            try:
                win32pdh.CloseQuery(query)
            except Exception:
                pass
def gauge_rect(center, radius):
    cx, cy = center
    return pygame.Rect(cx - radius - 10, cy - radius - 10, radius * 2 + 20, radius * 2 + 40)


def bar_rect(x1, y1, x2, y2, font):
    return pygame.Rect(x1, y1, x2 - x1, (y2 - y1) + font.get_height() + 4)


def graph_rect(x, y, w, h):
    return pygame.Rect(x, y, w, h)


def ticker_rect(y):
    return pygame.Rect(12, y, WIDTH - 24, TICKER_H - 12)


def draw_scanlines(surface):
    global _SCANLINE_SURF
    if _SCANLINE_SURF is None:
        _SCANLINE_SURF = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, SCANLINE_STEP):
            pygame.draw.line(_SCANLINE_SURF, SCANLINE_COLOR, (0, y), (WIDTH, y), 1)
    surface.blit(_SCANLINE_SURF, (0, 0))

def _render_text_cached(text: str, font: pygame.font.Font, color: tuple[int, int, int] = GREEN) -> pygame.Surface:
    key = (id(font), text, color)
    surf = _TEXT_CACHE.get(key)
    if surf is None:
        surf = font.render(text, True, color)
        _TEXT_CACHE[key] = surf
    return surf


def glow_text(surface, font, text, pos, color=GREEN, glow=DIM_GREEN, offset=1):
    x, y = pos
    if not text:
        return
    # simple "CRT glow": shadow underlay then main text
    surf_g = _render_text_cached(text, font, glow)
    surface.blit(surf_g, (x - offset, y - offset))
    surface.blit(surf_g, (x + offset, y + offset))
    surface.blit(_render_text_cached(text, font, color), pos)


def load_tinted_logo(path, tint=(0, 255, 70), opacity=220):
    """
    Load a white/gray PNG and tint it Fallout-green, preserving antialiasing.
    """
    surf = pygame.image.load(path).convert_alpha()
    tint_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    tint_surf.fill((*tint, 255))
    surf = surf.copy()
    surf.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.set_alpha(opacity)
    return surf


def scale_logo(logo_surf, max_dim):
    """Scale *logo_surf* to fit within a square of size *max_dim* without distortion."""
    lw, lh = logo_surf.get_size()
    scale = min(max_dim / lw, max_dim / lh)
    new_size = (int(lw * scale), int(lh * scale))
    return pygame.transform.smoothscale(logo_surf, new_size)

try:
    import win32evtlogutil  # type: ignore
except Exception:  # pragma: no cover - only on Windows
    win32evtlogutil = None

# ----------------------------
# Windows System Log Tailer (pywin32)
# ----------------------------
EVENT_TYPES = {
    1: "ERROR",
    2: "WARNING",
    4: "INFO",
    8: "AUDIT_SUCCESS",
    16: "AUDIT_FAILURE",
}

SEVERITY_MAP = {
    0: "SUCCESS",
    1: "INFO",
    2: "WARNING",
    3: "ERROR",
}


class SystemLogTail:
    """Tail Windows 'System' log for last hour, keeping newest events first."""
    def __init__(self, lookback_seconds=3600, max_events=100, source_patterns=None):
        self.lookback_seconds = lookback_seconds
        self.buffer = deque(maxlen=max_events)
        self._last_record = 0
        self._source_regexes = (
            [re.compile(pat, re.I) for pat in source_patterns]
            if source_patterns
            else None
        )

    def _source_wanted(self, src: str) -> bool:
        if not self._source_regexes:
            return True
        for rx in self._source_regexes:
            if rx.search(src or ""):
                return True
        return False

    def poll(self):
        if not (sys.platform == "win32" and WIN32_EVT_OK):
            return
        try:
            server = "localhost"
            logtype = "System"
            hlog = win32evtlog.OpenEventLog(server, logtype)
            flags = (win32evtlog.EVENTLOG_BACKWARDS_READ |
                     win32evtlog.EVENTLOG_SEQUENTIAL_READ)
            cutoff = datetime.datetime.now() - datetime.timedelta(seconds=self.lookback_seconds)
            newest_seen = self._last_record

            while True:
                events = win32evtlog.ReadEventLog(hlog, flags, 0)
                if not events:
                    break
                for ev in events:
                    evt_time = getattr(ev, "TimeGenerated", None)
                    # stop once we’ve passed our time window
                    if evt_time and evt_time < cutoff:
                        win32evtlog.CloseEventLog(hlog)
                        return

                    recno = getattr(ev, "RecordNumber", 0) or 0
                    if recno <= newest_seen:
                        win32evtlog.CloseEventLog(hlog)
                        return

                    full_id = getattr(ev, "EventID", 0) or 0
                    level = EVENT_TYPES.get(getattr(ev, "EventType", 0), "INFO")

                    source = getattr(ev, "SourceName", "") or ""
                    if not self._source_wanted(source):
                        continue

                    event_id = full_id & 0xFFFF
                    msg = ""
                    if win32evtlogutil:
                        try:
                            msg = win32evtlogutil.SafeFormatMessage(ev, logtype)[:160]
                        except Exception:
                            msg = ""
                    if not msg:
                        inserts = getattr(ev, "StringInserts", None)
                        if inserts:
                            try:
                                msg = " ".join(str(x) for x in inserts if x)[:160]
                            except Exception:
                                msg = str(inserts[0])[:160]

                    item = {
                        "record": recno,
                        "level": level,
                        "source": source,
                        "event_id": int(event_id),
                        "time": evt_time.strftime("%H:%M:%S") if evt_time else "--:--:--",
                        "message": msg,
                    }
                    self.buffer.appendleft(item)
                    self._last_record = max(self._last_record, recno)

            win32evtlog.CloseEventLog(hlog)
        except Exception:
            logging.exception("System log poll failed")

    def latest(self, n=25):
        return list(self.buffer)[:n]

    def ticker_text(self):
        # Build a compact, looping ticker string from the last hour of events
        parts = []
        for e in self.latest(25):
            parts.append(f"[{e['time']}] {e['level']}: {e['source']}({e['event_id']}) — {e['message']}")
        if not parts:
            return "EVENTS: No events in the last hour."
        return "   |   ".join(parts) + "   |   "

# ----------------------------
# Data providers
# ----------------------------
class RealStats:
    def __init__(self):
        # Prime cpu_percent to get meaningful values without blocking
        psutil.cpu_percent(interval=0.0)
        self.last_cpu_pct = 0.0
        self.last_cpu_ts = 0.0
        self.last_net = psutil.net_io_counters(pernic=False)
        self.last_net_ts = time.time()
        self.last_net_rates = (0.0, 0.0)
        self.last_mem = psutil.virtual_memory()
        self.last_mem_ts = time.time()
        self.last_disk_ts = 0.0
        self.last_disks = []
        self._gpu_val = None
        self._gpu_thread = threading.Thread(target=self._gpu_sampler, daemon=True)
        self._gpu_thread.start()

    def _gpu_sampler(self):
        alpha = 0.3

        def smooth(u):
            if u is None:
                self._gpu_val = None
            elif self._gpu_val is None:
                self._gpu_val = u
            else:
                self._gpu_val = alpha * u + (1 - alpha) * self._gpu_val

        if win32pdh is not None:
            query = counter = None
            try:
                query = win32pdh.OpenQuery()
                counter = win32pdh.AddCounter(
                    query,
                    r"\\GPU Engine(*engtype_3D)\\Utilization Percentage",
                    0,
                )
                win32pdh.CollectQueryData(query)
                while True:
                    time.sleep(1.0)
                    win32pdh.CollectQueryData(query)
                    try:
                        _, vals = win32pdh.GetFormattedCounterArray(counter, win32pdh.PDH_FMT_DOUBLE)
                        total = sum(v.get("value", 0.0) for v in vals)
                        util = clamp(total, 0.0, 100.0)
                    except Exception:
                        logging.exception("GPU PDH read failed")
                        util = None
                    smooth(util)
            except Exception:
                logging.exception("GPU PDH sampler failed")
            finally:
                if counter is not None:
                    try:
                        win32pdh.RemoveCounter(counter)
                    except Exception:
                        pass
                if query is not None:
                    try:
                        win32pdh.CloseQuery(query)
                    except Exception:
                        pass

        while True:
            smooth(_gpu_3d_util_windows())
            time.sleep(1.0)

    def _gpu(self):
        if self._gpu_val is None:
            return None
        return {
            "util_pct": self._gpu_val,
            "vram_used_gb": None,
            "vram_total_gb": None,
        }

    def gpu_data_src(self):
        return "TM-3D" if WMI_CLIENT is not None and self._gpu_val is not None else "OS"

    def _net(self):
        now = time.time()
        if now - self.last_net_ts >= NET_SAMPLE_SEC:
            cur = psutil.net_io_counters(pernic=False)
            dt = max(1e-6, now - self.last_net_ts)
            up_mbps = (cur.bytes_sent - self.last_net.bytes_sent) * 8.0 / dt / 1e6
            down_mbps = (cur.bytes_recv - self.last_net.bytes_recv) * 8.0 / dt / 1e6
            self.last_net = cur
            self.last_net_ts = now
            self.last_net_rates = (up_mbps, down_mbps)
        return self.last_net_rates

    def snapshot(self):
        now = time.time()
        if now - self.last_cpu_ts >= CPU_SAMPLE_SEC:
            self.last_cpu_pct = psutil.cpu_percent(interval=None)
            self.last_cpu_ts = now
        cpu_pct = self.last_cpu_pct
        if now - self.last_mem_ts >= MEM_SAMPLE_SEC:
            self.last_mem = psutil.virtual_memory()
            self.last_mem_ts = now
        mem = self.last_mem
        mem_used_gb = (mem.total - mem.available) / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_pct = mem.percent

        if now - self.last_disk_ts >= DISK_SAMPLE_SEC:
            disks = []
            for p in psutil.disk_partitions(all=False):
                mount = p.mountpoint
                try:
                    u = psutil.disk_usage(mount)
                    disks.append({
                        "name": mount,
                        "used_gb": u.used / (1024**3),
                        "total_gb": u.total / (1024**3),
                        "pct": pct(u.used, u.total),
                    })
                except Exception:
                    continue
            self.last_disks = disks
            self.last_disk_ts = now
        disks = self.last_disks

        up, down = self._net()
        gpu = self._gpu()

        return {
            "ts": time.time(),
            "cpu_pct": cpu_pct,
            "gpu_util_pct": None if not gpu else gpu["util_pct"],
            "ram_pct": mem_pct,
            "mem_used_gb": mem_used_gb,
            "mem_total_gb": mem_total_gb,
            "disks": disks,
            "net_up_mbps": up,
            "net_down_mbps": down,
        }

class DemoStats:
    def __init__(self):
        self.t0 = time.time()

    def snapshot(self):
        t = time.time() - self.t0
        cpu_pct = 50 + 45 * (0.5 + 0.5 * math.sin(t * 0.7))
        gpu_pct = 30 + 65 * (0.5 + 0.5 * math.sin(t * 0.4 + 0.8))
        ram_pct = 40 + 25 * (0.5 + 0.5 * math.sin(t * 0.3 + 1.6))

        up = max(0.1, 6 + 4 * math.sin(t * 0.9) + 0.5 * math.sin(t * 3.7))
        down = max(0.2, 80 + 30 * math.sin(t * 0.4 + 1.4) + 3 * math.sin(t * 2.1))

        disks = [
            {"name": "C:", "used_gb": 420 + 2 * math.sin(t * 0.1), "total_gb": 931, "pct": (420/931)*100},
            {"name": "D:", "used_gb": 1280 + 5 * math.sin(t * 0.07), "total_gb": 1863, "pct": (1280/1863)*100},
        ]
        return {
            "ts": time.time(),
            "cpu_pct": cpu_pct,
            "gpu_util_pct": gpu_pct,
            "ram_pct": ram_pct,
            "mem_used_gb": 12.3,  # not used for gauge anymore, but kept for bars if needed
            "mem_total_gb": 32.0,
            "disks": disks,
            "net_up_mbps": up,
            "net_down_mbps": down,
        }

    def gpu_data_src(self):
        return "OS"

# ----------------------------
# UI Widgets
# ----------------------------
def draw_header(surface, font_lg, font_sm, data_src_label="OS"):
    glow_text(surface, font_lg, "VAULT-TEC // SYSTEMS MONITOR", (14, 5))
    # data source badge
    badge = f"DATA SRC: {data_src_label}"
    bw, _ = font_sm.size(badge)
    glow_text(surface, font_sm, badge, (WIDTH - bw - 14, 32))
    # thin divider
    pygame.draw.line(surface, GREEN, (12, 58), (WIDTH - 12, 58), 1)

def _make_gauge_frame(radius: int) -> pygame.Surface:
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    cx, cy = radius, radius
    pygame.draw.arc(surf, GREEN, (0, 0, radius * 2, radius * 2), GAUGE_ANGLE_MIN, GAUGE_ANGLE_MAX, 3)
    ticks = 7
    for i in range(ticks):
        a = GAUGE_ANGLE_MIN + (GAUGE_SWEEP * i / (ticks - 1))
        x1 = cx + (radius - 10) * math.cos(a)
        y1 = cy + (radius - 10) * math.sin(a)
        x2 = cx + (radius - 2) * math.cos(a)
        y2 = cy + (radius - 2) * math.sin(a)
        pygame.draw.line(surf, GREEN, (x1, y1), (x2, y2), 2)
    return surf


def draw_gauge(surface, center, radius, value, vmin, vmax, label, unit, font, font_sm, frame_surf, hot=False):
    cx, cy = center
    surface.blit(frame_surf, (cx - radius, cy - radius))

    # needle
    if value is not None:
        frac = clamp((value - vmin) / (vmax - vmin), 0.0, 1.0)
        a = GAUGE_ANGLE_MIN + GAUGE_SWEEP * frac
        nx = cx + (radius - 16) * math.cos(a)
        ny = cy + (radius - 16) * math.sin(a)
        pygame.draw.line(surface, GREEN, (cx, cy), (nx, ny), 3)
        pygame.draw.circle(surface, GREEN, (int(cx), int(cy)), 4, 0)

    vtxt = "--" if value is None else (f"{int(round(value))}{unit}")
    vcolor = (255, 60, 60) if hot else GREEN
    glow_text(surface, font, vtxt, (cx - 20, cy + radius - 50), color=vcolor)

    lab_w, _ = font_sm.size(label)
    glow_text(surface, font_sm, label, (int(cx - lab_w/2), int(cy + radius - 70)))

def draw_bar(surface, font, x1, y1, x2, y2, pct_val, label, right_text):
    # frame
    pygame.draw.rect(surface, GREEN, (x1, y1, x2 - x1, y2 - y1), width=1)
    # fill
    fill_w = int((x2 - x1 - 8) * clamp(pct_val / 100.0, 0.0, 1.0))
    pygame.draw.rect(surface, GREEN, (x1 + 1, y1 + 1, fill_w, (y2 - y1) - 2), width=0)

    # text BELOW bar
    glow_text(surface, font, f"{label}", (x1 + 1, y2 + 1))
    rt_surf = _render_text_cached(right_text, font)
    surface.blit(rt_surf, (x2 - rt_surf.get_width(), y2 + 1))


def draw_graph(surface, font, x, y, w, h, up_hist, down_hist):
    """Centered baseline: upload spikes up, download spikes down."""
    # frame
    pygame.draw.rect(surface, GREEN, (x, y, w, h), width=1)

    # midline (zero)
    mid = y + h // 2
    pygame.draw.line(surface, GREEN, (x + 1, mid), (x + w - 2, mid), 1)

    # grid (symmetrical about midline)
    rows = 4  # total horizontal grid lines per half
    cols = 6
    # horizontal lines
    for i in range(1, rows):
        dy = int((h // 2) * i / rows)
        y_up = mid - dy
        y_dn = mid + dy
        pygame.draw.line(surface, DIM_GREEN, (x + 1, y_up), (x + w - 2, y_up), 1)
        pygame.draw.line(surface, DIM_GREEN, (x + 1, y_dn), (x + w - 2, y_dn), 1)
    # vertical lines
    for j in range(1, cols):
        xx = x + int(w * j / cols)
        pygame.draw.line(surface, DIM_GREEN, (xx, y + 1), (xx, y + h - 2), 1)

    # autoscale from max of both series
    max_up = max(up_hist) if up_hist else 1.0
    max_dn = max(down_hist) if down_hist else 1.0
    vmax = max(1.0, max_up, max_dn)  # Mb/s
    half_h = (h - 4) / 2.0  # breathing room

    def y_for_upload(val):
        return int(mid - (val / vmax) * half_h)

    def y_for_download(val):
        return int(mid + (val / vmax) * half_h)

    # Build x coordinates
    def x_for_index(i, n):
        return x + int(i * (w - 2) / max(1, n - 1))

    # plot upload (solid, above midline)
    n_up = len(up_hist)
    step_up = 2 if n_up >= GRAPH_POINTS else 1
    if n_up >= 2:
        pts = [(x_for_index(i, n_up), y_for_upload(up_hist[i])) for i in range(0, n_up, step_up)]
        if len(pts) >= 2:
            pygame.draw.lines(surface, GREEN, False, pts, 2)

    # plot download (dashed, below midline)
    n_dn = len(down_hist)
    step_dn = 2 if n_dn >= GRAPH_POINTS else 1
    if n_dn >= 2:
        pts = [(x_for_index(i, n_dn), y_for_download(down_hist[i])) for i in range(0, n_dn, step_dn)]
        for i in range(0, len(pts) - 1, 2):
            pygame.draw.line(surface, DIM_GREEN, pts[i], pts[i + 1], 2)

    # legend & vmax
    glow_text(surface, font, f"NET ↑ / ↓  (Mb/s)   ±max={vmax:.0f}", (x + 8, y + 6))

# ----------------------------
# Scrolling Ticker
# ----------------------------
class Ticker:
    def __init__(self, font):
        self.font = font
        self.text = "EVENTS: initializing..."
        self.speed = 80  # pixels per second
        self.x = WIDTH  # start off right edge
        self.surf = _render_text_cached(self.text, self.font)
        self.last_update = time.time()

    def set_text(self, text: str):
        if not text:
            text = " "
        self.text = text
        self.surf = _render_text_cached(self.text, self.font)
        # keep x as-is to avoid jump; if text shorter, loop will catch up

    def update(self):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        self.x -= self.speed * dt
        # Loop when the whole surface has scrolled past left edge
        if self.x < -self.surf.get_width():
            self.x = WIDTH

    def draw(self, surface, y):
        # draw ticker frame
        pygame.draw.rect(surface, GREEN, (12, y, WIDTH - 24, TICKER_H - 12), 1)
        # clip area
        clip = pygame.Rect(16, y + 4, WIDTH - 32, TICKER_H - 20)
        prev = surface.get_clip()
        surface.set_clip(clip)
        surface.blit(self.surf, (int(self.x), y + 8))
        # also draw a second copy to create seamless loop
        surface.blit(self.surf, (int(self.x) + self.surf.get_width() + 40, y + 8))
        surface.set_clip(prev)

def start_screen(surface, font_lg, logo):
    surface.fill(BG_COLOR)
    draw_scanlines(surface)

    lw, lh = logo.get_size()
    lx = (WIDTH - lw) // 2
    ly = (HEIGHT - lh) // 2 + LOGO_START_Y_OFFSET  # instructions: tweak LOGO_START_Y_OFFSET to move logo
    surface.blit(logo, (lx, ly))

    # START button rectangle in middle
    btn_w, btn_h = 200, 56
    btn_x = (WIDTH - btn_w) // 2
    btn_y = ly + lh + 30

    pygame.draw.rect(surface, GREEN, (btn_x, btn_y, btn_w, btn_h), width=2)
    # emboss lines
    pygame.draw.line(surface, DIM_GREEN, (btn_x + 6, btn_y + 6), (btn_x + btn_w - 6, btn_y + 6), 2)
    pygame.draw.line(surface, DIM_GREEN, (btn_x + 6, btn_y + btn_h - 6), (btn_x + btn_w - 6, btn_y + btn_h - 6), 2)

    txt = "START"
    tw, th = font_lg.size(txt)
    glow_text(surface, font_lg, txt, (btn_x + (btn_w - tw) // 2, btn_y + (btn_h - th) // 2))

    # footer hint
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True) or pygame.font.SysFont(None, FONT_SIZE_SMALL, bold=True)
    glow_text(surface, font, "Tap screen or click to begin", (WIDTH // 2 - 110, btn_y + btn_h + 14))

    return pygame.Rect(btn_x, btn_y, btn_w, btn_h)

# ----------------------------
# Main app
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run with simulated data (no sensors needed)")
    parser.add_argument(
        "--icon",
        metavar="PATH",
        help="Path to a .ico file for the window icon (defaults to 'pippy.ico' if present)",
    )
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("Vault-Tec Monitor")

    icon_path = args.icon
    if not icon_path:
        default_icon = os.path.join(os.path.dirname(__file__), "pippy.ico")
        if os.path.exists(default_icon):
            icon_path = default_icon
    elif not os.path.isabs(icon_path):
        icon_path = os.path.join(os.path.dirname(__file__), icon_path)
    if icon_path and os.path.exists(icon_path):
        try:
            pygame.display.set_icon(pygame.image.load(icon_path))
        except Exception as exc:  # pragma: no cover - icon failures are non-critical
            print(f"Could not load icon '{icon_path}': {exc}")

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # fonts
    try:
        font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=True)
        font_sm = pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True)
        font_lg = pygame.font.SysFont(FONT_NAME, FONT_SIZE_LARGE, bold=True)
    except Exception:
        font = pygame.font.SysFont(None, FONT_SIZE, bold=True)
        font_sm = pygame.font.SysFont(None, FONT_SIZE_SMALL, bold=True)
        font_lg = pygame.font.SysFont(None, FONT_SIZE_LARGE, bold=True)

    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    logo = load_tinted_logo(logo_path)
    logo_start = scale_logo(logo, LOGO_START_MAX)
    logo_main = scale_logo(logo, LOGO_MAIN_MAX)

    # data provider
    provider = DemoStats() if args.demo else RealStats()

    # event log + ticker
    ev_tail = SystemLogTail(lookback_seconds=3600, max_events=60) if WIN32_EVT_OK else None
    ticker = Ticker(font)

    # Pre-cache static text surfaces
    _render_text_cached("VAULT-TEC // SYSTEMS MONITOR", font_lg)
    for lab in ("CPU LOAD", "GPU LOAD", "RAM USAGE"):
        _render_text_cached(lab, font_sm)
    for badge in ("DATA SRC: OS", "DATA SRC: TM-3D"):
        _render_text_cached(badge, font_sm)

    gauge_frame = _make_gauge_frame(GAUGE_RADIUS)

    background = pygame.Surface((WIDTH, HEIGHT))
    background.fill(BG_COLOR)
    draw_scanlines(background)
    logo_x = (WIDTH - logo_main.get_width()) // 2 + LOGO_MAIN_X_OFFSET
    logo_y = GRAPH_Y - logo_main.get_height() + LOGO_MAIN_Y_OFFSET
    background.blit(logo_main, (logo_x, logo_y))
    pygame.draw.line(background, GREEN, (12, GRAPH_Y - 8), (WIDTH - 12, GRAPH_Y - 8), 1)

    # histories for graph
    up_hist = deque(maxlen=GRAPH_POINTS)
    dn_hist = deque(maxlen=GRAPH_POINTS)

    state = "start"
    start_btn_rect = None
    last_event_poll = 0.0
    need_full_redraw = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit(0)
            if state == "start" and event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                if start_btn_rect and start_btn_rect.collidepoint(event.pos):
                    state = "dash"
                    need_full_redraw = True

        if state == "start":
            screen.fill(BG_COLOR)
            draw_scanlines(screen)
            draw_header(screen, font_lg, font_sm, provider.gpu_data_src())
            start_btn_rect = start_screen(screen, font_lg, logo_start)
            pygame.display.update([screen.get_rect()])
            clock.tick(FPS)
            continue

        snap = provider.snapshot()

        dirty: list[pygame.Rect] = []
        if need_full_redraw:
            screen.blit(background, (0, 0))
            dirty.append(screen.get_rect())
            need_full_redraw = False

        header_rect = pygame.Rect(0, 0, WIDTH, 60)
        screen.blit(background, header_rect, header_rect)
        draw_header(screen, font_lg, font_sm, provider.gpu_data_src())
        dirty.append(header_rect)

        gy = 150
        centers = [(80, gy), (240, gy), (400, gy)]

        cpu_pct = snap.get("cpu_pct")
        gpu_util = snap.get("gpu_util_pct")
        ram_pct = snap.get("ram_pct")

        rect = gauge_rect(centers[0], GAUGE_RADIUS)
        screen.blit(background, rect, rect)
        draw_gauge(screen, centers[0], GAUGE_RADIUS, cpu_pct, 0, 100, "CPU LOAD", "%", font, font_sm, gauge_frame, hot=(cpu_pct is not None and cpu_pct >= CPU_HOT_PCT))
        dirty.append(rect)

        rect = gauge_rect(centers[1], GAUGE_RADIUS)
        screen.blit(background, rect, rect)
        draw_gauge(screen, centers[1], GAUGE_RADIUS, gpu_util, 0, 100, "GPU LOAD", "%", font, font_sm, gauge_frame, hot=(gpu_util is not None and gpu_util >= GPU_HOT_PCT))
        dirty.append(rect)

        rect = gauge_rect(centers[2], GAUGE_RADIUS)
        screen.blit(background, rect, rect)
        draw_gauge(screen, centers[2], GAUGE_RADIUS, ram_pct, 0, 100, "RAM USAGE", "%", font, font_sm, gauge_frame, hot=(ram_pct is not None and ram_pct >= RAM_HOT_PCT))
        dirty.append(rect)

        y = MID_BAND_Y + 10
        for d in (snap["disks"] or [])[:6]:
            bar_pct = clamp(d["pct"], 0, 100)
            rect = bar_rect(BAR_LEFT, y, BAR_RIGHT, y + BAR_H, font)
            screen.blit(background, rect, rect)
            draw_bar(screen, font, BAR_LEFT, y, BAR_RIGHT, y + BAR_H, bar_pct,
                     f"DSK {d['name']} {human_gb(d['used_gb'])}/{human_gb(d['total_gb'])}",
                     f"{bar_pct:.0f}%")
            dirty.append(rect)
            y += BAR_H + BAR_GAP

        up_hist.append(snap["net_up_mbps"])
        dn_hist.append(snap["net_down_mbps"])
        rect = graph_rect(12, GRAPH_Y, WIDTH - 24, GRAPH_H)
        screen.blit(background, rect, rect)
        draw_graph(screen, font_sm, 12, GRAPH_Y, WIDTH - 24, GRAPH_H, up_hist, dn_hist)
        dirty.append(rect)

        if ev_tail and (time.time() - last_event_poll > 5.0):
            ev_tail.poll()
            ticker.set_text(ev_tail.ticker_text())
            last_event_poll = time.time()
        elif not ev_tail:
            ticker.set_text("Install 'pywin32' to enable System log ticker (Windows only).")

        ticker.update()
        rect = ticker_rect(TICKER_Y + 2)
        screen.blit(background, rect, rect)
        ticker.draw(screen, TICKER_Y + 2)
        dirty.append(rect)

        pygame.display.update(dirty)
        clock.tick(FPS)

if __name__ == "__main__":
    main()
