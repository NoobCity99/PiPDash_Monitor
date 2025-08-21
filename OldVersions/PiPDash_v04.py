
import argparse
import math
import os
import re
import sys
import time
from collections import deque

import pygame
import psutil

# ----------------------------
# Optional GPU metrics via NVIDIA NVML
# ----------------------------
NVML_OK = False
try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

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

# ----------------------------
# Display config (portrait)
# ----------------------------
WIDTH, HEIGHT = 480, 800   # portrait: 480 wide x 800 tall
FPS = 30

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

LOGO_PATH = os.path.join("vt.png")

# ----------------------------
# Layout geometry
# ----------------------------
TOP_BAND_H = 230
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
# Alerts / thresholds
# ----------------------------
CPU_HOT_PCT = 95.0
GPU_HOT_PCT = 95.0
RAM_HOT_PCT = 90.0

# ----------------------------
# Utility
# ----------------------------

def load_tinted_logo(path, tint=(0, 255, 70), opacity=220):

    surf = pygame.image.load(path).convert_alpha()
    # Multiply the RGB by your tint: white stays tint, gray becomes dim green
    tint_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    tint_surf.fill((*tint, 255))
    # Multiply colors; keep alpha from the original
    surf = surf.copy()
    surf.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.set_alpha(opacity)
    return surf

def draw_logo_centered(surface, logo_surf, area_rect, pad=6, allow_upscale=True):

    if logo_surf is None or area_rect.height <= 10 or area_rect.width <= 10:
        return
    lw, lh = logo_surf.get_size()
    max_w = max(1, area_rect.w - 2 * pad)
    max_h = max(1, area_rect.h - 2 * pad)
    scale = min(max_w / lw, max_h / lh)
    if not allow_upscale:
        scale = min(1.0, scale)
    new_size = (max(1, int(lw * scale)), max(1, int(lh * scale)))
    scaled = pygame.transform.smoothscale(logo_surf, new_size)
    surface.blit(scaled, (area_rect.centerx - scaled.get_width() // 2,
                          area_rect.centery - scaled.get_height() // 2))

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def pct(a, b):
    b = max(1e-9, b)
    return 100.0 * (a / b)

def human_gb(x):
    return f"{x:.1f} GB"

def draw_scanlines(surface):
    for y in range(0, HEIGHT, SCANLINE_STEP):
        pygame.draw.line(surface, SCANLINE_COLOR, (0, y), (WIDTH, y), 1)

def glow_text(surface, font, text, pos, color=GREEN, glow=DIM_GREEN, offset=1):
    x, y = pos
    if not text:
        return
    # simple "CRT glow": shadow underlay then main text
    surf_g = font.render(text, True, glow)
    surface.blit(surf_g, (x - offset, y - offset))
    surface.blit(surf_g, (x + offset, y + offset))
    surface.blit(font.render(text, True, color), pos)

# ----------------------------
# Windows System Log Tailer (pywin32)
# ----------------------------
EVENT_TYPES = {
    1: "CRITICAL",
    2: "ERROR",
    3: "WARNING",
    4: "INFO",
    8: "AUDIT_SUCCESS",
    16: "AUDIT_FAILURE",
}

# Five categories to include from the System log
# (broad regexes covering common source names)
EVENT_SOURCE_PATTERNS = [
    r"^Kernel-Power$",                 # unexpected shutdowns
    r"^(Disk|Ntfs)$",                  # storage issues
    r"^WHEA-Logger$",                  # corrected hardware errors
    r"^(e1rexpress|Netwtw\w*|rtwlane|rtwlanu|athr|Killer|ndis|rtl\w+)$",  # NIC resets/timeouts
    r"^Service Control Manager$",      # service failures
]

SOURCE_REGEXES = [re.compile(pat, re.I) for pat in EVENT_SOURCE_PATTERNS]

class SystemLogTail:
    """Tail Windows 'System' log for last hour, keeping newest events first."""
    def __init__(self, lookback_seconds=3600, max_events=100):
        self.lookback_seconds = lookback_seconds
        self.buffer = deque(maxlen=max_events)
        self._last_record = 0

    def _source_wanted(self, src: str) -> bool:
        for rx in SOURCE_REGEXES:
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
            cutoff = time.time() - self.lookback_seconds
            newest_seen = self._last_record

            while True:
                events = win32evtlog.ReadEventLog(hlog, flags, 0)
                if not events:
                    break
                for ev in events:
                    # stop once we’ve passed our time window
                    if ev.TimeGenerated and ev.TimeGenerated < cutoff:
                        win32evtlog.CloseEventLog(hlog)
                        return

                    recno = getattr(ev, "RecordNumber", 0) or 0
                    if recno <= newest_seen:
                        win32evtlog.CloseEventLog(hlog)
                        return

                    level = EVENT_TYPES.get(ev.EventType, str(ev.EventType))
                    if level not in ("CRITICAL", "ERROR", "WARNING"):
                        continue

                    source = getattr(ev, "SourceName", "") or ""
                    if not self._source_wanted(source):
                        continue

                    event_id = getattr(ev, "EventID", 0) & 0xFFFF
                    # Basic message preview from first string insert (if any)
                    inserts = getattr(ev, "StringInserts", None)
                    msg = ""
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
                        "time": time.strftime("%H:%M:%S", time.localtime(ev.TimeGenerated)) if ev.TimeGenerated else "--:--:--",
                        "message": msg,
                    }
                    self.buffer.appendleft(item)
                    self._last_record = max(self._last_record, recno)

            win32evtlog.CloseEventLog(hlog)
        except Exception:
            # Swallow transient API errors; the UI should keep running
            pass

    def latest(self, n=25):
        return list(self.buffer)[:n]

    def ticker_text(self):
        # Build a compact, looping ticker string from the last hour of events
        parts = []
        for e in self.latest(25):
            parts.append(f"[{e['time']}] {e['level']}: {e['source']}({e['event_id']}) — {e['message']}")
        if not parts:
            return "EVENTS: No recent WARN/ERROR from selected sources in the last hour."
        return "   |   ".join(parts) + "   |   "

# ----------------------------
# Data providers
# ----------------------------
class RealStats:
    def __init__(self):
        # Prime cpu_percent to get meaningful values without blocking
        psutil.cpu_percent(interval=None)
        self.last_net = psutil.net_io_counters(pernic=False)
        self.last_t = time.time()

    def _gpu(self):
        if not NVML_OK:
            return None
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_used_gb = mem.used / (1024**3)
            vram_total_gb = mem.total / (1024**3)
            return {
                "util_pct": util,
                "vram_used_gb": vram_used_gb,
                "vram_total_gb": vram_total_gb,
            }
        except Exception:
            return None

    def _net(self):
        now = time.time()
        cur = psutil.net_io_counters(pernic=False)
        dt = max(1e-6, now - self.last_t)
        up_mbps = (cur.bytes_sent - self.last_net.bytes_sent) * 8.0 / dt / 1e6
        down_mbps = (cur.bytes_recv - self.last_net.bytes_recv) * 8.0 / dt / 1e6
        self.last_net, self.last_t = cur, now
        return up_mbps, down_mbps

    def snapshot(self):
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_used_gb = (mem.total - mem.available) / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_pct = mem.percent

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

# ----------------------------
# UI Widgets
# ----------------------------
def draw_header(surface, font_lg, font_sm, data_src_label="OS"):
    glow_text(surface, font_lg, "VAULT-TEC // SYSTEMS MONITOR", (14, 12))
    # data source badge (NVML/OS)
    badge = f"DATA SRC: {'NVML' if NVML_OK else data_src_label}"
    bw, _ = font_sm.size(badge)
    glow_text(surface, font_sm, badge, (WIDTH - bw - 14, 14))
    # thin divider
    pygame.draw.line(surface, GREEN, (12, 58), (WIDTH - 12, 58), 1)

def draw_gauge(surface, center, radius, value, vmin, vmax, label, unit, hot=False):
    cx, cy = center
    # frame arc
    pygame.draw.arc(surface, GREEN,
                    (cx - radius, cy - radius, radius * 2, radius * 2),
                    GAUGE_ANGLE_MIN, GAUGE_ANGLE_MAX, 3)
    # tick marks
    ticks = 7
    for i in range(ticks):
        a = GAUGE_ANGLE_MIN + (GAUGE_SWEEP * i / (ticks - 1))
        x1 = cx + (radius - 10) * math.cos(a)
        y1 = cy + (radius - 10) * math.sin(a)
        x2 = cx + (radius - 2) * math.cos(a)
        y2 = cy + (radius - 2) * math.sin(a)
        pygame.draw.line(surface, GREEN, (x1, y1), (x2, y2), 2)

    # needle
    if value is not None:
        frac = clamp((value - vmin) / (vmax - vmin), 0.0, 1.0)
        a = GAUGE_ANGLE_MIN + GAUGE_SWEEP * frac
        nx = cx + (radius - 16) * math.cos(a)
        ny = cy + (radius - 16) * math.sin(a)
        pygame.draw.line(surface, GREEN, (cx, cy), (nx, ny), 3)
        pygame.draw.circle(surface, GREEN, (int(cx), int(cy)), 4, 0)

    # label & value
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=True) or pygame.font.SysFont(None, FONT_SIZE, bold=True)
    font_sm = pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True) or pygame.font.SysFont(None, FONT_SIZE_SMALL, bold=True)

    vtxt = "--" if value is None else (f"{int(round(value))}{unit}")
    vcolor = (255, 60, 60) if hot else GREEN
    glow_text(surface, font, vtxt, (cx - 28, cy + radius - 10), color=vcolor)

    lab_w, _ = font_sm.size(label)
    glow_text(surface, font_sm, label, (int(cx - lab_w/2), int(cy + radius + 10)))

def draw_bar(surface, x1, y1, x2, y2, pct_val, label, right_text):
    # frame
    pygame.draw.rect(surface, GREEN, (x1, y1, x2 - x1, y2 - y1), width=1)
    # fill
    fill_w = int((x2 - x1 - 2) * clamp(pct_val / 100.0, 0.0, 1.0))
    pygame.draw.rect(surface, GREEN, (x1 + 1, y1 + 1, fill_w, (y2 - y1) - 2), width=0)

    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=True) or pygame.font.SysFont(None, FONT_SIZE, bold=True)
    # text BELOW bar
    glow_text(surface, font, f"{label}", (x1 + 6, y2 + 6))
    rt_surf = font.render(right_text, True, GREEN)
    surface.blit(rt_surf, (x2 - rt_surf.get_width(), y2 + 6))

    

def draw_graph(surface, x, y, w, h, up_hist, down_hist):
    # frame
    pygame.draw.rect(surface, GREEN, (x, y, w, h), width=1)
    # grid
    rows = 4
    cols = 6
    for i in range(1, rows):
        yy = y + int(h * i / rows)
        pygame.draw.line(surface, DIM_GREEN, (x, yy), (x + w, yy), 1)
    for j in range(1, cols):
        xx = x + int(w * j / cols)
        pygame.draw.line(surface, DIM_GREEN, (xx, y), (xx, y + h), 1)

    # autoscale based on max over window (avoid divide by zero)
    max_up = max(up_hist) if up_hist else 1.0
    max_dn = max(down_hist) if down_hist else 1.0
    vmax = max(1.0, max(max_up, max_dn))

    def to_y(val):
        return y + h - int((val / vmax) * (h - 2))

    # plot down (solid)
    if len(down_hist) >= 2:
        pts = []
        for i, v in enumerate(down_hist):
            px = x + int(i * (w - 2) / max(1, len(down_hist) - 1))
            py = to_y(v)
            pts.append((px, py))
        pygame.draw.lines(surface, GREEN, False, pts, 2)

    # plot up (dim/dashed)
    if len(up_hist) >= 2:
        pts = []
        for i, v in enumerate(up_hist):
            px = x + int(i * (w - 2) / max(1, len(up_hist) - 1))
            py = to_y(v)
            pts.append((px, py))
        # dashed: draw every other segment
        for i in range(0, len(pts) - 1, 2):
            pygame.draw.line(surface, DIM_GREEN, pts[i], pts[i + 1], 2)

    # legend & vmax
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True) or pygame.font.SysFont(None, FONT_SIZE_SMALL, bold=True)
    glow_text(surface, font, f"NET ↓ / ↑  (Mb/s)   scale max={vmax:.0f}", (x + 8, y + 6))

# ----------------------------
# Scrolling Ticker
# ----------------------------
class Ticker:
    def __init__(self, font):
        self.font = font
        self.text = "EVENTS: initializing..."
        self.speed = 80  # pixels per second
        self.x = WIDTH  # start off right edge
        self.surf = self.font.render(self.text, True, GREEN)
        self.last_update = time.time()

    def set_text(self, text: str):
        if not text:
            text = " "
        self.text = text
        self.surf = self.font.render(self.text, True, GREEN)
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

# ----------------------------
# Start screen with ASCII gear
# ----------------------------
GEAR = [
    "                                      ",
    "                                      ",
    "                                      ",
    "                                      ",
    "                                      ",
    "      <<<<<<<<  /oo0oo\\  >>>>>>>>>     ",
    "               /ooooooo\\              ",
    "  <<<<<<<<<<< |000o0o000| >>>>>>>>>>>>",
    "               \\ooooooo/               ",
    "     <<<<<<<<<  \\oo0oo/  >>>>>>>>>      ",
    "                                       ",
    "                                       ",
    "                                       ",
    "                                       ",
    "                                       ",
]

def start_screen(surface, font_lg):
    surface.fill(BG_COLOR)
    draw_scanlines(surface)
    # center the gear block
    gear_font = pygame.font.SysFont(FONT_NAME, 18, bold=True) or pygame.font.SysFont(None, 18, bold=True)
    gear_w = max(gear_font.size(line)[0] for line in GEAR)
    gear_h = len(GEAR) * (gear_font.get_height() + 0)

    gx = (WIDTH - gear_w) // 2
    gy = 160

    for i, line in enumerate(GEAR):
        glow_text(surface, gear_font, line, (gx, gy + i * (gear_font.get_height())))

    # START button rectangle in middle
    btn_w, btn_h = 200, 56
    btn_x = (WIDTH - btn_w) // 2
    btn_y = gy + gear_h + 30

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
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("Vault-Tec Monitor")
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

    # data provider
    provider = DemoStats() if args.demo else RealStats()

    # logo:
    try:
        logo = load_tinted_logo(LOGO_PATH, tint=(0, 255, 70), opacity=220)
    # logo = load_mask_logo(LOGO_PATH, color=(0, 255, 70), opacity=220)
    except Exception:
        logo = None  # Fail gracefully if asset missing


    # event log + ticker
    ev_tail = SystemLogTail(lookback_seconds=3600, max_events=60) if WIN32_EVT_OK else None
    ticker = Ticker(font)

    # histories for graph
    up_hist = deque(maxlen=GRAPH_POINTS)
    dn_hist = deque(maxlen=GRAPH_POINTS)

    state = "start"
    start_btn_rect = None
    last_event_poll = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit(0)
            if state == "start" and event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                if start_btn_rect and start_btn_rect.collidepoint(event.pos):
                    state = "dash"

        if state == "start":
            screen.fill(BG_COLOR)
            draw_scanlines(screen)
            draw_header(screen, font_lg, font_sm)
            start_btn_rect = start_screen(screen, font_lg)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # ---- Dashboard ----
        snap = provider.snapshot()

        screen.fill(BG_COLOR)
        draw_scanlines(screen)
        draw_header(screen, font_lg, font_sm)

        # --- Top gauges ---
        gy = 150  # nudged down for header spacing
        centers = [(80, gy), (240, gy), (400, gy)]

        cpu_pct = snap.get("cpu_pct")
        gpu_util = snap.get("gpu_util_pct")
        ram_pct = snap.get("ram_pct")

        draw_gauge(screen, centers[0], GAUGE_RADIUS, cpu_pct, 0, 100, "CPU LOAD", "%", hot=(cpu_pct is not None and cpu_pct >= CPU_HOT_PCT))
        draw_gauge(screen, centers[1], GAUGE_RADIUS, gpu_util, 0, 100, "GPU LOAD", "%", hot=(gpu_util is not None and gpu_util >= GPU_HOT_PCT))
        draw_gauge(screen, centers[2], GAUGE_RADIUS, ram_pct, 0, 100, "RAM USAGE", "%", hot=(ram_pct is not None and ram_pct >= RAM_HOT_PCT))

        # --- Middle bars (Disks only; RAM bar removed) ---
        y = MID_BAND_Y + 10
        # Disks (up to 6 rows)
        for d in (snap["disks"] or [])[:6]:
            bar_pct = clamp(d["pct"], 0, 100)
            draw_bar(screen, BAR_LEFT, y, BAR_RIGHT, y + BAR_H, bar_pct,
                     f"DSK {d['name']} {human_gb(d['used_gb'])}/{human_gb(d['total_gb'])}",
                     f"{bar_pct:.0f}%")
            y += BAR_H + BAR_GAP
        logo_top = y + 6
        logo_bottom = GRAPH_Y - 12  # stay clear of the graph header/legend
        available_h = logo_bottom - logo_top
        if logo and available_h >= 40:
            area_rect = pygame.Rect(24, logo_top, WIDTH - 48, available_h)
            draw_logo_centered(screen, logo, area_rect, pad=6, allow_upscale=True)


        # divider above graph
        pygame.draw.line(screen, GREEN, (12, GRAPH_Y - 8), (WIDTH - 12, GRAPH_Y - 8), 1)

        # --- Bottom graph (NET) ---
        up_hist.append(snap["net_up_mbps"])
        dn_hist.append(snap["net_down_mbps"])
        draw_graph(screen, 12, GRAPH_Y, WIDTH - 24, GRAPH_H, up_hist, dn_hist)

        # --- Event ticker at very bottom ---
        # Poll every 5 seconds
        if ev_tail and (time.time() - last_event_poll > 5.0):
            ev_tail.poll()
            ticker.set_text(ev_tail.ticker_text())
            last_event_poll = time.time()
        elif not ev_tail:
            # Friendly message when pywin32 isn't installed or non-Windows
            ticker.set_text("Install 'pywin32' to enable System log ticker (Windows only).")

        ticker.update()
        ticker.draw(screen, TICKER_Y + 2)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
