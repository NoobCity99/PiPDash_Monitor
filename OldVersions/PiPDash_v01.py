import argparse
import math
import os
import random
import sys
import time
from collections import deque

import pygame
import psutil

# Optional GPU metrics via NVIDIA NVML
NVML_OK = False
try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

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

# Gauges geometry (top band)
TOP_BAND_H = 230
GAUGE_RADIUS = 85
GAUGE_ANGLE_MIN = math.radians(210)   # -150°
GAUGE_ANGLE_MAX = math.radians(330)   # +150°
GAUGE_SWEEP = GAUGE_ANGLE_MAX - GAUGE_ANGLE_MIN

# Bars geometry (middle band)
MID_BAND_Y = TOP_BAND_H + 10
MID_BAND_H = 420
BAR_H = 28
BAR_GAP = 12
BAR_LEFT = 24
BAR_RIGHT = WIDTH - 24

# Net graph geometry (bottom band)
BOT_BAND_H = HEIGHT - (TOP_BAND_H + MID_BAND_H)
GRAPH_Y = HEIGHT - BOT_BAND_H + 6
GRAPH_H = BOT_BAND_H - 24
GRAPH_POINTS = WIDTH  # one point per x pixel

# Alerts
CPU_HOT_C = 85.0
GPU_HOT_C = 85.0

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


def draw_scanlines(surface):
    for y in range(0, HEIGHT, SCANLINE_STEP):
        pygame.draw.line(surface, SCANLINE_COLOR, (0, y), (WIDTH, y), 1)


def glow_text(surface, font, text, pos, color=GREEN, glow=DIM_GREEN, offset=1):
    x, y = pos
    # simple "CRT glow": shadow underlay then main text
    surf_g = font.render(text, True, glow)
    surface.blit(surf_g, (x - offset, y - offset))
    surface.blit(surf_g, (x + offset, y + offset))
    surface.blit(font.render(text, True, color), pos)

# ----------------------------
# Data providers
# ----------------------------


class RealStats:
    def __init__(self):
        self.last_net = psutil.net_io_counters()
        self.last_t = time.time()

    def _gpu(self):
        if not NVML_OK:
            return None
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = float(pynvml.nvmlDeviceGetTemperature(
                h, pynvml.NVML_TEMPERATURE_GPU))
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            p_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
            return {"temp_c": temp, "util_pct": float(util), "power_w": float(p_w)}
        except Exception:
            return None

    def _cpu_temp(self):
        # psutil temps often work on Linux; on Windows they're often unavailable.
        try:
            temps = psutil.sensors_temperatures()
            # Heuristic: prefer entries that look like package/core
            for k, arr in temps.items():
                for s in arr:
                    label = (s.label or "").lower()
                    if "package" in label or "cpu" in label or "core 0" in label:
                        return float(s.current)
            # fallback: any sensor
            for k, arr in temps.items():
                if arr:
                    return float(arr[0].current)
        except Exception:
            pass
        return None  # unknown

    def _net(self):
        now = time.time()
        cur = psutil.net_io_counters()
        dt = max(1e-6, now - self.last_t)
        up_mbps = (cur.bytes_sent - self.last_net.bytes_sent) * 8.0 / dt / 1e6
        down_mbps = (cur.bytes_recv - self.last_net.bytes_recv) * \
            8.0 / dt / 1e6
        self.last_net, self.last_t = cur, now
        return up_mbps, down_mbps

    def snapshot(self):
        cpu_temp = self._cpu_temp()
        gpu = self._gpu()

        mem = psutil.virtual_memory()
        disks = []
        for p in psutil.disk_partitions(all=False):
            # Skip pseudo/dev mounts
            if os.name == "nt":
                mount = p.mountpoint
            else:
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

        return {
            "ts": time.time(),
            "cpu_temp_c": cpu_temp,                      # None if unknown
            "gpu_temp_c": None if not gpu else gpu["temp_c"],
            "gpu_power_w": None if not gpu else gpu["power_w"],
            "mem_used_gb": (mem.total - mem.available) / (1024**3),
            "mem_total_gb": mem.total / (1024**3),
            "disks": disks,
            "net_up_mbps": up,
            "net_down_mbps": down,
        }


class DemoStats:
    def __init__(self):
        self.t0 = time.time()
        self._up = 0.0
        self._down = 0.0

    def snapshot(self):
        t = time.time() - self.t0
        cpu_temp = 50 + 25 * (0.5 + 0.5 * math.sin(t * 0.6))
        gpu_temp = 55 + 20 * (0.5 + 0.5 * math.sin(t * 0.5 + 1.2))
        gpu_power = 80 + 60 * (0.5 + 0.5 * math.sin(t * 0.8 + 2.3))
        mem_total = 32.0
        mem_used = 10 + 12 * (0.5 + 0.5 * math.sin(t * 0.3))

        # wobbly network traffic
        up = max(0.1, 4 + 3 * math.sin(t * 0.9) + random.uniform(-0.8, 0.8))
        down = max(0.2, 60 + 25 * math.sin(t * 0.4 + 1.4) +
                   random.uniform(-3, 3))

        disks = [
            {"name": "C:", "used_gb": 420 + 2 *
                math.sin(t * 0.1), "total_gb": 931, "pct": (420/931)*100},
            {"name": "D:", "used_gb": 1280 + 5 *
                math.sin(t * 0.07), "total_gb": 1863, "pct": (1280/1863)*100},
        ]
        return {
            "ts": time.time(),
            "cpu_temp_c": cpu_temp,
            "gpu_temp_c": gpu_temp,
            "gpu_power_w": gpu_power,
            "mem_used_gb": mem_used,
            "mem_total_gb": mem_total,
            "disks": disks,
            "net_up_mbps": up,
            "net_down_mbps": down,
        }

# ----------------------------
# UI Widgets
# ----------------------------


def draw_header(surface, font_lg):
    glow_text(surface, font_lg, "VAULT-TEC // SYSTEMS MONITOR", (14, 12))
    # thin divider
    pygame.draw.line(surface, GREEN, (12, 50), (WIDTH - 12, 50), 1)


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
        # hub
        pygame.draw.circle(surface, GREEN, (int(cx), int(cy)), 4, 0)

    # label & value
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=True) or pygame.font.SysFont(
        None, FONT_SIZE, bold=True)
    font_sm = pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True) or pygame.font.SysFont(
        None, FONT_SIZE_SMALL, bold=True)

    vtxt = "--" if value is None else (f"{int(round(value))}{unit}")
    vcolor = (255, 60, 60) if hot else GREEN
    glow_text(surface, font, vtxt, (cx - 28, cy + radius - 10), color=vcolor)
# center the label and place it UNDER the gauge
    lab_w, _ = font_sm.size(label)
    glow_text(surface, font_sm, label,
              (int(cx - lab_w/2), int(cy + radius + 10)))


def draw_bar(surface, x1, y1, x2, y2, pct_val, label, right_text):
    # frame
    pygame.draw.rect(surface, GREEN, (x1, y1, x2 - x1, y2 - y1), width=1)
    # fill
    fill_w = int((x2 - x1 - 2) * clamp(pct_val / 100.0, 0.0, 1.0))
    pygame.draw.rect(surface, GREEN, (x1 + 1, y1 + 1,
                     fill_w, (y2 - y1) - 2), width=0)

    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=True) or pygame.font.SysFont(
        None, FONT_SIZE, bold=True)
    glow_text(surface, font, f"{label}", (x1 + 6, y1 - 2))
    # right-aligned text
    rt_surf = font.render(right_text, True, GREEN)
    surface.blit(rt_surf, (x2 - rt_surf.get_width(), y1 - 2))


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
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True) or pygame.font.SysFont(
        None, FONT_SIZE_SMALL, bold=True)
    glow_text(surface, font,
              f"NET ↓ / ↑  (Mb/s)   scale max={vmax:.0f}", (x + 8, y + 6))


# ----------------------------
# Start screen with ASCII gear
# ----------------------------
GEAR = [
    "                                      ",
    "                                      ",
    "                                      ",
    "                                      ",
    "                                      ",
    "      <<<<<<<<  /oo0oo\  >>>>>>>>>     ",
    "               /ooooooo\            ",
    "  <<<<<<<<<<< |000o0o000| >>>>>>>>>>>>",
    "               \ooooooo/               ",
    "     <<<<<<<<<  \oo0oo/  >>>>>>>>>      ",
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
    gear_font = pygame.font.SysFont(
        FONT_NAME, 18, bold=True) or pygame.font.SysFont(None, 18, bold=True)
    gear_w = max(gear_font.size(line)[0] for line in GEAR)
    gear_h = len(GEAR) * (gear_font.get_height() + 0)

    gx = (WIDTH - gear_w) // 2
    gy = 160

    for i, line in enumerate(GEAR):
        glow_text(surface, gear_font, line,
                  (gx, gy + i * (gear_font.get_height())))

    # START button rectangle in middle
    btn_w, btn_h = 200, 56
    btn_x = (WIDTH - btn_w) // 2
    btn_y = gy + gear_h + 30

    pygame.draw.rect(surface, GREEN, (btn_x, btn_y, btn_w, btn_h), width=2)
    # emboss lines
    pygame.draw.line(surface, DIM_GREEN, (btn_x + 6, btn_y + 6),
                     (btn_x + btn_w - 6, btn_y + 6), 2)
    pygame.draw.line(surface, DIM_GREEN, (btn_x + 6, btn_y +
                     btn_h - 6), (btn_x + btn_w - 6, btn_y + btn_h - 6), 2)

    txt = "START"
    tw, th = font_lg.size(txt)
    glow_text(surface, font_lg, txt,
              (btn_x + (btn_w - tw) // 2, btn_y + (btn_h - th) // 2))

    # footer hint
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True) or pygame.font.SysFont(
        None, FONT_SIZE_SMALL, bold=True)
    glow_text(surface, font, "Tap screen or click to begin",
              (WIDTH // 2 - 110, btn_y + btn_h + 14))

    return pygame.Rect(btn_x, btn_y, btn_w, btn_h)

# ----------------------------
# Main app
# ----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Run with simulated data (no sensors needed)")
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

    # histories for graph
    up_hist = deque(maxlen=GRAPH_POINTS)
    dn_hist = deque(maxlen=GRAPH_POINTS)

    state = "start"
    start_btn_rect = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)
            if state == "start" and event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                if start_btn_rect and start_btn_rect.collidepoint(event.pos):
                    state = "dash"

        if state == "start":
            screen.fill(BG_COLOR)
            draw_scanlines(screen)
            draw_header(screen, font_lg)
            start_btn_rect = start_screen(screen, font_lg)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # ---- Dashboard ----
        snap = provider.snapshot()

        screen.fill(BG_COLOR)
        draw_scanlines(screen)
        draw_header(screen, font_lg)

        # --- Top gauges ---
        gy = 120
        centers = [(80, gy), (240, gy), (400, gy)]
        cpu_t = snap["cpu_temp_c"]
        gpu_t = snap["gpu_temp_c"]
        pwr_w = snap["gpu_power_w"]  # use GPU power if available

        draw_gauge(screen, centers[0], GAUGE_RADIUS, cpu_t, 20, 100, "CPU TEMP", "C",
                   hot=(cpu_t is not None and cpu_t >= CPU_HOT_C))
        draw_gauge(screen, centers[1], GAUGE_RADIUS, gpu_t, 20, 100, "GPU TEMP", "C",
                   hot=(gpu_t is not None and gpu_t >= GPU_HOT_C))
        draw_gauge(screen, centers[2], GAUGE_RADIUS, pwr_w, 0, 300, "POWER", "W",
                   hot=False)

        # warning banner if hot
        if (cpu_t is not None and cpu_t >= CPU_HOT_C) or (gpu_t is not None and gpu_t >= GPU_HOT_C):
            warn = "!! RAD DETECTED: THERMAL LIMIT APPROACHING !!"
            glow_text(screen, font, warn, (24, TOP_BAND_H - 28),
                      color=(255, 60, 60), glow=(60, 0, 0))

        # --- Middle bars (RAM + disks) ---
        y = MID_BAND_Y + 10

        # RAM
        mem_used = snap["mem_used_gb"]
        mem_tot = snap["mem_total_gb"]
        mem_pct = clamp(pct(mem_used, mem_tot), 0, 100)
        draw_bar(screen, BAR_LEFT, y, BAR_RIGHT, y + BAR_H, mem_pct,
                 f"MEM {human_gb(mem_used)}/{human_gb(mem_tot)}",
                 f"{mem_pct:.0f}%")
        y += BAR_H + BAR_GAP

        # Disks (up to 6 rows)
        for d in (snap["disks"] or [])[:6]:
            bar_pct = clamp(d["pct"], 0, 100)
            draw_bar(screen, BAR_LEFT, y, BAR_RIGHT, y + BAR_H, bar_pct,
                     f"DSK {d['name']} {human_gb(d['used_gb'])}/{human_gb(d['total_gb'])}",
                     f"{bar_pct:.0f}%")
            y += BAR_H + BAR_GAP

        # pad to graph area with a divider
        pygame.draw.line(screen, GREEN, (12, GRAPH_Y - 8),
                         (WIDTH - 12, GRAPH_Y - 8), 1)

        # --- Bottom graph (NET) ---
        up_hist.append(snap["net_up_mbps"])
        dn_hist.append(snap["net_down_mbps"])
        draw_graph(screen, 12, GRAPH_Y, WIDTH - 24, GRAPH_H, up_hist, dn_hist)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
