# Vault-Tec Pip-Boy System Monitor / Hardware Monitor

A retro-styled, green-on-black **system dashboard** / inspired by the Fallout Pip-Boy.  
Runs on Windows/Designed for a **5″ portrait display (480×800)** and just as happy on a laptop.
It shows:
- **Three gauges** (top): **CPU Load**, **GPU Load**, **RAM Usage** (0–100%)
- **Disk usage bars** (middle): one per drive
- **Network graph** (bottom): **upload spikes up**, **download spikes down**, autoscaled
- **Scrolling ticker** (very bottom): recent **Windows System** log events (last 2 hours) via *pywin32* Any **ERROR or CRITICAL** system events in the last 1 hour will activate _RED ALERT_ mode.

_**ANY RELEASE IS A DOWNLOADABLE SELF CONTAINED .EXE APPLICATION, NO CODING OR ASSEMBLY REQUIRED**_

<img width="964" height="834" alt="image" src="https://github.com/user-attachments/assets/31f3d8c6-3226-4a5b-9dfc-34bb4986f567" />


---

## Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run](#run)
- [Command-Line Options](#command-line-options)
- [Data Sources](#data-sources)
- [Windows System Log Ticker](#windows-system-log-ticker)
- [Layout & Styling Knobs](#layout--styling-knobs)
- [Design Notes](#design-notes)
- [Raspberry Pi & Case-Mod Ideas](#raspberry-pi--case-mod-ideas)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Fallout-style UI**: scanlines, glow text, mono font, gauge needles tapered to a point.
- **Centered network graph**: zero line in the middle; ↑ upload above, ↓ download below.
- **Event ticker**: pulls **last 2 hours** of important System events (Errors, Warnings & Critical Events) For Errors and Critical, red alerts display for 1 hour and scrolls them.
- **Smooth gauges**: optional throttling/EMA smoothing to avoid jitter.
- **Portrait-first**: defaults to `480×800` window for 5″ screens; easy to resize.

---

## Quick Start
IF DOWNLOADING SOURCE CODE
```bash
# 1) Install Python 3.10+
# 2) Install deps
pip install pygame psutil

# Optional: NVIDIA metrics
pip install pynvml

# Optional (Windows only): Event Viewer ticker
pip install pywin32

# 3) Run
python pipboy_ui_events_bidir_net_needles.py         # real stats (best effort)

# Or: simulated data for development
python pipboy_ui_events_bidir_net_needles.py --demo
```

> Escape (`Esc`) quits. Touch/click starts from the START screen.

---

## Requirements

- **Python**: 3.10 or newer
- **OS**: Windows 10/11, macOS, or Linux
- **Libraries**:
  - Required: `pygame`, `psutil`
  - Optional: `pynvml` (NVIDIA GPUs), `pywin32` (Windows Event Log ticker)

> Fonts: the app requests `"Courier New"`; if unavailable, Pygame falls back to a system monospace.

---

## Installation

```bash
pip install pygame psutil
pip install pynvml         # optional for NVIDIA GPU load/VRAM
pip install pywin32        # optional, Windows System log ticker
```

Windows PowerShell equivalent:

```powershell
py -m pip install pygame psutil
py -m pip install pynvml
py -m pip install pywin32
```

---

## Run

```bash
python pipboy_ui_events_bidir_net_needles.py          # window 480×800 portrait
python pipboy_ui_events_bidir_net_needles.py --demo   # fake data to design UI
```

If you’re on a small display and want true fullscreen:

```python
# change the window creation line if you want fullscreen later:
# screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
```

---

## Command-Line Options

```
--demo          Use simulated stats (works everywhere; great for UI iteration)
--icon PATH     Use a custom window icon (.ico). Defaults to 'pippy.ico' if found.
```

Planned:
``>
--fullscreen    Fullscreen kiosk mode
--scale 1.25    UI scale factor (fonts, spacing) for larger panels
```

---

## Data Sources

- **CPU Load / RAM** / **Disk usage** / **Network throughput** → `psutil` (cross-platform)
- **GPU Load / VRAM** (optional) → `pynvml` (NVIDIA)
- **System Events** (optional, Windows) → `pywin32` (Event Viewer API)

> CPU temperature is intentionally out of scope on Windows without a sensor driver. This project opts to avoid a resident helper app and focuses on stats available from the OS + optional vendor SDKs.

### Smoothing & Sample Rate

Two simple controls (add/adjust near the top if you want calmer needles):

```python
STATS_INTERVAL = 1.0      # seconds between snapshots (global throttle)

CPU_SAMPLE_SEC = 0.5      # CPU % sampling cadence
CPU_SMOOTH_ALPHA = 0.35   # EMA smoothing; lower = smoother
```

Exponential moving average and 1 Hz sampling keep CPU % from “hummingbird jitter.”

---

## Windows System Log Ticker

The ticker shows **ALL INFO & CRITICAL/ERROR/WARNING** from the **System** log in the last hour for these sources:

- `Kernel-Power` (unexpected shutdowns)
- `Disk` / `Ntfs` (I/O anomalies)
- `WHEA-Logger` (hardware corrected errors)
- Common NIC sources (Intel/Realtek/Killer; resets/timeouts)
- `Service Control Manager` (service start failures)

Install **pywin32** and run on Windows:

```bash
pip install pywin32
python pipboy_ui_events_bidir_net_needles.py
```

If pywin32 isn’t available (or non-Windows), the ticker will display a friendly hint and the rest of the dashboard keeps working.

You can add more sources by editing `EVENT_SOURCE_PATTERNS` (regex list) in the code.

---

## Layout & Styling Knobs

**Window size (portrait default):**

```python
WIDTH, HEIGHT = 480, 800
```

**Header & gauges:**

```python
# Header divider line Y
pygame.draw.line(surface, GREEN, (12, 58), (WIDTH - 12, 58), 1)

# Gauge row Y
gy = 150
centers = [(80, gy), (240, gy), (400, gy)]

# Gauge size
GAUGE_RADIUS = 85
```

**Bars (disks only):**

```python
TOP_BAND_H = 230        # height of top section (affects where bars start)
MID_BAND_Y = TOP_BAND_H + 10
MID_BAND_H = 360        # total bar section height
BAR_H = 28
BAR_GAP = 22
```

**Network graph & ticker:**

```python
TICKER_H = 40
BOT_BAND_H = HEIGHT - (TOP_BAND_H + MID_BAND_H) - TICKER_H
GRAPH_Y = HEIGHT - TICKER_H - BOT_BAND_H + 6
GRAPH_H = BOT_BAND_H - 18
```

**Needle styling:**

```python
NEEDLE_BASE_W = 14
NEEDLE_BASE_BACK = 10
NEEDLE_TIP_MARGIN = 16
```

**Colors & CRT vibe:**

```python
GREEN = (0, 255, 70)
DIM_GREEN = (0, 120, 30)
SCANLINE_STEP = 4
FONT_NAME = "Courier New"
```

> Tip: enable a quick **layout debug overlay** (boxes around bands) while tuning.

---

## Design Notes

- Gauges are semicircular with **tapered needles** (filled triangles) and a small hub ring.
- Disk bars show **% used** with labels **below** the bars for readability.
- The network graph draws a **midline (0 Mb/s)**; upload (solid) is above, download (dashed) below.
- The ticker scrolls left to right continuously; events refresh every ~5 seconds.

---

## Raspberry Pi & Case-Mod Ideas

This app was designed with a **5″ 800×480 portrait touch** in mind—perfect for mounting inside a custom PC case as a Pip-Boy-style status screen.

### Hardware

- **Pi model**: Pi Zero 2 W (easy USB gadget mode) or Pi 3/4 (HDMI display).  
- **Display**: 5″ 800×480 HDMI or DSI panel; set *portrait* orientation.  
- **Power**: tap the PC PSU 5V rail (SATA/Molex → buck/USB module) to power the Pi + screen; keep **USB header for data only** if tethering.

### Orientation

- X11 desktop: `xrandr -o left` for portrait.  
- Console/framebuffer: set rotation in `/boot/config.txt` (varies by driver; e.g., `display_hdmi_rotate=1`).

### Kiosk Boot

On Raspberry Pi OS Lite, create a systemd service to launch the app on boot (in a virtual framebuffer or with SDL on the console), or run the full desktop and autostart the script.

### Data Link Options

- **USB Serial Gadget** (Pi Zero 2 W): Pi appears as a COM device to the PC; feed JSON lines from a tiny PC helper (optional); or run the app standalone with local stats if the Pi *is* the host.  
- **USB Ethernet Gadget**: Pi exposes a virtual NIC; talk HTTP/REST if you later add a producer.  
- **LAN/Wi-Fi**: Fetch from a local API if you set one up; not required for this repo.

> This repo **does not require** any helper app. It displays host stats (the machine it runs on). If you embed a Pi as a *remote display* for a different PC, you’ll need a small bridge program to send that PC’s stats across USB/network.

### Case-Mod Tips

- 3D-print a **bezel** with faux rivets; keep airflow clear of GPU fans.  
- Add **GPIO buttons** to the Pi to flip pages or toggle views.  
- Boot splash: show a **Vault-Tec** logo while the app starts.  
- Easter eggs: flash “RAD DETECTED” when RAM>90% or CPU/GPU load>95%.

---

## Troubleshooting

**`AttributeError: module 'serial' has no attribute 'Serial'`**  
You installed `serial` instead of `pyserial`, or shadowed it with a local `serial.py`.  
Fix:
```bash
pip uninstall -y serial pyserial
pip install pyserial
```
Remove/rename any local `serial.py` file/folder.

**GPU numbers are `--`**  
- Install NVIDIA drivers + `pynvml`.  
- AMD/Intel GPUs aren’t covered by NVML; the app will just show N/A for GPU load.

**No events in ticker**  
- Install `pywin32` and run on Windows.  
- Some systems require Administrator to read all sources.  
- If nothing is in the last hour, the ticker will say so.

**Fonts look different**  
- The app requests `Courier New`; on non-Windows it may fall back to another monospace. Adjust `FONT_NAME` to taste.

**CPU needle is jumpy**  
- Use the `STATS_INTERVAL` global or the `CPU_SAMPLE_SEC/CPU_SMOOTH_ALPHA` EMA to calm it down.

---

## Roadmap

- Toggle pages (Process Top-Talkers, I/O rates per disk, Wi-Fi SSID/signal).  
- Config file for colors/thresholds/layout.  
- Optional log **details page** (scrollable last 50 events, color-coded).  
- On-screen buttons for brightness/theme.  
- Packaged Windows EXE (PyInstaller) and Pi image recipe.

---

## License

MIT — see `LICENSE` (or add one to your repo).

---

### Screenshots

(Add PNGs or GIFs here once you have them.)

```
docs/
  screenshot_start.png
  screenshot_dashboard.png
  screenshot_ticker.png
```
