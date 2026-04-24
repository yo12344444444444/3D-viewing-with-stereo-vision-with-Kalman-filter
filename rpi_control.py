"""
rpi_control.py  —  Robot Car Control via MKS Gen L + NEMA 17
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• RPi talks to MKS Gen L over USB serial (G-code / Marlin)
• X stepper = Left wheel,  Y stepper = Right wheel
• Serves phone web UI on port 8001 (connect to RobotCar WiFi)
• Stops automatically when robot_vision.py detects an obstacle
• ESP32 can send text commands via TCP port 9000

Wiring:
  RPi USB  ──►  MKS Gen L USB
  MKS X stepper output  ──►  Left  NEMA 17
  MKS Y stepper output  ──►  Right NEMA 17

Marlin settings needed (in Configuration.h):
  #define DEFAULT_AXIS_STEPS_PER_UNIT { 80, 80, 400, 100 }
  Set X/Y steps/mm for your wheel diameter + microstep setting.

Run:
    python3 rpi_control.py

Phone (on RobotCar WiFi) → http://192.168.4.1:8001
ESP32 commands           → TCP 192.168.4.1:9000
                           send: forward / backward / left / right / stop\n
"""

import threading, time, json, sys, serial, serial.tools.list_ports
import urllib.request

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ══════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════
# MKS Gen L serial port — change if needed
# Linux RPi:  usually /dev/ttyUSB0 or /dev/ttyACM0
SERIAL_PORT  = "/dev/ttyUSB0"
SERIAL_BAUD  = 115200

# Movement speed (mm/min — Marlin feedrate)
DRIVE_SPEED  = 3000   # forward / backward
TURN_SPEED   = 2000   # left / right

# How far each command moves (mm) — tune to your wheel size
DRIVE_STEP   = 50     # mm per forward/backward pulse
TURN_STEP    = 30     # mm per turn pulse (one wheel fwd, one bwd)

# robot_vision.py obstacle endpoint
VISION_URL   = "http://localhost:8000/status"
VISION_HZ    = 5

# Ports
CONTROL_PORT = 8001
ESP32_PORT   = 9000
RPi_IP       = "192.168.4.1"
# ══════════════════════════════════════════════════

_VALID_CMDS = {"forward", "backward", "left", "right", "stop"}


# ─────────────────────────────────────────────────
#  Serial finder — auto-detect MKS Gen L port
# ─────────────────────────────────────────────────
def find_serial_port() -> str:
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(k in desc for k in ("usb", "arduino", "ch340", "cp210", "ftdi")):
            print(f"[SERIAL] Auto-detected: {p.device} ({p.description})")
            return p.device
    print(f"[SERIAL] Using default: {SERIAL_PORT}")
    return SERIAL_PORT


# ─────────────────────────────────────────────────
#  MKS Gen L / Marlin G-code driver
# ─────────────────────────────────────────────────
class MKSDriver:
    """
    Sends G-code to MKS Gen L running Marlin.
    X axis → Left motor,  Y axis → Right motor.

    Movement logic (differential drive):
      forward  : X+ Y+   (both wheels forward)
      backward : X- Y-   (both wheels backward)
      left     : X- Y+   (left back, right fwd → turn left)
      right    : X+ Y-   (left fwd, right back → turn right)
      stop     : M410    (emergency stop)
    """

    def __init__(self, port: str, baud: int = 115200):
        self._lock     = threading.Lock()
        self._obstacle = False
        self._last_cmd = "stop"
        self._speed    = DRIVE_SPEED
        self._connected = False
        self._ser      = None

        try:
            self._ser = serial.Serial(port, baud, timeout=2)
            time.sleep(2)          # wait for Marlin to boot
            self._ser.flushInput()
            # Put Marlin in relative positioning mode
            self._send_raw("G91")  # relative moves
            self._send_raw("G21")  # millimetre units
            self._send_raw("M17")  # enable steppers
            self._connected = True
            print(f"[MKS] Connected on {port} @ {baud} baud")
        except serial.SerialException as e:
            print(f"[MKS] WARNING: Could not open {port}: {e}")
            print("[MKS] Running in SIMULATION mode — no real motors")

    # ── low-level serial write ─────────────────────
    def _send_raw(self, gcode: str):
        if self._ser and self._ser.is_open:
            cmd = (gcode.strip() + "\n").encode()
            self._ser.write(cmd)
            # Read and discard 'ok' response (non-blocking)
            try:
                self._ser.readline()
            except Exception:
                pass
        else:
            print(f"[SIM] {gcode}")

    # ── movement G-code ────────────────────────────
    def _move(self, x_mm: float, y_mm: float, speed: int):
        """
        Send a relative move:
          x_mm = left  wheel distance  (+ = forward)
          y_mm = right wheel distance  (+ = forward)
        """
        self._send_raw(f"G1 X{x_mm:.1f} Y{y_mm:.1f} F{speed}")

    def _stop(self):
        self._send_raw("M410")   # emergency stop (Marlin)

    # ── public API ────────────────────────────────
    def move(self, cmd: str) -> str:
        with self._lock:
            self._last_cmd = cmd
            if self._obstacle and cmd == "forward":
                self._stop()
                return "blocked"

            s = self._speed
            t = int(self._speed * 0.75)

            if   cmd == "forward":  self._move(+DRIVE_STEP, +DRIVE_STEP, s)
            elif cmd == "backward": self._move(-DRIVE_STEP, -DRIVE_STEP, s)
            elif cmd == "left":     self._move(-TURN_STEP,  +TURN_STEP,  t)
            elif cmd == "right":    self._move(+TURN_STEP,  -TURN_STEP,  t)
            elif cmd == "stop":     self._stop()

            return "ok"

    def set_obstacle(self, detected: bool):
        with self._lock:
            was = self._obstacle
            self._obstacle = detected
            if detected and not was:
                self._stop()
                print("[OBSTACLE] Vision detected obstacle — motors stopped")
            elif not detected and was:
                print("[OBSTACLE] Path clear")

    def set_speed(self, pct: int):
        """Set speed as percentage 20-100."""
        pct = max(20, min(100, pct))
        with self._lock:
            self._speed = int(DRIVE_SPEED * pct / 100)

    def disable_motors(self):
        """Cut stepper current (saves heat when idle)."""
        with self._lock:
            self._send_raw("M18")   # disable steppers

    def enable_motors(self):
        with self._lock:
            self._send_raw("M17")

    def close(self):
        with self._lock:
            self._stop()
            time.sleep(0.2)
            self._send_raw("M18")
            if self._ser:
                self._ser.close()

    @property
    def state(self) -> dict:
        with self._lock:
            return {
                "obstacle":   self._obstacle,
                "last_cmd":   self._last_cmd,
                "speed_pct":  int(self._speed / DRIVE_SPEED * 100),
                "connected":  self._connected,
            }


# Global driver instance
port   = find_serial_port()
driver = MKSDriver(port, SERIAL_BAUD)


# ─────────────────────────────────────────────────
#  Vision monitor — polls robot_vision.py /status
# ─────────────────────────────────────────────────
def _vision_monitor():
    while True:
        try:
            with urllib.request.urlopen(VISION_URL, timeout=1) as r:
                data = json.loads(r.read())
                driver.set_obstacle(bool(data.get("alert", False)))
        except Exception:
            pass
        time.sleep(1.0 / VISION_HZ)


# ─────────────────────────────────────────────────
#  ESP32 TCP socket server (port 9000)
# ─────────────────────────────────────────────────
def _handle_esp32(conn, addr):
    print(f"[ESP32] Connected: {addr}")
    try:
        buf = ""
        while True:
            chunk = conn.recv(64).decode("utf-8", errors="ignore")
            if not chunk:
                break
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                cmd = line.strip().lower()
                if cmd in _VALID_CMDS:
                    result = driver.move(cmd)
                    conn.sendall(f"{result}\n".encode())
    except Exception:
        pass
    finally:
        conn.close()
        print(f"[ESP32] Disconnected: {addr}")


def _esp32_server():
    import socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", ESP32_PORT))
    srv.listen(5)
    print(f"[ESP32] Socket server on port {ESP32_PORT}")
    while True:
        try:
            conn, addr = srv.accept()
            threading.Thread(target=_handle_esp32,
                             args=(conn, addr), daemon=True).start()
        except Exception as e:
            print(f"[ESP32] Error: {e}")


# ─────────────────────────────────────────────────
#  Phone web UI — D-pad controller
# ─────────────────────────────────────────────────
CONTROL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Robot Car</title>
<style>
  :root {
    --bg:#0a0d10; --panel:#111620; --border:#1e2a38;
    --accent:#00e5ff; --danger:#ff3355; --ok:#00e676;
    --text:#c8d8e8; --r:14px;
  }
  *{box-sizing:border-box;margin:0;padding:0;touch-action:manipulation;}
  html,body{height:100%;background:var(--bg);color:var(--text);
            font-family:'Segoe UI',sans-serif;overflow:hidden;}

  header{padding:10px 16px;background:var(--panel);
         border-bottom:1px solid var(--border);
         display:flex;align-items:center;gap:12px;}
  .logo{font-size:1rem;font-weight:800;letter-spacing:2px;color:var(--accent);}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--ok);transition:background .3s;}
  .dot.danger{background:var(--danger);animation:blink .5s infinite;}
  .dot.offline{background:#555;}
  .stxt{margin-left:auto;font-size:.78rem;}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

  .obs-bar{display:none;background:rgba(255,51,85,.9);text-align:center;
           padding:8px;font-weight:800;letter-spacing:2px;
           animation:blink .5s infinite;}
  .obs-bar.show{display:block;}

  .dpad{display:flex;flex-direction:column;align-items:center;
        justify-content:center;height:calc(100vh - 130px);gap:12px;}
  .row{display:flex;gap:12px;}

  .btn{width:96px;height:96px;border-radius:var(--r);
       border:2px solid var(--border);background:var(--panel);
       color:var(--accent);font-size:2.2rem;cursor:pointer;
       display:flex;align-items:center;justify-content:center;
       transition:all .1s;user-select:none;
       -webkit-tap-highlight-color:transparent;}
  .btn:active,.btn.active{background:rgba(0,229,255,.15);
                           border-color:var(--accent);transform:scale(.92);}
  .stop{color:var(--danger);font-size:.95rem;font-weight:800;letter-spacing:1px;}
  .stop:active{border-color:var(--danger);}

  .bottom-bar{position:fixed;bottom:0;left:0;right:0;
              background:var(--panel);border-top:1px solid var(--border);
              padding:10px 16px;display:flex;align-items:center;gap:14px;
              font-size:.75rem;}
  .bottom-bar input{flex:1;accent-color:var(--accent);}
  .conn-badge{font-size:.65rem;padding:3px 8px;border-radius:20px;
              border:1px solid var(--border);font-family:monospace;}
  .conn-badge.ok{border-color:var(--ok);color:var(--ok);}
  .conn-badge.err{border-color:var(--danger);color:var(--danger);}
</style>
</head>
<body>
<header>
  <div class="dot" id="dot"></div>
  <div class="logo">ROBOT CAR</div>
  <span class="conn-badge" id="conn-badge">MKS —</span>
  <div class="stxt" id="stxt">connecting…</div>
</header>
<div class="obs-bar" id="obs">⚠ OBSTACLE — FORWARD BLOCKED</div>

<div class="dpad">
  <div class="row">
    <button class="btn" id="bF"
      ontouchstart="hold('forward')"  ontouchend="release()"
      onmousedown="hold('forward')"   onmouseup="release()" onmouseleave="release()">▲</button>
  </div>
  <div class="row">
    <button class="btn" id="bL"
      ontouchstart="hold('left')"     ontouchend="release()"
      onmousedown="hold('left')"      onmouseup="release()" onmouseleave="release()">◀</button>
    <button class="btn stop" id="bS"
      ontouchstart="send('stop')"     ontouchend="send('stop')"
      onmousedown="send('stop')">STOP</button>
    <button class="btn" id="bR"
      ontouchstart="hold('right')"    ontouchend="release()"
      onmousedown="hold('right')"     onmouseup="release()" onmouseleave="release()">▶</button>
  </div>
  <div class="row">
    <button class="btn" id="bB"
      ontouchstart="hold('backward')" ontouchend="release()"
      onmousedown="hold('backward')"  onmouseup="release()" onmouseleave="release()">▼</button>
  </div>
</div>

<div class="bottom-bar">
  <span>Speed</span>
  <input type="range" min="20" max="100" value="80" id="spd"
         oninput="setSpd(this.value)">
  <span id="sval">80%</span>
</div>

<script>
  let timer = null;

  function send(dir) {
    fetch('/move', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({cmd: dir})
    });
  }

  function hold(dir) {
    send(dir);
    timer = setInterval(() => send(dir), 200);
  }

  function release() {
    clearInterval(timer);
    timer = null;
    send('stop');
  }

  function setSpd(v) {
    document.getElementById('sval').textContent = v + '%';
    fetch('/speed', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({speed: parseInt(v)})
    });
  }

  // Keyboard (WASD / arrows)
  const keyMap = {
    ArrowUp:'forward', ArrowDown:'backward',
    ArrowLeft:'left',  ArrowRight:'right',
    w:'forward', s:'backward', a:'left', d:'right'
  };
  const held = new Set();
  document.addEventListener('keydown', e => {
    const c = keyMap[e.key];
    if (c && !held.has(c)) { held.add(c); send(c); }
    if (e.key === ' ') { e.preventDefault(); send('stop'); }
  });
  document.addEventListener('keyup', e => {
    const c = keyMap[e.key];
    if (c) { held.delete(c); if (!held.size) send('stop'); }
  });

  // Status poll every 400 ms
  setInterval(async () => {
    try {
      const s = await (await fetch('/status')).json();
      document.getElementById('dot').className =
        'dot' + (s.obstacle ? ' danger' : '');
      document.getElementById('stxt').textContent =
        s.obstacle ? '⚠ OBSTACLE DETECTED' : '✓ PATH CLEAR';
      document.getElementById('obs').className =
        'obs-bar' + (s.obstacle ? ' show' : '');
      const badge = document.getElementById('conn-badge');
      badge.textContent = s.connected ? 'MKS ✓' : 'MKS ✗';
      badge.className   = 'conn-badge ' + (s.connected ? 'ok' : 'err');
    } catch(e) {}
  }, 400);
</script>
</body>
</html>"""


def _build_app() -> FastAPI:
    app = FastAPI(title="Robot Car")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return CONTROL_HTML

    @app.post("/move")
    async def move(body: dict):
        cmd = body.get("cmd", "stop").lower()
        if cmd not in _VALID_CMDS:
            return {"error": "invalid command"}
        result = driver.move(cmd)
        return {"cmd": cmd, "result": result}

    @app.post("/speed")
    async def set_speed(body: dict):
        driver.set_speed(int(body.get("speed", 80)))
        return {"ok": True}

    @app.get("/status")
    async def status():
        return driver.state

    return app


# ─────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────
def main():
    print("━" * 55)
    print("  Robot Car — MKS Gen L + NEMA 17")
    print(f"  Web UI  → http://{RPi_IP}:{CONTROL_PORT}")
    print(f"  ESP32   → TCP {RPi_IP}:{ESP32_PORT}")
    print(f"  Vision  → polling {VISION_URL}")
    print("━" * 55)

    threading.Thread(target=_vision_monitor, daemon=True).start()
    threading.Thread(target=_esp32_server,   daemon=True).start()

    app = _build_app()
    try:
        uvicorn.run(app, host="0.0.0.0", port=CONTROL_PORT, log_level="warning")
    finally:
        driver.close()
        print("\n[EXIT] Motors stopped.")


if __name__ == "__main__":
    main()
