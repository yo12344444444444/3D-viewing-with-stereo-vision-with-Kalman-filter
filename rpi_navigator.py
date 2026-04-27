"""
rpi_navigator.py  —  Navigation Controller  (runs on RPi)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Serves phone web UI on port 8002.
Polls PC AI server for detected objects.
User picks an object number → robot drives there.
Stops if robot_vision.py sees an obstacle.

Run:
    python3 rpi_navigator.py

Phone → http://10.3.141.1:8002
"""

import threading, time, json, sys, subprocess, re
import urllib.request
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# ── fix import path for Movments ──────────────────
sys.path.append("/home/yorai")
from mks.Movments import Movments

# ══════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════
# Put your PC's MAC address here (find it with  ipconfig /all  on Windows)
# Format: "aa:bb:cc:dd:ee:ff"  or  "AA-BB-CC-DD-EE-FF"  — both work
PC_MAC_ADDRESS = "F0-57-A6-A6-07-57"   # ← change this to your PC MAC

AI_SERVER_PORT = 5000
AP_SUBNET      = "10.3.141"            # RobotCar AP subnet (RaspAP)
ARP_RETRIES    = 5                     # how many times to try finding PC
ARP_RETRY_WAIT = 3                     # seconds between retries

VISION_URL     = None   # set automatically from PC_SERVER_URL below
CONTROL_PORT   = 8002
RPI_IP         = "10.3.141.1"

FRAME_W        = 320
DEADZONE       = 25
ARRIVE_CM      = 35
POLL_HZ        = 10
LOST_TIMEOUT   = 5.0
# ══════════════════════════════════════════════════


# ─────────────────────────────────────────────────
#  MAC → IP resolver  (ARP lookup)
# ─────────────────────────────────────────────────
def _normalise_mac(mac: str) -> str:
    """Convert any MAC format to lowercase colon-separated: aa:bb:cc:dd:ee:ff"""
    digits = re.sub(r"[^0-9a-fA-F]", "", mac)
    if len(digits) != 12:
        raise ValueError(f"Invalid MAC address: {mac!r}")
    return ":".join(digits[i:i+2].lower() for i in range(0, 12, 2))


def _ping_subnet(subnet: str):
    """Ping broadcast + all hosts to populate the ARP cache."""
    # broadcast ping (may need sudo or be disabled — try anyway)
    subprocess.run(f"ping -c 1 -b {subnet}.255",
                   shell=True, capture_output=True, timeout=3)
    # ping each host quickly (fills ARP table)
    for i in range(2, 51):
        subprocess.Popen(f"ping -c 1 -W 1 {subnet}.{i}",
                         shell=True, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    time.sleep(2)   # let pings complete


def _arp_lookup(mac_normalised: str) -> str | None:
    """Search ARP / neighbour table for a MAC, return its IP or None."""
    # try  ip neigh  first (modern Linux)
    for cmd in ("ip neigh", "arp -n"):
        try:
            out = subprocess.check_output(cmd, shell=True, text=True,
                                          stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if mac_normalised in line.lower():
                    # first token is the IP address
                    ip = line.split()[0]
                    if re.match(r"\d+\.\d+\.\d+\.\d+", ip):
                        return ip
        except Exception:
            continue
    return None


def find_pc_ip(mac: str,
               subnet: str   = AP_SUBNET,
               retries: int  = ARP_RETRIES,
               wait: float   = ARP_RETRY_WAIT) -> str:
    """
    Resolve PC MAC address → IP on the AP network.
    Pings the subnet to fill ARP cache, then searches the table.
    Retries several times in case the PC hasn't connected yet.
    """
    try:
        mac_norm = _normalise_mac(mac)
    except ValueError as e:
        print(f"[ARP] {e} — falling back to manual IP")
        return None

    print(f"[ARP] Looking for MAC {mac_norm} on {subnet}.0/24 ...")

    for attempt in range(1, retries + 1):
        print(f"[ARP] Attempt {attempt}/{retries} — scanning subnet...")
        _ping_subnet(subnet)
        ip = _arp_lookup(mac_norm)
        if ip:
            print(f"[ARP] ✓ Found PC at {ip}")
            return ip
        if attempt < retries:
            print(f"[ARP] Not found yet — waiting {wait}s "
                  f"(is PC connected to RobotCar WiFi?)")
            time.sleep(wait)

    print(f"[ARP] ✗ Could not find MAC {mac_norm} after {retries} attempts")
    return None


# ── resolve PC IP at startup ───────────────────────
_pc_ip = find_pc_ip(PC_MAC_ADDRESS)
if _pc_ip:
    PC_SERVER_URL = f"http://{_pc_ip}:{AI_SERVER_PORT}"
else:
    # fallback — user must set manually
    PC_SERVER_URL = f"http://{AP_SUBNET}.X:{AI_SERVER_PORT}"
    print(f"[ARP] FALLBACK: set PC_SERVER_URL manually in this file")

print(f"[ARP] AI Server URL → {PC_SERVER_URL}")
VISION_URL = f"{PC_SERVER_URL}/status"   # obstacle check uses same server


# ─────────────────────────────────────────────────
#  Navigation state
# ─────────────────────────────────────────────────
class NavState:
    def __init__(self):
        self._lock       = threading.Lock()
        self.target_id   = None      # which object number to chase
        self.navigating  = False
        self.status      = "idle"    # idle / moving / turning / arrived / blocked / lost
        self.obstacle    = False
        self.detections  = []        # latest from PC server
        self.last_seen   = 0.0       # time we last saw the target

    def set_target(self, obj_id: int):
        with self._lock:
            self.target_id  = obj_id
            self.navigating = True
            self.status     = "moving"
            self.last_seen  = time.time()

    def cancel(self):
        with self._lock:
            self.target_id  = None
            self.navigating = False
            self.status     = "idle"

    def get(self):
        with self._lock:
            return {
                "navigating": self.navigating,
                "target_id":  self.target_id,
                "status":     self.status,
                "obstacle":   self.obstacle,
                "detections": list(self.detections),
            }


nav   = NavState()
car   = Movments(speed=2000, step=20, turn=15)


# ─────────────────────────────────────────────────
#  Helpers — fetch from PC and vision server
# ─────────────────────────────────────────────────
def _fetch(url: str, timeout: float = 1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _poll_detections():
    """Continuously fetch object list from PC AI server."""
    while True:
        data = _fetch(f"{PC_SERVER_URL}/detections")
        if data:
            with nav._lock:
                nav.detections = data.get("objects", [])
        time.sleep(1.0 / POLL_HZ)


def _poll_obstacle():
    """Check robot_vision.py for obstacles."""
    while True:
        data = _fetch(VISION_URL)
        if data is not None:
            with nav._lock:
                nav.obstacle = bool(data.get("alert", False))
        time.sleep(0.2)


# ─────────────────────────────────────────────────
#  Navigation loop
# ─────────────────────────────────────────────────
def _find_target(detections, target_id):
    for d in detections:
        if d["id"] == target_id:
            return d
    return None


def _navigation_loop():
    center_x = FRAME_W // 2

    while True:
        time.sleep(1.0 / POLL_HZ)

        with nav._lock:
            if not nav.navigating:
                continue
            target_id   = nav.target_id
            obstacle    = nav.obstacle
            detections  = list(nav.detections)
            last_seen   = nav.last_seen

        # ── obstacle check ─────────────────────────
        if obstacle:
            car.stop()
            with nav._lock:
                nav.status = "blocked"
            continue

        # ── find target in latest detections ───────
        target = _find_target(detections, target_id)

        if target is None:
            # lost target
            if time.time() - last_seen > LOST_TIMEOUT:
                car.stop()
                with nav._lock:
                    nav.status     = "lost"
                    nav.navigating = False
            else:
                # slowly rotate to search
                car.turn_right()
                with nav._lock:
                    nav.status = "searching"
            continue

        # target found — update last seen time
        with nav._lock:
            nav.last_seen = time.time()

        # ── arrived check ──────────────────────────
        depth = target.get("depth_cm")
        if depth is not None and depth < ARRIVE_CM:
            car.stop()
            with nav._lock:
                nav.status     = "arrived"
                nav.navigating = False
            print(f"[NAV] Arrived at object #{target_id} ({target['label']})")
            continue

        # ── steer toward target ────────────────────
        error = target["cx"] - center_x   # + = target is RIGHT, - = target is LEFT

        if error > DEADZONE:
            # target is to the right
            car.turn_right()
            with nav._lock:
                nav.status = "turning_right"

        elif error < -DEADZONE:
            # target is to the left
            car.turn_left()
            with nav._lock:
                nav.status = "turning_left"

        else:
            # target is centered → go forward
            car.forward()
            with nav._lock:
                nav.status = "moving"


# ─────────────────────────────────────────────────
#  FastAPI — phone web UI
# ─────────────────────────────────────────────────
UI_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Robot Navigator</title>
<style>
  :root{{
    --bg:#0a0d10;--panel:#111620;--border:#1e2a38;
    --accent:#00e5ff;--danger:#ff3355;--ok:#00e676;--warn:#ff8c00;
    --text:#c8d8e8;--r:12px;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;touch-action:manipulation}}
  html,body{{height:100%;background:var(--bg);color:var(--text);
             font-family:'Segoe UI',sans-serif;}}

  header{{padding:10px 16px;background:var(--panel);
          border-bottom:1px solid var(--border);
          display:flex;align-items:center;gap:10px;}}
  .logo{{font-size:1rem;font-weight:800;letter-spacing:2px;color:var(--accent);}}
  .dot{{width:9px;height:9px;border-radius:50%;background:var(--ok);}}
  .dot.danger{{background:var(--danger);animation:blink .5s infinite;}}
  @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.2}}}}

  .feed{{width:100%;background:#000;display:block;}}

  .status-bar{{padding:8px 14px;background:#0d1118;
               border-bottom:1px solid var(--border);
               font-family:monospace;font-size:.8rem;
               display:flex;gap:16px;align-items:center;flex-wrap:wrap;}}
  .chip .label{{color:#4a6070;}}
  .chip .val{{color:var(--accent);margin-left:4px;}}
  .chip.danger .val{{color:var(--danger);}}
  .chip.ok .val{{color:var(--ok);}}

  .section{{padding:14px;}}
  .section-title{{font-size:.6rem;letter-spacing:3px;color:#4a6070;
                  text-transform:uppercase;margin-bottom:10px;
                  border-left:2px solid var(--accent);padding-left:8px;}}

  /* object cards */
  .obj-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;}}
  .obj-card{{background:var(--panel);border:1px solid var(--border);
             border-radius:var(--r);padding:10px 12px;cursor:pointer;
             transition:all .15s;}}
  .obj-card:active{{transform:scale(.96);}}
  .obj-card.selected{{border-color:var(--accent);background:rgba(0,229,255,.08);}}
  .obj-id{{font-size:1.4rem;font-weight:800;color:var(--accent);}}
  .obj-label{{font-size:.8rem;margin-top:2px;}}
  .obj-depth{{font-size:.7rem;color:#4a6070;font-family:monospace;margin-top:2px;}}

  /* go button */
  .go-btn{{width:100%;padding:16px;margin-top:14px;
           background:rgba(0,229,255,.1);border:2px solid var(--accent);
           color:var(--accent);border-radius:var(--r);
           font-size:1rem;font-weight:800;letter-spacing:2px;cursor:pointer;
           transition:all .15s;}}
  .go-btn:active{{background:rgba(0,229,255,.25);transform:scale(.98);}}
  .go-btn:disabled{{opacity:.3;cursor:not-allowed;}}

  .stop-btn{{width:100%;padding:12px;margin-top:8px;
             background:transparent;border:2px solid var(--danger);
             color:var(--danger);border-radius:var(--r);
             font-size:.9rem;font-weight:800;cursor:pointer;}}
  .stop-btn:active{{background:rgba(255,51,85,.15);}}

  /* status message */
  .status-msg{{margin-top:10px;padding:12px;background:var(--panel);
               border:1px solid var(--border);border-radius:var(--r);
               font-family:monospace;font-size:.82rem;text-align:center;}}
  .status-msg.arrived{{border-color:var(--ok);color:var(--ok);}}
  .status-msg.blocked{{border-color:var(--danger);color:var(--danger);
                       animation:blink .5s infinite;}}
  .status-msg.moving{{border-color:var(--accent);color:var(--accent);}}
  .status-msg.lost{{border-color:var(--warn);color:var(--warn);}}

  .spacer{{height:20px;}}

  .no-objects{{text-align:center;color:#4a6070;
               font-family:monospace;font-size:.8rem;padding:20px;}}
</style>
</head>
<body>

<header>
  <div class="dot" id="dot"></div>
  <div class="logo">ROBOT NAV</div>
  <span style="margin-left:auto;font-size:.7rem;color:#4a6070;" id="pc-status">AI Server —</span>
</header>

<img class="feed" src="{PC_SERVER_URL}/stream" id="feed">

<div class="status-bar">
  <div class="chip" id="chip-nav">
    <span class="label">NAV</span>
    <span class="val" id="val-nav">IDLE</span>
  </div>
  <div class="chip">
    <span class="label">TARGET</span>
    <span class="val" id="val-target">—</span>
  </div>
  <div class="chip" id="chip-obs">
    <span class="label">PATH</span>
    <span class="val" id="val-obs">CLEAR</span>
  </div>
</div>

<!-- DETECTED OBJECTS -->
<div class="section">
  <div class="section-title">Detected Objects — tap to select</div>
  <div class="obj-grid" id="obj-grid">
    <div class="no-objects" style="grid-column:1/-1">
      Waiting for AI server…
    </div>
  </div>
</div>

<!-- GO / STOP -->
<div class="section" style="padding-top:0">
  <button class="go-btn" id="go-btn" onclick="goToTarget()" disabled>
    GO TO SELECTED OBJECT
  </button>
  <button class="stop-btn" onclick="stopNow()">■ STOP</button>
  <div class="status-msg" id="status-msg">Select an object above then press GO</div>
</div>

<div class="spacer"></div>

<script>
  let selectedId   = null;
  let lastObjects  = [];

  // ── select object card ──────────────────────────
  function selectObj(id) {{
    selectedId = id;
    document.querySelectorAll('.obj-card').forEach(c => {{
      c.classList.toggle('selected', parseInt(c.dataset.id) === id);
    }});
    document.getElementById('go-btn').disabled = false;
  }}

  // ── send GO command ─────────────────────────────
  async function goToTarget() {{
    if (!selectedId) return;
    await fetch('/goto', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{id: selectedId}})
    }});
  }}

  // ── stop ────────────────────────────────────────
  async function stopNow() {{
    selectedId = null;
    document.getElementById('go-btn').disabled = true;
    await fetch('/stop', {{method:'POST'}});
  }}

  // ── render object cards ─────────────────────────
  function renderObjects(objects) {{
    const grid = document.getElementById('obj-grid');
    if (!objects || objects.length === 0) {{
      grid.innerHTML = '<div class="no-objects" style="grid-column:1/-1">No objects detected</div>';
      return;
    }}
    grid.innerHTML = objects.map(o => `
      <div class="obj-card ${{selectedId===o.id?'selected':''}}"
           data-id="${{o.id}}" onclick="selectObj(${{o.id}})">
        <div class="obj-id">#${{o.id}}</div>
        <div class="obj-label">${{o.label}}</div>
        <div class="obj-depth">${{o.depth_cm ? o.depth_cm+'cm away' : 'depth unknown'}}</div>
      </div>
    `).join('');
  }}

  // ── status message ──────────────────────────────
  const statusMap = {{
    idle:          ['Select an object then press GO', ''],
    moving:        ['▶ Moving toward target...', 'moving'],
    turning_left:  ['↩ Turning left...', 'moving'],
    turning_right: ['↪ Turning right...', 'moving'],
    searching:     ['🔍 Searching for target...', 'moving'],
    arrived:       ['✅ Arrived!', 'arrived'],
    blocked:       ['⚠ OBSTACLE — path blocked', 'blocked'],
    lost:          ['Target lost — please select again', 'lost'],
  }};

  // ── poll status every 400ms ─────────────────────
  setInterval(async () => {{
    try {{
      const s = await (await fetch('/status')).json();

      // dot
      document.getElementById('dot').className =
        'dot' + (s.obstacle ? ' danger' : '');

      // status chips
      document.getElementById('val-nav').textContent =
        s.navigating ? 'ACTIVE' : 'IDLE';
      document.getElementById('chip-nav').className =
        'chip ' + (s.navigating ? 'ok' : '');

      document.getElementById('val-target').textContent =
        s.target_id ? '#' + s.target_id : '—';

      document.getElementById('val-obs').textContent =
        s.obstacle ? '⚠ BLOCKED' : 'CLEAR';
      document.getElementById('chip-obs').className =
        'chip ' + (s.obstacle ? 'danger' : 'ok');

      // status message
      const [msg, cls] = statusMap[s.status] || ['—', ''];
      const msgEl = document.getElementById('status-msg');
      msgEl.textContent  = msg;
      msgEl.className    = 'status-msg ' + cls;

      // object cards
      renderObjects(s.detections);

      // pc server indicator
      document.getElementById('pc-status').textContent =
        s.detections.length > 0 ? 'AI ✓' : 'AI connecting...';

    }} catch(e) {{}}
  }}, 400);
</script>
</body>
</html>"""


app = FastAPI(title="Robot Navigator")


@app.get("/", response_class=HTMLResponse)
async def index():
    return UI_HTML


@app.post("/goto")
async def goto(body: dict):
    obj_id = int(body.get("id", 0))
    if obj_id < 1:
        return {"error": "invalid id"}
    nav.set_target(obj_id)
    print(f"[NAV] Target set → object #{obj_id}")
    return {"navigating": True, "target_id": obj_id}


@app.post("/stop")
async def stop():
    car.stop()
    nav.cancel()
    return {"status": "stopped"}


@app.get("/status")
async def status():
    return JSONResponse(nav.get())


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────
def main():
    print("━" * 55)
    print("  Robot Navigator")
    print(f"  Web UI    → http://{RPI_IP}:{CONTROL_PORT}")
    print(f"  AI Server → {PC_SERVER_URL}")
    print("━" * 55)

    threading.Thread(target=_poll_detections, daemon=True).start()
    threading.Thread(target=_poll_obstacle,   daemon=True).start()
    threading.Thread(target=_navigation_loop, daemon=True).start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=CONTROL_PORT, log_level="warning")
    finally:
        car.stop()
        car.motors_off()
        print("\n[EXIT] Stopped.")


if __name__ == "__main__":
    main()
