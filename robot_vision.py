"""
robot_vision.py  —  AI-powered 3-D obstacle detector for a robot car
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  * NO calibration needed.  No chessboard.  Just plug in camera IPs and run.
  * Uses MiDaS (neural network) for depth estimation — works on a single image.
  * Uses YOLOv8-nano for object detection — finds chairs, people, boxes, etc.
  * Uses both cameras to cross-validate depth and convert relative → real cm.
  * FastAPI web server — control & view from your phone on the same network.

Install (one time):
    pip install torch torchvision timm opencv-python ultralytics numpy fastapi uvicorn

Run:
    python robot_vision.py

Then open on your phone:
    http://<THIS-PC-IP>:8000

Controls (web UI or keyboard):
    C           calibrate (stand at 50 cm and press)
    [ / ]       fine-tune scale ×0.7 / ×1.4
    + / -       danger distance threshold ± 10 cm
    D           toggle depth-map panel on/off
    1 / 2       show left / right camera as main
    Q           quit OpenCV window
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import threading, time, sys, io, socket
import cv2
import numpy as np
import urllib.request

# ══════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════
LEFT_URL  = "http://10.100.102.38/stream"
RIGHT_URL = "http://10.100.102.32/stream"

# Camera baseline in mm (measure between the two lenses with a ruler).
BASELINE_MM = 112.0

# Which camera is physically rotated 90 degrees?
#   0 = left,  1 = right,  None = neither
ROTATED_CAM = 1
ROTATE_DIR  = cv2.ROTATE_90_CLOCKWISE

# Processing size — both cameras are resized to this before AI inference.
PROC_W, PROC_H = 320, 240

# Danger distance in cm — objects closer than this trigger an alert.
DANGER_CM = 60.0

# Minimum YOLO confidence to count a detection.
YOLO_CONF = 0.40

# Use GPU if available.
DEVICE = "cuda"   # "cuda" or "cpu"

# Web server port.
WEB_PORT = 8000
# ══════════════════════════════════════════════════


# ─────────────────────────────────────────────────
#  Shared state between vision loop and web server
# ─────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.danger_cm      = DANGER_CM
        self.main_cam       = 0           # 0=left, 1=right
        self.show_depth     = True
        self.calibrate_req  = False       # vision loop reads this flag
        self.scale_mult_req = None        # ']' → 1.4,  '[' → 0.7
        self.alert          = False
        self.scale_str      = "NOT CALIBRATED"
        self.detections_log = []          # latest console-style lines
        # Latest JPEG bytes of the composite frame for MJPEG streaming
        self._jpeg_buf      = None
        self._jpeg_lock     = threading.Lock()

    # ── frame buffer ─────────────────────────────
    def push_frame(self, bgr_frame: np.ndarray):
        ok, jpg = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._jpeg_lock:
                self._jpeg_buf = jpg.tobytes()

    def get_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg_buf


state = AppState()


# ─────────────────────────────────────────────────
#  FastAPI web server
# ─────────────────────────────────────────────────
def _build_app():
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

    app = FastAPI(title="Robot Vision")

    # ── HTML dashboard ────────────────────────────
    DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Robot Vision</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

  :root {
    --bg: #0a0d10;
    --panel: #111620;
    --border: #1e2a38;
    --accent: #00e5ff;
    --danger: #ff3355;
    --warn: #ff8c00;
    --ok: #00e676;
    --text: #c8d8e8;
    --muted: #4a6070;
    --radius: 10px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--bg); color: var(--text);
               font-family: 'Exo 2', sans-serif; overflow-x: hidden; }

  header {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  header .logo { font-size: 1.1rem; font-weight: 800; letter-spacing: 2px;
                 color: var(--accent); text-transform: uppercase; }
  header .sub  { font-size: 0.7rem; color: var(--muted); font-family: 'Share Tech Mono', monospace; }
  header .dot  { width: 9px; height: 9px; border-radius: 50%; background: var(--muted);
                 transition: background .3s; flex-shrink: 0; }
  header .dot.alive { background: var(--ok); box-shadow: 0 0 6px var(--ok); }

  .status-bar {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    padding: 6px 16px;
    background: #0d1118;
    border-bottom: 1px solid var(--border);
    display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
  }
  .status-bar .chip {
    display: flex; align-items: center; gap: 5px;
  }
  .status-bar .chip .label { color: var(--muted); }
  .status-bar .chip .val   { color: var(--accent); }
  .status-bar .chip.alert .val { color: var(--danger); animation: blink 0.6s infinite; }
  .status-bar .chip.ok    .val { color: var(--ok); }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

  /* ── video feed ── */
  .feed-wrap {
    position: relative;
    width: 100%;
    background: #000;
    border-bottom: 1px solid var(--border);
  }
  .feed-wrap img {
    width: 100%;
    display: block;
    max-height: 55vw;
    object-fit: contain;
  }
  .feed-overlay {
    position: absolute; bottom: 8px; right: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; color: rgba(255,255,255,.55);
  }
  .alert-banner {
    display: none;
    position: absolute; top: 0; left: 0; right: 0;
    padding: 5px; text-align: center;
    background: rgba(255,51,85,.85);
    font-weight: 800; letter-spacing: 3px; font-size: .9rem;
    animation: blink 0.5s infinite;
  }
  .alert-banner.show { display: block; }

  /* ── controls ── */
  .section { padding: 14px 14px 0; }
  .section-title {
    font-size: 0.6rem; letter-spacing: 3px; color: var(--muted);
    text-transform: uppercase; margin-bottom: 10px;
    border-left: 2px solid var(--accent); padding-left: 8px;
  }

  .btn-grid {
    display: grid;
    gap: 8px;
  }
  .btn-grid.two   { grid-template-columns: 1fr 1fr; }
  .btn-grid.three { grid-template-columns: 1fr 1fr 1fr; }
  .btn-grid.one   { grid-template-columns: 1fr; }

  button {
    font-family: 'Exo 2', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 12px 8px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    border-radius: var(--radius);
    cursor: pointer;
    transition: all .15s;
    letter-spacing: .5px;
    position: relative;
    overflow: hidden;
  }
  button:active { transform: scale(.96); }
  button.primary {
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: inset 0 0 0 0 var(--accent);
  }
  button.primary:active { background: rgba(0,229,255,.12); }
  button.danger-btn { border-color: var(--danger); color: var(--danger); }
  button.ok-btn     { border-color: var(--ok);     color: var(--ok); }
  button.warn-btn   { border-color: var(--warn);   color: var(--warn); }
  button .icon { font-size: 1rem; display: block; margin-bottom: 2px; }
  button .sub  { font-size: 0.62rem; color: var(--muted); font-family: 'Share Tech Mono', monospace; }

  .danger-row {
    display: flex; align-items: center; gap: 10px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 10px 14px;
  }
  .danger-row .val {
    flex: 1; text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem; color: var(--accent);
  }
  .danger-row .val span { font-size: .7rem; color: var(--muted); }
  .danger-row button { flex: 0 0 44px; padding: 8px; font-size: 1.1rem; }

  .log-box {
    margin-top: 10px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    line-height: 1.6;
    color: #7090a0;
    max-height: 90px;
    overflow-y: auto;
    min-height: 40px;
  }

  .spacer { height: 20px; }

  /* scan line effect on video */
  .feed-wrap::after {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 3px, rgba(0,0,0,.08) 3px, rgba(0,0,0,.08) 4px
    );
    pointer-events: none;
  }
</style>
</head>
<body>

<header>
  <div class="dot" id="dot"></div>
  <div>
    <div class="logo">Robot Vision</div>
    <div class="sub">AI Depth · YOLO · Stereo</div>
  </div>
</header>

<div class="status-bar">
  <div class="chip" id="chip-status">
    <span class="label">STATUS</span>
    <span class="val" id="val-status">—</span>
  </div>
  <div class="chip">
    <span class="label">SCALE</span>
    <span class="val" id="val-scale">—</span>
  </div>
  <div class="chip">
    <span class="label">CAM</span>
    <span class="val" id="val-cam">—</span>
  </div>
</div>

<div class="feed-wrap">
  <div class="alert-banner" id="alert-banner">⚠ OBSTACLE DETECTED</div>
  <img id="feed" src="/stream" alt="Live feed">
  <div class="feed-overlay" id="fps-label">live</div>
</div>

<!-- CALIBRATION -->
<div class="section">
  <div class="section-title">Calibration</div>
  <div class="btn-grid one">
    <button class="primary" onclick="api('/calibrate','POST')">
      <span class="icon">🎯</span>
      Calibrate — stand at 50 cm then tap
    </button>
  </div>
  <div style="margin-top:8px" class="btn-grid two">
    <button class="warn-btn" onclick="api('/scale','POST',{mult:0.7})">
      <span class="icon">◀◀</span>
      <span class="sub">Scale ×0.7</span>
    </button>
    <button class="warn-btn" onclick="api('/scale','POST',{mult:1.4})">
      <span class="icon">▶▶</span>
      <span class="sub">Scale ×1.4</span>
    </button>
  </div>
</div>

<!-- DANGER THRESHOLD -->
<div class="section" style="margin-top:14px">
  <div class="section-title">Danger Threshold</div>
  <div class="danger-row">
    <button class="danger-btn" onclick="api('/danger','POST',{delta:-10})">−</button>
    <div class="val" id="val-danger">—<span> cm</span></div>
    <button class="ok-btn"     onclick="api('/danger','POST',{delta:+10})">+</button>
  </div>
</div>

<!-- CAMERA / VIEW -->
<div class="section" style="margin-top:14px">
  <div class="section-title">Camera &amp; View</div>
  <div class="btn-grid two">
    <button onclick="api('/camera','POST',{cam:0})" id="btn-cam0">
      <span class="icon">📷</span>Left Camera
    </button>
    <button onclick="api('/camera','POST',{cam:1})" id="btn-cam1">
      <span class="icon">📷</span>Right Camera
    </button>
  </div>
  <div class="btn-grid one" style="margin-top:8px">
    <button onclick="api('/depth_toggle','POST')" id="btn-depth">
      <span class="icon">🌈</span>Toggle Depth Panel
    </button>
  </div>
</div>

<!-- LOG -->
<div class="section" style="margin-top:14px">
  <div class="section-title">Detection Log</div>
  <div class="log-box" id="log-box">waiting for detections…</div>
</div>

<div class="spacer"></div>

<script>
  async function api(path, method='POST', body=null) {
    try {
      const opts = { method };
      if (body) { opts.headers={'Content-Type':'application/json'}; opts.body=JSON.stringify(body); }
      const r = await fetch(path, opts);
      const j = await r.json();
      if (j.message) flashToast(j.message);
    } catch(e) { console.error(e); }
  }

  function flashToast(msg) {
    let t = document.getElementById('toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'toast';
      Object.assign(t.style, {
        position:'fixed', bottom:'20px', left:'50%', transform:'translateX(-50%)',
        background:'rgba(0,229,255,.15)', border:'1px solid #00e5ff',
        color:'#00e5ff', padding:'8px 20px', borderRadius:'20px',
        fontFamily:"'Share Tech Mono',monospace", fontSize:'.8rem',
        zIndex:999, transition:'opacity .4s'
      });
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.style.opacity = '1';
    clearTimeout(t._tid);
    t._tid = setTimeout(() => t.style.opacity='0', 2500);
  }

  // Poll status every second
  let frameCount = 0;
  setInterval(async () => {
    try {
      const s = await (await fetch('/status')).json();

      // dot
      document.getElementById('dot').classList.toggle('alive', s.alive);

      // status chip
      const chip = document.getElementById('chip-status');
      const valS = document.getElementById('val-status');
      valS.textContent = s.alert ? '!! OBSTACLE !!' : 'CLEAR';
      chip.className = 'chip ' + (s.alert ? 'alert' : 'ok');

      document.getElementById('val-scale').textContent  = s.scale_str;
      document.getElementById('val-cam').textContent    = s.main_cam === 0 ? 'LEFT' : 'RIGHT';
      document.getElementById('val-danger').innerHTML   = s.danger_cm + '<span> cm</span>';

      // alert banner
      document.getElementById('alert-banner').classList.toggle('show', s.alert);

      // log
      const lb = document.getElementById('log-box');
      if (s.log && s.log.length) {
        lb.innerHTML = s.log.slice().reverse().map(l =>
          `<div style="color:${l.includes('OBSTACLE')||l.includes('!!')?'#ff3355':'#7090a0'}">${l}</div>`
        ).join('');
      }
    } catch(e) {}
  }, 1000);
</script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    # ── MJPEG stream ──────────────────────────────
    def _mjpeg_gen():
        while True:
            jpg = state.get_jpeg()
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            time.sleep(0.04)   # ~25 fps cap

    @app.get("/stream")
    def video_stream():
        return StreamingResponse(
            _mjpeg_gen(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # ── status JSON ───────────────────────────────
    @app.get("/status")
    def get_status():
        with state.lock:
            return {
                "alive":     True,
                "alert":     state.alert,
                "danger_cm": round(state.danger_cm),
                "main_cam":  state.main_cam,
                "scale_str": state.scale_str,
                "show_depth":state.show_depth,
                "log":       list(state.detections_log[-8:]),
            }

    # ── calibrate ─────────────────────────────────
    @app.post("/calibrate")
    def calibrate():
        with state.lock:
            state.calibrate_req = True
        return {"message": "Calibrating at 50 cm…"}

    # ── danger threshold ──────────────────────────
    @app.post("/danger")
    def set_danger(body: dict):
        delta = float(body.get("delta", 0))
        with state.lock:
            state.danger_cm = max(10, state.danger_cm + delta)
            val = state.danger_cm
        return {"message": f"Danger threshold → {val:.0f} cm", "danger_cm": val}

    # ── camera select ─────────────────────────────
    @app.post("/camera")
    def set_camera(body: dict):
        cam = int(body.get("cam", 0))
        with state.lock:
            state.main_cam = cam
        return {"message": f"Camera → {'LEFT' if cam == 0 else 'RIGHT'}"}

    # ── depth toggle ──────────────────────────────
    @app.post("/depth_toggle")
    def depth_toggle():
        with state.lock:
            state.show_depth = not state.show_depth
            val = state.show_depth
        return {"message": f"Depth panel {'ON' if val else 'OFF'}"}

    # ── scale fine-tune ───────────────────────────
    @app.post("/scale")
    def set_scale(body: dict):
        mult = float(body.get("mult", 1.0))
        with state.lock:
            state.scale_mult_req = mult
        label = "×1.4 (increase)" if mult > 1 else "×0.7 (decrease)"
        return {"message": f"Scale {label}"}

    return app


def _run_web_server():
    import uvicorn
    app = _build_app()
    # Find local IP to print a helpful URL
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print(f"\n{'═'*60}")
    print(f"  📱  Web UI:  http://{local_ip}:{WEB_PORT}")
    print(f"  🎥  Stream:  http://{local_ip}:{WEB_PORT}/stream")
    print(f"{'═'*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT, log_level="warning")


# ─────────────────────────────────────────────────
#  Frame grabber — threaded MJPEG reader
# ─────────────────────────────────────────────────
_frames = [None, None]
_lock   = threading.Lock()

def _read_mjpeg(url: str, idx: int) -> None:
    while True:
        try:
            stream = urllib.request.urlopen(url, timeout=6)
            buf = b""
            while True:
                buf += stream.read(4096)
                a = buf.find(b'\xff\xd8')
                b_pos = buf.find(b'\xff\xd9')
                if a != -1 and b_pos != -1:
                    img = cv2.imdecode(
                        np.frombuffer(buf[a:b_pos+2], np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    buf = buf[b_pos+2:]
                    if img is not None:
                        with _lock:
                            _frames[idx] = img
        except Exception as e:
            print(f"[cam{idx}] {e}, retrying…")
            time.sleep(1)


def grab_frames():
    with _lock:
        fl = _frames[0]
        fr = _frames[1]
    if fl is None or fr is None:
        return None, None
    fl = fl.copy(); fr = fr.copy()
    if ROTATED_CAM == 0: fl = cv2.rotate(fl, ROTATE_DIR)
    elif ROTATED_CAM == 1: fr = cv2.rotate(fr, ROTATE_DIR)
    fl = cv2.resize(fl, (PROC_W, PROC_H))
    fr = cv2.resize(fr, (PROC_W, PROC_H))
    return fl, fr


# ─────────────────────────────────────────────────
#  MiDaS depth estimator
# ─────────────────────────────────────────────────
class DepthEstimator:
    def __init__(self, device: str = "cuda"):
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[MiDaS] loading model on {self.device}…")
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        self.model.eval().to(self.device)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self._torch = torch
        print("[MiDaS] ready.")

    def estimate(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        with self._torch.no_grad():
            pred = self.model(inp)
        depth = pred.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (bgr_image.shape[1], bgr_image.shape[0]))
        return depth


# ─────────────────────────────────────────────────
#  Depth scaler
# ─────────────────────────────────────────────────
class DepthScaler:
    def __init__(self, baseline_mm: float, focal_px: float):
        self.baseline_mm     = baseline_mm
        self.focal_px        = focal_px
        self.scale           = None
        self.manual_mult     = 1.0
        self._manual_locked  = False
        self._auto_buf       = []
        self._orb            = cv2.ORB_create(800)
        self._bf             = cv2.BFMatcher(cv2.NORM_HAMMING)

    def manual_calibrate(self, midas_depth: np.ndarray, target_cm: float = 50.0):
        h, w = midas_depth.shape
        cy, cx = h // 4, w // 4
        roi = midas_depth[cy:3*cy, cx:3*cx]
        med = float(np.median(roi[roi > 1e-3]))
        if med > 1e-3:
            self.scale = target_cm * med
            self.manual_mult = 1.0
            self._manual_locked = True
            msg = f"[CALIBRATED] scale={self.scale:.1f}  (target={target_cm:.0f}cm, midas={med:.2f})"
            print(f"  {msg}")
            return True, msg
        msg = "[CALIBRATE FAILED] no valid depth in centre of frame"
        print(f"  {msg}")
        return False, msg

    def auto_update(self, left_bgr, right_bgr, depth_left, depth_right):
        if self._manual_locked:
            return
        gl = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self._orb.detectAndCompute(gl, None)
        kp2, des2 = self._orb.detectAndCompute(gr, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return
        raw = self._bf.knnMatch(des1, des2, k=2)
        good = [m for pair in raw if len(pair)==2
                for m, n in [pair] if m.distance < 0.70 * n.distance]
        if len(good) < 5:
            return
        scales = []
        for m in good:
            pt_l = kp1[m.queryIdx].pt
            pt_r = kp2[m.trainIdx].pt
            if abs(pt_l[1] - pt_r[1]) > 15: continue
            disp = abs(pt_l[0] - pt_r[0])
            if disp < 2: continue
            z_mm = (self.focal_px * self.baseline_mm) / disp
            if not (80 <= z_mm <= 4000): continue
            z_cm = z_mm / 10.0
            y = min(int(pt_l[1]), depth_left.shape[0]-1)
            x = min(int(pt_l[0]), depth_left.shape[1]-1)
            mv = depth_left[y, x]
            if mv < 1e-3: continue
            scales.append(z_cm * mv)
        if len(scales) < 3: return
        arr = np.array(scales)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        clean = arr[(arr >= q1-1.5*iqr) & (arr <= q3+1.5*iqr)]
        if len(clean) < 3: return
        self._auto_buf.append(float(np.median(clean)))
        self._auto_buf = self._auto_buf[-40:]
        new_scale = float(np.median(self._auto_buf))
        if self.scale is None:
            self.scale = new_scale
        elif self.manual_mult == 1.0:
            self.scale = self.scale * 0.9 + new_scale * 0.1

    def to_cm(self, midas_depth: np.ndarray) -> np.ndarray:
        if self.scale is None:
            return None
        eff = self.scale * self.manual_mult
        safe = np.where(midas_depth > 1e-3, midas_depth, 1e-3)
        return eff / safe


# ─────────────────────────────────────────────────
#  YOLO object detector
# ─────────────────────────────────────────────────
class ObjectDetector:
    def __init__(self, conf: float = 0.4, device: str = "cuda"):
        from ultralytics import YOLO
        print("[YOLO] loading yolov8n…")
        self.model = YOLO("yolov8n.pt")
        self.model.to(device if device == "cuda" else "cpu")
        self.conf = conf
        print("[YOLO] ready.")

    def detect(self, bgr_image: np.ndarray) -> list:
        results = self.model(bgr_image, conf=self.conf, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id   = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf     = float(box.conf[0])
            detections.append({"name": cls_name, "conf": conf,
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return detections


# ─────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────
def colorize_depth(depth_cm, danger_cm):
    if depth_cm is None:
        return np.full((PROC_H, PROC_W, 3), 40, np.uint8)
    near = 10.0; far = max(danger_cm * 2.0, 100.0)
    norm = np.clip((depth_cm - near) / (far - near), 0, 1)
    norm = 1.0 - norm
    invalid = depth_cm < 1.0
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[invalid] = (25, 25, 25)
    return col


def draw_detections(canvas, detections, depth_cm, danger_cm):
    alert = False
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        name = det["name"]
        dist_label = ""; obj_dist = None
        if depth_cm is not None:
            roi   = depth_cm[y1:y2, x1:x2]
            valid = roi[(roi > 1) & (roi < 2000)]
            if valid.size > 10:
                obj_dist = float(np.median(valid))
                f_approx = PROC_W * 0.866
                w_cm = (x2-x1)*obj_dist/f_approx
                h_cm = (y2-y1)*obj_dist/f_approx
                dist_label = f" {obj_dist:.0f}cm ({w_cm:.0f}x{h_cm:.0f}cm)"
        if obj_dist is not None and obj_dist < danger_cm:
            col = (0, 0, 255); alert = True
        elif obj_dist is not None and obj_dist < danger_cm * 1.5:
            col = (0, 120, 255)
        else:
            col = (0, 200, 60)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        label = f"{name}{dist_label}"
        ly = max(y1 - 6, 14)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(canvas, (x1, ly-th-3), (x1+tw+2, ly+2), (0,0,0), -1)
        cv2.putText(canvas, label, (x1+1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
    return alert


# ─────────────────────────────────────────────────
#  Main vision loop
# ─────────────────────────────────────────────────
def main():
    print("━" * 60)
    print("  Robot Vision — AI depth + object detection + Web UI")
    print("━" * 60)

    # ── start web server in background ───────────
    threading.Thread(target=_run_web_server, daemon=True).start()

    # ── load AI models ───────────────────────────
    depth_net = DepthEstimator(DEVICE)
    detector  = ObjectDetector(YOLO_CONF, DEVICE)
    focal_px  = PROC_W / (2 * np.tan(np.radians(30)))
    scaler    = DepthScaler(BASELINE_MM, focal_px)

    # ── start camera threads ─────────────────────
    for i, url in enumerate([LEFT_URL, RIGHT_URL]):
        threading.Thread(target=_read_mjpeg, args=(url, i), daemon=True).start()

    print("Connecting to cameras…")
    while True:
        fl, fr = grab_frames()
        if fl is not None and fr is not None:
            break
        if cv2.waitKey(200) & 0xFF == ord('q'):
            return
    print("Both cameras online.\n")

    WIN       = "Robot Vision  [C=calibrate | [/]=scale | +/-=danger | D=depth | Q=quit]"
    frame_n   = 0
    latest_midas = None

    while True:
        fl, fr = grab_frames()
        if fl is None or fr is None:
            cv2.waitKey(30); continue

        # ── read state from web/keyboard ─────────
        with state.lock:
            danger_cm  = state.danger_cm
            main_cam   = state.main_cam
            show_depth = state.show_depth
            do_calib   = state.calibrate_req
            mult_req   = state.scale_mult_req
            state.calibrate_req  = False
            state.scale_mult_req = None

        # ── apply scale fine-tune (web request) ──
        if mult_req is not None:
            scaler.manual_mult *= mult_req
            eff = (scaler.scale or 0) * scaler.manual_mult
            print(f"  scale ×{mult_req} → effective={eff:.0f}")

        # ── MiDaS depth ───────────────────────────
        midas_l = depth_net.estimate(fl)
        midas_r = depth_net.estimate(fr)
        latest_midas = midas_l if main_cam == 0 else midas_r

        # ── calibration (web or keyboard) ────────
        if do_calib and latest_midas is not None:
            ok, msg = scaler.manual_calibrate(latest_midas, target_cm=50.0)
            with state.lock:
                state.detections_log.append(f"[CAL] {msg}")
                state.detections_log = state.detections_log[-30:]

        # ── auto-scale ────────────────────────────
        scaler.auto_update(fl, fr, midas_l, midas_r)

        # ── convert to cm ─────────────────────────
        depth_cm_l = scaler.to_cm(midas_l)
        depth_cm_r = scaler.to_cm(midas_r)

        main_img = fl if main_cam == 0 else fr
        depth_cm = depth_cm_l if main_cam == 0 else depth_cm_r

        # ── YOLO detection ────────────────────────
        detections = detector.detect(main_img)

        # ── draw ──────────────────────────────────
        vis   = main_img.copy()
        alert = draw_detections(vis, detections, depth_cm, danger_cm)

        if scaler.scale:
            eff = scaler.scale * scaler.manual_mult
            scale_str = f"scale={eff:.0f}"
        else:
            scale_str = "NOT CALIBRATED"

        status_txt = "!! OBSTACLE !!" if alert else "CLEAR"
        s_col      = (0, 0, 255) if alert else (0, 220, 0)
        cv2.putText(vis, status_txt, (6, PROC_H-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_col, 2)
        hud = f"danger={danger_cm:.0f}cm  {scale_str}"
        cv2.putText(vis, hud, (6, PROC_H-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160,160,160), 1)
        if scaler.scale is None:
            cv2.putText(vis, "Press C or tap WEB UI to calibrate", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,200,255), 1)

        DISP_SCALE = 2
        vis_big = cv2.resize(vis, (PROC_W*DISP_SCALE, PROC_H*DISP_SCALE))

        if show_depth:
            dvis = colorize_depth(depth_cm, danger_cm)
            cv2.putText(dvis, "DEPTH (red=close)", (4,16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
            dvis_big = cv2.resize(dvis, (PROC_W*DISP_SCALE, PROC_H*DISP_SCALE))
            display = np.hstack([vis_big, dvis_big])
        else:
            display = vis_big

        cv2.imshow(WIN, display)

        # ── push to web stream ────────────────────
        state.push_frame(display)

        # ── update web state ──────────────────────
        with state.lock:
            state.alert     = alert
            state.scale_str = scale_str

        # ── periodic detection log ────────────────
        frame_n += 1
        if frame_n % 20 == 0 and detections and depth_cm is not None:
            parts = []
            for d in detections:
                roi   = depth_cm[d["y1"]:d["y2"], d["x1"]:d["x2"]]
                valid = roi[(roi > 1) & (roi < 2000)]
                dist  = f" {np.median(valid):.0f}cm" if valid.size > 0 else ""
                parts.append(f'{d["name"]}{dist}')
            line = f"[{scale_str}]  {'  |  '.join(parts)}"
            print(f"  {line}")
            with state.lock:
                state.detections_log.append(line)
                state.detections_log = state.detections_log[-30:]

        # ── keyboard controls ─────────────────────
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('c'):
            if latest_midas is not None:
                ok, msg = scaler.manual_calibrate(latest_midas, 50.0)
                with state.lock:
                    state.detections_log.append(f"[CAL] {msg}")
            else:
                print("  No depth data yet…")
        elif key == ord(']'):
            scaler.manual_mult *= 1.4
            with state.lock: state.scale_mult_req = None   # clear any pending
            print(f"  scale ×1.4 → {(scaler.scale or 0)*scaler.manual_mult:.0f}")
        elif key == ord('['):
            scaler.manual_mult *= 0.7
            print(f"  scale ×0.7 → {(scaler.scale or 0)*scaler.manual_mult:.0f}")
        elif key in (ord('+'), ord('=')):
            with state.lock: state.danger_cm += 10
            print(f"  danger → {state.danger_cm:.0f} cm")
        elif key == ord('-'):
            with state.lock: state.danger_cm = max(10, state.danger_cm - 10)
            print(f"  danger → {state.danger_cm:.0f} cm")
        elif key == ord('d'):
            with state.lock: state.show_depth = not state.show_depth
        elif key == ord('1'):
            with state.lock: state.main_cam = 0; print("  main camera → LEFT")
        elif key == ord('2'):
            with state.lock: state.main_cam = 1; print("  main camera → RIGHT")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
