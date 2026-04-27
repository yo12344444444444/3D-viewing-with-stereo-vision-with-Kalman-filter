"""
ai_server.py  —  Unified AI Server  (runs on Windows PC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replaces BOTH robot_vision.py AND pc_ai_server.py.
Runs ONE MiDaS + ONE YOLO-World instance for everything.

  /           → web dashboard (distances, calibration, controls)
  /stream     → annotated MJPEG video
  /detections → JSON objects + depth  (rpi_navigator polls this)
  /status     → obstacle alert JSON   (rpi_navigator polls this)
  /calibrate  → trigger 50 cm calibration
  /scale_value→ calibrated scale (shared)
  /classes    → update YOLO detection classes at runtime
  /danger     → adjust obstacle threshold
  /scale      → fine-tune depth multiplier
  /health     → health check

Install (once):
    pip install torch torchvision timm opencv-python ultralytics fastapi uvicorn numpy

Run:
    python ai_server.py

On RPi:
    python3 rpi_navigator.py   (it polls this server for both detections + obstacle)
"""

import threading, time, subprocess, re, json
import urllib.request
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

# ══════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════
# ESP32-CAM MAC addresses — script resolves IPs automatically
LEFT_MAC   = "e0-5a-1b-6c-e4-2c "   # ← replace with LEFT  ESP32 MAC
RIGHT_MAC  = "34:98:7a:6b:61:40"   # ← replace with RIGHT ESP32 MAC (already known)

AP_SUBNET  = "10.3.141"
STREAM_PATH = "/stream"

# YOLO-World — objects to detect (edit freely, or use /classes endpoint)
CLASSES = [
    "person", "chair", "table", "sofa", "desk", "box", "bag",
    "bottle", "cup", "mug", "can", "laptop", "phone", "book",
    "remote", "keyboard", "backpack", "suitcase", "door", "wall"
]

SERVER_PORT  = 5000
DEVICE       = "cpu"       # "cuda" once paging file is fixed
PROC_W       = 320
PROC_H       = 240
YOLO_CONF    = 0.30
DANGER_CM    = 60.0        # objects closer than this = obstacle
ARRIVE_CM    = 35.0        # navigation "arrived" distance
BASELINE_MM  = 112.0       # distance between the two camera lenses (mm)

ARP_RETRIES    = 5
ARP_RETRY_WAIT = 3
# ══════════════════════════════════════════════════


# ─────────────────────────────────────────────────
#  MAC → IP resolver
# ─────────────────────────────────────────────────
def _norm_mac(mac: str) -> str:
    d = re.sub(r"[^0-9a-fA-F]", "", mac)
    if len(d) != 12:
        raise ValueError(f"Bad MAC: {mac!r}")
    return ":".join(d[i:i+2].lower() for i in range(0, 12, 2))

def _ping_subnet(subnet: str):
    procs = [subprocess.Popen(f"ping -n 1 -w 300 {subnet}.{i}",
             shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for i in range(2, 51)]
    for p in procs:
        try: p.wait(timeout=5)
        except: p.kill()

def _arp_lookup(mac_norm: str) -> str | None:
    mac_dash = mac_norm.replace(":", "-")
    try:
        out = subprocess.check_output("arp -a", shell=True, text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            ll = line.lower()
            if mac_norm in ll or mac_dash in ll:
                for part in line.split():
                    if re.match(r"\d+\.\d+\.\d+\.\d+", part):
                        return part
    except: pass
    return None

def find_ip(mac: str, subnet: str = AP_SUBNET,
            retries: int = ARP_RETRIES, wait: float = ARP_RETRY_WAIT) -> str | None:
    try: mac_norm = _norm_mac(mac)
    except ValueError as e: print(f"[ARP] {e}"); return None
    print(f"[ARP] Looking for {mac_norm} on {subnet}.0/24 ...")
    for attempt in range(1, retries + 1):
        print(f"[ARP] Attempt {attempt}/{retries}...")
        _ping_subnet(subnet)
        ip = _arp_lookup(mac_norm)
        if ip:
            print(f"[ARP] ✓  Found at {ip}")
            return ip
        if attempt < retries:
            print(f"[ARP] Not found — waiting {wait}s")
            time.sleep(wait)
    print(f"[ARP] ✗  Could not find {mac_norm}")
    return None

# Resolve camera IPs at startup
_left_ip  = find_ip(LEFT_MAC)
_right_ip = find_ip(RIGHT_MAC)
LEFT_URL  = f"http://{_left_ip}{STREAM_PATH}"  if _left_ip  else f"http://10.3.141.143{STREAM_PATH}"
RIGHT_URL = f"http://{_right_ip}{STREAM_PATH}" if _right_ip else f"http://10.3.141.150{STREAM_PATH}"
print(f"[CAM] Left  → {LEFT_URL}")
print(f"[CAM] Right → {RIGHT_URL}")


# ─────────────────────────────────────────────────
#  Shared state
# ─────────────────────────────────────────────────
class State:
    def __init__(self):
        self._lock         = threading.Lock()
        self._jpeg_lock    = threading.Lock()
        # video
        self._jpeg         = None
        # detections
        self.detections    = []    # [{id, label, conf, cx, cy, x1,y1,x2,y2, depth_cm, arrived, danger}]
        self.distances     = []    # [{name, dist_cm, danger}] sorted closest-first
        # obstacle
        self.alert         = False
        self.danger_cm     = DANGER_CM
        # depth calibration
        self.scale         = None
        self.manual_mult   = 1.0
        self.scale_str     = "NOT CALIBRATED"
        self.calibrate_req = False
        self.scale_mult_req = None
        # camera
        self.main_cam      = 0    # 0=left, 1=right
        self.show_depth    = True
        # log
        self.log           = []

    # ── thread-safe setters ───────────────────────
    def push_jpeg(self, frame):
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._jpeg_lock:
                self._jpeg = jpg.tobytes()

    def get_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def snapshot(self):
        with self._lock:
            return {
                "alert":      self.alert,
                "danger_cm":  round(self.danger_cm),
                "detections": list(self.detections),
                "distances":  list(self.distances),
                "scale_str":  self.scale_str,
                "main_cam":   self.main_cam,
                "show_depth": self.show_depth,
                "log":        list(self.log[-8:]),
                "alive":      True,
            }


state = State()


# ─────────────────────────────────────────────────
#  Camera readers (MJPEG)
# ─────────────────────────────────────────────────
_frames     = [None, None]
_frames_lock = threading.Lock()

def _read_cam(url: str, idx: int):
    while True:
        try:
            stream = urllib.request.urlopen(url, timeout=15)
            buf = b""
            while True:
                buf += stream.read(4096)
                a = buf.find(b'\xff\xd8')
                b = buf.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    img = cv2.imdecode(np.frombuffer(buf[a:b+2], np.uint8), cv2.IMREAD_COLOR)
                    buf = buf[b+2:]
                    if img is not None:
                        img = cv2.resize(img, (PROC_W, PROC_H))
                        with _frames_lock:
                            _frames[idx] = img
        except Exception as e:
            print(f"[CAM{idx}] {e} — retrying...")
            time.sleep(1)

def grab_frames():
    with _frames_lock:
        return (_frames[0].copy() if _frames[0] is not None else None,
                _frames[1].copy() if _frames[1] is not None else None)


# ─────────────────────────────────────────────────
#  MiDaS depth estimator  (loaded ONCE)
# ─────────────────────────────────────────────────
def _load_midas():
    import torch
    dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[MiDaS] loading on {dev}...")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    model.eval().to(dev)
    tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    print("[MiDaS] ready.")
    return model, tf, dev

def _midas_infer(model, tf, dev, bgr):
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = tf(rgb).to(dev)
    with torch.no_grad():
        pred = model(inp)
    d = pred.squeeze().cpu().numpy()
    return cv2.resize(d, (bgr.shape[1], bgr.shape[0]))


# ─────────────────────────────────────────────────
#  Depth scaler — stereo auto-calibration + manual
# ─────────────────────────────────────────────────
class DepthScaler:
    def __init__(self):
        self.scale        = None
        self.manual_mult  = 1.0
        self._locked      = False
        self._buf         = []
        self._orb         = cv2.ORB_create(800)
        self._bf          = cv2.BFMatcher(cv2.NORM_HAMMING)
        focal_px          = PROC_W / (2 * np.tan(np.radians(30)))
        self._focal       = focal_px

    def manual_calibrate(self, midas_depth, target_cm=50.0):
        h, w = midas_depth.shape
        cy, cx = h // 4, w // 4
        roi = midas_depth[cy:3*cy, cx:3*cx]
        med = float(np.median(roi[roi > 1e-3]))
        if med > 1e-3:
            self.scale       = target_cm * med
            self.manual_mult = 1.0
            self._locked     = True
            msg = f"Calibrated: scale={self.scale:.1f}  target={target_cm:.0f}cm"
            print(f"[CAL] {msg}")
            return True, msg
        msg = "Calibration failed — no valid depth in centre"
        print(f"[CAL] {msg}")
        return False, msg

    def auto_update(self, left_bgr, right_bgr, depth_l, depth_r):
        """Use ORB stereo matches to auto-estimate scale (only when not manually calibrated)."""
        if self._locked:
            return
        gl = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self._orb.detectAndCompute(gl, None)
        kp2, des2 = self._orb.detectAndCompute(gr, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return
        raw = self._bf.knnMatch(des1, des2, k=2)
        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.70 * n.distance]
        if len(good) < 5:
            return
        scales = []
        for m in good:
            pl = kp1[m.queryIdx].pt
            pr = kp2[m.trainIdx].pt
            if abs(pl[1] - pr[1]) > 15: continue
            disp = abs(pl[0] - pr[0])
            if disp < 2: continue
            z_cm = (self._focal * BASELINE_MM) / disp / 10.0
            if not (8 <= z_cm <= 400): continue
            y = min(int(pl[1]), depth_l.shape[0]-1)
            x = min(int(pl[0]), depth_l.shape[1]-1)
            mv = depth_l[y, x]
            if mv < 1e-3: continue
            scales.append(z_cm * mv)
        if len(scales) < 3: return
        arr = np.array(scales)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        clean = arr[(arr >= q1-1.5*iqr) & (arr <= q3+1.5*iqr)]
        if len(clean) < 3: return
        self._buf.append(float(np.median(clean)))
        self._buf = self._buf[-40:]
        new_s = float(np.median(self._buf))
        self.scale = new_s if self.scale is None else self.scale * 0.9 + new_s * 0.1

    def to_cm(self, midas_depth):
        if self.scale is None:
            return None
        eff  = self.scale * self.manual_mult
        safe = np.where(midas_depth > 1e-3, midas_depth, 1e-3)
        return eff / safe


# ─────────────────────────────────────────────────
#  YOLO-World detector  (loaded ONCE)
# ─────────────────────────────────────────────────
def _load_yolo():
    from ultralytics import YOLO
    print("[YOLO] loading yolov8s-worldv2.pt...")
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(CLASSES)
    print(f"[YOLO] ready — {len(CLASSES)} classes")
    return model


# ─────────────────────────────────────────────────
#  Main AI loop  (one MiDaS + one YOLO per frame)
# ─────────────────────────────────────────────────
def _ai_loop():
    midas, midas_tf, dev = _load_midas()
    yolo                 = _load_yolo()
    scaler               = DepthScaler()
    latest_midas         = None

    while True:
        left, right = grab_frames()
        if left is None:
            time.sleep(0.05)
            continue
        # use right as fallback if only left is available
        right = right if right is not None else left

        with state._lock:
            main_idx    = state.main_cam
            danger_cm   = state.danger_cm
            do_calib    = state.calibrate_req
            mult_req    = state.scale_mult_req
            show_depth  = state.show_depth
            state.calibrate_req  = False
            state.scale_mult_req = None

        main_frame = left if main_idx == 0 else right

        # ── apply manual scale tweak ──────────────
        if mult_req is not None:
            scaler.manual_mult *= mult_req

        # ── MiDaS — run ONCE on main frame ───────
        midas_raw   = _midas_infer(midas, midas_tf, dev, main_frame)
        midas_other = _midas_infer(midas, midas_tf, dev,
                                   right if main_idx == 0 else left)
        latest_midas = midas_raw

        # ── stereo auto-calibration ───────────────
        scaler.auto_update(left, right,
                           midas_raw if main_idx == 0 else midas_other,
                           midas_other if main_idx == 0 else midas_raw)

        # ── manual calibration request ────────────
        if do_calib:
            ok, msg = scaler.manual_calibrate(midas_raw, target_cm=50.0)
            with state._lock:
                state.log.append(f"[CAL] {msg}")
                state.log = state.log[-30:]

        # ── convert to cm ─────────────────────────
        depth_cm = scaler.to_cm(midas_raw)
        scale_str = f"scale={scaler.scale * scaler.manual_mult:.0f}" \
                    if scaler.scale else "NOT CALIBRATED"

        # ── YOLO-World detect ──────────────────────
        results   = yolo(main_frame, conf=YOLO_CONF, verbose=False)
        annotated = main_frame.copy()
        detections = []
        dist_list  = []
        alert      = False
        obj_id     = 1

        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
            cls_id  = int(box.cls[0])
            label   = results[0].names[cls_id]
            conf    = float(box.conf[0])
            cx, cy  = int((x1+x2)//2), int((y1+y2)//2)

            # depth for this object
            obj_depth = None
            if depth_cm is not None:
                roi   = depth_cm[y1:y2, x1:x2]
                valid = roi[(roi > 1) & (roi < 2000)]
                if valid.size > 5:
                    obj_depth = float(np.median(valid))

            is_danger  = obj_depth is not None and obj_depth < danger_cm
            is_arrived = obj_depth is not None and obj_depth < ARRIVE_CM
            if is_danger:
                alert = True

            detections.append({
                "id":       int(obj_id),
                "label":    str(label),
                "conf":     round(float(conf), 2),
                "cx":       int(cx),  "cy":      int(cy),
                "x1":       int(x1),  "y1":      int(y1),
                "x2":       int(x2),  "y2":      int(y2),
                "depth_cm": round(float(obj_depth), 1) if obj_depth is not None else None,
                "arrived":  bool(is_arrived),
                "danger":   bool(is_danger),
            })

            if obj_depth is not None:
                dist_list.append({
                    "name":    str(label),
                    "dist_cm": round(float(obj_depth), 1),
                    "danger":  bool(is_danger),
                })

            # draw box
            col = (0, 60, 255) if is_danger else (0, 220, 60)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            tag = f"#{obj_id} {label}"
            if obj_depth:
                tag += f" {obj_depth:.0f}cm"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), (0,0,0), -1)
            cv2.putText(annotated, tag, (x1+2, y1-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.circle(annotated, (cx, cy), 4, (0,255,255), -1)
            obj_id += 1

        dist_list.sort(key=lambda x: x["dist_cm"])

        # ── depth overlay ──────────────────────────
        if show_depth and depth_cm is not None:
            near, far = 10.0, max(danger_cm * 2, 100.0)
            norm = np.clip((depth_cm - near) / (far - near), 0, 1)
            col_map = cv2.applyColorMap(((1 - norm) * 255).astype(np.uint8),
                                        cv2.COLORMAP_TURBO)
            annotated = cv2.addWeighted(annotated, 0.65, col_map, 0.35, 0)

        # ── HUD ───────────────────────────────────
        status_txt = "!! OBSTACLE !!" if alert else "CLEAR"
        s_col = (0, 60, 255) if alert else (0, 220, 0)
        cv2.putText(annotated, status_txt, (6, PROC_H-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_col, 2)
        cv2.putText(annotated, f"{scale_str}  danger={danger_cm:.0f}cm",
                    (6, PROC_H-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160,160,160), 1)
        if scaler.scale is None:
            cv2.putText(annotated, "Tap CALIBRATE in web UI",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,200,255), 1)

        # crosshair
        cx2, cy2 = PROC_W//2, PROC_H//2
        cv2.line(annotated, (cx2-10, cy2), (cx2+10, cy2), (255,255,255), 1)
        cv2.line(annotated, (cx2, cy2-10), (cx2, cy2+10), (255,255,255), 1)

        # ── push to state ──────────────────────────
        with state._lock:
            state.detections  = detections
            state.distances   = dist_list
            state.alert       = alert
            state.scale_str   = scale_str
            state.scale       = scaler.scale
            state.manual_mult = scaler.manual_mult

        state.push_jpeg(annotated)


# ─────────────────────────────────────────────────
#  FastAPI
# ─────────────────────────────────────────────────
app = FastAPI(title="AI Server")

def _mjpeg_gen():
    while True:
        jpg = state.get_jpeg()
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.04)

@app.get("/stream")
def stream():
    return StreamingResponse(_mjpeg_gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detections")
def get_detections():
    with state._lock:
        return JSONResponse({"objects": list(state.detections)})

@app.get("/status")
def get_status():
    return JSONResponse(state.snapshot())

@app.get("/scale_value")
def scale_value():
    with state._lock:
        s, m = state.scale, state.manual_mult
    if s is None:
        return {"calibrated": False, "scale": None}
    return {"calibrated": True, "scale": s * m}

@app.post("/calibrate")
def calibrate():
    with state._lock:
        state.calibrate_req = True
    return {"message": "Calibrating at 50 cm — point camera at object 50 cm away"}

@app.post("/classes")
def set_classes(body: dict):
    new = body.get("classes", [])
    if new:
        CLASSES.clear(); CLASSES.extend(new)
    return {"classes": CLASSES}

@app.post("/danger")
def set_danger(body: dict):
    delta = float(body.get("delta", 0))
    with state._lock:
        state.danger_cm = max(10, state.danger_cm + delta)
        val = state.danger_cm
    return {"message": f"Danger threshold → {val:.0f} cm", "danger_cm": val}

@app.post("/scale")
def set_scale(body: dict):
    mult = float(body.get("mult", 1.0))
    with state._lock:
        state.scale_mult_req = mult
    return {"message": f"Scale ×{mult}"}

@app.post("/camera")
def set_camera(body: dict):
    cam = int(body.get("cam", 0))
    with state._lock:
        state.main_cam = cam
    return {"message": f"Camera → {'LEFT' if cam == 0 else 'RIGHT'}"}

@app.post("/depth_toggle")
def depth_toggle():
    with state._lock:
        state.show_depth = not state.show_depth
        val = state.show_depth
    return {"message": f"Depth overlay {'ON' if val else 'OFF'}"}

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Robot Vision</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');
  :root {
    --bg: #0a0d10; --panel: #111620; --border: #1e2a38;
    --accent: #00e5ff; --danger: #ff3355; --warn: #ff8c00;
    --ok: #00e676; --text: #c8d8e8; --muted: #4a6070; --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--text);
               font-family: 'Exo 2', sans-serif; overflow-x: hidden; }
  header { display: flex; align-items: center; gap: 10px;
           padding: 12px 16px; background: var(--panel);
           border-bottom: 1px solid var(--border);
           position: sticky; top: 0; z-index: 10; }
  header .logo { font-size: 1.1rem; font-weight: 800; letter-spacing: 2px;
                 color: var(--accent); text-transform: uppercase; }
  header .sub  { font-size: 0.7rem; color: var(--muted);
                 font-family: 'Share Tech Mono', monospace; }
  header .dot  { width: 9px; height: 9px; border-radius: 50%;
                 background: var(--muted); transition: background .3s; flex-shrink: 0; }
  header .dot.alive  { background: var(--ok); box-shadow: 0 0 6px var(--ok); }
  header .dot.danger { background: var(--danger); animation: blink .5s infinite; }
  .status-bar { font-family: 'Share Tech Mono', monospace; font-size: 0.8rem;
                padding: 6px 16px; background: #0d1118;
                border-bottom: 1px solid var(--border);
                display: flex; gap: 18px; flex-wrap: wrap; align-items: center; }
  .status-bar .chip { display: flex; align-items: center; gap: 5px; }
  .status-bar .chip .label { color: var(--muted); }
  .status-bar .chip .val   { color: var(--accent); }
  .status-bar .chip.alert .val { color: var(--danger); animation: blink 0.6s infinite; }
  .status-bar .chip.ok    .val { color: var(--ok); }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
  .feed-wrap { position: relative; width: 100%; background: #000;
               border-bottom: 1px solid var(--border); }
  .feed-wrap img { width: 100%; display: block; max-height: 55vw; object-fit: contain; }
  .alert-banner { display: none; position: absolute; top: 0; left: 0; right: 0;
                  padding: 5px; text-align: center; background: rgba(255,51,85,.85);
                  font-weight: 800; letter-spacing: 3px; font-size: .9rem;
                  animation: blink 0.5s infinite; }
  .alert-banner.show { display: block; }
  .section { padding: 14px 14px 0; }
  .section-title { font-size: 0.6rem; letter-spacing: 3px; color: var(--muted);
                   text-transform: uppercase; margin-bottom: 10px;
                   border-left: 2px solid var(--accent); padding-left: 8px; }
  /* distances */
  .dist-item { display: flex; align-items: center; gap: 10px; padding: 8px 10px;
               margin-bottom: 6px; background: var(--panel);
               border: 1px solid var(--border); border-radius: var(--radius); }
  .dist-name { flex: 1; font-size: .85rem; }
  .dist-cm   { font-family: 'Share Tech Mono', monospace; font-size: 1.1rem;
               font-weight: 800; min-width: 64px; text-align: right; }
  .dist-bar  { width: 60px; height: 6px; background: #1e2a38;
               border-radius: 3px; overflow: hidden; }
  .dist-fill { height: 100%; border-radius: 3px; }
  /* buttons */
  .btn-grid { display: grid; gap: 8px; }
  .btn-grid.two { grid-template-columns: 1fr 1fr; }
  .btn-grid.one { grid-template-columns: 1fr; }
  button { font-family: 'Exo 2', sans-serif; font-weight: 600; font-size: 0.82rem;
           padding: 12px 8px; border: 1px solid var(--border);
           background: var(--panel); color: var(--text);
           border-radius: var(--radius); cursor: pointer; transition: all .15s; }
  button:active { transform: scale(.96); }
  button.primary    { border-color: var(--accent); color: var(--accent); }
  button.primary:active { background: rgba(0,229,255,.12); }
  button.danger-btn { border-color: var(--danger); color: var(--danger); }
  button.ok-btn     { border-color: var(--ok);     color: var(--ok); }
  button.warn-btn   { border-color: var(--warn);   color: var(--warn); }
  button .icon { font-size: 1rem; display: block; margin-bottom: 2px; }
  button .sub  { font-size: 0.62rem; color: var(--muted);
                 font-family: 'Share Tech Mono', monospace; }
  .danger-row { display: flex; align-items: center; gap: 10px;
                background: var(--panel); border: 1px solid var(--border);
                border-radius: var(--radius); padding: 10px 14px; }
  .danger-row .val { flex: 1; text-align: center;
                     font-family: 'Share Tech Mono', monospace;
                     font-size: 1.4rem; color: var(--accent); }
  .danger-row .val span { font-size: .7rem; color: var(--muted); }
  .danger-row button { flex: 0 0 44px; padding: 8px; font-size: 1.1rem; }
  .log-box { background: var(--panel); border: 1px solid var(--border);
             border-radius: var(--radius); padding: 10px;
             font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
             line-height: 1.6; color: #7090a0; max-height: 90px;
             overflow-y: auto; min-height: 40px; }
  .spacer { height: 20px; }
  .feed-wrap::after { content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg,transparent,transparent 3px,
      rgba(0,0,0,.08) 3px,rgba(0,0,0,.08) 4px); pointer-events: none; }
  .no-det { color: var(--muted); font-family: 'Share Tech Mono', monospace;
            font-size: .78rem; padding: 10px 0; }
  .warn-note { color: var(--warn); font-family: 'Share Tech Mono', monospace;
               font-size: .68rem; padding: 4px 0; }
</style>
</head>
<body>
<header>
  <div class="dot" id="dot"></div>
  <div>
    <div class="logo">Robot Vision</div>
    <div class="sub">AI Depth · YOLO-World · Stereo</div>
  </div>
  <span style="margin-left:auto;font-family:'Share Tech Mono',monospace;
               font-size:.7rem;color:var(--muted)" id="scale-hdr">NOT CALIBRATED</span>
</header>

<div class="status-bar">
  <div class="chip" id="chip-status">
    <span class="label">STATUS</span>
    <span class="val"  id="val-status">—</span>
  </div>
  <div class="chip">
    <span class="label">SCALE</span>
    <span class="val"  id="val-scale">—</span>
  </div>
  <div class="chip">
    <span class="label">CAM</span>
    <span class="val"  id="val-cam">—</span>
  </div>
  <div class="chip">
    <span class="label">OBJECTS</span>
    <span class="val"  id="val-count">0</span>
  </div>
</div>

<div class="feed-wrap">
  <div class="alert-banner" id="alert-banner">⚠ OBSTACLE DETECTED</div>
  <img src="/stream" alt="Live feed">
</div>

<!-- LIVE DISTANCES -->
<div class="section" style="margin-top:14px">
  <div class="section-title">Live Distances — closest first</div>
  <div id="dist-panel">
    <div class="no-det">Waiting for AI models to load…</div>
  </div>
</div>

<!-- CALIBRATION -->
<div class="section" style="margin-top:14px">
  <div class="section-title">Calibration</div>
  <div class="btn-grid one">
    <button class="primary" onclick="api('/calibrate','POST')">
      <span class="icon">🎯</span>
      Calibrate — stand object at 50 cm then tap
    </button>
  </div>
  <div class="btn-grid two" style="margin-top:8px">
    <button class="warn-btn" onclick="api('/scale','POST',{mult:0.7})">
      <span class="icon">◀◀</span><span class="sub">Scale ×0.7</span>
    </button>
    <button class="warn-btn" onclick="api('/scale','POST',{mult:1.4})">
      <span class="icon">▶▶</span><span class="sub">Scale ×1.4</span>
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
    <button onclick="api('/camera','POST',{cam:0})">
      <span class="icon">📷</span>Left Camera
    </button>
    <button onclick="api('/camera','POST',{cam:1})">
      <span class="icon">📷</span>Right Camera
    </button>
  </div>
  <div class="btn-grid one" style="margin-top:8px">
    <button onclick="api('/depth_toggle','POST')">
      <span class="icon">🌈</span>Toggle Depth Overlay
    </button>
  </div>
</div>

<!-- DETECTION LOG -->
<div class="section" style="margin-top:14px">
  <div class="section-title">Detection Log</div>
  <div class="log-box" id="log-box">waiting for detections…</div>
</div>
<div class="spacer"></div>

<script>
  async function api(path, method, body) {
    try {
      const r = await fetch(path, {
        method,
        headers: body ? {'Content-Type':'application/json'} : {},
        body: body ? JSON.stringify(body) : undefined
      });
      const j = await r.json();
      if (j.message) flashToast(j.message);
    } catch(e) { console.error(e); }
  }

  function flashToast(msg) {
    let t = document.getElementById('toast');
    if (!t) {
      t = document.createElement('div'); t.id = 'toast';
      Object.assign(t.style, {
        position:'fixed', bottom:'20px', left:'50%', transform:'translateX(-50%)',
        background:'rgba(0,229,255,.15)', border:'1px solid #00e5ff',
        color:'#00e5ff', padding:'8px 20px', borderRadius:'20px',
        fontFamily:"'Share Tech Mono',monospace", fontSize:'.8rem',
        zIndex:999, transition:'opacity .4s'
      });
      document.body.appendChild(t);
    }
    t.textContent = msg; t.style.opacity = '1';
    clearTimeout(t._tid);
    t._tid = setTimeout(() => t.style.opacity = '0', 2500);
  }

  setInterval(async () => {
    try {
      const s = await (await fetch('/status')).json();

      // header dot
      const isAlert = s.alert;
      document.getElementById('dot').className = 'dot ' + (isAlert ? 'danger' : 'alive');
      document.getElementById('scale-hdr').textContent = s.scale_str;

      // status chips
      const chip = document.getElementById('chip-status');
      document.getElementById('val-status').textContent = isAlert ? '!! OBSTACLE !!' : 'CLEAR';
      chip.className = 'chip ' + (isAlert ? 'alert' : 'ok');
      document.getElementById('val-scale').textContent = s.scale_str;
      document.getElementById('val-cam').textContent   = s.main_cam === 0 ? 'LEFT' : 'RIGHT';
      document.getElementById('val-count').textContent = (s.distances || []).length;

      // alert banner
      document.getElementById('alert-banner').classList.toggle('show', isAlert);

      // danger threshold display
      document.getElementById('val-danger').innerHTML = s.danger_cm + '<span> cm</span>';

      // ── distances panel ────────────────────────
      const dp  = document.getElementById('dist-panel');
      const cal = s.scale_str !== 'NOT CALIBRATED';
      if (s.distances && s.distances.length) {
        dp.innerHTML = s.distances.map(o => {
          const col = o.danger
            ? '#ff3355'
            : (o.dist_cm < s.danger_cm * 1.5 ? '#ff8c00' : '#00e676');
          const bar = Math.max(4, Math.min(100, Math.round(100 - o.dist_cm / 3)));
          return `<div class="dist-item" style="border-color:${col}44">
            <div class="dist-name">${o.name}</div>
            <div class="dist-cm" style="color:${col}">
              ${cal ? o.dist_cm + ' cm' : '? cm'}
            </div>
            <div class="dist-bar">
              <div class="dist-fill" style="width:${bar}%;background:${col}"></div>
            </div>
          </div>`;
        }).join('');
        if (!cal) {
          dp.innerHTML += '<div class="warn-note">⚠ Tap Calibrate above for real distances</div>';
        }
      } else {
        dp.innerHTML = '<div class="no-det">No objects detected</div>';
      }

      // log
      const lb = document.getElementById('log-box');
      if (s.log && s.log.length) {
        lb.innerHTML = s.log.slice().reverse().map(l =>
          `<div style="color:${
            l.includes('OBSTACLE') || l.includes('!!') ? '#ff3355' : '#7090a0'
          }">${l}</div>`
        ).join('');
      }
    } catch(e) {}
  }, 1000);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); local_ip = s.getsockname()[0]; s.close()
    except: local_ip = "localhost"

    print("━" * 55)
    print("  Unified AI Server (MiDaS + YOLO-World)")
    print(f"  Dashboard  → http://{local_ip}:{SERVER_PORT}")
    print(f"  Stream     → http://{local_ip}:{SERVER_PORT}/stream")
    print(f"  Detections → http://{local_ip}:{SERVER_PORT}/detections")
    print(f"  Left cam   → {LEFT_URL}")
    print(f"  Right cam  → {RIGHT_URL}")
    print("━" * 55)
    print(f"\n  On RPi set PC_SERVER_URL = 'http://{local_ip}:{SERVER_PORT}'")
    print()

    threading.Thread(target=_read_cam, args=(LEFT_URL,  0), daemon=True).start()
    threading.Thread(target=_read_cam, args=(RIGHT_URL, 1), daemon=True).start()
    threading.Thread(target=_ai_loop, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
