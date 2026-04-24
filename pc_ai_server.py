"""
pc_ai_server.py  —  AI Detection Server  (runs on Windows PC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs YOLO-World + MiDaS on your PC (GPU if available).
Reads ESP32 camera stream from the RobotCar WiFi AP.
RPi navigator polls this server to get detected objects + depth.

Install (Windows, once):
    pip install ultralytics fastapi uvicorn torch torchvision timm opencv-python numpy requests

Run:
    python pc_ai_server.py

Then on RPi set PC_SERVER_URL = "http://<YOUR-PC-IP>:5000"
Your PC IP on RobotCar AP: run  ipconfig  and look for 192.168.4.x
"""

import threading, time, urllib.request, io, subprocess, re
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

# ══════════════════════════════════════════════════
#  CONFIG — only change these two MAC addresses
# ══════════════════════════════════════════════════
# ESP32 camera MAC address
# Find it in Arduino serial monitor on boot:
#   "WiFi connected — MAC: AA:BB:CC:DD:EE:FF"
CAM_MAC_ADDRESS = "aa:bb:cc:dd:ee:ff"
CAM_STREAM_PATH = "/stream"             # ESP32 stream endpoint

# What objects to search for (YOLO-World — add anything you want!)
CLASSES   = [
    "cup", "mug", "bottle", "can",
    "chair", "table", "sofa", "desk",
    "person", "laptop", "phone", "box",
    "remote", "keyboard", "book", "bag"
]

SERVER_PORT    = 5000
AP_SUBNET      = "192.168.4"
DEVICE         = "cuda"    # "cuda" or "cpu"
PROC_W, PROC_H = 320, 240
YOLO_CONF      = 0.30
ARRIVE_DEPTH   = 35
ARP_RETRIES    = 5
ARP_RETRY_WAIT = 3
# ══════════════════════════════════════════════════


# ─────────────────────────────────────────────────
#  MAC → IP resolver  (Windows ARP)
# ─────────────────────────────────────────────────
def _normalise_mac(mac: str) -> str:
    digits = re.sub(r"[^0-9a-fA-F]", "", mac)
    if len(digits) != 12:
        raise ValueError(f"Invalid MAC address: {mac!r}")
    return ":".join(digits[i:i+2].lower() for i in range(0, 12, 2))


def _ping_subnet_windows(subnet: str):
    """Ping all hosts in range to populate Windows ARP cache."""
    procs = []
    for i in range(2, 51):
        p = subprocess.Popen(
            f"ping -n 1 -w 300 {subnet}.{i}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        procs.append(p)
    # wait for all pings to finish
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()


def _arp_lookup_windows(mac_normalised: str) -> str | None:
    """
    Read Windows ARP table (arp -a).
    Windows shows MACs as  aa-bb-cc-dd-ee-ff  with dashes.
    """
    # also accept dash-separated version
    mac_dashes = mac_normalised.replace(":", "-")
    try:
        out = subprocess.check_output("arp -a", shell=True, text=True,
                                      stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            line_low = line.lower()
            if mac_normalised in line_low or mac_dashes in line_low:
                # Windows arp -a format:
                #   192.168.4.3     aa-bb-cc-dd-ee-ff    dynamic
                parts = line.split()
                for part in parts:
                    if re.match(r"\d+\.\d+\.\d+\.\d+", part):
                        return part
    except Exception:
        pass
    return None


def find_ip_by_mac(mac: str,
                   subnet: str  = AP_SUBNET,
                   retries: int = ARP_RETRIES,
                   wait: float  = ARP_RETRY_WAIT) -> str | None:
    try:
        mac_norm = _normalise_mac(mac)
    except ValueError as e:
        print(f"[ARP] {e}")
        return None

    print(f"[ARP] Looking for MAC {mac_norm} on {subnet}.0/24 ...")

    for attempt in range(1, retries + 1):
        print(f"[ARP] Attempt {attempt}/{retries} — scanning subnet...")
        _ping_subnet_windows(subnet)
        ip = _arp_lookup_windows(mac_norm)
        if ip:
            print(f"[ARP] ✓ Found ESP32 at {ip}")
            return ip
        if attempt < retries:
            print(f"[ARP] Not found — waiting {wait}s "
                  f"(is ESP32 connected to RobotCar WiFi?)")
            time.sleep(wait)

    print(f"[ARP] ✗ Could not find {mac_norm} after {retries} attempts")
    return None


# ── resolve ESP32 IP at startup ────────────────────
_cam_ip = find_ip_by_mac(CAM_MAC_ADDRESS)
if _cam_ip:
    CAM_URL = f"http://{_cam_ip}{CAM_STREAM_PATH}"
else:
    CAM_URL = f"http://{AP_SUBNET}.X{CAM_STREAM_PATH}"
    print("[ARP] FALLBACK: set CAM_URL manually")

print(f"[ARP] Camera URL → {CAM_URL}")


# ─────────────────────────────────────────────────
#  Shared state
# ─────────────────────────────────────────────────
class State:
    def __init__(self):
        self._lock       = threading.Lock()
        self.frame       = None          # latest BGR frame
        self.detections  = []            # latest object list
        self.annotated   = None          # frame with boxes drawn
        self._jpeg       = None

    def push(self, frame, detections, annotated):
        ok, jpg = cv2.imencode('.jpg', annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._lock:
            self.frame      = frame
            self.detections = detections
            self.annotated  = annotated
            self._jpeg      = jpg.tobytes() if ok else self._jpeg

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_detections(self):
        with self._lock:
            return list(self.detections)


state = State()


# ─────────────────────────────────────────────────
#  MJPEG camera reader
# ─────────────────────────────────────────────────
_raw_frame      = None
_raw_frame_lock = threading.Lock()


def _read_camera():
    global _raw_frame
    while True:
        try:
            stream = urllib.request.urlopen(CAM_URL, timeout=6)
            buf = b""
            while True:
                buf += stream.read(4096)
                a = buf.find(b'\xff\xd8')
                b = buf.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg  = buf[a:b + 2]
                    buf  = buf[b + 2:]
                    img  = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                        cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, (PROC_W, PROC_H))
                        with _raw_frame_lock:
                            _raw_frame = img
        except Exception as e:
            print(f"[CAM] {e} — retrying...")
            time.sleep(1)


# ─────────────────────────────────────────────────
#  YOLO-World loader
# ─────────────────────────────────────────────────
def _load_yolo():
    from ultralytics import YOLO
    print("[YOLO-World] loading yolov8s-worldv2.pt ...")
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(CLASSES)
    print(f"[YOLO-World] ready — tracking: {CLASSES}")
    return model


# ─────────────────────────────────────────────────
#  MiDaS depth estimator
# ─────────────────────────────────────────────────
def _load_midas():
    import torch
    dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[MiDaS] loading on {dev} ...")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    model.eval().to(dev)
    tf = torch.hub.load("intel-isl/MiDaS", "transforms",
                        trust_repo=True).small_transform
    print("[MiDaS] ready.")
    return model, tf, dev


def _midas_depth(model, tf, dev, bgr):
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = tf(rgb).to(dev)
    with torch.no_grad():
        pred = model(inp)
    d = pred.squeeze().cpu().numpy()
    return cv2.resize(d, (bgr.shape[1], bgr.shape[0]))


# ─────────────────────────────────────────────────
#  Detection + depth loop
# ─────────────────────────────────────────────────
def _detection_loop():
    yolo                = _load_yolo()
    midas, midas_tf, dev = _load_midas()
    scale               = None   # midas → cm scale (auto-estimated)
    scale_buf           = []

    while True:
        with _raw_frame_lock:
            frame = _raw_frame
        if frame is None:
            time.sleep(0.05)
            continue

        frame = frame.copy()

        # ── depth ─────────────────────────────────
        raw_depth = _midas_depth(midas, midas_tf, dev, frame)

        # ── auto-scale depth to cm (rough estimate) ─
        # assume median depth in frame ≈ 150 cm (1.5 m) as a baseline
        med = float(np.median(raw_depth[raw_depth > 1e-3]))
        if med > 1e-3:
            scale_buf.append(150.0 * med)
            scale_buf = scale_buf[-60:]
            scale = float(np.median(scale_buf))

        depth_cm = None
        if scale:
            safe     = np.where(raw_depth > 1e-3, raw_depth, 1e-3)
            depth_cm = scale / safe

        # ── YOLO-World detect ──────────────────────
        results    = yolo(frame, conf=YOLO_CONF, verbose=False)
        detections = []
        annotated  = frame.copy()

        obj_id = 1
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id   = int(box.cls[0])
            label    = results[0].names[cls_id]
            conf     = float(box.conf[0])
            cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2

            # depth of this object
            obj_depth = None
            if depth_cm is not None:
                roi   = depth_cm[y1:y2, x1:x2]
                valid = roi[(roi > 1) & (roi < 2000)]
                if valid.size > 5:
                    obj_depth = float(np.median(valid))

            detections.append({
                "id":       obj_id,
                "label":    label,
                "conf":     round(conf, 2),
                "cx":       cx,
                "cy":       cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "depth_cm": round(obj_depth, 1) if obj_depth else None,
                "arrived":  (obj_depth is not None and obj_depth < ARRIVE_DEPTH),
            })

            # draw box + number
            col = (0, 220, 60)
            if obj_depth and obj_depth < ARRIVE_DEPTH:
                col = (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            tag = f"#{obj_id} {label}"
            if obj_depth:
                tag += f" {obj_depth:.0f}cm"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0,0,0), -1)
            cv2.putText(annotated, tag, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # center dot
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
            obj_id += 1

        # center crosshair
        cv2.line(annotated, (PROC_W//2 - 10, PROC_H//2),
                 (PROC_W//2 + 10, PROC_H//2), (255,255,255), 1)
        cv2.line(annotated, (PROC_W//2, PROC_H//2 - 10),
                 (PROC_W//2, PROC_H//2 + 10), (255,255,255), 1)

        state.push(frame, detections, annotated)


# ─────────────────────────────────────────────────
#  FastAPI server
# ─────────────────────────────────────────────────
app = FastAPI(title="AI Detection Server")


def _mjpeg_gen():
    while True:
        jpg = state.get_jpeg()
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg + b"\r\n")
        time.sleep(0.04)


@app.get("/stream")
def video_stream():
    return StreamingResponse(_mjpeg_gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detections")
def get_detections():
    return JSONResponse({"objects": state.get_detections()})


@app.post("/classes")
def set_classes(body: dict):
    """Update what YOLO-World looks for at runtime."""
    new_classes = body.get("classes", [])
    if new_classes:
        CLASSES.clear()
        CLASSES.extend(new_classes)
    return {"classes": CLASSES}


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.get("/", response_class=HTMLResponse)
def index():
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>AI Server</title>
<style>body{{background:#0a0d10;color:#c8d8e8;font-family:monospace;padding:20px}}
h2{{color:#00e5ff}} img{{width:640px;border:1px solid #1e2a38}}</style>
</head><body>
<h2>AI Detection Server</h2>
<p>Classes: {', '.join(CLASSES)}</p>
<img src="/stream"><br><br>
<a href="/detections" style="color:#00e5ff">/detections JSON</a>
</body></html>"""


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print("━" * 55)
    print("  AI Detection Server (YOLO-World + MiDaS)")
    print(f"  URL      → http://{local_ip}:{SERVER_PORT}")
    print(f"  Stream   → http://{local_ip}:{SERVER_PORT}/stream")
    print(f"  Objects  → http://{local_ip}:{SERVER_PORT}/detections")
    print(f"  Camera   → {CAM_URL}")
    print("━" * 55)
    print(f"\n  Set on RPi:  PC_SERVER_URL = 'http://{local_ip}:{SERVER_PORT}'")
    print()

    threading.Thread(target=_read_camera,    daemon=True).start()
    threading.Thread(target=_detection_loop, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
