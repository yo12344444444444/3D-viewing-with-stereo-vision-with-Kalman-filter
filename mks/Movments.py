import serial, time

mks = serial.Serial("/dev/ttyUSB0", 115200, timeout=3)
time.sleep(2)
mks.flushInput()


class Movments:
    def __init__(self, speed=3000, step=50, turn=30):
        self.speed = speed
        self.step  = step
        self.turn  = turn

        # ── setup (must be called on instance, not class body) ──
        self.gcode("G21")   # millimetres
        self.gcode("G91")   # relative mode
        self.gcode("M17")   # enable motors

    # ════════════════════════════════════════════════
    #  X = Front-Left    Y = Front-Right
    #  Z = Rear-Left    E0 = Rear-Right
    #
    #  Left side  = X + Z
    #  Right side = Y + E0
    # ════════════════════════════════════════════════

    def gcode(self, cmd):
        mks.write((cmd + "\n").encode())
        mks.readline()

    def forward(self):
        self.gcode(f"G1 X{self.step} Y{self.step} Z{self.step} E{self.step} F{self.speed}")

    def backward(self):
        # ❌ was: -X{} -Y{} -F{}  →  wrong G-code syntax
        self.gcode(f"G1 X-{self.step} Y-{self.step} Z-{self.step} E-{self.step} F{self.speed}")

    def turn_left(self):
        # left wheels back, right wheels forward
        # ❌ was: F{self.turn}  →  feedrate should be speed not turn distance
        self.gcode(f"G1 X-{self.turn} Y{self.turn} Z-{self.turn} E{self.turn} F{self.speed}")

    def turn_right(self):
        # left wheels forward, right wheels back
        # ❌ was: F{self.turn}  →  feedrate should be speed not turn distance
        self.gcode(f"G1 X{self.turn} Y-{self.turn} Z{self.turn} E-{self.turn} F{self.speed}")

    def stop(self):
        self.gcode("M410")

    def motors_off(self):
        self.gcode("M18")

    def motors_on(self):
        self.gcode("M17")
