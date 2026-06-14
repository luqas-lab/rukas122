# ============================================================
#  AdaptGrip — PS4 + Reinforcement Learning Control
#  Author  : Muhammad Luqmanul Hakeem bin Ramli
#  Project : AdaptGrip — RL-Based Adaptive Force Control
#  Run on  : Linux PC (Ubuntu), Python 3
# ============================================================
#
#  WHAT THIS CODE DOES:
#  This is the main control program that runs on the PC.
#  It connects three things together:
#    1. PS4 controller  — user controls the arm manually
#    2. Arduino Mega    — receives commands and moves the arm
#    3. RL (AI) model   — automatically controls grip force
#
#  The program reads PS4 button/stick inputs, translates them
#  into joint angle commands, and sends them to Arduino via USB.
#  When the user presses Cross (X), the AI takes over the gripper.

# ── Imports ─────────────────────────────────────────────────
import serial                        # USB communication with Arduino
import time                          # For delays and timing
import pygame                        # Reads PS4 controller input
import numpy as np                   # Math and array operations
from stable_baselines3 import PPO    # Loads the trained RL model

# ── Connection Settings ──────────────────────────────────────
PORT       = '/dev/ttyUSB0'
BAUD       = 9600
MODEL_PATH = '/home/luq/fyp2/ppo_force_control.zip'

# ── Demo Mode Setting ────────────────────────────────────────
# Set True to use fixed angle (no FSR/RL). Set False for full AI grip.
DEMO_MODE          = False
DEMO_GRIPPER_ANGLE = 100  # 175=open, 0=closed — 100 gives a moderate grip

# ── FSR Calibration Values ───────────────────────────────────
# Loaded from fsr_calibration.txt at startup.
# Fallback hardcoded values from your physical calibration session:
#   baseline_l=0, baseline_r=166, scale=0.001047 N/unit
FSR_BASELINE_L    =   0
FSR_BASELINE_R    = 166
FSR_SCALE         = 0.001047   # Newtons per raw ADC unit
FSR_CALIB_PATH    = '/home/luq/fyp2/fsr_calibration.txt'

def load_fsr_calibration():
    """Load FSR calibration from file, fall back to hardcoded values."""
    import json, os
    global FSR_BASELINE_L, FSR_BASELINE_R, FSR_SCALE
    if os.path.exists(FSR_CALIB_PATH):
        try:
            with open(FSR_CALIB_PATH) as f:
                c = json.load(f)
            FSR_BASELINE_L = c.get('baseline_l', FSR_BASELINE_L)
            FSR_BASELINE_R = c.get('baseline_r', FSR_BASELINE_R)
            FSR_SCALE      = c.get('scale_N_per_unit', FSR_SCALE)
            print(f"FSR calibration loaded: "
                  f"baseline L={FSR_BASELINE_L} R={FSR_BASELINE_R} "
                  f"scale={FSR_SCALE:.6f} N/unit")
        except Exception as e:
            print(f"FSR calibration file error: {e} — using hardcoded values")
    else:
        print(f"FSR calibration file not found — using hardcoded values")
    print(f"  Detection threshold: raw > 50 counts as contact")

def raw_fsr_to_newton(fsr_l, fsr_r):
    """Convert raw FSR ADC readings to estimated grip force in Newtons."""
    baseline_avg = (FSR_BASELINE_L + FSR_BASELINE_R) / 2.0
    avg_raw      = (fsr_l + fsr_r) / 2.0
    net          = max(avg_raw - baseline_avg, 0.0)   # never negative
    return net * FSR_SCALE

# ── Joint Limits (Calibrated) ────────────────────────────────
# Each joint has three values:
#   rest = home position angle (where arm goes when HOME is pressed)
#   min  = minimum safe angle (never go below this)
#   max  = maximum safe angle (never go above this)
# These values were found by physical calibration of the real arm.
LIMITS = {
    'SHOULDER_R': {'rest':   0, 'min':  0, 'max': 180},  # CH0 — right shoulder
    'SHOULDER_L': {'rest': 180, 'min': 50, 'max': 190},  # CH1 — left shoulder (inverted, max>180 allows downward movement)
    'ELBOW':      {'rest':   0, 'min':  0, 'max': 110},  # CH2 — elbow joint
    'WRIST_P':    {'rest':  96, 'min':  0, 'max': 170},  # CH3 — wrist pitch (up/down)
    'WRIST_R':    {'rest':   0, 'min':  0, 'max': 175},  # CH4 — wrist roll (rotate)
    'GRIPPER':    {'rest': 175, 'min':  0, 'max': 175},  # CH5 — gripper (175=open, 0=closed)
}

# ── Stepper Motor Settings ───────────────────────────────────
STEPPER_STEPS = 150    # Steps per command — increased for faster base rotation
STEPPER_MAX   =  550   # Maximum steps clockwise from home
STEPPER_MIN   = -300   # Maximum steps counter-clockwise from home

# Tracks the stepper's current position in steps from home
stepper_pos = 0

# ── Object Profiles for AI Grip ──────────────────────────────
# These define how the RL model should grip different objects.
# mass     = object weight in kg (affects required grip force)
# damage   = maximum force in Newtons before object is damaged
# fragility = 0.0 (robust) to 1.0 (very fragile), used by RL model
OBJECTS = {
    1: {"name": "Very Fragile", "mass": 0.05, "damage":  5.0, "fragility": 0.05},
    2: {"name": "Fragile",      "mass": 0.1,  "damage": 10.0, "fragility": 0.10},
    3: {"name": "Medium",       "mass": 0.2,  "damage": 20.0, "fragility": 0.20},
    4: {"name": "Robust",       "mass": 0.4,  "damage": 40.0, "fragility": 0.40},
    5: {"name": "Very Robust",  "mass": 0.8,  "damage": 80.0, "fragility": 0.80},
}

# Stores the current angle of each joint (starts at rest position)
angles = {j: LIMITS[j]['rest'] for j in LIMITS}

# ── send() ───────────────────────────────────────────────────
# Sends a single joint movement command to Arduino via serial.
# Clips the angle to safe range before sending.
# Example: send(ser, 'ELBOW', 45) → sends "ELBOW:45\n" to Arduino
def send(ser, joint, angle):
    angle = int(np.clip(angle, LIMITS[joint]['min'], LIMITS[joint]['max']))
    angles[joint] = angle                          # Update local angle tracker
    ser.write(f"{joint}:{angle}\n".encode())       # Send command as bytes

# ── send_stepper() ───────────────────────────────────────────
# Rotates the base stepper motor one step in the given direction.
# Enforces software limits (STEPPER_MIN to STEPPER_MAX).
# cw = True for clockwise, False for counter-clockwise
def send_stepper(ser, cw):
    global stepper_pos
    new_pos = stepper_pos + (STEPPER_STEPS if cw else -STEPPER_STEPS)
    new_pos = int(np.clip(new_pos, STEPPER_MIN, STEPPER_MAX))  # Enforce limits
    steps   = abs(new_pos - stepper_pos)
    if steps > 0:
        direction = "CW" if cw else "CCW"
        ser.write(f"STEPPER:{direction}:{steps}\n".encode())
        stepper_pos = new_pos

# ── go_home() ────────────────────────────────────────────────
# Moves all joints back to their calibrated home (rest) positions.
# Also returns stepper to position 0.
# Called when Triangle button is pressed on PS4 controller.
def go_home(ser):
    global stepper_pos
    print("Going home (slow)...")
    # Move each joint slowly by stepping in small increments toward the rest angle
    SLOW_STEP   = 2    # degrees per increment
    SLOW_DELAY  = 0.03 # seconds between each increment
    for joint in LIMITS:
        target = LIMITS[joint]['rest']
        current = angles[joint]
        # Walk toward target in small steps
        while current != target:
            if current < target:
                current = min(current + SLOW_STEP, target)
            else:
                current = max(current - SLOW_STEP, target)
            send(ser, joint, current)
            time.sleep(SLOW_DELAY)
    ser.write(b"STEPPERHOME\n")
    stepper_pos = 0
    print("Home done!")

# ── force_to_angle() ─────────────────────────────────────────
# Converts a grip force value (in Newtons) to a servo angle.
# Higher force = smaller angle (gripper closes more).
# Lower force  = larger angle  (gripper opens more).
# open_angle = the angle at which the gripper just makes contact
#              with the object (0 force). Squeezing further from
#              there increases force, down to fully closed (0°).
def force_to_angle(force_n, max_force=10.0, open_angle=175):
    min_a = LIMITS['GRIPPER']['min']   # 0°  = fully closed
    max_a = open_angle                 # angle of first contact = 0 force
    angle = max_a - ((force_n - 0.1) / (max_force - 0.1)) * (max_a - min_a)
    return int(np.clip(angle, min_a, max_a))

# ── read_fsr() ───────────────────────────────────────────────
# Asks Arduino to read both FSR force sensors and returns the values.
# Sends "FSR" command and waits for "FSR:left:right" response.
# Returns (left_raw, right_raw) as integers (0-1023).
def read_fsr(ser):
    ser.write(b"FSR\n")
    time.sleep(0.1)
    fsr_l, fsr_r = 0, 0
    deadline = time.time() + 0.3          # Wait max 0.3 seconds for response
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            if line.startswith("FSR:"):
                parts = line.split(":")
                if len(parts) == 3:
                    try:
                        fsr_l = int(parts[1])
                        fsr_r = int(parts[2])
                    except:
                        pass
                    break
    return fsr_l, fsr_r

# ── calibrate_fsr() ─────────────────────────────────────────
# Interactive FSR calibration routine.
# Step 1: reads baseline (gripper open, no contact) → FSR_ZERO
# Step 2: user grips each object and presses Cross to record the value
# Step 3: prints a calibration table and saves to fsr_calibration.txt
#
# The raw-to-Newton formula after calibration:
#   force_N = (raw - FSR_ZERO) / (FSR_MAX - FSR_ZERO) * MAX_FORCE_N
# where MAX_FORCE_N is set by the heaviest object gripped during calibration.
def calibrate_fsr(ser, ctrl):
    import json, os

    CALIB_GRIP_STEP = 5     # degrees per L2/R2 press inside calibration
    last_grip_time  = 0

    def handle_gripper():
        """Allow L2/R2 to open/close gripper during calibration waits."""
        nonlocal last_grip_time
        now = time.time()
        if now - last_grip_time < 0.04:
            return
        l2 = ctrl.get_axis(2)
        r2 = ctrl.get_axis(5)
        if l2 > 0.1:
            scaled = int(CALIB_GRIP_STEP * ((l2 + 1.0) / 2.0) * 3) + 1
            new_angle = int(np.clip(angles['GRIPPER'] + scaled,
                                    LIMITS['GRIPPER']['min'], LIMITS['GRIPPER']['max']))
            angles['GRIPPER'] = new_angle
            ser.write(f"GRIPPER:{new_angle}\n".encode())
            last_grip_time = now
        elif r2 > 0.1:
            scaled = int(CALIB_GRIP_STEP * ((r2 + 1.0) / 2.0) * 3) + 1
            new_angle = int(np.clip(angles['GRIPPER'] - scaled,
                                    LIMITS['GRIPPER']['min'], LIMITS['GRIPPER']['max']))
            angles['GRIPPER'] = new_angle
            ser.write(f"GRIPPER:{new_angle}\n".encode())
            last_grip_time = now

    print("\n" + "=" * 50)
    print("  FSR CALIBRATION MODE")
    print("=" * 50)
    print("  L2 = close gripper   R2 = open gripper")
    print("  Cross = record reading")
    print("  Circle = skip object")
    print("=" * 50)
    print("\nStep 1: Open gripper fully (R2), then press Cross...")

    # Wait for Cross — gripper controllable while waiting
    while True:
        pygame.event.pump()
        handle_gripper()
        if ctrl.get_button(0):
            break
        time.sleep(0.04)
    time.sleep(0.15)  # debounce

    fsr_l, fsr_r = read_fsr(ser)
    baseline_l, baseline_r = fsr_l, fsr_r
    print(f"  Baseline recorded → Left:{baseline_l}  Right:{baseline_r}")
    time.sleep(0.5)

    calib = {"baseline_l": baseline_l, "baseline_r": baseline_r, "objects": []}

    objects_to_test = [
        {"name": "Very Fragile", "expected_N": 0.5},
        {"name": "Fragile",      "expected_N": 1.0},
        {"name": "Medium",       "expected_N": 2.0},
        {"name": "Robust",       "expected_N": 5.0},
        {"name": "Very Robust",  "expected_N": 10.0},
    ]

    print("\nStep 2: For each object — place it, use L2 to grip, press Cross to record.")
    print("        Press Circle to skip.\n")

    for obj in objects_to_test:
        # Open gripper before each object
        angles['GRIPPER'] = 0
        ser.write(b"GRIPPER:0\n")
        time.sleep(0.5)

        print(f"  → Place '{obj['name']}' in gripper, use L2 to close, then Cross...")
        last_fsr_print = 0
        while True:
            pygame.event.pump()
            handle_gripper()

            # Print live FSR every 0.4s so user can see sensor responding
            now = time.time()
            if now - last_fsr_print > 0.4:
                fsr_l, fsr_r = read_fsr(ser)
                print(f"    FSR live → Left:{fsr_l:4d}  Right:{fsr_r:4d}  "
                      f"Gripper:{angles['GRIPPER']}°", end='\r')
                last_fsr_print = now

            if ctrl.get_button(0):   # Cross = record
                time.sleep(0.1)
                fsr_l, fsr_r = read_fsr(ser)
                avg = (fsr_l + fsr_r) // 2
                calib["objects"].append({
                    "name":       obj["name"],
                    "expected_N": obj["expected_N"],
                    "raw_l":      fsr_l,
                    "raw_r":      fsr_r,
                    "raw_avg":    avg,
                })
                print(f"    Recorded → Left:{fsr_l}  Right:{fsr_r}  Avg:{avg}")
                time.sleep(0.4)
                break
            if ctrl.get_button(1):   # Circle = skip
                print(f"    Skipped.")
                time.sleep(0.4)
                break
            time.sleep(0.04)

    # Open gripper after last object
    angles['GRIPPER'] = 0
    ser.write(b"GRIPPER:0\n")
    time.sleep(0.3)

    # Compute linear scale factor using ALL recorded points (least-squares
    # fit through the origin) instead of just the single highest reading.
    # expected_N for each test object matches that object's required grip
    # force (mass * 9.81 * 1.2) in OBJECTS — so this scale is calibrated
    # across the same force range the RL grip loop operates in.
    if calib["objects"]:
        sum_xy = sum((o["raw_avg"] - baseline_l) * o["expected_N"] for o in calib["objects"])
        sum_xx = sum((o["raw_avg"] - baseline_l) ** 2          for o in calib["objects"])
        scale_N_per_unit = sum_xy / max(sum_xx, 1)

        print("\n" + "=" * 50)
        print("  CALIBRATION RESULTS")
        print("=" * 50)
        print(f"  Baseline (open): L={baseline_l}  R={baseline_r}")
        print(f"  Scale factor   : {scale_N_per_unit:.6f} N per raw unit")
        print(f"\n  Object readings:")
        for o in calib["objects"]:
            net = o["raw_avg"] - baseline_l
            est = net * scale_N_per_unit
            print(f"    {o['name']:15s}  raw_avg={o['raw_avg']:4d}  "
                  f"net={net:4d}  estimated={est:.3f}N  expected={o['expected_N']}N")

        calib["scale_N_per_unit"] = scale_N_per_unit

        # Save calibration file next to this script
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "fsr_calibration.txt")
        with open(save_path, "w") as f:
            json.dump(calib, f, indent=2)
        print(f"\n  Saved to: {save_path}")
    else:
        print("  No objects recorded — calibration skipped.")

    print("=" * 50)
    print("  Calibration done. Returning to normal mode.\n")

# ── ForceGUI ─────────────────────────────────────────────────
# Small pygame window that shows the live FSR force reading as a
# gauge bar, with markers for the required grip force and the
# object's damage limit. Used during AI grip so judges can see the
# Newton readout in real time.
class ForceGUI:
    def __init__(self, width=480, height=240):
        self.width  = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AdaptGrip — Live Force Feedback")
        self.font_big   = pygame.font.SysFont("consolas", 34, bold=True)
        self.font_med   = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 15)

    def update(self, actual_force, required_force, damage_limit, status, obj_name=""):
        pygame.event.pump()
        screen = self.screen
        screen.fill((26, 26, 46))

        title = self.font_med.render(f"AdaptGrip — {obj_name}", True, (255, 255, 255))
        screen.blit(title, (16, 10))

        # Gauge bar
        bar_x, bar_y, bar_w, bar_h = 20, 50, self.width - 40, 36
        max_scale = max(damage_limit * 1.15, 1.0)
        pygame.draw.rect(screen, (60, 60, 80), (bar_x, bar_y, bar_w, bar_h), border_radius=6)

        frac   = float(np.clip(actual_force / max_scale, 0.0, 1.0))
        fill_w = int(bar_w * frac)
        if actual_force >= damage_limit:
            color = (231, 76, 60)     # red — over damage limit
        elif actual_force >= required_force:
            color = (46, 204, 113)    # green — holding object
        else:
            color = (52, 152, 219)    # blue — still ramping up
        if fill_w > 0:
            pygame.draw.rect(screen, color, (bar_x, bar_y, fill_w, bar_h), border_radius=6)

        # Markers for required force and damage limit
        req_x = bar_x + int(bar_w * np.clip(required_force / max_scale, 0, 1))
        dmg_x = bar_x + int(bar_w * np.clip(damage_limit / max_scale, 0, 1))
        pygame.draw.line(screen, (0, 210, 255), (req_x, bar_y - 6), (req_x, bar_y + bar_h + 6), 3)
        pygame.draw.line(screen, (231, 76, 60), (dmg_x, bar_y - 6), (dmg_x, bar_y + bar_h + 6), 3)
        pygame.draw.rect(screen, (200, 200, 220), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=6)

        # Numeric readout
        force_txt = self.font_big.render(f"{actual_force:5.2f} N", True, (255, 255, 255))
        screen.blit(force_txt, (16, bar_y + bar_h + 14))

        info_txt = self.font_small.render(
            f"Required: {required_force:.2f} N    Damage limit: {damage_limit:.2f} N",
            True, (170, 170, 190))
        screen.blit(info_txt, (16, bar_y + bar_h + 64))

        status_txt = self.font_med.render(status, True, (241, 196, 15))
        screen.blit(status_txt, (16, bar_y + bar_h + 92))

        pygame.display.flip()


# ── find_contact() ───────────────────────────────────────────
# Closes the gripper gradually from fully open until the FSR
# sensors detect the object (force >= CONTACT_FORCE_N).
# Returns the angle at first contact — this becomes the "0 force"
# reference point for force_to_angle(), so the RL squeeze range
# adapts to the size of whatever object is in the gripper.
CONTACT_FORCE_N  = 0.05   # Newtons — minimum force counted as "touching"
CONTACT_STEP_DEG = 3      # degrees per contact-search step
CONTACT_DELAY    = 0.08   # seconds between steps

def find_contact(ser, ctrl, obj, req_force, gui=None):
    angle = LIMITS['GRIPPER']['max']   # start fully open (175°)
    send(ser, 'GRIPPER', angle)
    time.sleep(0.3)
    print("Searching for object contact...")

    while angle > LIMITS['GRIPPER']['min']:
        pygame.event.pump()
        if ctrl.get_button(3):   # Square = emergency stop
            ser.write(b"STOP\n")
            return "ESTOP"
        if ctrl.get_button(1):   # Circle = cancel
            return None

        fsr_l, fsr_r = read_fsr(ser)
        force = raw_fsr_to_newton(fsr_l, fsr_r)
        if gui:
            gui.update(force, req_force, obj['damage'],
                       "Searching for contact...", obj['name'])

        if force >= CONTACT_FORCE_N:
            print(f"Contact detected at angle {angle}° (force {force:.3f}N)")
            return angle

        angle = max(angle - CONTACT_STEP_DEG, LIMITS['GRIPPER']['min'])
        send(ser, 'GRIPPER', angle)
        time.sleep(CONTACT_DELAY)

    print("Fully closed without detecting contact — object may be missing/too small")
    return angle


# ── run_rl_grip() ────────────────────────────────────────────
# This is the AI grip function — runs when Cross button is pressed.
# Uses the trained PPO reinforcement learning model to decide
# how much grip force to apply based on the object type.
#
# The RL model takes 5 inputs (observation):
#   [mass, required_force, fragility, previous_force, step_progress]
# And outputs a force value (0.1N to damage_limit).
# That force is converted to a servo angle and sent to the gripper.
def run_rl_grip(ser, model, obj, ctrl, gui=None):
    req_force = obj['mass'] * 9.81 * 1.2

    # DEMO_MODE: skip RL, use a fixed safe angle until FSR is calibrated
    if DEMO_MODE:
        print(f"\nDEMO GRIP — fixed angle {DEMO_GRIPPER_ANGLE}° "
              f"(DEMO_MODE=True, FSR not calibrated)")
        print("Press Circle to release\n")
        send(ser, 'GRIPPER', DEMO_GRIPPER_ANGLE)
        fsr_l, fsr_r = read_fsr(ser)
        print(f"FSR readings → Left:{fsr_l}  Right:{fsr_r}")
        print("Holding... press Circle to release")
        while True:
            pygame.event.pump()
            if ctrl.get_button(1):   # Circle = release
                print("Released")
                break
            if gui:
                fsr_l, fsr_r = read_fsr(ser)
                force = raw_fsr_to_newton(fsr_l, fsr_r)
                gui.update(force, req_force, obj['damage'], "DEMO GRIP — holding", obj['name'])
            time.sleep(0.05)
        return

    print(f"\nAI GRIP — {obj['name']} | damage limit {obj['damage']}N | "
          f"required {req_force:.2f}N")

    # ── Phase 1: close gradually until the gripper touches the object ──
    contact_angle = find_contact(ser, ctrl, obj, req_force, gui)
    if contact_angle == "ESTOP":
        return "ESTOP"
    if contact_angle is None:
        print("Released by user during contact search")
        return

    print("Press Circle to release\n")
    step       = 0
    prev_force = 0.0

    while step < 10:
        pygame.event.pump()
        if ctrl.get_button(3):   # Square = emergency stop
            ser.write(b"STOP\n")
            return "ESTOP"
        if ctrl.get_button(1):
            print("Released by user")
            break

        fsr_l, fsr_r  = read_fsr(ser)
        actual_force  = raw_fsr_to_newton(fsr_l, fsr_r)  # real N from calibration

        obs = np.array([
            obj['mass'],
            req_force,
            obj['fragility'],
            prev_force,
            step / 10.0
        ], dtype=np.float32)

        action, _  = model.predict(obs, deterministic=True)
        force      = float(np.clip(action[0], 0.1, obj['damage'] * 0.9))
        grip_angle = force_to_angle(force, max_force=obj['damage'], open_angle=contact_angle)
        send(ser, 'GRIPPER', grip_angle)

        print(f"Step {step+1:2d} | "
              f"FSR raw:({fsr_l},{fsr_r}) | "
              f"actual:{actual_force:.3f}N | "
              f"model:{force:.2f}N | "
              f"angle:{grip_angle}°")

        if gui:
            gui.update(actual_force, req_force, obj['damage'],
                       f"Gripping... step {step+1}/10", obj['name'])

        # Use actual force when contact detected, model force otherwise
        # This prevents prev_force staying 0 while gripper is still closing
        prev_force = actual_force if actual_force > 0.01 else force
        step += 1

        # Stop early — grip is stable when actual force meets requirement
        if actual_force >= req_force and step >= 3:
            print(f"Stable grasp! {actual_force:.3f}N >= {req_force:.3f}N required")
            if gui:
                gui.update(actual_force, req_force, obj['damage'], "STABLE GRASP!", obj['name'])
            break

        # Safety stop — never exceed 90% of damage limit
        if actual_force >= obj['damage'] * 0.9:
            print(f"FORCE LIMIT REACHED — {actual_force:.3f}N, stopping")
            if gui:
                gui.update(actual_force, req_force, obj['damage'], "DAMAGE LIMIT!", obj['name'])
            break

        time.sleep(0.05)

# ── auto_pick_and_place() ────────────────────────────────────
# Automated pick-and-place sequence for demo.
# Triggered by Options button (Btn 9).
#
# Sequence:
#   1. Open gripper and raise arm to approach position
#   2. Lower arm down to object
#   3. Close gripper (DEMO_GRIPPER_ANGLE or full RL grip)
#   4. Lift arm back up
#   5. Rotate base to place position
#   6. Lower arm to place
#   7. Open gripper (release object)
#   8. Lift and rotate base back home
#
# Adjust the waypoint angles below to match your physical setup.
# PLACE_STEPS = how many stepper steps to rotate to the place position.
PLACE_STEPS = 250   # steps CW to rotate from pick → place position

def auto_pick_and_place(ser):
    global stepper_pos

    def wp(joint, angle, delay=0.8):
        """Send one waypoint and wait for servo to reach it."""
        send(ser, joint, angle)
        time.sleep(delay)

    def stepper_move_wait(steps, cw, delay=1.5):
        direction = "CW" if cw else "CCW"
        ser.write(f"STEPPER:{direction}:{steps}\n".encode())
        if cw:
            stepper_pos += steps
        else:
            stepper_pos -= steps
        time.sleep(delay)

    print("\n" + "=" * 40)
    print("  AUTO PICK AND PLACE — starting")
    print("=" * 40)

    # ── 1. Open gripper and raise to approach ─────────────────
    print("Step 1: Open gripper, move to approach position")
    wp('GRIPPER',    0)         # open gripper fully
    wp('WRIST_P',   96)         # wrist to neutral
    wp('ELBOW',     30)         # elbow mid
    wp('SHOULDER_R', 60)        # raise shoulder
    wp('SHOULDER_L', 120)
    time.sleep(0.5)

    # ── 2. Lower arm to object ─────────────────────────────────
    print("Step 2: Lower to object")
    wp('ELBOW',     75)
    wp('WRIST_P',  130)
    time.sleep(0.5)

    # ── 3. Close gripper ──────────────────────────────────────
    print(f"Step 3: Grip object (angle={DEMO_GRIPPER_ANGLE}°)")
    wp('GRIPPER', DEMO_GRIPPER_ANGLE, delay=1.0)
    fsr_l, fsr_r = read_fsr(ser)
    print(f"        FSR → Left:{fsr_l}  Right:{fsr_r}")
    time.sleep(0.5)

    # ── 4. Lift arm with object ───────────────────────────────
    print("Step 4: Lift")
    wp('WRIST_P',   96)
    wp('ELBOW',     30)
    time.sleep(0.5)

    # ── 5. Rotate base to place position ─────────────────────
    print(f"Step 5: Rotate base CW {PLACE_STEPS} steps")
    stepper_move_wait(PLACE_STEPS, cw=True, delay=2.0)

    # ── 6. Lower arm to place position ───────────────────────
    print("Step 6: Lower to place")
    wp('ELBOW',     75)
    wp('WRIST_P',  130)
    time.sleep(0.5)

    # ── 7. Release object ─────────────────────────────────────
    print("Step 7: Release")
    wp('GRIPPER', 0, delay=1.0)
    time.sleep(0.3)

    # ── 8. Lift and return base ───────────────────────────────
    print("Step 8: Return home")
    wp('WRIST_P',   96)
    wp('ELBOW',     30)
    stepper_move_wait(PLACE_STEPS, cw=False, delay=2.0)

    # Go back to full rest
    go_home(ser)

    print("=" * 40)
    print("  AUTO PICK AND PLACE — done")
    print("=" * 40 + "\n")

# ── main() ───────────────────────────────────────────────────
# Main program — runs everything.
# Steps:
#   1. Load RL model (GPU first, CPU fallback)
#   2. Connect to Arduino via USB serial
#   3. Connect to PS4 controller via pygame
#   4. Send START to Arduino (arm moves to home)
#   5. Enter main control loop — read PS4 inputs and send joint commands
def main():
    global stepper_pos

    print("=" * 50)
    print("    AdaptGrip — PS4 + RL Control")
    print("=" * 50)

    # Load FSR calibration values from file (or use hardcoded fallback)
    load_fsr_calibration()

    # ── Step 1: Load RL Model ───────────────────────────────
    # Try GPU first (faster), fall back to CPU if not available
    model = None
    try:
        model = PPO.load(MODEL_PATH, device='cuda')
        print("Model loaded on GPU!")
    except Exception:
        try:
            model = PPO.load(MODEL_PATH, device='cpu')
            print("Model loaded on CPU!")
        except Exception as e:
            print(f"Model load failed: {e}")
            print("Running without RL — AI grip disabled")

    # ── Step 2: Connect to Arduino ──────────────────────────
    print(f"\nConnecting Arduino on {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        time.sleep(2)         # Wait for Arduino to reset after USB connect
        ser.flushInput()      # Clear any old data in serial buffer
        print("Arduino connected!")
    except Exception as e:
        print(f"Arduino failed: {e}")
        return

    # ── Step 3: Connect to PS4 Controller ───────────────────
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No PS4 detected!")
        return
    ctrl = pygame.joystick.Joystick(0)
    ctrl.init()
    print(f"PS4: {ctrl.get_name()}")

    # ── Step 3b: Open Live Force Feedback GUI ───────────────
    gui = None
    try:
        gui = ForceGUI()
        print("Force feedback GUI opened.")
    except Exception as e:
        print(f"Could not open Force GUI: {e} — continuing without it")

    # ── Step 4: Start System ─────────────────────────────────
    # Sends START to Arduino → arm moves to home position
    print("\nSending START — arm moving to home...")
    ser.write(b"START\n")
    time.sleep(4)             # Wait for arm to finish moving home
    stepper_pos = 0
    print("System ready!\n")

    # ── Control Settings ─────────────────────────────────────
    AI_MODE      = False       # True when RL model is controlling gripper
    selected_obj = OBJECTS[3]  # Default object = Medium
    DEAD         = 0.25        # Deadzone for right stick (smaller, offset-corrected)
    DEAD_LEFT    = 0.35        # Deadzone for left stick (higher — noisier axis)
    STEP         = 3           # Degrees per loop for right stick joints
    SHOULDER_STEP = 4          # Degrees per loop for shoulders (heavier load, needs more)
    GRIP_STEP    = 5           # Degrees per step — increased to match demo speed

    prev_buttons      = [0] * ctrl.get_numbuttons()
    last_stepper_time = 0
    last_grip_time    = 0

    print("=" * 50)
    print("CONTROLS:")
    print("  Left  stick U/D  → Shoulder up/down")
    print("  Left  stick L/R  → Wrist Roll")
    print("  Right stick U/D  → Elbow")
    print("  Right stick L/R  → Wrist Pitch")
    print("  L1 held          → Base CCW (max -300)")
    print("  R1 held          → Base CW  (max +550)")
    print("  L2 held          → Gripper CLOSE")
    print("  R2 held          → Gripper OPEN")
    print("  Cross  (Btn 0)   → AI grip ON (or demo grip if DEMO_MODE=True)")
    print("  Circle (Btn 1)   → Release gripper")
    print("  Triangle(Btn 2)  → Go to Home (slow)")
    print("  Square (Btn 3)   → EMERGENCY STOP")
    print("  Options(Btn 9)   → Auto pick and place sequence")
    print("  L3 (Btn 11)      → Select Very Fragile")
    print("  R3 (Btn 12)      → Select Fragile")
    print("  L3 + R3 together → FSR Calibration mode")
    print("=" * 50)
    print(f"Object: {selected_obj['name']}\n")

    running = True

    # ── Step 5: Main Control Loop ────────────────────────────
    # Runs continuously until Square (emergency stop) is pressed.
    # Each loop: read PS4 → check buttons → move joints
    while running:
        pygame.event.pump()  # Update pygame with latest controller state

        # Read analog stick values (-1.0 to +1.0)
        lx = ctrl.get_axis(0)   # Left stick left/right  → Wrist Roll
        ly = ctrl.get_axis(1)   # Left stick up/down     → Shoulder
        rx = ctrl.get_axis(3)   # Right stick left/right → Wrist Pitch
        ry = ctrl.get_axis(4)   # Right stick up/down    → Elbow

        # FIX: Right stick rests at -1.0. Offset to a 0.0–2.0 range where
        # 1.0 is the true center. Subtract 1.0 before use to get -1.0 to +1.0.
        rx_fixed = rx + 1.0   # rest=1.0, left=-0.0, right=2.0 → subtract 1.0 → -1 to +1
        ry_fixed = ry + 1.0

        # FIX: Read L2/R2 as analog trigger axes (not buttons).
        # On PS4 via pygame on Linux, L2=axis2 and R2=axis5.
        # Value is -1.0 (released) to +1.0 (fully pressed).
        l2 = ctrl.get_axis(2)
        r2 = ctrl.get_axis(5)

        # Read all buttons (1=pressed, 0=not pressed)
        curr = [ctrl.get_button(i) for i in range(ctrl.get_numbuttons())]

        # Detect NEW button presses (just pressed this loop, not held)
        new  = [curr[i] and not prev_buttons[i] for i in range(len(curr))]

        # L3 + R3 held together = enter FSR calibration mode
        if curr[11] and curr[12]:
            print("Entering FSR calibration...")
            calibrate_fsr(ser, ctrl)
            # reset button state to avoid spurious triggers after returning
            prev_buttons = [ctrl.get_button(i) for i in range(ctrl.get_numbuttons())]
            continue

        # Object selection (L3 or R3 alone)
        if new[11]:
            selected_obj = OBJECTS[1]
            print(f"Object: {selected_obj['name']}")
        if new[12]:
            selected_obj = OBJECTS[2]
            print(f"Object: {selected_obj['name']}")

        # Square = Emergency Stop — turns off everything immediately
        if new[3]:
            print("EMERGENCY STOP!")
            ser.write(b"STOP\n")
            running = False
            break

        # Triangle = Go to home position
        if new[2]:
            AI_MODE = False
            go_home(ser)

        # Options (Btn 9) = Auto pick and place sequence
        if new[9]:
            AI_MODE = False
            auto_pick_and_place(ser)

        # Cross (Btn 0) = Activate AI grip mode
        if new[0] and not AI_MODE:
            if model is not None:
                AI_MODE = True
                result = run_rl_grip(ser, model, selected_obj, ctrl, gui)
                AI_MODE = False
                if result == "ESTOP":
                    print("EMERGENCY STOP!")
                    running = False
                    break
            else:
                print("No RL model — AI grip unavailable")

        # Circle (Btn 1) = Release gripper (open)
        if new[1]:
            AI_MODE = False
            send(ser, 'GRIPPER', LIMITS['GRIPPER']['rest'])
            print("Gripper open")

        # Manual joint control (only when AI is NOT active)
        if not AI_MODE:
            now = time.time()

            # Left stick U/D → Both shoulders move together (synchronised)
            # Push up (ly<0): SHOULDER_R increases, SHOULDER_L decreases (inverted servo)
            # SHOULDER_L max is set to 190 so it can move in both directions from rest=180
            if abs(ly) > DEAD_LEFT:
                send(ser, 'SHOULDER_R', angles['SHOULDER_R'] - (ly * SHOULDER_STEP))
                send(ser, 'SHOULDER_L', angles['SHOULDER_L'] + (ly * SHOULDER_STEP))

            # Left stick L/R → Wrist Roll
            if abs(lx) > DEAD_LEFT:
                send(ser, 'WRIST_R', angles['WRIST_R'] + (lx * STEP))

            # FIX: Right stick U/D → Elbow
            # Deadzone checked around the true center (1.0 after offset).
            # Subtract 1.0 to get a -1.0 to +1.0 value for movement direction.
            if abs(ry_fixed - 1.0) > DEAD:
                send(ser, 'ELBOW', angles['ELBOW'] + ((ry_fixed - 1.0) * STEP))

            # FIX: Right stick L/R → Wrist Pitch (direction inverted to match physical)
            if abs(rx_fixed - 1.0) > DEAD:
                send(ser, 'WRIST_P', angles['WRIST_P'] - ((rx_fixed - 1.0) * STEP))

            # L1/R1 = Base rotation (stepper motor)
            # Rate-limited to once every 0.06s — reduced for faster base rotation
            if now - last_stepper_time > 0.06:
                if curr[4]:   # L1 = CCW
                    send_stepper(ser, False)
                    print(f"Base CCW | pos:{stepper_pos}")
                    last_stepper_time = now
                elif curr[5]: # R1 = CW
                    send_stepper(ser, True)
                    print(f"Base CW  | pos:{stepper_pos}")
                    last_stepper_time = now

            # L2/R2 = Gripper open/close (analog triggers, not buttons)
            # Scale step by how hard the trigger is pressed for variable speed.
            # Rate-limited to once every 0.04s for faster response.
            if now - last_grip_time > 0.04:
                if l2 > 0.1:    # L2 = close gripper (decrease angle, 0=closed)
                    scaled = int(GRIP_STEP * ((l2 + 1.0) / 2.0) * 3) + 1
                    send(ser, 'GRIPPER', angles['GRIPPER'] - scaled)
                    last_grip_time = now
                elif r2 > 0.1:  # R2 = open gripper (increase angle, 175=open)
                    scaled = int(GRIP_STEP * ((r2 + 1.0) / 2.0) * 3) + 1
                    send(ser, 'GRIPPER', angles['GRIPPER'] + scaled)
                    last_grip_time = now

        prev_buttons = curr    # Save current buttons for next loop comparison
        time.sleep(0.05)       # 50ms loop delay = ~20 updates per second

    # ── Cleanup ──────────────────────────────────────────────
    ser.close()      # Close USB serial connection
    pygame.quit()    # Shut down pygame
    print("\nAdaptGrip stopped")

if __name__ == "__main__":
    main()
