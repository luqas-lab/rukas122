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
PORT       = '/dev/ttyUSB0'                    # Arduino USB port on Linux
BAUD       = 9600                              # Must match Arduino Serial.begin(9600)
MODEL_PATH = '/home/luq/fyp2/ppo_force_control.zip'  # Trained PPO model file

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
    'GRIPPER':    {'rest':   0, 'min':  0, 'max': 175},  # CH5 — gripper open/close
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
    print("Going home...")
    for joint in LIMITS:
        send(ser, joint, LIMITS[joint]['rest'])  # Move each joint to rest angle
        time.sleep(0.2)                          # Small delay between each joint
    ser.write(b"STEPPERHOME\n")                  # Tell Arduino to return stepper to 0
    stepper_pos = 0
    print("Home done!")

# ── force_to_angle() ─────────────────────────────────────────
# Converts a grip force value (in Newtons) to a servo angle.
# Higher force = smaller angle (gripper closes more).
# Lower force  = larger angle  (gripper opens more).
# Force range: 0.1N (barely touching) to 100N (maximum grip)
def force_to_angle(force_n):
    min_a = LIMITS['GRIPPER']['min']   # 0°
    max_a = LIMITS['GRIPPER']['max']   # 175°
    angle = max_a - ((force_n - 0.1) / (100.0 - 0.1)) * (max_a - min_a)
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
    print("\n" + "=" * 50)
    print("  FSR CALIBRATION MODE")
    print("=" * 50)
    print("Step 1: Make sure gripper is OPEN (no object)")
    print("Press Cross to record baseline...")

    # Wait for Cross press
    while True:
        pygame.event.pump()
        if ctrl.get_button(0):
            break
        time.sleep(0.05)
    time.sleep(0.1)  # debounce

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

    print("\nStep 2: Grip each object firmly and press Cross to record.")
    print("        Press Circle to skip an object.\n")

    for obj in objects_to_test:
        print(f"  → Grip '{obj['name']}' object now, then press Cross...")
        while True:
            pygame.event.pump()
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
            time.sleep(0.05)

    # Compute linear scale factor from highest reading
    if calib["objects"]:
        max_entry = max(calib["objects"], key=lambda x: x["raw_avg"])
        fsr_range = max(max_entry["raw_avg"] - baseline_l, 1)
        scale_N_per_unit = max_entry["expected_N"] / fsr_range

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

# ── run_rl_grip() ────────────────────────────────────────────
# This is the AI grip function — runs when Cross button is pressed.
# Uses the trained PPO reinforcement learning model to decide
# how much grip force to apply based on the object type.
#
# The RL model takes 5 inputs (observation):
#   [mass, required_force, fragility, previous_force, step_progress]
# And outputs a force value (0.1N to damage_limit).
# That force is converted to a servo angle and sent to the gripper.
def run_rl_grip(ser, model, obj, ctrl):
    print(f"\nAI GRIP — {obj['name']} | {obj['damage']}N limit")
    print("Press Circle to release\n")
    step       = 0
    prev_force = 0.0
    req_force  = obj['mass'] * 9.81 * 1.2  # Required force = weight × safety factor

    while step < 10:                         # Maximum 10 control steps
        pygame.event.pump()
        if ctrl.get_button(0):               # Cross button = manual release (cancel AI grip)
            print("Released by user")
            break

        fsr_l, fsr_r = read_fsr(ser)         # Read current grip force from sensors

        # Build observation array for RL model
        obs = np.array([
            obj['mass'],        # Object weight
            req_force,          # Target grip force needed
            obj['fragility'],   # How fragile the object is
            prev_force,         # What force was applied last step
            step / 10.0         # Progress (0.0 to 1.0)
        ], dtype=np.float32)

        # Ask RL model to predict the best action (force value)
        action, _  = model.predict(obs, deterministic=True)
        force      = float(np.clip(action[0], 0.1, obj['damage'] * 0.9))
        grip_angle = force_to_angle(force)   # Convert force to servo angle
        send(ser, 'GRIPPER', grip_angle)     # Send to gripper

        print(f"Step {step+1} | FSR:{fsr_l},{fsr_r} | "
              f"Force:{force:.2f}N | Angle:{grip_angle}")

        prev_force = force
        step += 1

        # Stop early if grip is stable
        if force >= req_force and step >= 3:
            print("Stable grasp!")
            break

        time.sleep(0.05)

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
    GRIP_STEP    = 2           # Degrees per press for gripper open/close

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
    print("  Cross  (Btn 0)   → AI grip ON")
    print("  Circle (Btn 1)   → Release gripper")
    print("  Triangle(Btn 2)  → Go to Home")
    print("  Square (Btn 3)   → EMERGENCY STOP")
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

        # FIX: Cross and Circle were swapped — corrected button indices
        # Circle (Btn 1) = Activate AI grip mode
        if new[1] and not AI_MODE:
            if model is not None:
                AI_MODE = True
                run_rl_grip(ser, model, selected_obj, ctrl)
                AI_MODE = False
            else:
                print("No RL model — AI grip unavailable")

        # Cross (Btn 0) = Release gripper (open)
        if new[0]:
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

            # FIX: L2/R2 = Gripper open/close (analog triggers, not buttons)
            # Trigger value > 0.1 means it is being pressed.
            # Rate-limited to once every 0.08s for smooth movement.
            if now - last_grip_time > 0.08:
                if l2 > 0.1:    # L2 held = close gripper
                    send(ser, 'GRIPPER', angles['GRIPPER'] + GRIP_STEP)
                    last_grip_time = now
                elif r2 > 0.1:  # R2 held = open gripper
                    send(ser, 'GRIPPER', angles['GRIPPER'] - GRIP_STEP)
                    last_grip_time = now

        prev_buttons = curr    # Save current buttons for next loop comparison
        time.sleep(0.05)       # 50ms loop delay = ~20 updates per second

    # ── Cleanup ──────────────────────────────────────────────
    ser.close()      # Close USB serial connection
    pygame.quit()    # Shut down pygame
    print("\nAdaptGrip stopped")

if __name__ == "__main__":
    main()
