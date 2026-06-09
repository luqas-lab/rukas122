# ============================================================
#  AdaptGrip — Demo Pick and Place
#  Author  : Muhammad Luqmanul Hakeem bin Ramli
#  Run on  : Linux PC (Ubuntu), Python 3
# ============================================================
#
#  HOW TO USE:
#  1. First run waypoint_recorder.py to record and save waypoints.json
#  2. Then run this script: python3 demo_pick_place.py
#  3. Drive arm near the object using PS4 controller
#  4. Press OPTIONS (Btn 9) to run the full auto pick-and-place
#  5. Press Square to emergency stop
#
#  DEMO_GRIPPER_ANGLE — tune this until arm holds object without crushing
#  Start at 60, test on cup, increase by 5° if it slips.

import serial
import time
import pygame
import numpy as np
import json
import os

# ── Settings ─────────────────────────────────────────────────
PORT               = '/dev/ttyUSB0'
BAUD               = 9600
DEMO_GRIPPER_ANGLE = 60      # degrees — tune this for your object
WAYPOINTS_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'waypoints.json')

# ── Joint Limits ─────────────────────────────────────────────
LIMITS = {
    'SHOULDER_R': {'rest':   0, 'min':  0, 'max': 180},
    'SHOULDER_L': {'rest': 180, 'min': 50, 'max': 190},
    'ELBOW':      {'rest':   0, 'min':  0, 'max': 110},
    'WRIST_P':    {'rest':  96, 'min':  0, 'max': 170},
    'WRIST_R':    {'rest':   0, 'min':  0, 'max': 175},
    'GRIPPER':    {'rest':   0, 'min':  0, 'max': 175},
}

STEPPER_STEPS = 150
STEPPER_MAX   =  550
STEPPER_MIN   = -300
stepper_pos   = 0

angles = {j: LIMITS[j]['rest'] for j in LIMITS}


def send(ser, joint, angle):
    angle = int(np.clip(angle, LIMITS[joint]['min'], LIMITS[joint]['max']))
    angles[joint] = angle
    ser.write(f"{joint}:{angle}\n".encode())


def send_stepper_abs(ser, target_steps):
    """Move stepper to an absolute step position."""
    global stepper_pos
    diff = target_steps - stepper_pos
    if diff == 0:
        return
    cw    = diff > 0
    steps = abs(diff)
    direction = "CW" if cw else "CCW"
    ser.write(f"STEPPER:{direction}:{steps}\n".encode())
    stepper_pos = target_steps
    time.sleep(abs(steps) * 0.01 + 1.0)   # wait proportional to distance


def send_stepper(ser, cw):
    global stepper_pos
    new_pos = stepper_pos + (STEPPER_STEPS if cw else -STEPPER_STEPS)
    new_pos = int(np.clip(new_pos, STEPPER_MIN, STEPPER_MAX))
    steps = abs(new_pos - stepper_pos)
    if steps > 0:
        direction = "CW" if cw else "CCW"
        ser.write(f"STEPPER:{direction}:{steps}\n".encode())
        stepper_pos = new_pos


def go_to_waypoint(ser, wp, delay=0.6):
    """Move all joints smoothly to a saved waypoint position."""
    print(f"  → Moving to: {wp['name']}")
    joints = ['SHOULDER_R', 'SHOULDER_L', 'ELBOW', 'WRIST_P', 'WRIST_R', 'GRIPPER']
    for joint in joints:
        if joint in wp:
            send(ser, joint, wp[joint])
            time.sleep(0.15)
    # Move stepper to saved position
    if 'STEPPER' in wp and wp['STEPPER'] != stepper_pos:
        send_stepper_abs(ser, wp['STEPPER'])
    time.sleep(delay)


def go_home(ser):
    global stepper_pos
    SLOW_STEP  = 2
    SLOW_DELAY = 0.03
    print("Going home (slow)...")
    for joint in LIMITS:
        target  = LIMITS[joint]['rest']
        current = angles[joint]
        while current != target:
            current = min(current + SLOW_STEP, target) if current < target \
                      else max(current - SLOW_STEP, target)
            send(ser, joint, current)
            time.sleep(SLOW_DELAY)
    ser.write(b"STEPPERHOME\n")
    stepper_pos = 0
    print("Home done!")


def read_fsr(ser):
    ser.write(b"FSR\n")
    time.sleep(0.1)
    fsr_l, fsr_r = 0, 0
    deadline = time.time() + 0.3
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


def auto_pick_and_place(ser, wps):
    """
    Run the full pick-and-place sequence using saved waypoints.
    wps is a dict keyed by waypoint name.
    """
    print("\n" + "=" * 45)
    print(f"  AUTO PICK AND PLACE  |  grip={DEMO_GRIPPER_ANGLE}°")
    print("=" * 45)

    def wp(name, extra_delay=0.0):
        if name in wps:
            go_to_waypoint(ser, wps[name])
            if extra_delay:
                time.sleep(extra_delay)
        else:
            print(f"  WARNING: waypoint '{name}' not found — skipping")

    # 1. Move to approach position (above object, gripper open)
    print("\n[1/8] Approach")
    send(ser, 'GRIPPER', 0)
    time.sleep(0.3)
    wp('1_approach')

    # 2. Lower to pick position
    print("[2/8] Lower to object")
    wp('2_pick_down')

    # 3. Close gripper
    print(f"[3/8] Grip at {DEMO_GRIPPER_ANGLE}°")
    send(ser, 'GRIPPER', DEMO_GRIPPER_ANGLE)
    time.sleep(1.0)
    fsr_l, fsr_r = read_fsr(ser)
    print(f"      FSR → Left:{fsr_l}  Right:{fsr_r}")
    # Update waypoint GRIPPER value so go_to_waypoint won't open it
    if '3_gripping' in wps:
        wps['3_gripping']['GRIPPER'] = DEMO_GRIPPER_ANGLE
    wp('3_gripping', extra_delay=0.3)

    # 4. Lift with object
    print("[4/8] Lift")
    wp('4_lifted')

    # 5. Move to place approach (base rotates here)
    print("[5/8] Rotate to place position")
    wp('5_place_approach', extra_delay=0.3)

    # 6. Lower to place position
    print("[6/8] Lower to place")
    wp('6_place_down')

    # 7. Release
    print("[7/8] Release")
    send(ser, 'GRIPPER', 0)
    time.sleep(1.0)

    # 8. Lift and return home
    print("[8/8] Return home")
    wp('8_return_up', extra_delay=0.3)
    go_home(ser)

    print("\n" + "=" * 45)
    print("  SEQUENCE COMPLETE")
    print("=" * 45 + "\n")


def load_waypoints():
    if not os.path.exists(WAYPOINTS_FILE):
        print(f"ERROR: waypoints.json not found at {WAYPOINTS_FILE}")
        print("Run waypoint_recorder.py first to record positions.")
        return None
    with open(WAYPOINTS_FILE, 'r') as f:
        wp_list = json.load(f)
    # Convert list to dict keyed by name for easy lookup
    wps = {wp['name']: wp for wp in wp_list}
    print(f"Loaded {len(wps)} waypoints: {list(wps.keys())}")
    return wps


def main():
    global stepper_pos

    print("=" * 45)
    print("  AdaptGrip — Demo Pick and Place")
    print("=" * 45)
    print(f"  Grip angle : {DEMO_GRIPPER_ANGLE}°")
    print(f"  Waypoints  : {WAYPOINTS_FILE}")
    print("=" * 45)

    # Load waypoints
    wps = load_waypoints()
    if wps is None:
        return

    # Connect Arduino
    print(f"\nConnecting to Arduino on {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        time.sleep(2)
        ser.flushInput()
        print("Arduino connected!")
    except Exception as e:
        print(f"Arduino failed: {e}")
        return

    # Connect PS4
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No PS4 controller detected!")
        return
    ctrl = pygame.joystick.Joystick(0)
    ctrl.init()
    print(f"PS4: {ctrl.get_name()}")

    # Start
    ser.write(b"START\n")
    time.sleep(4)
    stepper_pos = 0

    DEAD       = 0.25
    DEAD_LEFT  = 0.35
    STEP       = 3
    SHOULDER_STEP = 4
    GRIP_STEP  = 2

    prev_buttons      = [0] * ctrl.get_numbuttons()
    last_stepper_time = 0
    last_grip_time    = 0

    print("\n" + "=" * 45)
    print("  CONTROLS")
    print("  Left  stick U/D  → Shoulder")
    print("  Left  stick L/R  → Wrist Roll")
    print("  Right stick U/D  → Elbow")
    print("  Right stick L/R  → Wrist Pitch")
    print("  L1 / R1          → Base CCW / CW")
    print("  L2 / R2          → Gripper close / open")
    print("  Triangle (Btn 2) → Go to Home")
    print("  OPTIONS  (Btn 9) → RUN pick and place")
    print("  Square   (Btn 3) → EMERGENCY STOP")
    print("=" * 45)
    print(f"\nDrive arm near object then press OPTIONS to start.\n")

    running = True
    while running:
        pygame.event.pump()

        lx = ctrl.get_axis(0)
        ly = ctrl.get_axis(1)
        rx = ctrl.get_axis(3)
        ry = ctrl.get_axis(4)
        rx_fixed = rx + 1.0
        ry_fixed = ry + 1.0
        l2 = ctrl.get_axis(2)
        r2 = ctrl.get_axis(5)

        curr = [ctrl.get_button(i) for i in range(ctrl.get_numbuttons())]
        new  = [curr[i] and not prev_buttons[i] for i in range(len(curr))]

        if new[3]:
            print("EMERGENCY STOP!")
            ser.write(b"STOP\n")
            running = False
            break

        if new[2]:
            go_home(ser)

        if new[9]:
            auto_pick_and_place(ser, wps)

        now = time.time()

        if abs(ly) > DEAD_LEFT:
            send(ser, 'SHOULDER_R', angles['SHOULDER_R'] - (ly * SHOULDER_STEP))
            send(ser, 'SHOULDER_L', angles['SHOULDER_L'] + (ly * SHOULDER_STEP))

        if abs(lx) > DEAD_LEFT:
            send(ser, 'WRIST_R', angles['WRIST_R'] + (lx * STEP))

        if abs(ry_fixed - 1.0) > DEAD:
            send(ser, 'ELBOW', angles['ELBOW'] + ((ry_fixed - 1.0) * STEP))

        if abs(rx_fixed - 1.0) > DEAD:
            send(ser, 'WRIST_P', angles['WRIST_P'] - ((rx_fixed - 1.0) * STEP))

        if now - last_stepper_time > 0.06:
            if curr[4]:
                send_stepper(ser, False)
                last_stepper_time = now
            elif curr[5]:
                send_stepper(ser, True)
                last_stepper_time = now

        if now - last_grip_time > 0.04:
            if l2 > 0.1:
                scaled = int(GRIP_STEP * ((l2 + 1.0) / 2.0) * 3) + 1
                send(ser, 'GRIPPER', angles['GRIPPER'] + scaled)
                last_grip_time = now
            elif r2 > 0.1:
                scaled = int(GRIP_STEP * ((r2 + 1.0) / 2.0) * 3) + 1
                send(ser, 'GRIPPER', angles['GRIPPER'] - scaled)
                last_grip_time = now

        prev_buttons = curr
        time.sleep(0.05)

    ser.close()
    pygame.quit()
    print("\nDemo stopped.")


if __name__ == "__main__":
    main()
