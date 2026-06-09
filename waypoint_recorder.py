# ============================================================
#  AdaptGrip — Waypoint Recorder
#  Author  : Muhammad Luqmanul Hakeem bin Ramli
#  Run on  : Linux PC (Ubuntu), Python 3
# ============================================================
#
#  HOW TO USE:
#  1. Run this script: python3 waypoint_recorder.py
#  2. Drive the arm to each position using the PS4 controller
#  3. The terminal shows ALL current joint angles live every second
#  4. When the arm is in the right position, press OPTIONS (Btn 9)
#     to save that position as the next waypoint
#  5. Repeat for all positions you need
#  6. Press SHARE (Btn 8) to save all recorded waypoints to
#     waypoints.json — this file is loaded by the demo script
#  7. Press Square to emergency stop and exit
#
#  CONTROLS (same as main controller):
#  Left  stick U/D  → Shoulder up/down
#  Left  stick L/R  → Wrist Roll
#  Right stick U/D  → Elbow
#  Right stick L/R  → Wrist Pitch
#  L1 / R1          → Base rotate CCW / CW
#  L2 / R2          → Gripper close / open
#  OPTIONS (Btn 9)  → SAVE current position as waypoint
#  SHARE   (Btn 8)  → WRITE all waypoints to waypoints.json
#  Triangle(Btn 2)  → Go to home position
#  Square  (Btn 3)  → Emergency stop + exit

import serial
import time
import pygame
import numpy as np
import json
import os

# ── Connection Settings ──────────────────────────────────────
PORT = '/dev/ttyUSB0'
BAUD = 9600

# ── Joint Limits (must match main controller) ────────────────
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
waypoints = []   # list of saved positions

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waypoints.json')


def send(ser, joint, angle):
    angle = int(np.clip(angle, LIMITS[joint]['min'], LIMITS[joint]['max']))
    angles[joint] = angle
    ser.write(f"{joint}:{angle}\n".encode())


def send_stepper(ser, cw):
    global stepper_pos
    new_pos = stepper_pos + (STEPPER_STEPS if cw else -STEPPER_STEPS)
    new_pos = int(np.clip(new_pos, STEPPER_MIN, STEPPER_MAX))
    steps = abs(new_pos - stepper_pos)
    if steps > 0:
        direction = "CW" if cw else "CCW"
        ser.write(f"STEPPER:{direction}:{steps}\n".encode())
        stepper_pos = new_pos


def go_home(ser):
    global stepper_pos
    SLOW_STEP  = 2
    SLOW_DELAY = 0.03
    print("Going home...")
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


def print_angles():
    """Print all current joint angles and stepper position clearly."""
    print("\n" + "─" * 45)
    print(f"  SHOULDER_R (CH0) : {angles['SHOULDER_R']:>4}°")
    print(f"  SHOULDER_L (CH1) : {angles['SHOULDER_L']:>4}°")
    print(f"  ELBOW      (CH2) : {angles['ELBOW']:>4}°")
    print(f"  WRIST_P    (CH3) : {angles['WRIST_P']:>4}°")
    print(f"  WRIST_R    (CH4) : {angles['WRIST_R']:>4}°")
    print(f"  GRIPPER    (CH5) : {angles['GRIPPER']:>4}°")
    print(f"  STEPPER    (BASE): {stepper_pos:>4} steps")
    print("─" * 45)


def save_waypoint(name):
    """Save current position as a named waypoint."""
    wp = {
        "name":       name,
        "SHOULDER_R": angles['SHOULDER_R'],
        "SHOULDER_L": angles['SHOULDER_L'],
        "ELBOW":      angles['ELBOW'],
        "WRIST_P":    angles['WRIST_P'],
        "WRIST_R":    angles['WRIST_R'],
        "GRIPPER":    angles['GRIPPER'],
        "STEPPER":    stepper_pos,
    }
    waypoints.append(wp)
    print(f"\n*** WAYPOINT {len(waypoints)} SAVED: '{name}' ***")
    print_angles()
    return wp


def write_waypoints_file():
    """Write all saved waypoints to waypoints.json."""
    with open(SAVE_PATH, 'w') as f:
        json.dump(waypoints, f, indent=2)
    print(f"\n{'=' * 45}")
    print(f"  SAVED {len(waypoints)} waypoints to:")
    print(f"  {SAVE_PATH}")
    print(f"{'=' * 45}")
    for i, wp in enumerate(waypoints):
        print(f"  [{i+1}] {wp['name']}")
        print(f"       SR={wp['SHOULDER_R']} SL={wp['SHOULDER_L']} "
              f"EL={wp['ELBOW']} WP={wp['WRIST_P']} "
              f"WR={wp['WRIST_R']} GR={wp['GRIPPER']} "
              f"ST={wp['STEPPER']}")
    print(f"{'=' * 45}\n")


# ── Waypoint names in order ──────────────────────────────────
# Press OPTIONS repeatedly — each press saves the next name in this list.
# Add more names if you need more waypoints.
WAYPOINT_NAMES = [
    "1_approach",       # above object, gripper open
    "2_pick_down",      # lowered to object
    "3_gripping",       # gripper closed on object
    "4_lifted",         # raised with object
    "5_place_approach", # above drop location (after base rotation)
    "6_place_down",     # lowered to drop location
    "7_released",       # gripper open at drop location
    "8_return_up",      # raised after releasing
]


def main():
    global stepper_pos

    print("=" * 45)
    print("  AdaptGrip — Waypoint Recorder")
    print("=" * 45)
    print("  Drive arm to each position, then press:")
    print("  OPTIONS (Btn 9) → Save waypoint")
    print("  SHARE   (Btn 8) → Write to file")
    print("  Triangle(Btn 2) → Go home")
    print("  Square  (Btn 3) → Emergency stop")
    print("=" * 45)

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

    # Start arm
    ser.write(b"START\n")
    time.sleep(4)
    stepper_pos = 0
    print("\nSystem ready! Drive arm and press OPTIONS to save waypoints.\n")

    DEAD       = 0.25
    DEAD_LEFT  = 0.35
    STEP       = 3
    SHOULDER_STEP = 4
    GRIP_STEP  = 2

    prev_buttons     = [0] * ctrl.get_numbuttons()
    last_stepper_time = 0
    last_grip_time    = 0
    last_print_time   = 0
    wp_index          = 0   # which waypoint name to use next
    running           = True

    print_angles()   # show starting angles

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

        # ── Square = Emergency Stop ──────────────────────────
        if new[3]:
            print("EMERGENCY STOP!")
            ser.write(b"STOP\n")
            running = False
            break

        # ── Triangle = Home ──────────────────────────────────
        if new[2]:
            go_home(ser)
            print_angles()

        # ── OPTIONS (Btn 9) = Save current waypoint ──────────
        if new[9]:
            if wp_index < len(WAYPOINT_NAMES):
                save_waypoint(WAYPOINT_NAMES[wp_index])
                wp_index += 1
                remaining = len(WAYPOINT_NAMES) - wp_index
                if remaining > 0:
                    print(f"  Next waypoint to save: '{WAYPOINT_NAMES[wp_index]}'")
                    print("  Drive to position and press OPTIONS again.\n")
                else:
                    print("  All waypoints recorded!")
                    print("  Press SHARE (Btn 8) to write to file.\n")
            else:
                print("  All waypoints already saved. Press SHARE to write file.")

        # ── SHARE (Btn 8) = Write waypoints to file ──────────
        if new[8]:
            if waypoints:
                write_waypoints_file()
            else:
                print("  No waypoints saved yet — drive to positions and press OPTIONS first.")

        now = time.time()

        # ── Shoulders ────────────────────────────────────────
        if abs(ly) > DEAD_LEFT:
            send(ser, 'SHOULDER_R', angles['SHOULDER_R'] - (ly * SHOULDER_STEP))
            send(ser, 'SHOULDER_L', angles['SHOULDER_L'] + (ly * SHOULDER_STEP))

        # ── Wrist Roll ───────────────────────────────────────
        if abs(lx) > DEAD_LEFT:
            send(ser, 'WRIST_R', angles['WRIST_R'] + (lx * STEP))

        # ── Elbow ────────────────────────────────────────────
        if abs(ry_fixed - 1.0) > DEAD:
            send(ser, 'ELBOW', angles['ELBOW'] + ((ry_fixed - 1.0) * STEP))

        # ── Wrist Pitch ──────────────────────────────────────
        if abs(rx_fixed - 1.0) > DEAD:
            send(ser, 'WRIST_P', angles['WRIST_P'] - ((rx_fixed - 1.0) * STEP))

        # ── Stepper base ─────────────────────────────────────
        if now - last_stepper_time > 0.06:
            if curr[4]:
                send_stepper(ser, False)
                last_stepper_time = now
            elif curr[5]:
                send_stepper(ser, True)
                last_stepper_time = now

        # ── Gripper ──────────────────────────────────────────
        if now - last_grip_time > 0.04:
            if l2 > 0.1:
                scaled = int(GRIP_STEP * ((l2 + 1.0) / 2.0) * 3) + 1
                send(ser, 'GRIPPER', angles['GRIPPER'] + scaled)
                last_grip_time = now
            elif r2 > 0.1:
                scaled = int(GRIP_STEP * ((r2 + 1.0) / 2.0) * 3) + 1
                send(ser, 'GRIPPER', angles['GRIPPER'] - scaled)
                last_grip_time = now

        # ── Print angles every second ─────────────────────────
        if now - last_print_time > 1.0:
            print_angles()
            if wp_index < len(WAYPOINT_NAMES):
                print(f"  Waypoints saved: {wp_index}/{len(WAYPOINT_NAMES)}")
                print(f"  Next to save   : '{WAYPOINT_NAMES[wp_index]}'")
            last_print_time = now

        prev_buttons = curr
        time.sleep(0.05)

    ser.close()
    pygame.quit()
    print("\nWaypoint recorder stopped.")
    if waypoints and not os.path.exists(SAVE_PATH):
        print("WARNING: You have unsaved waypoints! Run again and press SHARE to save.")


if __name__ == "__main__":
    main()
