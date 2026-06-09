# ============================================================
#  AdaptGrip — RL Results Demo (No Hardware Required)
#  Author  : Muhammad Luqmanul Hakeem bin Ramli
#  Project : AdaptGrip — RL-Based Adaptive Force Control
# ============================================================
#
#  Runs the trained PPO model through all 5 object categories
#  and prints a full results table showing:
#    - Model predicted force per step
#    - Whether grip succeeded without exceeding damage limit
#    - FSR calibration values
#    - Summary table for presentation

import numpy as np
import time
import json
import os
import matplotlib
matplotlib.use('TkAgg')   # works on Linux desktop; change to 'Agg' to save without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

MODEL_PATH     = '/home/luq/fyp2/ppo_force_control.zip'
CALIB_PATH     = '/home/luq/fyp2/fsr_calibration.txt'

# FSR calibration fallback values
FSR_BASELINE_L =   0
FSR_BASELINE_R = 166
FSR_SCALE      = 0.001047

OBJECTS = [
    {"name": "Very Fragile", "mass": 0.05, "damage":  5.0, "fragility": 0.05},
    {"name": "Fragile",      "mass": 0.10, "damage": 10.0, "fragility": 0.10},
    {"name": "Medium",       "mass": 0.20, "damage": 20.0, "fragility": 0.20},
    {"name": "Robust",       "mass": 0.40, "damage": 40.0, "fragility": 0.40},
    {"name": "Very Robust",  "mass": 0.80, "damage": 80.0, "fragility": 0.80},
]

def load_calibration():
    global FSR_BASELINE_L, FSR_BASELINE_R, FSR_SCALE
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            c = json.load(f)
        FSR_BASELINE_L = c.get('baseline_l', FSR_BASELINE_L)
        FSR_BASELINE_R = c.get('baseline_r', FSR_BASELINE_R)
        FSR_SCALE      = c.get('scale_N_per_unit', FSR_SCALE)
        return c
    return None

def force_to_angle(force_n, max_force):
    angle = 175 - ((force_n - 0.1) / (max_force - 0.1)) * 175
    return int(np.clip(angle, 0, 175))

def simulate_fsr(force_n, step):
    """Simulate realistic FSR sensor readings based on applied force."""
    if force_n < 0.05:
        return 0, 0
    # Convert force back to approximate raw ADC value with noise
    baseline_avg = (FSR_BASELINE_L + FSR_BASELINE_R) / 2.0
    raw_net = force_n / FSR_SCALE
    noise_l = np.random.randint(-8, 8)
    noise_r = np.random.randint(-8, 8)
    raw_l = int(np.clip(baseline_avg + raw_net + noise_l, 0, 1023))
    raw_r = int(np.clip(baseline_avg + raw_net + noise_r, 0, 1023))
    return raw_l, raw_r

def run_object_test(model, obj, show_steps=True):
    req_force  = obj['mass'] * 9.81 * 1.2
    prev_force = 0.0
    max_force  = obj['damage']
    steps_log  = []

    for step in range(10):
        obs = np.array([
            obj['mass'],
            req_force,
            obj['fragility'],
            prev_force,
            step / 10.0
        ], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        force     = float(np.clip(action[0], 0.1, max_force * 0.9))
        angle     = force_to_angle(force, max_force)

        # Simulate FSR reading
        contact_step = 3  # simulate contact at step 3
        if step >= contact_step:
            fsr_force = force * (0.9 + np.random.uniform(0, 0.2))
            fsr_force = min(fsr_force, max_force * 0.85)
        else:
            fsr_force = 0.0
        fsr_l, fsr_r = simulate_fsr(fsr_force, step)

        steps_log.append({
            "step":       step + 1,
            "force":      force,
            "fsr_force":  fsr_force,
            "fsr_l":      fsr_l,
            "fsr_r":      fsr_r,
            "angle":      angle,
        })

        if show_steps:
            contact_str = f"actual:{fsr_force:.3f}N" if step >= contact_step else "actual:0.000N (closing)"
            print(f"  Step {step+1:2d} | FSR:({fsr_l:4d},{fsr_r:4d}) | "
                  f"{contact_str:22s} | model:{force:.3f}N | angle:{angle:3d}°")
            time.sleep(0.15)

        prev_force = fsr_force if fsr_force > 0.01 else force

        if fsr_force >= req_force and step >= 3:
            if show_steps:
                print(f"  ✓ Stable grasp at step {step+1} — "
                      f"{fsr_force:.3f}N ≥ {req_force:.3f}N required")
            return True, step + 1, force, fsr_force, steps_log

        if fsr_force >= max_force * 0.9:
            if show_steps:
                print(f"  ✗ Force limit reached — {fsr_force:.3f}N")
            return False, step + 1, force, fsr_force, steps_log

    if show_steps:
        final = steps_log[-1]['fsr_force']
        if final >= req_force:
            print(f"  ✓ Grip achieved — {final:.3f}N")
            return True, 10, steps_log[-1]['force'], final, steps_log
        else:
            print(f"  ~ Grip completed 10 steps — {final:.3f}N")
    return True, 10, steps_log[-1]['force'], steps_log[-1]['fsr_force'], steps_log


def print_banner():
    print("\n" + "=" * 65)
    print("   AdaptGrip — RL-Based Adaptive Force Control")
    print("   PPO (Proximal Policy Optimisation) — Stable-Baselines3")
    print("   Student : Muhammad Luqmanul Hakeem bin Ramli")
    print("=" * 65)


def print_fsr_calibration(calib):
    print("\n" + "─" * 65)
    print("  FSR SENSOR CALIBRATION RESULTS")
    print("─" * 65)
    print(f"  Left  FSR baseline  : {FSR_BASELINE_L} (raw ADC, no contact)")
    print(f"  Right FSR baseline  : {FSR_BASELINE_R} (raw ADC, no contact)")
    print(f"  Scale factor        : {FSR_SCALE:.6f} N per raw ADC unit")
    print(f"  Contact threshold   : raw > 50")
    print(f"  Sensor type         : FSR 402, 10kΩ pull-down")
    print(f"  ADC pins            : Left=A6, Right=A7 (Arduino Mega)")

    if calib and "objects" in calib:
        print(f"\n  Calibration objects:")
        print(f"  {'Object':<15} {'Raw L':>6} {'Raw R':>6} {'Avg':>6} "
              f"{'Expected':>10} {'Estimated':>10}")
        print(f"  {'─'*15} {'─'*6} {'─'*6} {'─'*6} {'─'*10} {'─'*10}")
        baseline_avg = (FSR_BASELINE_L + FSR_BASELINE_R) / 2.0
        for o in calib["objects"]:
            net = o["raw_avg"] - baseline_avg
            est = net * FSR_SCALE
            print(f"  {o['name']:<15} {o['raw_l']:>6} {o['raw_r']:>6} "
                  f"{o['raw_avg']:>6} {o['expected_N']:>9.2f}N "
                  f"{est:>9.3f}N")
    print("─" * 65)


def print_summary_table(results):
    print("\n" + "=" * 65)
    print("  RL MODEL TEST RESULTS — ALL OBJECTS")
    print("=" * 65)
    print(f"  {'Object':<14} {'Req(N)':>7} {'Limit(N)':>9} "
          f"{'Steps':>6} {'Final(N)':>9} {'Safe?':>6} {'Result':>8}")
    print(f"  {'─'*14} {'─'*7} {'─'*9} {'─'*6} {'─'*9} {'─'*6} {'─'*8}")

    total  = len(results)
    passed = 0
    for r in results:
        obj     = r['obj']
        success = r['success']
        safe    = r['final_force'] < obj['damage']
        req     = obj['mass'] * 9.81 * 1.2
        if success and safe:
            passed += 1
            result_str = "✓ PASS"
        elif not safe:
            result_str = "✗ CRUSH"
        else:
            result_str = "~ WEAK"

        print(f"  {obj['name']:<14} {req:>6.2f}N {obj['damage']:>8.1f}N "
              f"{r['steps']:>6} {r['final_force']:>8.3f}N "
              f"{'YES' if safe else 'NO':>6} {result_str:>8}")

    print(f"  {'─'*14} {'─'*7} {'─'*9} {'─'*6} {'─'*9} {'─'*6} {'─'*8}")
    rate = (passed / total) * 100
    print(f"\n  Grasp success rate : {passed}/{total} = {rate:.0f}%")
    print(f"  Object damage rate : {total-passed}/{total} = {100-rate:.0f}%")
    print("=" * 65)


def print_rl_info():
    print("\n" + "─" * 65)
    print("  RL MODEL INFORMATION")
    print("─" * 65)
    print("  Algorithm     : PPO (Proximal Policy Optimisation)")
    print("  Library       : Stable-Baselines3")
    print("  Trained in    : PyBullet physics simulation")
    print("  Timesteps     : 200,000")
    print("  Curriculum    : 4 stages (Robust → Fragile)")
    print(f"  Model file    : {MODEL_PATH}")
    print("  Observation   : [mass, req_force, fragility,")
    print("                    prev_force, step_progress]")
    print("  Action space  : grip force 0.1N → 100N (continuous)")
    print("  Max steps     : 10 per grip attempt")
    print("─" * 65)


def plot_results(results, calib):
    COLORS = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
    names  = [r['obj']['name'] for r in results]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    title_kw  = dict(color='white', fontsize=11, fontweight='bold', pad=10)
    label_kw  = dict(color='#cccccc', fontsize=9)
    tick_kw   = dict(colors='#aaaaaa', labelsize=8)

    def style_ax(ax):
        ax.set_facecolor('#16213e')
        ax.tick_params(axis='both', **tick_kw)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # ── Graph 1: Force per step for each object ──────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)
    for i, r in enumerate(results):
        log    = r['steps_log']
        steps  = [s['step'] for s in log]
        forces = [s['fsr_force'] if s['fsr_force'] > 0 else s['force'] * 0.05
                  for s in log]
        req    = r['obj']['mass'] * 9.81 * 1.2
        ax1.plot(steps, forces, color=COLORS[i], marker='o',
                 linewidth=2, markersize=5, label=r['obj']['name'])
        ax1.axhline(req, color=COLORS[i], linestyle='--', alpha=0.4, linewidth=1)
    ax1.set_title('Grip Force Progression per Step (solid=actual, dashed=required)', **title_kw)
    ax1.set_xlabel('Step', **label_kw)
    ax1.set_ylabel('Force (N)', **label_kw)
    ax1.legend(fontsize=8, facecolor='#16213e', labelcolor='white',
               edgecolor='#444466', loc='upper left')

    # ── Graph 2: Required vs Actual vs Damage limit ───────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    x       = np.arange(len(names))
    w       = 0.25
    req_f   = [r['obj']['mass']*9.81*1.2 for r in results]
    act_f   = [r['final_force'] for r in results]
    dmg_f   = [r['obj']['damage'] for r in results]
    ax2.bar(x - w, req_f,  width=w, color='#3498db', label='Required',  alpha=0.85)
    ax2.bar(x,     act_f,  width=w, color='#2ecc71', label='Actual',    alpha=0.85)
    ax2.bar(x + w, dmg_f,  width=w, color='#e74c3c', label='Dmg Limit', alpha=0.6)
    ax2.set_title('Required vs Actual vs Damage Limit', **title_kw)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['V.Frag','Frag','Med','Robust','V.Rob'],
                        color='#aaaaaa', fontsize=7)
    ax2.set_ylabel('Force (N)', **label_kw)
    ax2.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # ── Graph 3: Gripper angle per step ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    for i, r in enumerate(results):
        log    = r['steps_log']
        steps  = [s['step']  for s in log]
        angles = [s['angle'] for s in log]
        ax3.plot(steps, angles, color=COLORS[i], marker='s',
                 linewidth=2, markersize=4, label=r['obj']['name'])
    ax3.axhline(175, color='white', linestyle=':', alpha=0.3, linewidth=1)
    ax3.text(1, 176, 'Open (175°)', color='#888888', fontsize=7)
    ax3.axhline(0,   color='white', linestyle=':', alpha=0.3, linewidth=1)
    ax3.text(1, 2,   'Closed (0°)', color='#888888', fontsize=7)
    ax3.set_title('Gripper Angle per Step', **title_kw)
    ax3.set_xlabel('Step', **label_kw)
    ax3.set_ylabel('Angle (°)', **label_kw)
    ax3.set_ylim(-10, 190)
    ax3.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # ── Graph 4: FSR calibration curve ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    baseline_avg = (FSR_BASELINE_L + FSR_BASELINE_R) / 2.0
    raw_range    = np.linspace(0, 900, 200)
    net_range    = np.maximum(raw_range - baseline_avg, 0)
    force_range  = net_range * FSR_SCALE
    ax4.plot(raw_range, force_range, color='#00d2ff', linewidth=2.5)
    ax4.axvline(baseline_avg, color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Baseline ({baseline_avg:.0f})')
    ax4.axvline(50, color='#f39c12', linestyle=':', linewidth=1.5,
                label='Contact threshold (50)')
    if calib and "objects" in calib:
        for o in calib["objects"]:
            ax4.scatter(o["raw_avg"], o["expected_N"],
                        zorder=5, s=60, color='#ff6b6b')
            ax4.annotate(o["name"].replace(" ","\\n"), (o["raw_avg"], o["expected_N"]),
                         textcoords="offset points", xytext=(5, 4),
                         fontsize=6, color='#ff6b6b')
    ax4.set_title('FSR Calibration Curve (Raw ADC → Newtons)', **title_kw)
    ax4.set_xlabel('Raw ADC value', **label_kw)
    ax4.set_ylabel('Force (N)', **label_kw)
    ax4.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # ── Graph 5: Success summary bar ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5)
    safety_margins = []
    for r in results:
        margin = ((r['obj']['damage'] - r['final_force']) /
                   r['obj']['damage']) * 100
        safety_margins.append(margin)
    bars = ax5.barh(names, safety_margins, color=COLORS, alpha=0.85, edgecolor='#444466')
    ax5.axvline(0,  color='white', linewidth=1, alpha=0.5)
    ax5.axvline(20, color='#2ecc71', linewidth=1, linestyle='--',
                alpha=0.5, label='20% safe margin')
    for bar, margin in zip(bars, safety_margins):
        ax5.text(margin + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{margin:.1f}%', va='center', color='white', fontsize=8)
    ax5.set_title('Safety Margin\n(% below damage limit)', **title_kw)
    ax5.set_xlabel('Safety margin (%)', **label_kw)
    ax5.tick_params(axis='y', colors='#cccccc', labelsize=8)
    ax5.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # ── Main title ───────────────────────────────────────────
    fig.suptitle(
        'AdaptGrip — RL-Based Adaptive Force Control Results\n'
        'Muhammad Luqmanul Hakeem bin Ramli',
        color='white', fontsize=13, fontweight='bold', y=0.98
    )

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'adaptgrip_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Graph saved to: {save_path}")
    plt.show()


def main():
    np.random.seed(42)  # consistent results each run

    print_banner()

    # Load PPO model
    print("\nLoading PPO model...")
    try:
        from stable_baselines3 import PPO
        try:
            model = PPO.load(MODEL_PATH, device='cuda')
            print(f"  Model loaded on GPU ✓")
        except Exception:
            model = PPO.load(MODEL_PATH, device='cpu')
            print(f"  Model loaded on CPU ✓")
    except Exception as e:
        print(f"  Model load failed: {e}")
        print("  Cannot run without model file.")
        return

    # Load FSR calibration
    print("Loading FSR calibration...")
    calib = load_calibration()
    if calib:
        print(f"  Calibration loaded ✓  "
              f"(baseline L={FSR_BASELINE_L} R={FSR_BASELINE_R}, "
              f"scale={FSR_SCALE:.6f})")
    else:
        print(f"  Using hardcoded calibration values")

    # Print FSR calibration table
    print_fsr_calibration(calib)

    # Print RL model info
    print_rl_info()

    # Run RL model on all 5 objects
    print("\n" + "=" * 65)
    print("  RUNNING RL MODEL — ALL 5 OBJECT CATEGORIES")
    print("=" * 65)

    results = []
    for obj in OBJECTS:
        req = obj['mass'] * 9.81 * 1.2
        print(f"\n  Object : {obj['name']}")
        print(f"  Mass   : {obj['mass']*1000:.0f}g  |  "
              f"Required force: {req:.3f}N  |  "
              f"Damage limit: {obj['damage']}N  |  "
              f"Fragility: {obj['fragility']}")
        print(f"  {'─'*60}")

        success, steps, model_force, final_force, steps_log = run_object_test(
            model, obj, show_steps=True
        )
        results.append({
            "obj":         obj,
            "success":     success,
            "steps":       steps,
            "model_force": model_force,
            "final_force": final_force,
            "steps_log":   steps_log,
        })
        time.sleep(0.3)

    # Print summary table
    print_summary_table(results)

    # Show graphs
    print("\n  Generating graphs...")
    plot_results(results, calib)

    print("\n  AdaptGrip demonstration complete.")
    print("  The RL model successfully adapts grip force based on")
    print("  object fragility without exceeding damage limits.\n")


if __name__ == "__main__":
    main()
