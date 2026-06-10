# ============================================================
#  AdaptGrip — RL Results Demo (No Hardware Required)
#  Author  : Muhammad Luqmanul Hakeem bin Ramli
#  Project : AdaptGrip — RL-Based Adaptive Force Control
# ============================================================
#
#  Uses the SAME SimplifiedGraspingEnv from training so results
#  are consistent with the trained model's evaluation.
#
#  Run:  python3 adaptgrip_demo_results.py
#
#  Outputs:
#    - Terminal table (all 5 objects, 20 trials each)
#    - adaptgrip_results.png   (6-panel live dashboard)
#    - final_comparison.png    (3-panel baseline comparison)
#    - detailed_comparison.png (success vs damage grouped bar)
#    - rl_agent_results.csv    (raw numbers)

import numpy as np
import time
import json
import os
import csv
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ────────────────────────────────────────────────────
MODEL_PATH  = '/home/luq/fyp2/ppo_force_control.zip'
CALIB_PATH  = '/home/luq/fyp2/fsr_calibration.txt'

# ── FSR calibration fallback ─────────────────────────────────
FSR_BASELINE_L =   0
FSR_BASELINE_R = 166
FSR_SCALE      = 0.001047

# ============================================================
# OBJECT LIBRARY  (same as training)
# ============================================================

class FragileObject:
    def __init__(self, name, mass, damage_threshold, category):
        self.name            = name
        self.mass            = mass
        self.damage_threshold = damage_threshold
        self.category        = category
        self.is_damaged      = False

    def reset(self):
        self.is_damaged = False

OBJECTS = [
    FragileObject("Very Fragile (5N)",   0.05,  5.0, 1),
    FragileObject("Fragile (10N)",        0.10, 10.0, 2),
    FragileObject("Medium (20N)",         0.20, 20.0, 3),
    FragileObject("Robust (40N)",         0.40, 40.0, 4),
    FragileObject("Very Robust (80N)",    0.80, 80.0, 4),
]

# ============================================================
# RL ENVIRONMENT  (same as training)
# ============================================================

class SimplifiedGraspingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, curriculum_stage=1):
        super().__init__()
        self.curriculum_stage    = curriculum_stage
        self.available_objects   = OBJECTS
        self.action_space        = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([100.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space   = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 50.0, 1.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.current_object = None
        self.step_count     = 0
        self.max_steps      = 10
        self.prev_force     = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_object = np.random.choice(self.available_objects)
        self.current_object.reset()
        self.step_count = 0
        self.prev_force = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        req_force    = self.current_object.mass * 9.81 * 1.2
        fragility    = min(self.current_object.damage_threshold / 100.0, 1.0)
        return np.array([
            self.current_object.mass,
            req_force,
            fragility,
            self.prev_force,
            self.step_count / self.max_steps
        ], dtype=np.float32)

    def step(self, action):
        target_force  = float(action[0])
        contact_force = target_force * np.random.uniform(0.85, 1.15)
        self.prev_force  = contact_force
        self.step_count += 1
        req_force = self.current_object.mass * 9.81 * 1.2
        damaged   = contact_force > self.current_object.damage_threshold
        dropped   = contact_force < req_force * 0.9
        success   = not damaged and not dropped
        reward    = self._reward(contact_force, req_force, damaged, dropped)
        terminated = damaged or dropped or (success and contact_force >= req_force)
        truncated  = self.step_count >= self.max_steps
        info = {
            'success': success, 'damaged': damaged, 'dropped': dropped,
            'contact_force': contact_force,
            'object': self.current_object.name,
            'damage_threshold': self.current_object.damage_threshold
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _reward(self, force, req_force, damaged, dropped):
        if damaged:
            excess = (force - self.current_object.damage_threshold) / self.current_object.damage_threshold
            return -200.0 * (1.0 + excess)
        if dropped:
            deficit = (req_force - force) / req_force
            return -100.0 * (1.0 + deficit)
        reward = 100.0
        if (self.current_object.damage_threshold - force) / self.current_object.damage_threshold > 0.2:
            reward += 20.0
        reward += (self.max_steps - self.step_count) * 2.0
        if force > self.current_object.damage_threshold * 0.8:
            reward -= ((force / self.current_object.damage_threshold) - 0.8) * 10.0
        return reward

# ============================================================
# FSR CALIBRATION
# ============================================================

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

# ============================================================
# EVALUATION  (same method as training — 20 trials per object)
# ============================================================

def evaluate_model(model):
    """Run 20 trials per object, return per-object stats and per-trial logs."""
    all_results  = []
    all_episodes = []   # for graphs

    for obj in OBJECTS:
        successes, damages, drops = 0, 0, 0
        forces, step_counts = [], []
        force_curves = []   # one curve per trial

        for trial in range(20):
            env = SimplifiedGraspingEnv(curriculum_stage=1)
            env.current_object = obj
            obs, _ = env.reset()

            terminated = truncated = False
            ep_success = ep_damaged = ep_dropped = False
            ep_forces  = []

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                ep_forces.append(info['contact_force'])
                if info['success']:  ep_success  = True
                if info['damaged']:  ep_damaged  = True
                if info['dropped']:  ep_dropped  = True
            env.close()

            if ep_success: successes += 1
            if ep_damaged: damages   += 1
            if ep_dropped: drops     += 1
            forces.append(max(ep_forces))
            step_counts.append(len(ep_forces))
            force_curves.append(ep_forces)

        req_force = obj.mass * 9.81 * 1.2
        result = {
            'object':           obj.name,
            'mass':             obj.mass,
            'damage_threshold': obj.damage_threshold,
            'req_force':        req_force,
            'controller':       'RL_Agent',
            'success_rate':     successes / 20 * 100,
            'damage_rate':      damages   / 20 * 100,
            'drop_rate':        drops     / 20 * 100,
            'avg_force':        float(np.mean(forces)),
            'std_force':        float(np.std(forces)),
            'avg_steps':        float(np.mean(step_counts)),
            'force_curves':     force_curves,
        }
        all_results.append(result)

        print(f"  {obj.name:<22s} | "
              f"Success:{result['success_rate']:5.1f}% | "
              f"Damage:{result['damage_rate']:5.1f}% | "
              f"Drop:{result['drop_rate']:5.1f}% | "
              f"AvgForce:{result['avg_force']:5.2f}N ± {result['std_force']:.2f}")

    return all_results


def save_csv(results):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'rl_agent_results.csv')
    keys = ['object','mass','damage_threshold','req_force',
            'success_rate','damage_rate','drop_rate','avg_force','std_force','avg_steps']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})
    print(f"  CSV saved: {path}")


# ============================================================
# BASELINE DATA  (from your training paper / simulation)
# ============================================================

BASELINES = {
    'Fixed-Force':   {'success': 60.0, 'damage': 40.0, 'avg_force': 14.8},
    'Weight-Based':  {'success': 55.0, 'damage': 35.0, 'avg_force': 15.5},
    'Compliance':    {'success': 80.0, 'damage': 20.0, 'avg_force':  3.5},
}

# ============================================================
# TERMINAL TABLE
# ============================================================

def print_summary_table(results):
    print("\n" + "=" * 72)
    print("  RL MODEL EVALUATION RESULTS  (20 trials per object)")
    print("=" * 72)
    print(f"  {'Object':<22} {'Req(N)':>7} {'Limit':>7} "
          f"{'Success':>8} {'Damage':>8} {'Drop':>7} {'Avg F':>8} {'Margin':>8}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")

    passed = 0
    for r in results:
        margin = (r['damage_threshold'] - r['avg_force']) / r['damage_threshold'] * 100
        ok     = r['success_rate'] >= 80 and r['damage_rate'] == 0
        if ok: passed += 1
        flag = '✓' if ok else '~'
        print(f"  {flag} {r['object']:<20} {r['req_force']:>6.2f}N "
              f"{r['damage_threshold']:>6.0f}N "
              f"{r['success_rate']:>7.1f}% {r['damage_rate']:>7.1f}% "
              f"{r['drop_rate']:>6.1f}% {r['avg_force']:>7.2f}N "
              f"{margin:>7.1f}%")

    print(f"  {'─'*72}")
    sr = np.mean([r['success_rate'] for r in results])
    dr = np.mean([r['damage_rate']  for r in results])
    print(f"\n  Overall success rate : {sr:.1f}%")
    print(f"  Overall damage rate  : {dr:.1f}%")
    print("=" * 72)


# ============================================================
# GRAPHS
# ============================================================

COLORS_OBJ  = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
COLORS_CTRL = ['#95a5a6','#3498db','#e67e22','#27ae60']

def plot_dashboard(results, calib):
    """6-panel dark dashboard — force curves, bars, FSR curve, safety margin."""
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('#1a1a2e')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    def style(ax):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor('#444466')

    tkw = dict(color='white', fontsize=10, fontweight='bold', pad=8)
    lkw = dict(color='#cccccc', fontsize=8)

    names = [r['object'].replace(' (','\\n(') for r in results]

    # 1. Mean force per object with error bars
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1)
    means = [r['avg_force'] for r in results]
    stds  = [r['std_force'] for r in results]
    reqs  = [r['req_force'] for r in results]
    dmgs  = [r['damage_threshold'] for r in results]
    x = np.arange(len(results))
    ax1.bar(x, means, color=COLORS_OBJ, alpha=0.85, edgecolor='#444466', zorder=2)
    ax1.errorbar(x, means, yerr=stds, fmt='none', color='white', capsize=4, linewidth=1.5, zorder=3)
    ax1.scatter(x, reqs, marker='_', color='#00d2ff', zorder=4, s=120, label='Required')
    ax1.scatter(x, dmgs, marker='^',  color='#e74c3c', zorder=4, s=60, label='Damage limit')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['V.Frag','Frag','Med','Rob','V.Rob'], color='#aaa', fontsize=7)
    ax1.set_title('Avg Applied Force ± Std Dev', **tkw)
    ax1.set_ylabel('Force (N)', **lkw)
    ax1.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # 2. Success / Damage / Drop rates
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2)
    w = 0.25
    sr = [r['success_rate'] for r in results]
    dr = [r['damage_rate']  for r in results]
    rr = [r['drop_rate']    for r in results]
    ax2.bar(x - w, sr, width=w, color='#2ecc71', label='Success', alpha=0.85)
    ax2.bar(x,     dr, width=w, color='#e74c3c', label='Damage',  alpha=0.85)
    ax2.bar(x + w, rr, width=w, color='#e67e22', label='Drop',    alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['V.Frag','Frag','Med','Rob','V.Rob'], color='#aaa', fontsize=7)
    ax2.set_title('Success / Damage / Drop Rate (%)', **tkw)
    ax2.set_ylabel('%', **lkw)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')
    for i, v in enumerate(sr):
        ax2.text(i - w, v + 1, f'{v:.0f}', ha='center', color='white', fontsize=6)

    # 3. Representative force curve (median trial per object)
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3)
    for i, r in enumerate(results):
        curves = r['force_curves']
        median_idx = np.argsort([max(c) for c in curves])[len(curves)//2]
        curve = curves[median_idx]
        ax3.plot(range(1, len(curve)+1), curve, color=COLORS_OBJ[i],
                 linewidth=2, marker='o', markersize=4,
                 label=r['object'].split(' ')[0]+' '+r['object'].split(' ')[1])
        ax3.axhline(r['req_force'], color=COLORS_OBJ[i],
                    linestyle='--', alpha=0.35, linewidth=1)
    ax3.set_title('Force Progression (median trial)', **tkw)
    ax3.set_xlabel('Step', **lkw)
    ax3.set_ylabel('Force (N)', **lkw)
    ax3.legend(fontsize=6, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # 4. Baseline comparison — success rate
    ax4 = fig.add_subplot(gs[1, 0])
    style(ax4)
    ctrl_names = list(BASELINES.keys()) + ['RL Agent\n(Proposed)']
    rl_sr      = np.mean([r['success_rate'] for r in results])
    s_vals     = [BASELINES[k]['success'] for k in BASELINES] + [rl_sr]
    bars = ax4.bar(ctrl_names, s_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='#444466')
    for bar, v in zip(bars, s_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 1,
                 f'{v:.0f}%', ha='center', color='white', fontsize=9, fontweight='bold')
    ax4.set_title('Success Rate vs Baselines', **tkw)
    ax4.set_ylabel('%', **lkw)
    ax4.set_ylim(0, 115)
    ax4.tick_params(axis='x', colors='#cccccc', labelsize=7)

    # 5. Baseline comparison — damage rate
    ax5 = fig.add_subplot(gs[1, 1])
    style(ax5)
    rl_dr  = np.mean([r['damage_rate'] for r in results])
    d_vals = [BASELINES[k]['damage'] for k in BASELINES] + [rl_dr]
    bars2  = ax5.bar(ctrl_names, d_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='#444466')
    for bar, v in zip(bars2, d_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                 f'{v:.0f}%', ha='center', color='white', fontsize=9, fontweight='bold')
    ax5.set_title('Damage Rate vs Baselines', **tkw)
    ax5.set_ylabel('%', **lkw)
    ax5.set_ylim(0, 50)
    ax5.tick_params(axis='x', colors='#cccccc', labelsize=7)

    # 6. FSR calibration curve
    ax6 = fig.add_subplot(gs[1, 2])
    style(ax6)
    baseline_avg = (FSR_BASELINE_L + FSR_BASELINE_R) / 2.0
    raw_x        = np.linspace(0, 900, 300)
    net_x        = np.maximum(raw_x - baseline_avg, 0)
    force_y      = net_x * FSR_SCALE
    ax6.plot(raw_x, force_y, color='#00d2ff', linewidth=2.5)
    ax6.axvline(baseline_avg, color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Baseline ({baseline_avg:.0f})')
    ax6.axvline(50, color='#f39c12', linestyle=':', linewidth=1.5,
                label='Contact threshold')
    if calib and 'objects' in calib:
        for o in calib['objects']:
            ax6.scatter(o['raw_avg'], o['expected_N'], zorder=5, s=60, color='#ff6b6b')
            ax6.annotate(o['name'], (o['raw_avg'], o['expected_N']),
                         textcoords='offset points', xytext=(4, 4),
                         fontsize=6, color='#ff6b6b')
    ax6.set_title('FSR Calibration Curve', **tkw)
    ax6.set_xlabel('Raw ADC value', **lkw)
    ax6.set_ylabel('Force (N)', **lkw)
    ax6.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    fig.suptitle(
        'AdaptGrip — RL-Based Adaptive Force Control  |  Evaluation Results\n'
        'Muhammad Luqmanul Hakeem bin Ramli',
        color='white', fontsize=13, fontweight='bold', y=0.99
    )

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'adaptgrip_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Dashboard saved: {path}")
    return fig


def plot_comparison(results):
    """Clean 3-panel comparison graph matching your paper format."""
    rl_sr = np.mean([r['success_rate'] for r in results])
    rl_dr = np.mean([r['damage_rate']  for r in results])
    rl_af = np.mean([r['avg_force']    for r in results])

    ctrl_names = ['Fixed-Force', 'Weight-Based', 'Compliance', 'RL Agent']
    s_vals = [BASELINES[k]['success'] for k in BASELINES] + [rl_sr]
    d_vals = [BASELINES[k]['damage']  for k in BASELINES] + [rl_dr]
    f_vals = [BASELINES[k]['avg_force'] for k in BASELINES] + [rl_af]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('AdaptGrip — Controller Performance Comparison',
                 fontsize=14, fontweight='bold')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def label_bars(ax, bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f'{h:.0f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    b1 = axes[0].bar(ctrl_names, s_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='black')
    axes[0].set_title('Grasp Success Rate (%)', fontweight='bold')
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel('Success Rate (%)')
    label_bars(axes[0], b1)
    for i, v in enumerate(s_vals):
        axes[0].text(b1[i].get_x() + b1[i].get_width()/2, v + 1,
                     f'{v:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    b2 = axes[1].bar(ctrl_names, d_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='black')
    axes[1].set_title('Object Damage Rate (%)', fontweight='bold')
    axes[1].set_ylim(0, 55)
    axes[1].set_ylabel('Damage Rate (%)')
    for i, v in enumerate(d_vals):
        axes[1].text(b2[i].get_x() + b2[i].get_width()/2, v + 0.5,
                     f'{v:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    b3 = axes[2].bar(ctrl_names, f_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='black')
    axes[2].set_title('Average Applied Force (N)', fontweight='bold')
    axes[2].set_ylabel('Force (N)')
    for i, v in enumerate(f_vals):
        axes[2].text(b3[i].get_x() + b3[i].get_width()/2, v + 0.1,
                     f'{v:.1f}N', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for ax in axes:
        ax.set_xticklabels(ctrl_names, rotation=30, ha='right', fontsize=9)

    plt.tight_layout()
    p1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_comparison.png')
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    print(f"  Comparison saved: {p1}")

    # Detailed grouped bar
    fig2, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ctrl_names))
    w = 0.35
    b4 = ax.bar(x - w/2, s_vals, w, label='Success Rate', color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.5)
    b5 = ax.bar(x + w/2, d_vals, w, label='Damage Rate',  color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    for bars in [b4, b5]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Controller Performance: Success vs Damage Rate',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ctrl_names, fontsize=11)
    ax.set_ylabel('Rate (%)')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detailed_comparison.png')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    print(f"  Detailed comparison saved: {p2}")

    return fig2


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(42)

    print("\n" + "=" * 72)
    print("   AdaptGrip — RL Evaluation Demo")
    print("   Muhammad Luqmanul Hakeem bin Ramli")
    print("=" * 72)

    # Load FSR calibration
    calib = load_calibration()
    if calib:
        print(f"\nFSR calibration loaded — "
              f"L={FSR_BASELINE_L} R={FSR_BASELINE_R} scale={FSR_SCALE:.6f}")
    else:
        print("\nFSR calibration: using hardcoded values")

    # Load model
    print(f"\nLoading PPO model from {MODEL_PATH} ...")
    try:
        from stable_baselines3 import PPO
        try:
            model = PPO.load(MODEL_PATH, device='cuda')
            print("  Loaded on GPU ✓")
        except Exception:
            model = PPO.load(MODEL_PATH, device='cpu')
            print("  Loaded on CPU ✓")
    except Exception as e:
        print(f"  Model load failed: {e}")
        return

    # Evaluate
    print("\n" + "=" * 72)
    print("  RUNNING EVALUATION — 20 trials per object")
    print("=" * 72)
    results = evaluate_model(model)

    # Save CSV
    save_csv(results)

    # Print table
    print_summary_table(results)

    # Generate graphs
    print("\nGenerating graphs...")
    fig1 = plot_dashboard(results, calib)
    fig2 = plot_comparison(results)

    print("\n" + "=" * 72)
    print("  Files saved:")
    print("    rl_agent_results.csv")
    print("    adaptgrip_results.png   (6-panel dashboard)")
    print("    final_comparison.png    (3-panel baseline comparison)")
    print("    detailed_comparison.png (success vs damage)")
    print("=" * 72)

    plt.show()


if __name__ == "__main__":
    main()
