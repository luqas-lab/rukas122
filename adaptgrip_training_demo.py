# ============================================================
#  AdaptGrip — Live Training Demo (5-minute version)
#  Author  : Muhammad Luqmanul Hakeem bin Ramli
#  Project : AdaptGrip — RL-Based Adaptive Force Control
# ============================================================
#
#  Shows the PPO agent learning adaptive force control LIVE.
#  Graphs update in real-time so judges can watch training progress.
#
#  Runtime: ~2-3 minutes on CPU
#  Run:     python3 adaptgrip_training_demo.py

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETTINGS — tune here
# ============================================================
TOTAL_TIMESTEPS = 50_000   # ~2-3 min on CPU; raise to 100k for better results
EVAL_TRIALS     = 20       # trials per object in final evaluation
CURRICULUM_STAGES = [
    (0,       4, "Stage 1 — Robust objects only"),
    (12_500,  3, "Stage 2 — Adding Medium objects"),
    (25_000,  2, "Stage 3 — Adding Fragile objects"),
    (37_500,  1, "Stage 4 — All objects (Very Fragile included)"),
]

# ============================================================
# OBJECT LIBRARY
# ============================================================

class FragileObject:
    def __init__(self, name, mass, damage_threshold, category):
        self.name             = name
        self.mass             = mass
        self.damage_threshold = damage_threshold
        self.category         = category

    def reset(self):
        pass

OBJECTS = [
    FragileObject("Very Fragile (5N)",  0.05,  5.0, 1),
    FragileObject("Fragile (10N)",      0.10, 10.0, 2),
    FragileObject("Medium (20N)",       0.20, 20.0, 3),
    FragileObject("Robust (40N)",       0.40, 40.0, 4),
    FragileObject("Very Robust (80N)",  0.80, 80.0, 4),
]

def get_objects_for_stage(stage):
    return [o for o in OBJECTS if o.category >= (5 - stage)]

# ============================================================
# ENVIRONMENT
# ============================================================

class GraspEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, stage=4):
        super().__init__()
        self.stage            = stage
        self.available        = get_objects_for_stage(stage)
        self.action_space     = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([100.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0]*5, dtype=np.float32),
            high=np.array([2.0, 50.0, 1.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32)
        self.obj        = None
        self.step_count = 0
        self.max_steps  = 10
        self.prev_force = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obj        = np.random.choice(self.available)
        self.step_count = 0
        self.prev_force = 0.0
        return self._obs(), {}

    def _obs(self):
        req = self.obj.mass * 9.81 * 1.2
        return np.array([
            self.obj.mass,
            req,
            min(self.obj.damage_threshold / 100.0, 1.0),
            self.prev_force,
            self.step_count / self.max_steps
        ], dtype=np.float32)

    def step(self, action):
        force         = float(action[0]) * np.random.uniform(0.85, 1.15)
        self.prev_force = force
        self.step_count += 1
        req     = self.obj.mass * 9.81 * 1.2
        damaged = force > self.obj.damage_threshold
        dropped = force < req * 0.9
        success = not damaged and not dropped
        reward  = self._reward(force, req, damaged, dropped)
        terminated = damaged or dropped or (success and force >= req)
        truncated  = self.step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, {
            'success': success, 'damaged': damaged, 'dropped': dropped,
            'force': force, 'threshold': self.obj.damage_threshold
        }

    def _reward(self, force, req, damaged, dropped):
        if damaged:
            return -200.0 * (1 + (force - self.obj.damage_threshold) / self.obj.damage_threshold)
        if dropped:
            return -100.0 * (1 + (req - force) / req)
        r = 100.0
        if (self.obj.damage_threshold - force) / self.obj.damage_threshold > 0.2:
            r += 20.0
        r += (self.max_steps - self.step_count) * 2.0
        return r

# ============================================================
# LIVE GRAPH CALLBACK
# ============================================================

class LiveCallback(BaseCallback):
    def __init__(self, fig, axes, verbose=0):
        super().__init__(verbose)
        self.fig   = fig
        self.axes  = axes

        # Data buffers
        self.timesteps    = []
        self.rewards      = []
        self.success_rates = []
        self.damage_rates  = []
        self.drop_rates    = []
        self.stage_lines  = []   # (timestep, stage_label)

        self._ep_rewards   = []
        self._ep_successes = []
        self._ep_damages   = []
        self._ep_drops     = []
        self._window       = 50   # rolling average window
        self._current_stage = 4
        self._stage_labels  = {s[1]: s[2] for s in CURRICULUM_STAGES}
        self._eval_model    = None

    def _on_step(self):
        # Curriculum stage transitions
        for ts, stage, label in CURRICULUM_STAGES:
            if self.num_timesteps >= ts and stage != self._current_stage:
                if stage < self._current_stage:
                    self._current_stage = stage
                    self.stage_lines.append((self.num_timesteps, label))
                    for env in self.model.get_env().envs:
                        env.stage     = stage
                        env.available = get_objects_for_stage(stage)
                    print(f"\n  ▶ {label}")

        # Collect episode data from infos
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self._ep_rewards.append(info['episode']['r'])
            if 'success' in info:
                self._ep_successes.append(int(info['success']))
                self._ep_damages.append(int(info['damaged']))
                self._ep_drops.append(int(info['dropped']))

        # Update graphs every 500 steps
        if self.num_timesteps % 500 == 0 and len(self._ep_rewards) >= self._window:
            self.timesteps.append(self.num_timesteps)
            self.rewards.append(np.mean(self._ep_rewards[-self._window:]))
            n = max(len(self._ep_successes), 1)
            w = min(self._window, n)
            self.success_rates.append(np.mean(self._ep_successes[-w:]) * 100)
            self.damage_rates.append( np.mean(self._ep_damages[-w:])   * 100)
            self.drop_rates.append(   np.mean(self._ep_drops[-w:])     * 100)
            self._update_graphs()

        return True

    def _update_graphs(self):
        ts = self.timesteps
        if len(ts) < 2:
            return

        ax_reward, ax_rates, ax_stage, ax_bar = self.axes

        # ── Reward curve ────────────────────────────────────
        ax_reward.cla()
        ax_reward.set_facecolor('#16213e')
        ax_reward.plot(ts, self.rewards, color='#00d2ff', linewidth=2)
        ax_reward.fill_between(ts, self.rewards, alpha=0.15, color='#00d2ff')
        for sl_ts, sl_label in self.stage_lines:
            ax_reward.axvline(sl_ts, color='#f39c12', linestyle='--',
                              alpha=0.6, linewidth=1)
        ax_reward.set_title('Episode Reward (rolling avg)', color='white',
                             fontsize=10, fontweight='bold')
        ax_reward.set_xlabel('Timestep', color='#aaa', fontsize=8)
        ax_reward.set_ylabel('Reward', color='#aaa', fontsize=8)
        ax_reward.tick_params(colors='#aaa', labelsize=7)
        for s in ax_reward.spines.values(): s.set_edgecolor('#444466')

        # ── Success / Damage / Drop rates ───────────────────
        ax_rates.cla()
        ax_rates.set_facecolor('#16213e')
        ax_rates.plot(ts, self.success_rates, color='#2ecc71',
                      linewidth=2, label='Success %')
        ax_rates.plot(ts, self.damage_rates,  color='#e74c3c',
                      linewidth=2, label='Damage %')
        ax_rates.plot(ts, self.drop_rates,    color='#e67e22',
                      linewidth=2, label='Drop %')
        ax_rates.axhline(100, color='#2ecc71', linestyle=':', alpha=0.3)
        for sl_ts, _ in self.stage_lines:
            ax_rates.axvline(sl_ts, color='#f39c12', linestyle='--',
                             alpha=0.6, linewidth=1)
        ax_rates.set_title('Success / Damage / Drop Rate', color='white',
                            fontsize=10, fontweight='bold')
        ax_rates.set_xlabel('Timestep', color='#aaa', fontsize=8)
        ax_rates.set_ylabel('%', color='#aaa', fontsize=8)
        ax_rates.set_ylim(-5, 110)
        ax_rates.legend(fontsize=7, facecolor='#16213e',
                        labelcolor='white', edgecolor='#444466')
        ax_rates.tick_params(colors='#aaa', labelsize=7)
        for s in ax_rates.spines.values(): s.set_edgecolor('#444466')

        # ── Curriculum stage progress bar ───────────────────
        ax_stage.cla()
        ax_stage.set_facecolor('#16213e')
        progress = self.num_timesteps / TOTAL_TIMESTEPS * 100
        stage_colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71']
        stage_names  = [s[2].split('—')[1].strip() for s in CURRICULUM_STAGES]
        stage_ts     = [s[0] for s in CURRICULUM_STAGES]
        for i, (sname, sts) in enumerate(zip(stage_names, stage_ts)):
            pct = sts / TOTAL_TIMESTEPS * 100
            ax_stage.barh(0, 25, left=pct, color=stage_colors[i],
                          alpha=0.7, height=0.5, edgecolor='#222')
            ax_stage.text(pct + 12.5, 0, sname, ha='center', va='center',
                          color='white', fontsize=7, fontweight='bold')
        ax_stage.axvline(progress, color='white', linewidth=2.5)
        ax_stage.text(progress + 0.5, 0.35, f'{progress:.0f}%',
                      color='white', fontsize=9, fontweight='bold')
        ax_stage.set_xlim(0, 100)
        ax_stage.set_ylim(-0.5, 0.7)
        ax_stage.set_title(f'Curriculum Progress  |  '
                           f'Timestep {self.num_timesteps:,} / {TOTAL_TIMESTEPS:,}',
                           color='white', fontsize=10, fontweight='bold')
        ax_stage.set_xlabel('Training progress (%)', color='#aaa', fontsize=8)
        ax_stage.tick_params(colors='#aaa', labelsize=7)
        ax_stage.set_yticks([])
        for s in ax_stage.spines.values(): s.set_edgecolor('#444466')

        # ── Current performance snapshot bar ────────────────
        ax_bar.cla()
        ax_bar.set_facecolor('#16213e')
        latest = {
            'Success': self.success_rates[-1] if self.success_rates else 0,
            'Damage':  self.damage_rates[-1]  if self.damage_rates  else 0,
            'Drop':    self.drop_rates[-1]    if self.drop_rates    else 0,
        }
        bar_colors = ['#2ecc71','#e74c3c','#e67e22']
        bars = ax_bar.bar(list(latest.keys()), list(latest.values()),
                          color=bar_colors, alpha=0.85, edgecolor='#444466')
        for bar, v in zip(bars, latest.values()):
            ax_bar.text(bar.get_x() + bar.get_width()/2, v + 1,
                        f'{v:.0f}%', ha='center', color='white',
                        fontsize=11, fontweight='bold')
        ax_bar.set_title('Current Performance Snapshot', color='white',
                          fontsize=10, fontweight='bold')
        ax_bar.set_ylim(0, 115)
        ax_bar.tick_params(colors='#aaa', labelsize=9)
        for s in ax_bar.spines.values(): s.set_edgecolor('#444466')

        plt.pause(0.001)


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate(model):
    results = []
    COLORS  = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']

    print("\n" + "=" * 65)
    print("  FINAL EVALUATION — 20 trials per object")
    print("=" * 65)

    for obj in OBJECTS:
        successes, damages, drops, forces = 0, 0, 0, []
        force_curve_all = []

        for _ in range(EVAL_TRIALS):
            env = GraspEnv(stage=1)
            env.available = [obj]
            obs, _ = env.reset()
            terminated = truncated = False
            ep_forces = []
            ep_success = ep_damaged = ep_dropped = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                ep_forces.append(info['force'])
                if info['success']: ep_success = True
                if info['damaged']: ep_damaged = True
                if info['dropped']: ep_dropped = True
            env.close()

            if ep_success: successes += 1
            if ep_damaged: damages   += 1
            if ep_dropped: drops     += 1
            forces.append(max(ep_forces))
            force_curve_all.append(ep_forces)

        sr = successes / EVAL_TRIALS * 100
        dr = damages   / EVAL_TRIALS * 100
        rr = drops     / EVAL_TRIALS * 100
        af = float(np.mean(forces))
        sf = float(np.std(forces))
        req = obj.mass * 9.81 * 1.2
        margin = (obj.damage_threshold - af) / obj.damage_threshold * 100

        results.append({
            'obj': obj, 'success': sr, 'damage': dr, 'drop': rr,
            'avg_force': af, 'std_force': sf, 'req_force': req,
            'margin': margin, 'curves': force_curve_all,
        })
        print(f"  {obj.name:<22} | Success:{sr:5.1f}% | "
              f"Damage:{dr:4.1f}% | AvgForce:{af:5.2f}N | Margin:{margin:5.1f}%")

    print("=" * 65)
    print(f"  Overall success : {np.mean([r['success'] for r in results]):.1f}%")
    print(f"  Overall damage  : {np.mean([r['damage']  for r in results]):.1f}%")
    print("=" * 65)
    return results


# ============================================================
# RESULT GRAPHS
# ============================================================

BASELINES = {
    'Fixed-Force':  {'success': 60.0, 'damage': 40.0},
    'Weight-Based': {'success': 55.0, 'damage': 35.0},
    'Compliance':   {'success': 80.0, 'damage': 20.0},
}

def plot_final_results(results):
    COLORS_OBJ  = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
    COLORS_CTRL = ['#95a5a6','#3498db','#e67e22','#27ae60']

    rl_sr = np.mean([r['success'] for r in results])
    rl_dr = np.mean([r['damage']  for r in results])

    # ── 3-panel comparison ────────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('AdaptGrip — Controller Comparison',
                  fontsize=14, fontweight='bold')

    ctrl_names = ['Fixed-Force', 'Weight-Based', 'Compliance', 'RL Agent\n(Proposed)']
    s_vals = [BASELINES[k]['success'] for k in BASELINES] + [rl_sr]
    d_vals = [BASELINES[k]['damage']  for k in BASELINES] + [rl_dr]
    af_vals = [14.8, 15.5, 3.5,
               float(np.mean([r['avg_force'] for r in results]))]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def add_labels(ax, bars, fmt='{:.0f}%'):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    fmt.format(h), ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    b1 = axes[0].bar(ctrl_names, s_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='black')
    axes[0].set_title('Grasp Success Rate', fontweight='bold', fontsize=12)
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel('Success Rate (%)')
    add_labels(axes[0], b1)

    b2 = axes[1].bar(ctrl_names, d_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='black')
    axes[1].set_title('Object Damage Rate', fontweight='bold', fontsize=12)
    axes[1].set_ylim(0, 55)
    axes[1].set_ylabel('Damage Rate (%)')
    add_labels(axes[1], b2)

    b3 = axes[2].bar(ctrl_names, af_vals, color=COLORS_CTRL, alpha=0.85, edgecolor='black')
    axes[2].set_title('Average Applied Force', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Force (N)')
    add_labels(axes[2], b3, fmt='{:.1f}N')

    for ax in axes:
        ax.set_xticklabels(ctrl_names, rotation=20, ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: final_comparison.png")

    # ── Per-object result dashboard ───────────────────────────
    fig2 = plt.figure(figsize=(16, 9))
    fig2.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38)

    def style(ax):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaa', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#444466')

    tkw = dict(color='white', fontsize=10, fontweight='bold')
    lkw = dict(color='#ccc', fontsize=8)
    names_short = ['V.Frag','Frag','Med','Rob','V.Rob']
    x = np.arange(5)

    # Success/Damage/Drop
    ax1 = fig2.add_subplot(gs[0, :2])
    style(ax1)
    w = 0.25
    sr = [r['success'] for r in results]
    dr = [r['damage']  for r in results]
    rr = [r['drop']    for r in results]
    ax1.bar(x - w, sr, w, color='#2ecc71', label='Success %', alpha=0.85)
    ax1.bar(x,     dr, w, color='#e74c3c', label='Damage %',  alpha=0.85)
    ax1.bar(x + w, rr, w, color='#e67e22', label='Drop %',    alpha=0.85)
    for i, v in enumerate(sr):
        ax1.text(i - w, v + 1, f'{v:.0f}%', ha='center', color='white', fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(names_short, color='#aaa')
    ax1.set_title('Success / Damage / Drop Rate per Object', **tkw)
    ax1.set_ylim(0, 115); ax1.set_ylabel('%', **lkw)
    ax1.legend(fontsize=8, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # Safety margin
    ax2 = fig2.add_subplot(gs[0, 2])
    style(ax2)
    margins = [r['margin'] for r in results]
    ax2.barh([r['obj'].name.split(' (')[0] for r in results],
             margins, color=COLORS_OBJ, alpha=0.85, edgecolor='#444466')
    ax2.axvline(20, color='#2ecc71', linestyle='--', alpha=0.6,
                linewidth=1.5, label='20% safe margin')
    for i, v in enumerate(margins):
        ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', color='white', fontsize=8)
    ax2.set_title('Safety Margin\n(% below damage limit)', **tkw)
    ax2.set_xlabel('%', **lkw)
    ax2.tick_params(axis='y', colors='#ccc', labelsize=7)
    ax2.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # Force curves (median trial)
    ax3 = fig2.add_subplot(gs[1, :2])
    style(ax3)
    for i, r in enumerate(results):
        curves    = r['curves']
        med_idx   = np.argsort([max(c) for c in curves])[len(curves)//2]
        curve     = curves[med_idx]
        ax3.plot(range(1, len(curve)+1), curve, color=COLORS_OBJ[i],
                 linewidth=2, marker='o', markersize=4,
                 label=r['obj'].name.split(' (')[0])
        ax3.axhline(r['req_force'], color=COLORS_OBJ[i],
                    linestyle='--', alpha=0.35, linewidth=1)
        ax3.axhline(r['obj'].damage_threshold, color=COLORS_OBJ[i],
                    linestyle=':', alpha=0.2, linewidth=1)
    ax3.set_title('Force Per Step — median trial  '
                  '(dashed=required, dotted=damage limit)', **tkw)
    ax3.set_xlabel('Step', **lkw); ax3.set_ylabel('Force (N)', **lkw)
    ax3.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    # Avg force vs required vs damage
    ax4 = fig2.add_subplot(gs[1, 2])
    style(ax4)
    w2 = 0.25
    af = [r['avg_force'] for r in results]
    rf = [r['req_force'] for r in results]
    df = [r['obj'].damage_threshold for r in results]
    ax4.bar(x - w2, rf, w2, color='#3498db', label='Required', alpha=0.85)
    ax4.bar(x,      af, w2, color='#2ecc71', label='Actual',   alpha=0.85)
    ax4.bar(x + w2, df, w2, color='#e74c3c', label='Damage limit', alpha=0.5)
    ax4.set_xticks(x); ax4.set_xticklabels(names_short, color='#aaa', fontsize=7)
    ax4.set_title('Required vs Actual vs Damage Limit', **tkw)
    ax4.set_ylabel('Force (N)', **lkw)
    ax4.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

    fig2.suptitle(
        'AdaptGrip — RL Evaluation Results  |  Muhammad Luqmanul Hakeem bin Ramli',
        color='white', fontsize=12, fontweight='bold', y=0.99)

    plt.savefig('adaptgrip_results.png', dpi=150, bbox_inches='tight',
                facecolor=fig2.get_facecolor())
    print("  Saved: adaptgrip_results.png")

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("   AdaptGrip — Live Training Demo")
    print("   Muhammad Luqmanul Hakeem bin Ramli")
    print(f"   Training for {TOTAL_TIMESTEPS:,} timesteps (~2-3 min)")
    print("=" * 65)
    print("\nCurriculum stages:")
    for ts, stage, label in CURRICULUM_STAGES:
        print(f"  {ts:>6,} steps → {label}")
    print("\nClose the training window to skip ahead to results.\n")

    # ── Set up live training figure ──────────────────────────
    plt.ion()
    fig, axes_flat = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        'AdaptGrip — PPO Training Live\n'
        'Muhammad Luqmanul Hakeem bin Ramli',
        color='white', fontsize=12, fontweight='bold')
    axes = [axes_flat[0,0], axes_flat[0,1], axes_flat[1,0], axes_flat[1,1]]
    for ax in axes:
        ax.set_facecolor('#16213e')
        for s in ax.spines.values(): s.set_edgecolor('#444466')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.pause(0.5)

    # ── Train ───────────────────────────────────────────────
    env = DummyVecEnv([lambda: Monitor(GraspEnv(stage=4))])
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=512,          # smaller rollout = more frequent updates = livelier graphs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        verbose=0,
    )

    cb = LiveCallback(fig, axes)
    t0 = time.time()
    print("Training started...\n")

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=True)
    except Exception as e:
        print(f"\nTraining interrupted: {e}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    plt.ioff()

    model.save("ppo_force_control_demo")
    print("Model saved: ppo_force_control_demo.zip")

    # ── Evaluate & show results ──────────────────────────────
    results = evaluate(model)
    plot_final_results(results)


if __name__ == "__main__":
    main()
