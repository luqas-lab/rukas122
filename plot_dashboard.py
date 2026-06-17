import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_PATH = "adaptgrip_results_20260617_145601.csv"
CATEGORY_ORDER = ["Very Fragile", "Fragile", "Medium", "Robust", "Very Robust"]
NAMES_SHORT = ["V.Frag", "Frag", "Med", "Rob", "V.Rob"]
COLORS_OBJ = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.append(r)

stats = defaultdict(lambda: {"success": 0, "fail": 0, "forces": []})
for r in rows:
    cat = r["object_name"]
    stats[cat]["forces"].append(float(r["actual_force_N"]))
    stats[cat]["success" if r["outcome"] == "SUCCESS" else "fail"] += 1

req_map = {r["object_name"]: float(r["required_force_N"]) for r in rows}
dmg_map = {r["object_name"]: float(r["damage_limit_N"]) for r in rows}

success_rates, fail_rates, avg_forces, std_forces = [], [], [], []
required_forces, damage_limits = [], []
for cat in CATEGORY_ORDER:
    s, f = stats[cat]["success"], stats[cat]["fail"]
    total = s + f
    success_rates.append(100 * s / total)
    fail_rates.append(100 * f / total)
    forces = stats[cat]["forces"]
    avg_forces.append(np.mean(forces))
    std_forces.append(np.std(forces))
    required_forces.append(req_map[cat])
    damage_limits.append(dmg_map[cat])

overall_success = np.mean(success_rates)
overall_fail = np.mean(fail_rates)

def style(ax):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#aaa', labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor('#444466')

tkw = dict(color='white', fontsize=10, fontweight='bold')
lkw = dict(color='#ccc', fontsize=8)
x = np.arange(5)

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('AdaptGrip — Real-World Grasp Test Results',
             color='white', fontsize=13, fontweight='bold')

# ── Panel 1: Success rate per category ──────────────────────
ax1 = fig.add_subplot(2, 2, 1)
style(ax1)
bars = ax1.bar(NAMES_SHORT, success_rates, color=COLORS_OBJ, alpha=0.9, edgecolor='#444466')
for bar, rate in zip(bars, success_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, rate + 2, f"{rate:.0f}%",
              ha='center', color='white', fontsize=9, fontweight='bold')
ax1.axhline(overall_success, color='#2ecc71', linestyle=':', alpha=0.6, linewidth=1.5)
ax1.text(4.4, overall_success + 2, f"Avg: {overall_success:.0f}%",
          color='#2ecc71', fontsize=8, ha='right')
ax1.set_title('Success Rate by Category', **tkw)
ax1.set_ylabel('%', **lkw)
ax1.set_ylim(0, 115)

# ── Panel 2: Success vs Fail rate ───────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
style(ax2)
w = 0.3
ax2.bar(x - w/2, success_rates, w, color='#2ecc71', label='Success %', alpha=0.9)
ax2.bar(x + w/2, fail_rates, w, color='#e74c3c', label='Fail %', alpha=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels(NAMES_SHORT, color='#aaa')
ax2.set_title('Success / Fail Rate per Category', **tkw)
ax2.set_ylim(0, 115)
ax2.set_ylabel('%', **lkw)
ax2.legend(fontsize=8, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

# ── Panel 3: Required vs Actual vs Damage limit (log scale) ─
ax3 = fig.add_subplot(2, 2, 3)
style(ax3)
w2 = 0.25
ax3.bar(x - w2, required_forces, w2, color='#3498db', label='Required', alpha=0.9)
ax3.bar(x, avg_forces, w2, yerr=std_forces, color='#9b59b6', label='Actual (FSR)', alpha=0.9, capsize=3)
ax3.bar(x + w2, damage_limits, w2, color='#e74c3c', label='Damage limit', alpha=0.6)
ax3.set_yscale('log')
ax3.set_xticks(x)
ax3.set_xticklabels(NAMES_SHORT, color='#aaa')
ax3.set_title('Required vs Actual vs Damage Limit (log N)', **tkw)
ax3.set_ylabel('Force (N)', **lkw)
ax3.legend(fontsize=7, facecolor='#16213e', labelcolor='white', edgecolor='#444466')

# ── Panel 4: Overall performance snapshot ───────────────────
ax4 = fig.add_subplot(2, 2, 4)
style(ax4)
overall = {'Success': overall_success, 'Fail': overall_fail}
bars4 = ax4.bar(list(overall.keys()), list(overall.values()),
                 color=['#2ecc71', '#e74c3c'], alpha=0.9, edgecolor='#444466')
for bar, v in zip(bars4, overall.values()):
    ax4.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.0f}%",
              ha='center', color='white', fontsize=11, fontweight='bold')
ax4.set_title('Overall Performance Snapshot (100 trials)', **tkw)
ax4.set_ylim(0, 115)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("results_dashboard.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved: results_dashboard.png")
