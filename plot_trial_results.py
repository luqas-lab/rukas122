import csv
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_PATH = "adaptgrip_results_20260617_145601.csv"
CATEGORY_ORDER = ["Very Fragile", "Fragile", "Medium", "Robust", "Very Robust"]

rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

stats = defaultdict(lambda: {"success": 0, "fail": 0, "forces": []})
for r in rows:
    cat = r["object_name"]
    stats[cat]["forces"].append(float(r["actual_force_N"]))
    if r["outcome"] == "SUCCESS":
        stats[cat]["success"] += 1
    else:
        stats[cat]["fail"] += 1

success_rates = []
fail_counts = []
success_counts = []
avg_forces = []
required_forces = []
damage_limits = []

for cat in CATEGORY_ORDER:
    s = stats[cat]["success"]
    f = stats[cat]["fail"]
    total = s + f
    success_rates.append(100 * s / total)
    success_counts.append(s)
    fail_counts.append(f)
    avg_forces.append(sum(stats[cat]["forces"]) / len(stats[cat]["forces"]))

req_map = {r["object_name"]: float(r["required_force_N"]) for r in rows}
dmg_map = {r["object_name"]: float(r["damage_limit_N"]) for r in rows}
required_forces = [req_map[c] for c in CATEGORY_ORDER]
damage_limits = [dmg_map[c] for c in CATEGORY_ORDER]

overall_success = sum(success_counts)
overall_total = sum(success_counts) + sum(fail_counts)
overall_rate = 100 * overall_success / overall_total

# ── Figure 1: Success Rate Bar Chart ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
bars = ax.bar(CATEGORY_ORDER, success_rates, color=colors, edgecolor='black', linewidth=0.8)

for bar, rate, s, total in zip(bars, success_rates, success_counts, [s + f for s, f in zip(success_counts, fail_counts)]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{rate:.0f}%\n({s}/{total})", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(overall_rate, color='black', linestyle='--', linewidth=1, alpha=0.6)
ax.text(len(CATEGORY_ORDER) - 0.5, overall_rate + 1.5, f"Overall: {overall_rate:.0f}%",
        ha='right', fontsize=9, style='italic')

ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("AdaptGrip — Real-World Grasp Success Rate by Object Category\n(20 trials per category, 100 total)",
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results_success_rate.png", dpi=150)
plt.close()

# ── Figure 2: Force comparison (Required vs Damage Limit vs Actual avg) ──
fig, ax = plt.subplots(figsize=(9.5, 5.5))
x = range(len(CATEGORY_ORDER))
width = 0.25

ax.bar([i - width for i in x], required_forces, width, label='Required Force', color='#3498db')
ax.bar(x, avg_forces, width, label='Avg Actual Force (FSR)', color='#9b59b6')
ax.bar([i + width for i in x], damage_limits, width, label='Damage Limit', color='#e74c3c')

ax.set_yscale('log')
ax.set_xticks(list(x))
ax.set_xticklabels(CATEGORY_ORDER)
ax.set_ylabel("Force (N, log scale)", fontsize=12)
ax.set_title("Required vs Measured vs Damage-Limit Force per Category", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3, which='both')
plt.tight_layout()
plt.savefig("results_force_comparison.png", dpi=150)
plt.close()

# ── Figure 3: Success/Fail stacked bar ────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(CATEGORY_ORDER, success_counts, label='Success', color='#2ecc71', edgecolor='black')
ax.bar(CATEGORY_ORDER, fail_counts, bottom=success_counts, label='Fail', color='#e74c3c', edgecolor='black')

for i, cat in enumerate(CATEGORY_ORDER):
    ax.text(i, success_counts[i] / 2, str(success_counts[i]), ha='center', va='center', color='white', fontweight='bold')
    ax.text(i, success_counts[i] + fail_counts[i] / 2, str(fail_counts[i]), ha='center', va='center', color='white', fontweight='bold')

ax.set_ylabel("Number of Trials", fontsize=12)
ax.set_title("Success vs Fail Count per Category (out of 20 trials each)", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results_success_fail_count.png", dpi=150)
plt.close()

print("Saved: results_success_rate.png, results_force_comparison.png, results_success_fail_count.png")
print(f"\nOverall success rate: {overall_rate:.1f}% ({overall_success}/{overall_total})")
for cat, rate, s, f in zip(CATEGORY_ORDER, success_rates, success_counts, fail_counts):
    print(f"  {cat:15s} {rate:5.1f}%  ({s} success, {f} fail)")
