import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# hourly collision counts (representative afternoon/evening spike)
hours = np.arange(24)
collisions = np.array([
    800, 600, 450, 380, 320, 400,
    1800, 3200, 4800, 5200, 4900, 4200,
    3800, 3500, 4100, 5600, 7800, 10672,
    9800, 7200, 5400, 3800, 2400, 1400,
])

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 14), facecolor="#f5f5f5")
fig.suptitle("LA28 Venue Collision Risk Analysis", fontsize=32, fontweight="bold", y=0.98)

# polar bar chart (full clock)
ax_polar = fig.add_axes([0.08, 0.05, 0.84, 0.88], projection="polar")
ax_polar.set_facecolor("#f5f5f5")

# clock convention: 12pm at top, clockwise
theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
# rotate so 12am = top, then shift so noon (hour 12) is at top-left
# standard polar: 0=right, CCW. We want 0am at top, going clockwise.
# angle for hour h: theta = pi/2 - h * (2pi/24), mod 2pi
theta_clock = np.pi / 2 - hours * (2 * np.pi / 24)

bar_width = 2 * np.pi / 24 * 0.85

colors = []
for c in collisions:
    if c >= 8000:
        colors.append("#e63946")
    elif c >= 5000:
        colors.append("#ff6b35")
    elif c >= 3000:
        colors.append("#f4a261")
    else:
        colors.append("#8ecae6")

bars = ax_polar.bar(theta_clock, collisions, width=bar_width, bottom=0,
                    color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)

ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.set_rlabel_position(135)
ax_polar.yaxis.set_tick_params(labelsize=13, labelcolor="#555")
ax_polar.set_rlim(0, 12500)
ax_polar.set_rticks([2000, 4000, 6000, 8000, 10000, 12000])
ax_polar.grid(color="#ccc", linestyle="--", linewidth=0.6, alpha=0.7)

hour_labels = ["12am","1am","2am","3am","4am","5am","6am","7am","8am","9am","10am","11am",
               "12pm","1pm","2pm","3pm","4pm","5pm","6pm","7pm","8pm","9pm","10pm","11pm"]
ax_polar.set_xticks(theta_clock)
ax_polar.set_xticklabels(hour_labels, fontsize=13, color="#333")


ax_polar.set_title("", pad=60)

# color legend for polar
from matplotlib.patches import Patch
polar_legend = [
    Patch(color="#e63946", label="≥ 8,000"),
    Patch(color="#ff6b35", label="5,000–8,000"),
    Patch(color="#f4a261", label="3,000–5,000"),
    Patch(color="#8ecae6", label="< 3,000"),
]
ax_polar.legend(handles=polar_legend, title="Collisions / hr", title_fontsize=13,
                loc="lower right", bbox_to_anchor=(1.18, -0.04),
                fontsize=13, framealpha=0.9)

plt.savefig("/Users/narensara/Desktop/OlympiFlow/collision_heatmap.png",
            dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved collision_heatmap.png")
