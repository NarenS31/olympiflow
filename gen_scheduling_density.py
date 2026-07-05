import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sports = [
    "Canoeing", "Wrestling", "Handball", "Sailing", "Judo",
    "Hockey", "Shooting", "Football", "Cycling", "Rowing",
    "Swimming", "Athletics"
]

phases = [
    "Week 1\n(Days 1–4)",
    "Week 2\n(Days 5–8)",
    "Week 3\n(Days 9–12)",
    "Week 4\n(Days 13–16)",
]

data = np.array([
    [16, 29, 25, 12],
    [17, 30, 26, 13],
    [18, 31, 26, 13],
    [19, 33, 28, 14],
    [20, 34, 29, 15],
    [20, 34, 29, 15],
    [20, 34, 29, 15],
    [24, 41, 36, 18],
    [26, 45, 38, 19],
    [27, 48, 41, 20],
    [47, 82, 71, 35],
    [114, 198, 170, 85],
])

fig, ax = plt.subplots(figsize=(18, 14))

data = data.T  # flip so phases are rows, sports are columns

im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(len(sports)))
ax.set_xticklabels(sports, fontsize=22, fontweight="bold", rotation=30, ha="right")
ax.set_yticks(range(len(phases)))
ax.set_yticklabels(phases, fontsize=22, fontweight="bold")

for i in range(len(phases)):
    for j in range(len(sports)):
        val = data[i, j]
        color = "white" if val > 100 else "#333333"
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Athletes per Phase", fontsize=20, fontweight="bold")
cbar.ax.tick_params(labelsize=17)

ax.set_title("Estimated Daily Athlete Load by Sport & Phase",
             fontsize=26, fontweight="bold", pad=20)
ax.set_xlabel("Sport", fontsize=22, fontweight="bold", labelpad=12)
ax.set_ylabel("Games Phase", fontsize=22, fontweight="bold", labelpad=12)

fig.text(0.5, 0.01, "Source: FBLA Athlete Events Dataset",
         ha="center", fontsize=16, color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("scheduling_density.png", dpi=150, bbox_inches="tight")
print("Saved scheduling_density.png")
