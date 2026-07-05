import matplotlib.pyplot as plt

venues = ["Rose Bowl", "SoFi", "Crypto.com", "Long Beach", "Coliseum"]

athlete  = [16.0, 15.5, 15.0, 14.5, 31.2]
collision = [7.5,  8.5,  9.5,  9.5, 26.5]
transit  = [10.4, 9.9,  10.5, 11.3, 20.4]
totals   = [33.9, 33.9, 35.0, 35.3, 78.1]

fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(top=0.82)  # room for both title lines

colors = ["#5B8FD4", "#F0944D", "#7B5EA7"]

bars_a = ax.barh(venues, athlete,  color=colors[0], label="Athlete Volume (40%)")
bars_c = ax.barh(venues, collision, left=athlete, color=colors[1], label="Collision Density (30%)")
bars_t = ax.barh(venues, transit,
                 left=[a + c for a, c in zip(athlete, collision)],
                 color=colors[2], label="Transit Gap (30%)")

for i, (total, venue) in enumerate(zip(totals, venues)):
    color = "#CC0000" if total >= 50 else "#2E7D32"
    ax.text(total + 0.8, i, f"{total}", va="center", fontsize=12,
            fontweight="bold", color=color)

ax.axvline(50, color="#E05C7A", linestyle="--", linewidth=1.5, label="Threshold (50)")
ax.text(50.5, len(venues) - 0.5, "Threshold (50)", color="#E05C7A", fontsize=9)


ax.set_xlabel("Composite Risk Score", fontsize=11)
ax.set_xlim(0, 95)
ax.legend(loc="lower right", fontsize=9)


ax.set_title("Composite Venue Risk Score", fontsize=16, fontweight="bold", pad=28)

ax.text(1.0, -0.1,
        "Source: LAPD Crime Data / FBLA Athlete Events / DASH Routes",
        transform=ax.transAxes, ha="right", fontsize=8, color="#888888")

plt.tight_layout()
plt.savefig("venue_risk_scores.png", dpi=150, bbox_inches="tight")
print("Saved venue_risk_scores.png")
