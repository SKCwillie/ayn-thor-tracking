import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DB_PATH = "shipping_info.db"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    date,
    make,
    model,
    color,
    begin,
    end,
    units_shipped
FROM shipments
ORDER BY date
"""

df = pd.read_sql(query, conn)
conn.close()

df["date"] = pd.to_datetime(df["date"])
df["begin"] = pd.to_numeric(df["begin"], errors="coerce")
df["end"] = pd.to_numeric(df["end"], errors="coerce")
df["units_shipped"] = pd.to_numeric(df["units_shipped"], errors="coerce")

df["color"] = df["color"].astype(str).str.strip().str.lower()

df = df.sort_values(["make", "model", "color", "date"])

df["prev_end"] = df.groupby(["make", "model", "color"])["end"].shift(1)
df["calc_units"] = df["end"] - df["prev_end"]
df["calc_units"] = df["calc_units"].fillna(df["units_shipped"])
df["calc_units"] = df["calc_units"].clip(lower=0)

df["cumulative"] = df.groupby(["make", "model", "color"])["calc_units"].cumsum()

# Get unique colors
colors = df["color"].unique()
num_colors = len(colors)

# Create subplots
cols = 2
rows = (num_colors + 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharex=True)
axes = axes.flatten()

# If only one color, axes isn't a list
if num_colors == 1:
    axes = [axes]

MODEL_ORDER = ["Lite", "Base", "Pro", "Max"]
MODEL_COLORS = {
    "Lite": "#FFD700",
    "Base": "#D32D2F",
    "Pro": "#6A5ACD",
    "Max": "#3CB371"
}

plt.style.use('default')
fig.patch.set_facecolor('#F0F0F0')  # GameCube silver background

for ax, color in zip(axes, colors):
    color_df = df[df["color"] == color]
    plotted_labels = []
    model_cum_y = []
    model_end_labels = []
    for model in MODEL_ORDER:
        for (make, m), group in color_df.groupby(["make", "model"]):
            if m == model:
                label = f"{model}"
                ax.plot(
                    group["date"],
                    group["cumulative"],
                    label=label,
                    color=MODEL_COLORS[model],
                    linewidth=4,
                    zorder=3
                )
                plotted_labels.append(label)
                # Annotate the latest 'end' value at the rightmost point
                if not group.empty:
                    last_row = group.iloc[-1]
                    ax.annotate(
                        f"{int(last_row['end'])}",
                        xy=(last_row["date"], last_row["cumulative"]),
                        xytext=(10, 0),
                        textcoords='offset points',
                        va='center',
                        ha='left',
                        fontsize=12,
                        fontweight='bold',
                        color=MODEL_COLORS[model],
                        fontname='Comic Sans MS',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#F0F0F0', ec=MODEL_COLORS[model], lw=1)
                    )
                # Collect y and label for right y-axis
                model_cum_y.extend(list(group["cumulative"]))
                model_end_labels.extend(list(group["end"]))
    # Set up right y-axis with order numbers, but only show every 4th point to reduce clutter
    if model_cum_y:
        ax2 = ax.twinx()
        # Remove duplicate y values and sort
        y_pairs = sorted(set(zip(model_cum_y, model_end_labels)), key=lambda x: x[0])
        # Show every 4th point (and always the last one)
        if len(y_pairs) > 1:
            y_pairs_to_show = y_pairs[::4]
            if y_pairs[-1] not in y_pairs_to_show:
                y_pairs_to_show.append(y_pairs[-1])
        else:
            y_pairs_to_show = y_pairs
        y_ticks, y_labels = zip(*y_pairs_to_show)
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels([str(int(lbl)) for lbl in y_labels], fontsize=10, color="#222", fontname='Comic Sans MS')
        ax2.set_ylabel("Order Number", fontsize=14, fontweight='bold', color='#222', fontname='Comic Sans MS')
        ax2.tick_params(axis='y', colors='#222')
        ax2.spines['right'].set_linewidth(2)
        ax2.spines['right'].set_color('#222')
        ax2.grid(False)
    ax.set_title(f"{color.title()}", fontsize=16, fontweight='bold', color='#222', fontname='Comic Sans MS')
    ax.grid(True, linestyle='-', linewidth=1.2, color='#888', alpha=0.5)
    ax.set_facecolor('#E0E0E0')  # Slightly darker gray for panel
    # Only show legend for plotted models, in order
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = [handles[labels.index(m)] for m in MODEL_ORDER if m in labels]
    ordered_labels = [m for m in MODEL_ORDER if m in labels]
    ax.legend(ordered_handles, ordered_labels, fontsize=12, frameon=True, fancybox=False, edgecolor='#222', facecolor='#F0F0F0', borderpad=1, loc='upper left')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#222')
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#222')

# Format dates
for ax in axes:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

fig.suptitle("AYN Thor Shipping Progress", fontsize=22, fontweight='bold', color='#222', fontname='Comic Sans MS')
fig.supylabel("Units Shipped", fontsize=16, fontweight='bold', color='#222', fontname='Comic Sans MS')

def plot_shipping_progress(output_path=None):
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        plt.savefig(output_path)
    #plt.show()

if __name__ == "__main__":
    plot_shipping_progress(output_path="shipping_progress.png")
