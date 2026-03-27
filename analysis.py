import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def format_ax(ax, title=None, ylabel=None):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', color='#222', fontname='Comic Sans MS')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold', color='#222', fontname='Comic Sans MS')
    ax.grid(True, linestyle='-', linewidth=1.2, color='#888', alpha=0.5)
    ax.set_facecolor('#E0E0E0')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#222')
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#222')

DB_PATH = "shipping_info.db"

MODEL_ORDER = ["Lite", "Base", "Pro", "Max"]
MODEL_COLORS = {
    "Lite": "#FFD700",
    "Base": "#D32D2F",
    "Pro": "#6A5ACD",
    "Max": "#3CB371"
}

def get_df():
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
    return df

def plot_shipping_progress(output_path=None):
    df = get_df()
    colors = df["color"].unique()
    num_colors = len(colors)
    cols = 2
    rows = (num_colors + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharex=True)
    axes = axes.flatten()
    if num_colors == 1:
        axes = [axes]
    plt.style.use('default')
    fig.patch.set_facecolor('#F0F0F0')
    for ax, color in zip(axes, colors):
        color_df = df[df["color"] == color]
        plotted_labels = []
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
                    if not group.empty:
                        last_row = group.iloc[-1]
                        ax.annotate(
                            f"Order #{int(last_row['end'])}",
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
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = [handles[labels.index(m)] for m in MODEL_ORDER if m in labels]
        ordered_labels = [m for m in MODEL_ORDER if m in labels]
        ax.legend(ordered_handles, ordered_labels, fontsize=12, frameon=True, fancybox=False, edgecolor='#222', facecolor='#F0F0F0', borderpad=1, loc='upper left')
        format_ax(ax, title=f"{color.title()}")
    fig.suptitle("AYN Thor Shipping Progress", fontsize=22, fontweight='bold', color='#222', fontname='Comic Sans MS')
    fig.supylabel("Units Shipped", fontsize=16, fontweight='bold', color='#222', fontname='Comic Sans MS')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path:
        plt.savefig(output_path)

def plot_orders(model, color, output_path=None):
    df = get_df()
    model = model.capitalize()
    color = color.lower()
    filtered = df[(df["model"] == model) & (df["color"] == color)]
    if filtered.empty:
        print(f"No data for model '{model}' and color '{color}'")
        return
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(filtered["date"], filtered["end"], color=MODEL_COLORS.get(model, '#333'), linewidth=4, zorder=3)
    format_ax(ax, title=f"{model} {color.title()} Orders", ylabel="Order Number")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_black_models(output_path=None):
    df = get_df()
    black_df = df[df["color"] == "black"]
    if black_df.empty:
        print("No data for black models.")
        return
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        model_df = black_df[black_df["model"] == model]
        if model_df.empty:
            ax.set_visible(False)
            continue
        ax.plot(
            model_df["date"],
            model_df["end"],  # Use order number as y-axis
            color=MODEL_COLORS[model],
            linewidth=4,
            zorder=3,
            label=model
        )
        format_ax(ax, title=f"Black {model}", ylabel="Order Number")
        ax.legend([model], fontsize=12, frameon=True, fancybox=False, edgecolor='#222', facecolor='#F0F0F0', borderpad=1, loc='upper left')
    fig.suptitle("AYN Thor Black Models Order Progress", fontsize=22, fontweight='bold', color='#222', fontname='Comic Sans MS')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path:
        plt.savefig(output_path)

def plot_color_models(output_path=None):
    df = get_df()
    special_colors = ["white", "clear purple", "rainbow"]
    special_models = ["Pro", "Max"]
    color_titles = {"white": "White", "clear purple": "Clear Purple", "rainbow": "Rainbow"}
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    for row_idx, color in enumerate(special_colors):
        for col_idx, model in enumerate(special_models):
            ax = axes[row_idx, col_idx]
            model_df = df[(df["color"] == color) & (df["model"] == model)]
            if model_df.empty:
                ax.set_visible(False)
                continue
            ax.plot(
                model_df["date"],
                model_df["end"],
                color=MODEL_COLORS[model],
                linewidth=4,
                zorder=3,
                label=model
            )
            format_ax(ax, title=f"{color_titles[color]} {model}", ylabel="Order Number")
            ax.legend([model], fontsize=12, frameon=True, fancybox=False, edgecolor='#222', facecolor='#F0F0F0', borderpad=1, loc='upper left')
    fig.suptitle("AYN Thor Special Colors Order Progress", fontsize=22, fontweight='bold', color='#222', fontname='Comic Sans MS')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path:
        plt.savefig(output_path)

def make_graphs():
    plot_shipping_progress(output_path="assets/shipping_progress.png")
    plot_black_models(output_path="assets/black_models_orders.png")
    plot_color_models(output_path="assets/color_models_orders.png")


if __name__ == "__main__":
    make_graphs()
