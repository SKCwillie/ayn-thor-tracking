import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from datetime import datetime, timedelta

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

MODEL_PATH = "assets/shipping_model.joblib"

def load_trained_models():
    artifact = joblib.load(MODEL_PATH)
    models = artifact["models"]
    meta = artifact["training_meta"]
    return models, meta

def normalize(value: str) -> str:
    return str(value).replace(" ", "").replace("-", "").replace("_", "").lower()

def get_model_prediction_line(models, meta, make, model, color, min_date, max_date):
    lookup_key = (
        normalize(make),
        normalize(model),
        normalize(color)
    )
    normalized_to_actual = {
        (
            normalize(k[0]),
            normalize(k[1]),
            normalize(k[2])
        ): k for k in models.keys()
    }
    if lookup_key not in normalized_to_actual:
        print(f"  Model key not found: {lookup_key}")
        print(f"  Available normalized keys (first 5): {list(normalized_to_actual.keys())[:5]}")
        return None, None
    actual_key = normalized_to_actual[lookup_key]
    reg = models[actual_key]
    # Extend 14 days past the last data point (max_date)
    future_limit = pd.Timestamp(max_date) + pd.Timedelta(days=14)
    date_range = pd.date_range(min_date, future_limit, freq="7D")
    ordinals = date_range.map(pd.Timestamp.toordinal)
    coef = reg.coef_[0]
    intercept = reg.intercept_
    if coef == 0:
        return None, None
    order_numbers = (ordinals - intercept) / coef
    return date_range, order_numbers

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

def get_month_range(dates, extend_days=0):
    min_date = dates.min()
    max_date = dates.max()
    start = pd.Timestamp(year=min_date.year, month=min_date.month, day=1)
    # Extend max_date by extend_days if requested
    if extend_days > 0:
        max_date = max_date + pd.Timedelta(days=extend_days)
    if max_date.month == 12:
        end = pd.Timestamp(year=max_date.year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=max_date.year, month=max_date.month + 1, day=1)
    return start, end

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
    # Extend x-axis 21 days past last data
    start, end = get_month_range(df["date"], extend_days=21)
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
        ax.set_xlim(start, end)
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
    print("Plotting black models...")
    df = get_df()
    black_df = df[df["color"] == "black"]
    if black_df.empty:
        print("No data for black models.")
        return
    models, meta = load_trained_models()
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    # Extend x-axis 21 days past last data
    if not black_df.empty:
        start, end = get_month_range(black_df["date"], extend_days=21)
    else:
        start, end = None, None
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        model_df = black_df[black_df["model"] == model]
        if model_df.empty:
            ax.set_visible(False)
            continue
        # Calculate x-axis limits, skipping the first month
        if not model_df.empty:
            min_date = model_df["date"].min()
            # Set start to the first day of the month after min_date
            if min_date.month == 12:
                start = pd.Timestamp(year=min_date.year + 1, month=1, day=1)
            else:
                start = pd.Timestamp(year=min_date.year, month=min_date.month + 1, day=1)
        ax.plot(
            model_df["date"],
            model_df["end"],
            color=MODEL_COLORS[model],
            linewidth=4,
            zorder=3,
            label="Shipped"
        )
        make = model_df["make"].iloc[0] if not model_df.empty else "Thor"
        min_date = pd.Timestamp("2026-01-01")
        # Use the graph's x-axis end for trendline extension
        trendline_max_date = end
        dates, preds = get_model_prediction_line(
            models, meta, make, model, "black", min_date, trendline_max_date
        )
        r2_str = ""
        slope_str = ""
        if dates is not None and preds is not None:
            ax.plot(
                dates,
                preds,
                color=MODEL_COLORS[model],
                linestyle=":",
                linewidth=2,
                label="Projected",
                zorder=2,
                alpha=0.8,
            )
            reg_key = (
                make, model, "black"
            )
            reg_key_norm = (
                normalize(make), normalize(model), normalize("black")
            )
            normalized_to_actual = {
                (
                    normalize(k[0]),
                    normalize(k[1]),
                    normalize(k[2])
                ): k for k in models.keys()
            }
            if reg_key_norm in normalized_to_actual:
                actual_key = normalized_to_actual[reg_key_norm]
                reg = models[actual_key]
                coef = reg.coef_[0]
                units_per_day = 1.0 / coef if coef != 0 else 0
                slope_str = f"{units_per_day:.1f} units/day"
                filtered_df = model_df[model_df["date"] >= pd.Timestamp("2026-01-01")]
                if not filtered_df.empty:
                    X = filtered_df[["order_number"]] if "order_number" in filtered_df else pd.DataFrame({"order_number": filtered_df["end"]})
                    y_true = filtered_df["date"].map(pd.Timestamp.toordinal)
                    r2 = reg.score(X, y_true)
                    r2_str = f"$R^2$ = {r2:.3f}"
        else:
            print(f"  No model found for Black {model}")
        format_ax(ax, title=f"Black {model}", ylabel="Order Number")
        if r2_str or slope_str:
            ax.text(
                0.98, 0.02, f"{slope_str}\n{r2_str}",
                transform=ax.transAxes,
                fontsize=13,
                color="#333",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="#f8f9fb", ec="#bbb", lw=1, alpha=0.85)
            )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=12, frameon=True, fancybox=False, edgecolor='#222', facecolor='#F0F0F0', borderpad=1, loc='upper left')
        if start is not None and end is not None:
            ax.set_xlim(start, end)
    fig.suptitle("AYN Thor Black Models Order Progress", fontsize=22, fontweight='bold', color='#222', fontname='Comic Sans MS')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path:
        plt.savefig(output_path)
        print(f"Saved black models graph to {output_path}")

def plot_color_models(output_path=None):
    print("Plotting special color models...")
    df = get_df()
    special_colors = ["white", "clear purple", "rainbow"]
    special_models = ["Pro", "Max"]
    color_titles = {"white": "White", "clear purple": "Clear Purple", "rainbow": "Rainbow"}
    models, meta = load_trained_models()
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    filtered_df = df[df["color"].isin(special_colors)]
    # Extend x-axis 21 days past last data
    if not filtered_df.empty:
        start, end = get_month_range(filtered_df["date"], extend_days=21)
    else:
        start, end = None, None
    for row_idx, color in enumerate(special_colors):
        for col_idx, model in enumerate(special_models):
            ax = axes[row_idx, col_idx]
            model_df = df[(df["color"] == color) & (df["model"] == model)]
            if model_df.empty:
                ax.set_visible(False)
                continue
            # Calculate x-axis limits, skipping the first month
            if not model_df.empty:
                min_date = model_df["date"].min()
                # Set start to the first day of the month after min_date
                if min_date.month == 12:
                    start = pd.Timestamp(year=min_date.year + 1, month=1, day=1)
                else:
                    start = pd.Timestamp(year=min_date.year, month=min_date.month + 1, day=1)
            ax.plot(
                model_df["date"],
                model_df["end"],
                color=MODEL_COLORS[model],
                linewidth=4,
                zorder=3,
                label="Shipped"
            )
            make = model_df["make"].iloc[0] if not model_df.empty else "Thor"
            min_date = pd.Timestamp("2026-01-01")
            # Use the graph's x-axis end for trendline extension
            trendline_max_date = end
            dates, preds = get_model_prediction_line(
                models, meta, make, model, color, min_date, trendline_max_date
            )
            r2_str = ""
            slope_str = ""
            if dates is not None and preds is not None:
                ax.plot(
                    dates,
                    preds,
                    color=MODEL_COLORS[model],
                    linestyle=":",
                    linewidth=2,
                    label="Projected",
                    zorder=2,
                    alpha=0.8,
                )
                reg_key = (
                    make, model, color
                )
                reg_key_norm = (
                    normalize(make), normalize(model), normalize(color)
                )
                normalized_to_actual = {
                    (
                        normalize(k[0]),
                        normalize(k[1]),
                        normalize(k[2])
                    ): k for k in models.keys()
                }
                if reg_key_norm in normalized_to_actual:
                    actual_key = normalized_to_actual[reg_key_norm]
                    reg = models[actual_key]
                    coef = reg.coef_[0]
                    units_per_day = 1.0 / coef if coef != 0 else 0
                    slope_str = f"{units_per_day:.1f} units/day"
                    filtered_df2 = model_df[model_df["date"] >= pd.Timestamp("2026-01-01")]
                    if not filtered_df2.empty:
                        if "order_number" in filtered_df2:
                            X = filtered_df2[["order_number"]]
                        else:
                            X = pd.DataFrame({"order_number": filtered_df2["end"]})
                        y_true = filtered_df2["date"].map(pd.Timestamp.toordinal)
                        r2 = reg.score(X, y_true)
                        r2_str = f"$R^2$ = {r2:.3f}"
            else:
                print(f"  No model found for {color_titles[color]} {model}")
            format_ax(ax, title=f"{color_titles[color]} {model}", ylabel="Order Number")
            if r2_str or slope_str:
                ax.text(
                    0.98, 0.02, f"{slope_str}\n{r2_str}",
                    transform=ax.transAxes,
                    fontsize=13,
                    color="#333",
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f8f9fb", ec="#bbb", lw=1, alpha=0.85)
                )
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=12, frameon=True, fancybox=False, edgecolor='#222', facecolor='#F0F0F0', borderpad=1, loc='upper left')
            if start is not None and end is not None:
                ax.set_xlim(start, end)
    fig.suptitle("AYN Thor Special Colors Order Progress", fontsize=22, fontweight='bold', color='#222', fontname='Comic Sans MS')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path:
        plt.savefig(output_path)
        print(f"Saved color models graph to {output_path}")

def make_graphs():
    print("Generating all graphs...")
    plot_shipping_progress(output_path="assets/shipping_progress.png")
    plot_black_models(output_path="assets/black_models_orders.png")
    plot_color_models(output_path="assets/color_models_orders.png")
    print("All graphs generated.")

if __name__ == "__main__":
    make_graphs()
