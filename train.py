import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


DEFAULT_DB_PATH = "shipping_info.db"
DEFAULT_TABLE_NAME = "shipments"
DEFAULT_OUTPUT_PATH = "assets/shipping_model.joblib"


def load_raw_data(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Load shipment data from SQLite.

    Expected schema:
        make   TEXT
        model  TEXT
        color  TEXT
        begin  INTEGER
        end    INTEGER
        date   TEXT or DATE
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        query = f"""
        SELECT
            make,
            model,
            color,
            begin,
            "end" AS end_order,
            date
        FROM {table_name}
        ORDER BY date, begin
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError(f"No rows found in table '{table_name}'")

    required_cols = {"make", "model", "color", "begin", "end_order", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head()
        raise ValueError(f"Some dates could not be parsed:\n{bad}")
    df = df[df["date"] >= "2026-01-01"].copy()


    df["begin"] = pd.to_numeric(df["begin"], errors="coerce")
    df["end_order"] = pd.to_numeric(df["end_order"], errors="coerce")

    if df["begin"].isna().any() or df["end_order"].isna().any():
        bad = df[df["begin"].isna() | df["end_order"].isna()].head()
        raise ValueError(f"Some begin/end values are not numeric:\n{bad}")

    df["begin"] = df["begin"].astype(int)
    df["end_order"] = df["end_order"].astype(int)

    invalid_ranges = df[df["end_order"] < df["begin"]]
    if not invalid_ranges.empty:
        raise ValueError(f"Found rows where end < begin:\n{invalid_ranges.head()}")

    # Normalize text fields a little
    for col in ["make", "model", "color"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def expand_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each begin/end range so each order number gets its own row.
    """
    expanded_rows = []

    for _, row in df.iterrows():
        for order_number in range(row["begin"], row["end_order"] + 1):
            expanded_rows.append(
                {
                    "make": row["make"],
                    "model": row["model"],
                    "color": row["color"],
                    "order_number": order_number,
                    "ship_date": row["date"],
                }
            )

    expanded_df = pd.DataFrame(expanded_rows)

    if expanded_df.empty:
        raise ValueError("Expanded dataframe is empty")

    expanded_df["date_ordinal"] = expanded_df["ship_date"].map(pd.Timestamp.toordinal)

    return expanded_df


def train_models(expanded_df: pd.DataFrame) -> dict:
    """
    Train one model per (make, model, color).

    Returns an artifact dict ready to be saved with joblib.
    """
    models = {}
    training_meta = {}

    grouped = expanded_df.groupby(["make", "model", "color"], dropna=False)

    for key, group in grouped:
        group = group.sort_values("order_number").copy()

        X = group[["order_number"]]
        y = group["date_ordinal"]

        row_count = len(group)
        min_order = int(group["order_number"].min())
        max_order = int(group["order_number"].max())
        min_date = group["ship_date"].min()
        max_date = group["ship_date"].max()

        # Use LinearRegression for all groups to support extrapolation.
        model = LinearRegression()
        model_type = "linear_regression"

        model.fit(X, y)

        models[key] = model
        training_meta[key] = {
            "model_type": model_type,
            "row_count": row_count,
            "min_order": min_order,
            "max_order": max_order,
            "min_date": min_date.isoformat(),
            "max_date": max_date.isoformat(),
        }

    if not models:
        raise ValueError("No models were trained")

    artifact = {
        "models": models,
        "training_meta": training_meta,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_version": "v1",
        "feature_columns": ["order_number"],
    }

    return artifact


def print_summary(raw_df: pd.DataFrame, expanded_df: pd.DataFrame, artifact: dict) -> None:
    """
    Print a useful training summary.
    """
    print("\nTraining summary")
    print("-" * 60)
    print(f"Raw rows:        {len(raw_df):,}")
    print(f"Expanded rows:   {len(expanded_df):,}")
    print(f"Models trained:  {len(artifact['models']):,}")

    print("\nAvailable combinations:")
    for key in sorted(artifact["training_meta"].keys()):
        meta = artifact["training_meta"][key]
        print(
            f"  {key} | "
            f"type={meta['model_type']} | "
            f"rows={meta['row_count']} | "
            f"orders={meta['min_order']}-{meta['max_order']}"
        )


def create_model():
    parser = argparse.ArgumentParser(description="Train shipping prediction models from SQLite")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite DB")
    parser.add_argument("--table", default=DEFAULT_TABLE_NAME, help="Table name")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save trained model artifact",
    )
    parser.add_argument(
        "--save-expanded-csv",
        default=None,
        help="Optional path to save expanded training data as CSV",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.db} / table '{args.table}'...")
    raw_df = load_raw_data(args.db, args.table)

    print("Expanding shipment ranges into one row per order...")
    expanded_df = expand_ranges(raw_df)

    if args.save_expanded_csv:
        # Ensure the path is in the assets folder
        save_path = Path(args.save_expanded_csv)
        assets_dir = Path("assets")
        if not save_path.parent or str(save_path.parent) == ".":
            save_path = assets_dir / save_path.name
        else:
            # If user provided a path, ensure 'assets' is in the path
            if assets_dir not in save_path.parents:
                save_path = assets_dir / save_path.name
        assets_dir.mkdir(exist_ok=True)
        expanded_df.to_csv(save_path, index=False)
        print(f"Expanded data saved to {save_path}")

    print("Training models...")
    artifact = train_models(expanded_df)

    output_path = Path(args.output)
    joblib.dump(artifact, output_path)
    print(f"Saved artifact to: {output_path}")
    print(f"Trained at: {artifact['trained_at']}")
    print(f"Model version: {artifact['model_version']}")
    print(f"Model artifact saved to {output_path}")

    print_summary(raw_df, expanded_df, artifact)


if __name__ == "__main__":
    create_model()
