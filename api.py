import re
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

MODEL_PATH = "assets/shipping_model.joblib"

app = FastAPI(title="Shipping Prediction API")

artifact = None
models = {}
training_meta = {}
canonical_names = {}


# -------------------------
# Normalization Function
# -------------------------
def normalize(value: str) -> str:
    """
    Normalize strings for matching:
    - remove whitespace
    - remove dashes
    - remove underscores
    - lowercase
    """
    return re.sub(r"[\s\-_]", "", value).lower()


# -------------------------
# Load Model
# -------------------------
def load_artifact():
    global artifact, models, training_meta, canonical_names

    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run train_model.py first."
        )

    artifact = joblib.load(model_file)

    raw_models = artifact["models"]
    raw_meta = artifact["training_meta"]

    models.clear()
    training_meta.clear()
    canonical_names.clear()

    for key, model in raw_models.items():
        make, model_name, color = key

        normalized_key = (
            normalize(make),
            normalize(model_name),
            normalize(color),
        )

        models[normalized_key] = model
        training_meta[normalized_key] = raw_meta[key]
        canonical_names[normalized_key] = key


@app.on_event("startup")
def startup():
    load_artifact()


# -------------------------
# Root
# -------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "model_version": artifact["model_version"],
        "trained_at": artifact["trained_at"],
        "available_models": len(models),
    }


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


# -------------------------
# Prediction Endpoint
# Pattern: /ClearPurple/Max/1910
# -------------------------
@app.get("/{color}/{model}/{shipment_number}")
def predict(color: str, model: str, shipment_number: int):

    normalized_color = normalize(color)
    normalized_model = normalize(model)

    # find matching keys
    matching_keys = [
        key for key in models.keys()
        if key[2] == normalized_color and key[1] == normalized_model
    ]

    if not matching_keys:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Color/model combination not found",
                "requested": {
                    "color": color,
                    "model": model
                }
            }
        )

    # If multiple makes exist, choose first
    key = matching_keys[0]

    reg = models[key]

    pred_ordinal = reg.predict(
        pd.DataFrame({"order_number": [shipment_number]})
    )[0]

    pred_date = pd.Timestamp.fromordinal(int(round(pred_ordinal)))

    meta = training_meta[key]

    canonical = canonical_names[key]

    in_training_range = (
        meta["min_order"] <= shipment_number <= meta["max_order"]
    )

    return {
        "make": canonical[0],
        "model": canonical[1],
        "color": canonical[2],
        "shipment_number": shipment_number,
        "predicted_ship_date": pred_date.date().isoformat(),
        "in_training_range": in_training_range,
        "training_order_range": [
            meta["min_order"],
            meta["max_order"]
        ],
        "model_type": meta["model_type"],
        "model_version": artifact["model_version"],
        "trained_at": artifact["trained_at"],
    }


# -------------------------
# List Available Models
# -------------------------
@app.get("/models")
def list_models():

    results = []

    for key in models.keys():
        canonical = canonical_names[key]
        meta = training_meta[key]

        results.append({
            "make": canonical[0],
            "model": canonical[1],
            "color": canonical[2],
            "rows": meta["row_count"],
            "order_range": [
                meta["min_order"],
                meta["max_order"]
            ]
        })

    return {
        "count": len(results),
        "models": results
    }


# -------------------------
# Latest Shipments
# -------------------------
@app.get("/latest")
def latest_shipments():
    """
    Return the latest shipping info grouped by color, with models ordered as Lite, Base, Pro, Max.
    """
    # Define desired model order (case-insensitive)
    model_order = ["lite", "base", "pro", "max"]

    # Group by color
    color_dict = {}
    for key in models.keys():
        canonical = canonical_names[key]
        meta = training_meta[key]
        color = canonical[2]
        model = canonical[1]
        make = canonical[0]
        entry = {
            "make": make,
            "model": model,
            "latest_order": meta["max_order"],
            "latest_ship_date": meta["max_date"],
            "rows": meta["row_count"]
        }
        if color not in color_dict:
            color_dict[color] = []
        color_dict[color].append(entry)

    # Sort models for each color by the specified order
    def model_sort_key(entry):
        model_name = entry["model"].lower()
        try:
            return model_order.index(model_name)
        except ValueError:
            return len(model_order)  # unknown models go last

    latest_payload = []
    for color, models_list in color_dict.items():
        sorted_models = sorted(models_list, key=model_sort_key)
        latest_payload.append({
            "color": color,
            "models": sorted_models
        })

    return {
        "count": len(latest_payload),
        "latest_shipments": latest_payload
    }
