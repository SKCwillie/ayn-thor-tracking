## Quick Start: Get a Shipping Prediction
Copy and paste url to get predictions for your order

https://ayn-thor-tracking.onrender.com/{{color}}/{{model}}/{{first4digits_of_order_number}}

## Example:
```
http://ayn-thor-tracker.com/ClearPurple/Max/1910
```

## Current Shipping Progress
![Shipping Progress Graph](assets/shipping_progress.png)
![Black Models Orders](assets/black_models_orders.png)
![Color Models Orders](assets/color_models_orders.png)

---

## API Endpoints

The following endpoints are available in the Shipping Prediction API:

### Root

- **GET /**  
  Returns API status and model info.

  **Example:**
  ```
  GET / 
  ```

  **Response:**
  ```json
  {
    "status": "running",
    "model_version": "v1",
    "trained_at": "2024-06-10T12:00:00Z",
    "available_models": 42
  }
  ```

---

### Health Check

- **GET /health**  
  Returns health status.

  **Example:**
  ```
  GET /health
  ```

  **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

---

### Predict Shipping Date

- **GET /{color}/{model}/{shipment_number}**  
  Predicts the shipping date for a given color, model, and shipment number.

  **Example:**
  ```
  GET /ClearPurple/Max/1910
  ```

  **Response:**
  ```json
  {
    "make": "Thor",
    "model": "Max",
    "color": "ClearPurple",
    "shipment_number": 1910,
    "predicted_ship_date": "2026-04-29",
    "in_training_range": false,
    "training_order_range": [1000, 1500],
    "model_type": "linear_regression",
    "model_version": "v1",
    "trained_at": "2024-06-10T12:00:00Z"
  }
  ```

---

### List Available Models

- **GET /models**  
  Lists all available (make, model, color) combinations and their order ranges.

  **Example:**
  ```
  GET /models
  ```

  **Response:**
  ```json
  {
    "count": 2,
    "models": [
      {
        "make": "Thor",
        "model": "Max",
        "color": "ClearPurple",
        "rows": 500,
        "order_range": [1000, 1500]
      },
      {
        "make": "Thor",
        "model": "Pro",
        "color": "Black",
        "rows": 300,
        "order_range": [2000, 2300]
      }
    ]
  }
  ```

---
