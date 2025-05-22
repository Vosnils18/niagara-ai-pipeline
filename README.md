# Niagara AI Pipeline

This project demonstrates a proof-of-concept (PoC) for integrating AI-powered predictive insights with a Niagara-based building automation system and Reflow dashboard.

## 🔍 Objective

Use historical sensor data (energy, water, air quality) to predict future values or detect anomalies using Python. Push results back to Niagara for visualization in Reflow.

---

## 📁 Project Structure

```
niagara-ai-pipeline/
├── data/
│   ├── sample_niagara_data.json       # Simulated Niagara history export
│   └── prediction.json                # Model output
│
├── models/
│   ├── train_model.py                 # (Optional) model training script
│   └── model.pkl                      # Trained model
│
├── src/
│   ├── preprocess.py                  # Data formatting/cleaning
│   ├── predict.py                     # Generate predictions
│   └── send_to_niagara.py             # Push predictions back to Niagara
│
├── scripts/
│   └── run_pipeline.py                # Executes entire pipeline
│
├── utils/
│   └── logger.py                      # Logging/helpers
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🧪 Example Workflow

1. `sample_niagara_data.json` simulates real Niagara-exported sensor data.
2. `predict.py` reads it, runs a model, and outputs a prediction.
3. `send_to_niagara.py` (stub) simulates pushing prediction back to Niagara via HTTP POST.
4. Predictions can be visualized in Reflow as a virtual point or KPI.

---

## ⚙️ Getting Started

```bash
# 1. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run prediction pipeline
python scripts/run_pipeline.py
```

---

## 📌 Requirements

- Python 3.8+
- `pandas`, `scikit-learn`, `requests`

---

## 🔐 Notes

- Niagara must expose a servlet or endpoint for real deployment.
- Reflow should be configured to display predicted point values or graphs.
