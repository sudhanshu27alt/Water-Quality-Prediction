# 💧 Water Quality Prediction

A machine learning system that predicts whether water is suitable for drinking by analysing parameters such as pH, turbidity and dissolved solids. It includes a data pipeline, a SQLite database, model comparison, and an interactive Streamlit web dashboard.

> **Disclaimer:** This is an educational screening tool. It does not replace laboratory water testing.

---

## Features

- Data cleaning (missing values filled with the median)
- SQLite database storage (`water.db`)
- Compares 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Automatically saves the best model
- Web dashboard with 3 tabs:
  - **Predict**: enter water test values and get Safe / Not safe with a confidence score
  - **Data explorer**: histograms and correlation heatmap
  - **Model performance**: accuracy, F1 score, feature importance, confusion matrix

## Project structure

```
water_quality_project/
├── 1_prepare_data.py    # load + clean data, create database
├── 2_train_model.py     # train, compare and save the best model
├── app.py               # Streamlit web app (front end + back end)
├── requirements.txt     # Python libraries needed
├── water_clean.csv      # cleaned dataset (generated)
├── water.db             # SQLite database (generated)
├── water_model.joblib   # trained model (generated)
└── metrics.json         # scores for the dashboard (generated)
```

## Dataset

The project uses the **Water Potability** dataset from Kaggle (about 3,276 samples).

| Column | Meaning |
|---|---|
| ph | Acidity/alkalinity (safe range about 6.5–8.5) |
| Hardness | Calcium and magnesium content |
| Solids | Total dissolved solids (ppm) |
| Chloramines | Disinfectant level (mg/L) |
| Sulfate | Sulfate concentration (mg/L) |
| Conductivity | Electrical conductivity (µS/cm) |
| Organic_carbon | Organic carbon (mg/L) |
| Trihalomethanes | Disinfection by-products (µg/L) |
| Turbidity | Cloudiness of water (NTU) |
| Potability | **Target**: 1 = safe, 0 = not safe |

**To use the real data:** download `water_potability.csv` from Kaggle (search "Water Potability") and place it in the project folder before running step 1.
If the file is missing, the script generates **synthetic practice data** with the same columns. State this clearly if you use it in a report.

## Installation

1. Install Python 3.9 or newer from python.org (tick **Add Python to PATH**).
2. Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

## How to run

Run these commands in order:

```bash
python 1_prepare_data.py     # cleans data, creates water.db
python 2_train_model.py      # trains models, saves water_model.joblib
streamlit run app.py         # opens the dashboard in your browser
```

## Deployment (Streamlit Community Cloud, free)

1. Create a **public** GitHub repository.
2. Upload: `app.py`, `requirements.txt`, `water.db`, `water_model.joblib`, `metrics.json`.
3. Go to share.streamlit.io and sign in with GitHub.
4. Click **New app**, choose your repository, set main file to `app.py`, click **Deploy**.
5. You get a public link such as `https://your-app.streamlit.app`.

## Results

| Data | Typical accuracy |
|---|---|
| Synthetic practice data | about 80% (Gradient Boosting) |
| Real Kaggle dataset | about 60–70% (the data is noisy, which is normal) |

Your exact numbers appear after running step 2 and in the **Model performance** tab.

## Technologies

Python, Pandas, NumPy, Scikit-learn, SQLite, Plotly, Streamlit, Joblib

## Limitations and future work

- Only 9 chemical/physical parameters are used; real safety also depends on bacteria and heavy metals.
- Future work: live IoT sensor input, more advanced models, explainability (SHAP), mobile app.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python is not recognized` | Reinstall Python and tick *Add to PATH* |
| `No module named ...` | Run `pip install -r requirements.txt` again |
| Model file not found | Run steps 1 and 2 before step 3 |

## License

For educational use.
