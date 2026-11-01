💧 Water Quality Prediction

A machine learning system that predicts whether water is suitable for drinking by analysing parameters such as pH, turbidity and dissolved solids. It includes a data pipeline, a SQLite database, model comparison, and an interactive Streamlit web dashboard.

This is an educational screening tool. It does not replace laboratory water testing.

Features
Data cleaning (missing values filled with the median)
SQLite database storage (water.db)
Compares 3 models: Logistic Regression, Random Forest, Gradient Boosting
Automatically saves the best model
Web dashboard with 3 tabs:
Predict: enter water test values and get Safe / Not safe with a confidence score
Data explorer: histograms and correlation heatmap
Model performance: accuracy, F1 score, feature importance, confusion matrix
Project structure
water_quality_project/
├── 1_prepare_data.py    # load + clean data, create database
├── 2_train_model.py     # train, compare and save the best model
├── app.py               # Streamlit web app (front end + back end)
├── requirements.txt     # Python libraries needed
├── water_clean.csv      # cleaned dataset (generated)
├── water.db             # SQLite database (generated)
├── water_model.joblib   # trained model (generated)
└── metrics.json         # scores for the dashboard (generated)
Dataset

The project uses the Water Potability dataset from Kaggle (about 3,276 samples).

Column	Meaning
ph	Acidity/alkalinity (safe range about 6.5–8.5)
Hardness	Calcium and magnesium content
Solids	Total dissolved solids (ppm)
Chloramines	Disinfectant level (mg/L)
Sulfate	Sulfate concentration (mg/L)
Conductivity	Electrical conductivity (µS/cm)
Organic_carbon	Organic carbon (mg/L)
Trihalomethanes	Disinfection by-products (µg/L)
Turbidity	Cloudiness of water (NTU)
Potability	Target: 1 = safe, 0 = not safe
