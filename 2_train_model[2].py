"""STEP 2: Train several ML models, compare them, keep the best one.
Run:  python 2_train_model.py"""
import json, sqlite3
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

with sqlite3.connect("water.db") as con:
    df = pd.read_sql("SELECT * FROM water_quality", con)

X, y = df.drop(columns="Potability"), df["Potability"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}

results, best_name, best_f1, best_model = {}, None, -1, None
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    acc, f1 = accuracy_score(y_te, p), f1_score(y_te, p)
    results[name] = {"accuracy": acc, "f1": f1}
    print(f"{name:20s} accuracy={acc:.3f}  f1={f1:.3f}")
    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, m

print("\nBEST MODEL:", best_name)
pred = best_model.predict(X_te)
print(classification_report(y_te, pred, target_names=["Not safe", "Safe"]))
cm = confusion_matrix(y_te, pred).tolist()
print("Confusion matrix:", cm)

# which parameters matter most? (Random Forest is always used for this chart)
rf = models["Random Forest"]
importance = dict(zip(X.columns, rf.feature_importances_.round(4).tolist()))

joblib.dump(best_model, "water_model.joblib")
json.dump({"best_model": best_name, "results": results, "confusion_matrix": cm,
           "importance": importance, "columns": list(X.columns)}, open("metrics.json", "w"))
print("Saved water_model.joblib and metrics.json")
