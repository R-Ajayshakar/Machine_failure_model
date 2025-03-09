import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

#  Load Dataset
data_path = 'PMS-IP-project\\predictive_maintenance.csv'
df = pd.read_csv(data_path)

# 🔹 Select Features & Target
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Target'  # Binary Classification: Machine Failure (1) or Not (0)

# Train-Validation-Test Split (80-10-10)
X = df[features]
y = df[[target, 'Failure Type']]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=df['Failure Type'], random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, stratify=y_trainval['Failure Type'], random_state=42)

# 🏋️‍♂️ Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#  Define Models
models = {
    "LogisticRegression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

#  Hyperparameter Tuning Function
def tune_and_fit(clf, X, y, params):
    f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
    start_time = time.time()
    grid_model = GridSearchCV(clf, param_grid=params, cv=5, scoring=f2_scorer)
    grid_model.fit(X, y['Target'])
    print(f"Best params for {clf.__class__.__name__}: {grid_model.best_params_}")
    print(f" Training time: {round(time.time()-start_time, 2)}s")
    return grid_model

#  Define Hyperparameters for GridSearch
params_grid = {
    "LogisticRegression": {"C": [0.1, 1, 10]},
    "KNN": {"n_neighbors": [3, 5, 7]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "RandomForest": {"n_estimators": [50, 100, 200]},
    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
}

#  Train Models & Tune Hyperparameters
best_models = {}
for name, model in models.items():
    print(f"🔹 Training {name}...")
    best_models[name] = tune_and_fit(model, X_train, y_train, params_grid[name])

#  Model Evaluation Function
def eval_preds(model, X, y_true):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y_true['Target'], y_pred)
    auc = roc_auc_score(y_true['Target'], proba)
    f1 = f1_score(y_true['Target'], y_pred, pos_label=1)
    f2 = fbeta_score(y_true['Target'], y_pred, pos_label=1, beta=2)
    cm = confusion_matrix(y_true['Target'], y_pred)
    return cm, {"ACC": acc, "AUC": auc, "F1": f1, "F2": f2}

#  Evaluate on Test Set
metrics_dict = {}
for name, model in best_models.items():
    print(f"📌 Evaluating {name}...")
    cm, metrics = eval_preds(model, X_test, y_test)
    metrics_dict[name] = metrics
    print(f"🔎 {name} Metrics: {metrics}")

#  Save Model Performance to CSV
metrics_df = pd.DataFrame(metrics_dict).T.round(3)
metrics_df.to_csv('PMS-IP-project\\model_performance.csv')
print("✅ Model Performance saved to CSV.")

# Predict on Test Data & Save Results
predictions = {}
for name, model in best_models.items():
    predictions[name] = model.predict(X_test)

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('PMS-IP-project\\model_predictions.csv', index=False)
print("✅ Model Predictions saved to CSV.")

#  Best Model Selection
best_model_name = metrics_df["AUC"].idxmax()  # Select model with highest AUC
best_model = best_models[best_model_name]
print(f"🏆 Best Model: {best_model_name}")

# Save Final Model Predictions
final_predictions = best_model.predict(X_test)
df_test = X_test.copy()
df_test["Actual Failure"] = y_test["Target"]
df_test["Predicted Failure"] = final_predictions
df_test.to_csv('PMS-IP-project\\final_predictions.csv', index=False)
print(" Final Predictions saved to CSV.")
