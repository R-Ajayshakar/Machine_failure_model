import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 📂 Load Dataset
data_path = 'PMS-IP-project\\predictive_maintenance.csv'
df = pd.read_csv(data_path)

# 🔹 Select Features & Target
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Target'

# 🔀 Train-Validation-Test Split (80-10-10)
X = df[features]
y = df[[target, 'Failure Type']]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=df['Failure Type'], random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, stratify=y_trainval['Failure Type'], random_state=42)

# 🏋️‍♂️ Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ✅ Define Models
models = {
    "LogisticRegression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 🏆 Model Training Function
def fit_models(clf, clf_str, X_train, X_val, y_train, y_val):
    metrics = pd.DataFrame(columns=clf_str)
    for model, model_name in zip(clf, clf_str):
        model.fit(X_train, y_train['Target'])
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val['Target'], y_val_pred)
        f1 = f1_score(y_val['Target'], y_val_pred)
        metrics[model_name] = {"Accuracy": acc, "F1 Score": f1}
    return metrics

# 📌 Train Models on Different Feature Sets
clf = [models["LogisticRegression"], models["KNN"], models["SVM"], models["RandomForest"], models["XGBoost"]]
clf_str = ['LR', 'KNN', 'SVC', 'RFC', 'XGB']

# 🔹 1️⃣ Train on Raw Dataset
metrics_0 = fit_models(clf, clf_str, X_train, X_val, y_train, y_val)

# 🔹 2️⃣ Train on Temperature Product Dataset
XX_train = pd.DataFrame(X_train, columns=features).drop(columns=['Process temperature', 'Air temperature'])
XX_val = pd.DataFrame(X_val, columns=features).drop(columns=['Process temperature', 'Air temperature'])
XX_train['Temperature'] = X_train[:, 1] * X_train[:, 0]  # Process temp * Air temp
XX_val['Temperature'] = X_val[:, 1] * X_val[:, 0]
metrics_1 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# 🔹 3️⃣ Train on Power Product Dataset
XX_train = pd.DataFrame(X_train, columns=features).drop(columns=['Rotational speed', 'Torque'])
XX_val = pd.DataFrame(X_val, columns=features).drop(columns=['Rotational speed', 'Torque'])
XX_train['Power'] = X_train[:, 2] * X_train[:, 3]  # Rotational Speed * Torque
XX_val['Power'] = X_val[:, 2] * X_val[:, 3]
metrics_2 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# 🔹 4️⃣ Train on Both Engineered Features
XX_train = pd.DataFrame(X_train, columns=features).drop(columns=['Process temperature', 'Air temperature', 'Rotational speed', 'Torque'])
XX_val = pd.DataFrame(X_val, columns=features).drop(columns=['Process temperature', 'Air temperature', 'Rotational speed', 'Torque'])
XX_train['Temperature'] = X_train[:, 1] * X_train[:, 0]
XX_val['Temperature'] = X_val[:, 1] * X_val[:, 0]
XX_train['Power'] = X_train[:, 2] * X_train[:, 3]
XX_val['Power'] = X_val[:, 2] * X_val[:, 3]
metrics_3 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# 📊 Plot Classification Metrics
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
fig.suptitle('Classification Metrics')

for j, model in enumerate(clf_str):
    ax = axs[j // 3, j - 3 * (j // 3)]
    model_metrics = pd.DataFrame(data=[metrics_0[model], metrics_1[model], metrics_2[model], metrics_3[model]])
    model_metrics.index = ['Original', 'Temperature', 'Power', 'Both']
    model_metrics.transpose().plot(ax=ax, kind='bar', rot=0)
    ax.title.set_text(model)
    ax.get_legend().remove()

fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
axs.flatten()[-2].legend(title='Dataset', loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)
plt.show()

# 📂 Save Results
metrics_df = pd.DataFrame({
    "Original": metrics_0.mean(axis=1),
    "Temperature": metrics_1.mean(axis=1),
    "Power": metrics_2.mean(axis=1),
    "Both": metrics_3.mean(axis=1)
}).round(3)

metrics_df.to_csv('PMS-IP-project\\feature_selection_metrics.csv')
print("✅ Feature Selection Metrics saved to CSV.")
