import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('PMS-IP-project\\predictive_maintenance.csv')

# Identify numeric features
num_features = [col for col in df.columns if df[col].dtype == 'float64']

# Plot histograms for numeric features
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
fig.suptitle('Numeric Features Histogram')
for j, feature in enumerate(num_features):
    sns.histplot(ax=axs[j // 3, j % 3], data=df, x=feature, kde=True)
plt.tight_layout()
plt.show()

# Plot boxplots for numeric features
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
fig.suptitle('Numeric Features Boxplot')
for j, feature in enumerate(num_features):
    sns.boxplot(ax=axs[j // 3, j % 3], data=df, x=feature)
plt.tight_layout()
plt.show()

print("Outlier inspection completed. Analyze the plots to decide further actions.")
