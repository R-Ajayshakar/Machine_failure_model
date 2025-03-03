import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data_path = 'PMS\\PMS-IP-project\\predictive_maintenance.csv'
df = pd.read_csv(data_path)

# Define numerical features for PCA
num_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]']  # Ensure these columns exist in your dataset

# 🔹 Step 1: Apply PCA
pca = PCA(n_components=len(num_features))  # Use all numerical features
X_pca = pd.DataFrame(data=pca.fit_transform(df[num_features]), 
                     columns=['PC'+str(i+1) for i in range(len(num_features))])

# Explained variance ratio per component
var_exp = pd.Series(data=100 * pca.explained_variance_ratio_, 
                    index=['PC'+str(i+1) for i in range(len(num_features))])

# Print variance explained
print('🔹 Explained variance ratio per component:', round(var_exp, 2), sep='\n')
print('🔹 Explained variance with 3 components:', round(var_exp.values[:3].sum(), 2), '%')

# 🔹 Step 2: Reduce Data to 3D Representation
pca3 = PCA(n_components=3)
X_pca3 = pd.DataFrame(data=pca3.fit_transform(df[num_features]), 
                      columns=['PC1','PC2','PC3'])

# 🔹 Step 3: Understand Feature Contributions to Principal Components (Loadings)
fig, axs = plt.subplots(ncols=3, figsize=(18, 4))
fig.suptitle('Loadings Magnitude (Feature Contributions)')

# Extract PCA loadings
pca_loadings = pd.DataFrame(data=pca3.components_, columns=num_features)

# Plot loadings for each principal component
for j in range(3):
    ax = axs[j]
    sns.barplot(ax=ax, x=pca_loadings.columns, y=pca_loadings.values[j])
    ax.tick_params(axis='x', rotation=90)
    ax.title.set_text('PC' + str(j+1))

plt.show()

# 🔹 Step 4: Rename Components for Better Understanding
X_pca3.rename(columns={'PC1':'Temperature', 'PC2':'Power', 'PC3':'Tool Wear'}, inplace=True)

# 🔹 Step 5: 3D Visualization of PCA Results
color_map = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple'}
colors = df['Failure Type'].map(color_map)

fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca3['Temperature'], X_pca3['Power'], X_pca3['Tool Wear'], c=colors)

ax.set_xlabel('Temperature')
ax.set_ylabel('Power')
ax.set_zlabel('Tool Wear')
ax.set_title('Data in 3D PCA Space')

plt.show()

# 🔹 Step 6: Generate Correlation Heatmap
plt.figure(figsize=(7, 4))
sns.heatmap(data=df.corr(), mask=np.triu(df.corr()), annot=True, cmap='BrBG')

plt.title('Correlation Heatmap')
plt.show()
