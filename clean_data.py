import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
data_path = 'PMS-IP-project\\predictive_maintenance.csv'
data = pd.read_csv(data_path)
# Remove first character and set to numeric dtype
data['Product ID'] = data['Product ID'].apply(lambda x: x[1:])
data['Product ID'] = pd.to_numeric(data['Product ID'])

# Histogram of ProductID
sns.histplot(data=data, x='Product ID', hue='Type')
plt.show()
# Save the modified dataset after processing
data.to_csv('PMS-IP-project/predictive_maintenance.csv', index=False)

#data.to_csv('PMS-IP-project/predictive_maintenance_cleaned.csv', index=False)
