import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
data_path = 'PMS-IP-project\\predictive_maintenance.csv'
data = pd.read_csv(data_path)

 
# Drop ID columns
df = data.copy()
df.drop(columns=['UDI','Product ID'], inplace=True)

# Save the modified dataset after processing
df.to_csv('PMS-IP-project\\predictive_maintenance.csv', index=False)



# Pie chart of Type percentage
value = data['Type'].value_counts()
Type_percentage = 100*value/data.Type.shape[0]
labels = Type_percentage.index.array
x = Type_percentage.array
plt.pie(x, labels = labels, colors=sns.color_palette('tab10')[0:3], autopct='%.0f%%')
plt.title('Machine Type percentage')
plt.show()