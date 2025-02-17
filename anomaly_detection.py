import pandas as pd


data_path = 'PMS-IP-project\\predictive_maintenance.csv'
df = pd.read_csv(data_path)

# Identify features and target columns
features = [col for col in df.columns if df[col].dtype == 'float64' or col == 'Type']
target = ['Target', 'Failure Type']

# Identify rows where Failure Type is 'Random Failures' but Target is 0
idx_RNF = df.loc[df['Failure Type'] == 'Random Failures'].index
first_drop = df.loc[idx_RNF, target].shape[0]
print('Number of observations where RNF=1 but Machine failure=0:', first_drop)

# Drop these rows
df.drop(index=idx_RNF, inplace=True)

# Identify ambiguous cases where Target=1 but Failure Type is 'No Failure'
idx_ambiguous = df.loc[(df['Target'] == 1) & (df['Failure Type'] == 'No Failure')].index
second_drop = df.loc[idx_ambiguous, target].shape[0]
print('Number of ambiguous observations:', second_drop)

# Drop these rows
df.drop(index=idx_ambiguous, inplace=True)

# Global percentage of removed observations
n = df.shape[0]
print('Global percentage of removed observations:', (100 * (first_drop + second_drop) / n))

# Reset index
df.reset_index(drop=True, inplace=True)



# Save the modified dataset after processing
df.to_csv('PMS-IP-project\\predictive_maintenance.csv', index=False)
print("Data cleaning completed. Cleaned file saved back")