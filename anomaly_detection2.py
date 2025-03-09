import pandas as pd 

# Load dataset
data_path = 'PMS-IP-project\original_predictive_maintenance_dataset.csv'
df = pd.read_csv(data_path)

# Identify rows where Failure Type is 'Random Failures' but Machine Failure is 0
idx_RNF = df[(df['Failure Type'] == 'Random Failures') & (df['Target'] == 0)].index
print(f'Fixing {len(idx_RNF)} cases where RNF exists but Machine Failure is 0')

# Fix: Set Machine Failure to 1
df.loc[idx_RNF, 'Target'] = 1

# Identify cases where Target = 1 but Failure Type is 'No Failure'
idx_ambiguous = df[(df['Target'] == 1) & (df['Failure Type'] == 'No Failure')].index
print(f'Fixing {len(idx_ambiguous)} cases where Machine Failure = 1 but Failure Type is missing')

# Fix: Assign "Unknown Failure" instead of dropping
df.loc[idx_ambiguous, 'Failure Type'] = 'Unknown Failure'

# Save the modified dataset
df.to_csv('PMS-IP-project\\predictive_maintenance_cleaned.csv', index=False)
print("Data inconsistencies resolved. Cleaned file saved.")
