import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

# Load the dataset
#dataset_path = r"D:\AAA Semester 6\IP Project\PMS-IP-project\predictive_maintenance.csv"
#dataset_path = 'PMS-IP-project\\predictive_maintenance.csv'  # Your dataset path
df=pd.read_csv('PMS\PMS-IP-project\predictive_maintenance.csv')
#df = pd.read_csv(dataset_path)  # Replace 'your_dataset.csv' with your actual file name

# Display original class distribution
print("Original Class Distribution:", df['Failure Type'].value_counts())

# Calculate the desired resampling size
n_working = df['Failure Type'].value_counts()['No Failure']
desired_length = round(n_working / 0.8)  # Calculate target dataset size for 80:20 ratio
spc = round((desired_length - n_working) / 4)  # Samples per failure class

# Define the sampling strategy
balance_cause = {
    'No Failure': n_working,
    'Overstrain Failure': spc,
    'Heat Dissipation Failure': spc,
    'Power Failure': spc,
    'Tool Wear Failure': spc
}

# Apply SMOTENC (categorical features: Machine Type at index 0 and column 7)
sm = SMOTENC(categorical_features=[0, 7], sampling_strategy=balance_cause, random_state=0)
df_res, y_res = sm.fit_resample(df, df['Failure Type'])

# Display results after resampling
print("Resampled Class Distribution:", df_res['Failure Type'].value_counts())
