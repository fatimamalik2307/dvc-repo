import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import joblib
import os

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

print("Loading Pakistan House Price data...")
data = pd.read_csv('Entities.csv')

print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())

# Basic preprocessing for Pakistan House Price Dataset
print("Preprocessing data...")

# Handle missing values
data = data.dropna(subset=['price', 'bedrooms', 'baths', 'Total_Area'])

# Select relevant features for house price prediction
feature_columns = [
    'bedrooms', 'baths', 'Total_Area', 
    'property_type', 'location', 'city', 'province_name'
]

target_column = 'price'

print(f"Using features: {feature_columns}")
print(f"Using target: {target_column}")

X = data[feature_columns]
y = data[target_column]

# Handle categorical variables
categorical_columns = ['property_type', 'location', 'city', 'province_name']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=params['prepare']['test_size'],
    random_state=params['prepare']['random_state']
)

# Save processed data
os.makedirs('data/prepared', exist_ok=True)
joblib.dump(X_train, 'data/prepared/X_train.pkl')
joblib.dump(X_test, 'data/prepared/X_test.pkl') 
joblib.dump(y_train, 'data/prepared/y_train.pkl')
joblib.dump(y_test, 'data/prepared/y_test.pkl')
joblib.dump(list(X.columns), 'data/prepared/feature_columns.pkl')  # Save actual feature names
joblib.dump(target_column, 'data/prepared/target_column.pkl')

print("Data preparation completed!")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of features after encoding: {X_train.shape[1]}")