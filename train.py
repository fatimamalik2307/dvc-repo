import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import json
import os

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

print("Loading prepared data...")
X_train = joblib.load('data/prepared/X_train.pkl')
y_train = joblib.load('data/prepared/y_train.pkl')
feature_columns = joblib.load('data/prepared/feature_columns.pkl')
target_column = joblib.load('data/prepared/target_column.pkl')

print("Training model...")
model = RandomForestRegressor(
    n_estimators=params['train']['n_estimators'],
    max_depth=params['train']['max_depth'],
    random_state=params['train']['random_state']
)

model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')

print("Model training completed!")