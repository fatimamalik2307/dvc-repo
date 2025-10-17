import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

print("Loading test data and model...")
X_test = joblib.load('data/prepared/X_test.pkl')
y_test = joblib.load('data/prepared/y_test.pkl')
model = joblib.load('models/model.pkl')

print("Making predictions...")
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)

metrics = {
    'mse': float(mse),
    'mae': float(mae),
    'r2': float(r2)
}

print("Metrics:", metrics)

# Save metrics
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Evaluation completed!")
