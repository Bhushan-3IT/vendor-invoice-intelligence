import joblib

# Load your model
model = joblib.load('freight_cost_prediction/models/predict_freight_model.pkl')

# Print what column names the model expects
print("Model expects these column names:", model.feature_names_in_)