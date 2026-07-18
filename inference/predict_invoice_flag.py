import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'invoice_flagging', 'models', 'predict_invoice_flag_model.pkl')

def load_model(model_path=MODEL_PATH):
    """
    Load trained invoice flag prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new invoice

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    dict with flag prediction
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    features = ['invoice_quantity', 'invoice_dollars', 'Freight', 'total_item_quantity', 'total_item_dollars']
    prediction = model.predict(input_df[features])
    return {'Predicted_Flag': prediction}

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [50, 1000, 10],
        "invoice_dollars": [352.95, 3000, 50],
        "Freight": [1.73, 500, 20],
        "total_item_quantity": [162, 600, 5],
        "total_item_dollars": [2476.0, 1500, 30]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)