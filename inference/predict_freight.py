import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'freight_cost_prediction', 'models', 'predict_freight_model.pkl')

def load_model(model_path=MODEL_PATH):
    """
    Load trained freight predict model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new customer

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    dict with freight cost
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df[['Quantity', 'Dollars']])
    return {'Predicted_Freight': prediction}

if __name__ == "__main__":
    sample_data = {
        "Quantity": [1200, 500, 100],
        "Dollars": [18500, 9000, 3000]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)