import joblib
import pandas as pd

MODEL_PATH = "../invoice_flagging/models/predict_flag_invoice.pkl"
SCALER_PATH = "../invoice_flagging/models/scaler.pkl"


def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """
    Load trained invoice flagging model and scaler.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)

    return model, scaler


def predict_invoice_flag(input_data):
    """
    Predict whether an invoice is risky.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame
    """

    model, scaler = load_model()

    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    input_df["Predicted_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [10],
        "invoice_dollars": [1000],
        "Freight": [50],
        "total_item_quantity": [10],
        "total_item_dollars": [995],
        "avg_receiving_delay": [12]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)







