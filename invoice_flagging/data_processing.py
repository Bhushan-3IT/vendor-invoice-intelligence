import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_invoice_data():
    conn=sqlite3.connect(r"C:\Users\LOQ\OneDrive\Desktop\ML_Project\data\inventory.db")

    query = """
    WITH purchase_agg as(
    SELECT
        p.PONumber,
        COUNT(DISTINCT p.Brand) AS total_brands,
        SUM(p.Quantity) AS total_item_quantity,
        SUM(p.Dollars) AS total_item_dollars,
        AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
    )
    SELECT 
        vi.PONumber,
        vi.Quantity as invoice_quantity,
        vi.Dollars as invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate)-julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate)-julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
    on vi.PONumber=pa.PONumber
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1

    if row["avg_receiving_delay"] > 10:
        return 1

    return 0


def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df

def split_data(df, features, target):
    X = df[features]
    y = df[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )





def scale_features(X_train, X_test, scaler_path):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create models folder if it doesn't exist
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled
    
 



































# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "id": "e5ee8000-8e8b-4c12-9f0d-e26a231ba1ef",
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "Python 3 (ipykernel)",
#    "language": "python",
#    "name": "python3"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.13.7"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 5
# }
