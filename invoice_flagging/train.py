from modeling_evaluation import train_random_forest,evaluate_classifier
import joblib


from data_processing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

TARGET = "flag_invoice"


def main():

    # Load Data
    df = load_invoice_data()

    # Create Labels
    df = apply_labels(df)

    # Split Data
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # Scale Features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        "models/scaler.pkl"
    )

    # Train Model
    grid_search = train_random_forest(
        X_train_scaled,
        y_train
    )

    # Evaluate Model
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save Best Model
    joblib.dump(
        grid_search.best_estimator_,
        "models/predict_flag_invoice.pkl"
    )

    print("✅ Model saved successfully!")


if __name__ == "__main__":
    main()