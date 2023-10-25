# Script to train machine learning model.
import joblib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data import process_data
from model import train_model, compute_model_metrics, inference, save_model

data_path = Path(__file__).absolute().parent.parent / "data" / "census_CLEAN.csv"
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
label = "salary"

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)

# Save the encoder (we'll need it for inference)
encoder_path = Path(__file__).absolute().parent.parent / "model" / "encoder.pkl"
save_model(encoder, encoder_path)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label=label,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train a model
model = train_model(X_train, y_train)
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_pred, y_test)
print(f"{precision=}, {recall=}, {fbeta=}")

# Save the model
model_path = Path(__file__).absolute().parent.parent / "model" / "model.pkl"
save_model(model, model_path)
