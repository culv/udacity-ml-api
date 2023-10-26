# Script to train machine learning model.
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data import process_data
from model import (
    train_model,
    compute_model_metrics,
    compute_column_slice_performance,
    inference,
    save_model,
    RANDOM_STATE,
)

data_path = Path(__file__).absolute().parent.parent / "data" / "census_CLEAN.csv"
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=RANDOM_STATE)

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

# Compute model performance on the test set
precision, recall, fbeta = compute_model_metrics(y_pred, y_test)
print(f"{precision=}, {recall=}, {fbeta=}")

# Compute slice performance for each category
cat_perfs = {}
for cat in cat_features:
    cat_perf = compute_column_slice_performance(model, X_test, y_test, test, cat)
    cat_perfs[cat] = cat_perf

slice_perf_path = data_path = Path(__file__).absolute().parent / "slice_performance.txt"
with open(slice_perf_path, "w") as f:
    f.write(json.dumps(cat_perfs, indent=4))

# Save the model
model_path = Path(__file__).absolute().parent.parent / "model" / "model.pkl"
save_model(model, model_path)
