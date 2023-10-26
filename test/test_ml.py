from pathlib import Path

import numpy as np
import pytest

from ml.model import compute_model_metrics, inference, load_model

model_path = Path(__file__).absolute().parent.parent / "model" / "model.pkl"


@pytest.mark.parametrize(
    "ground_truth,pred,precision,recall,fbeta",
    [(np.ones(5), np.ones(5), 1.0, 1.0, 1.0), (np.zeros(5), np.ones(5), 0.0, 1.0, 0.0)],
)
def test_compute_model_metrics(ground_truth, pred, precision, recall, fbeta):
    precision_, recall_, fbeta_ = compute_model_metrics(ground_truth, pred)
    assert precision == precision_ and recall == recall_ and fbeta == fbeta_


def test_model_inference_dtype():
    model = load_model(model_path)
    X = np.ones((1, 108))
    pred = inference(model, X)
    assert isinstance(pred, np.ndarray)


def test_model_inference_shape():
    model = load_model(model_path)
    X = np.ones((20, 108))
    pred = inference(model, X)
    assert pred.shape == (20,)
