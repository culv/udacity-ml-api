import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 111


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_model(model, fpath):
    """Save the model as a pickle file to the given filepath.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier or sklearn.preprocessing.OneHotEncoder
        Trained machine learning model or encoder.
    fpath : Path
        Location to save the model.
    Returns
    -------
    """
    joblib.dump(model, fpath)

def load_model(fpath):
    """Load the pickle file at the given filepath.

    Inputs
    ------
    fpath : Path
        Location to save the model.
    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier or sklearn.preprocessing.OneHotEncoder
        Trained machine learning model or encoder.
    """
    model = joblib.load(fpath)
    return model