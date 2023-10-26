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


def compute_column_slice_performance(model, X, y, df, column):
    # First make sure the dataframe index is a RangeIndex going from 0 to len(df)-1
    df_ = df.copy()
    df_.reset_index(drop=True, inplace=True)

    # Next make sure X and df have the same length
    if X.shape[0] != df_.shape[0]:
        raise Exception(f"Expected processed data X and original dataframe df to have \
the same length. Got {X.shape[0]=} and {df.shape[0]=}")

    # Start by getting unique values for the column
    slices = df_[column].unique().tolist()
    column_slice_perf = {}
    for slice_ in slices:
        # Grab only the data for the desired slice
        slice_index = df_[df_[column] == slice_].index
        X_slice = X[slice_index, :]
        y_slice = y[slice_index]

        # Compute metrics on the slice
        y_pred = model.predict(X_slice)
        prec, recall, fbeta = compute_model_metrics(y_slice, y_pred)
        column_slice_perf[slice_] = {"slice_size": X_slice.shape[0], "precision": prec, "recall": recall, "fbeta": fbeta}

    return column_slice_perf

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