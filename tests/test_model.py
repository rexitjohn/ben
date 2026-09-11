from sklearn.model_selection import train_test_split

from ben.data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic_canine_dataset
from ben.evaluate import classification_metrics
from ben.model import build_model


def test_model_trains_and_scores():
    df = make_synthetic_canine_dataset(rows=240, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_size=0.25, stratify=df[TARGET_COLUMN], random_state=10)
    model = build_model(); model.fit(X_train, y_train); p = model.predict_proba(X_test)[:,1]
    metrics = classification_metrics(y_test, p)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["n"] == len(y_test)
