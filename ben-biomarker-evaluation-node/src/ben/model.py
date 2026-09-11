from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURE_COLUMNS

def build_model(C: float = 1.0, max_iter: int = 2000, class_weight: str = "balanced") -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preprocess = ColumnTransformer([("numeric", numeric, FEATURE_COLUMNS)], remainder="drop")
    classifier = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight, solver="liblinear", random_state=42)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])
