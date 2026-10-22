from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier, XGBRFClassifier
except ImportError:
    XGBClassifier = None
    XGBRFClassifier = None


def paper_knn():
    return KNeighborsClassifier(n_neighbors=2, p=1, weights="distance")


def paper_logistic_regression():
    return LogisticRegression(C=1000, max_iter=2000, penalty="l2")


def paper_sgd():
    return SGDClassifier(
        alpha=0.0001,
        learning_rate="optimal",
        loss="hinge",
        penalty="l2",
    )


def paper_random_forest():
    return RandomForestClassifier(
        criterion="gini",
        max_features="log2",
        n_estimators=1000,
    )


def paper_svc():
    return SVC(C=0.1, coef0=0.0, degree=4, gamma="scale", kernel="poly")


def paper_xgb():
    if XGBClassifier is None:
        raise ImportError("Install xgboost to use this model.")
    return XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        n_estimators=100,
        subsample=0.8,
    )


def paper_xgbrf():
    if XGBRFClassifier is None:
        raise ImportError("Install xgboost to use this model.")
    # The published parameter table also lists `min_samples_leaf=2`.
    # XGBRFClassifier does not expose that sklearn RandomForest parameter,
    # so it is intentionally not silently translated to another XGBoost knob.
    return XGBRFClassifier(
        colsample_bynode=0.6,
        learning_rate=0.01,
        max_depth=5,
        n_estimators=100,
        subsample=1.0,
    )
