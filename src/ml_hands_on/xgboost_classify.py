import logging
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils._bunch import Bunch
from xgboost import Booster

logger = logging.getLogger(__name__)
hander = logging.StreamHandler()
if os.environ.get("DEBUG"):
    logger.setLevel(logging.DEBUG)
    hander.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
    hander.setLevel(logging.WARNING)
logger.addHandler(hander)


def calculate_xgboost_accuracy(dataset: Bunch, *, test_size: float = 0.2) -> tuple[float, Booster]:
    logger.debug(f"Dataset description: {dataset.DESCR}")

    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    xgb_train = xgb.DMatrix(X_train, label=y_train, feature_names=dataset.feature_names)
    xgb_test = xgb.DMatrix(X_test, label=y_test, feature_names=dataset.feature_names)
    param = {
        "objective": "binary:logistic",
    }
    model = xgb.train(param, xgb_train)
    y_pred_probability = model.predict(xgb_test)
    y_pred = np.where(y_pred_probability > 0.5, 1, 0)  # adjust thsreshold?

    acc = float(accuracy_score(y_test, y_pred))
    return acc, model


def main() -> None:
    from ml_hands_on import PROJECT_ROOT

    OUTDIR = PROJECT_ROOT / "output"
    OUTDIR.mkdir(exist_ok=True, parents=True)

    cancer_dataset = cast(Bunch, load_breast_cancer())
    acc, model = calculate_xgboost_accuracy(cancer_dataset)

    print(f"acc: {acc}")

    fig, ax1 = plt.subplots(figsize=(20, 10))
    xgb.plot_importance(model, ax=ax1)
    plt.show()
    fig.savefig(str(OUTDIR / "feature_importance.png"))


if __name__ == "__main__":
    main()
