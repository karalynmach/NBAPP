import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.features import split_X_y, FEATURE_COLS
from src.utils import regression_metrics

def train_and_evaluate(dataset_df, models_dir: str) -> dict:
    X, y = split_X_y(dataset_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline: Ridge Regression
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    # XGBoost model
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4
    )

    ridge.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    ridge_pred = ridge.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    results = {
        "ridge": regression_metrics(y_test, ridge_pred),
        "xgboost": regression_metrics(y_test, xgb_pred)
    }

    # Save models + features for later inference
    joblib.dump(ridge, f"{models_dir}/ridge.pkl")
    joblib.dump(xgb, f"{models_dir}/xgboost.pkl")
    joblib.dump(FEATURE_COLS, f"{models_dir}/feature_cols.pkl")

    return results