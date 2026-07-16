from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error)

directory = Path("processed_data")
file = directory / "main_dataset.csv"

main = pd.read_csv(file)

# chronological split
train_set = main[main["timestamp"] < "2022-12-31"] # 8 years / ~11.465 years = 70%
val_set = main[(main["timestamp"] >= "2023-01-01") & (main["timestamp"] < "2025-01-01")] # 2 years / ~11.465 years = 17.4%
test_set = main[main["timestamp"] >= "2025-01-01"] # 1.465 years / 11.465 years = 12.8%
print(len(train_set))
print(len(val_set))
print(len(test_set))

target_cols = [f"demand_target_hour_{hour}" for hour in range(1, 25)]
feature_cols = [
    col for col in main.columns if col not in
    ["timestamp", "ontario_demand_mw"] + target_cols
]

training_data = train_set[feature_cols]
training_labels = train_set[target_cols]

val_data = val_set[feature_cols]
val_labels = val_set[target_cols]

testing_data = test_set[feature_cols]
testing_labels = test_set[target_cols]

print(training_labels.shape)
print(val_labels.shape)
print(testing_labels.shape)

# insantiate model
model = MultiOutputRegressor(
    XGBRegressor(
        random_state=42,
        n_jobs = -1,
    )
)

params = {
    "estimator__n_estimators": [300, 500, 800, 1200],
    "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "estimator__max_depth": [4, 6, 8, 10],
    "estimator__min_child_weight": [1, 3, 5, 7],
    "estimator__subsample": [0.7, 0.8, 0.9, 1.0], # prevent overfitting
    "estimator__colsample_bytree": [0.6, 0.8, 1.0], # prevent overfitting
    "estimator__gamma": [0, 0.1, 0.5, 1], # minimum decrease in loss for algo to make another partition on leaf node
    "estimator__reg_alpha": [0, 0.01, 0.1, 1],
    "estimator__reg_lambda": [1, 2, 5, 10],
}

tscv = TimeSeriesSplit(n_splits=5)

# set up random search
rand_srch = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=20,
    scoring="neg_mean_squared_error", # industry standard
    cv=tscv, # prevents timeline getting shuffled
    verbose=2,
    random_state=42
)

# fit model
rand_srch.fit(training_data, training_labels,
    verbose=False
)
print("Best parameters:", rand_srch.best_params_)
best_model = rand_srch.best_estimator_
direc = Path("xgboost_model")
fl = direc / "forecast_best_xgboost_model_j11.json"
best_model.save_model(fl)
feature_columns = list(training_data.columns)
fl2 = direc / "forecast_feature_columns_j11.json"
with open(fl2, "w") as f:
    json.dump(feature_columns, f)

xgboost_params = {
    key.replace("estimator__", ""): value
    for key, value in rand_srch.best_params_.items()
}

# Best parameters: {'estimator__subsample': 0.9, 'estimator__reg_lambda': 5, 'estimator__reg_alpha':
# 0, 'estimator__n_estimators': 1200, 'estimator__min_child_weight': 3, 'estimator__max_depth': 6,
# 'estimator__learning_rate': 0.03, 'estimator__gamma': 0.1, 'estimator__colsample_bytree': 0.8}

final_model = MultiOutputRegressor(
    XGBRegressor(
        subsample=0.9,
        reg_lambda=5,
        reg_alpha=0, 
        n_estimators=1200, 
        min_child_weight=3, 
        max_depth=6,
        learning_rate=0.03, 
        gamma=0.1, 
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
)

final_model.fit(training_data, training_labels)

direc = Path("xgboost_model")
fl = direc / "forecast_best_xgboost_model_j12.joblib"
joblib.dump(final_model, fl)
feature_columns = list(training_data.columns)
fl2 = direc / "forecast_feature_columns_j12.json"
with open(fl2, "w") as f:
    json.dump(feature_columns, f)

# predictions
validation_prediction = final_model.predict(val_data)
testing_prediction = final_model.predict(testing_data)

def eval(labels, prediction):
    return {
        "MAE": mean_absolute_error(labels, prediction),
        "MAPE": mean_absolute_percentage_error(labels, prediction) * 100,
        "RMSE": np.sqrt(mean_squared_error(labels, prediction)),
        "R2": r2_score(labels, prediction)
    }

print(f"XGBOOST VALIDATION METRICS")
validation_metrics = eval(val_labels, validation_prediction)
print(validation_metrics)

print(f"XGBOOST TEST METRICS")
test_metrics = eval(testing_labels, testing_prediction)
print(test_metrics)

hourly_metrics = []
for hour in range(24):
    hourly_metrics.append({
        "Horizon": hour + 1,
        "Validation MAE":
            mean_absolute_error(
                val_labels.iloc[:,hour],
                validation_prediction[:,hour],
            ),
        "Validation MAPE":
            mean_absolute_percentage_error(
                val_labels.iloc[:,hour],
                validation_prediction[:,hour],
            )*100,
        "Validation RMSE":
            np.sqrt(
                mean_squared_error(
                    val_labels.iloc[:,hour],
                    validation_prediction[:,hour],
                )
            ),
        "Validation R2":
            r2_score(
                val_labels.iloc[:,hour],
                validation_prediction[:,hour],
            ),
        "Test MAE":
            mean_absolute_error(
                testing_labels.iloc[:,hour],
                testing_prediction[:,hour],
            ),
        "Test MAPE":
            mean_absolute_percentage_error(
                testing_labels.iloc[:,hour],
                testing_prediction[:,hour],
            )*100,
        "Test RMSE":
            np.sqrt(
                mean_squared_error(
                    testing_labels.iloc[:,hour],
                    testing_prediction[:,hour],
                )
            ),
        "Test R2":
            r2_score(
                testing_labels.iloc[:,hour],
                testing_prediction[:,hour],
            ),
    })
hor_metrics = pd.DataFrame(hourly_metrics)
print(hor_metrics)

directory = Path("xgboost_model")
output_file = directory / "horizon_metrics_2.csv"
hor_metrics.to_csv(output_file, index=False)