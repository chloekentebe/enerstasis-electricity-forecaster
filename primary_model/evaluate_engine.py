import torch
from scenario_engine import ScenarioEngine, extract_windows
import pandas as pd
from pathlib import Path
from training import classify_feature_names
from model import TemporalFusionTransformer

directory = Path("processed_data")
file = directory / "main_dataset.csv"
main = pd.read_csv(file)

impor_dir = Path("notebooks")
f = impor_dir / "xgboost_feature_importance.csv"
df_import = pd.read_csv(f)
top_feats = df_import.sort_values("importance", ascending=False).head(100)["feature"].tolist()
keep = ["ontario_demand_mw", "season_fall", "phev_registration", "total_large_load_mw"]
main_trimmed = main[top_feats + keep].copy()
print(main_trimmed)

# chronological split
train_set = main[main["timestamp"] < "2022-12-31"] 
val_set = main[(main["timestamp"] >= "2023-01-01") & (main["timestamp"] < "2025-01-01")]
test_set = main[main["timestamp"] >= "2025-01-01"]

print(f"length of train/val/test set: {len(train_set), len(val_set), len(test_set)}")

train_set = train_set[top_feats + keep]
val_set = val_set[top_feats + keep]
test_set = test_set[top_feats + keep]

features = top_feats + keep
# USING TRAIN SET MEAN AND STD for z-score normalization
f_mean = train_set[features].mean()
f_std = train_set[features].std()

train_set[features] = (train_set[features] - f_mean) / f_std
val_set[features] = (val_set[features] - f_mean) / f_std
test_set[features] = (test_set[features] - f_mean) / f_std
print(f"f_mean: {f_mean}, f_std:{f_std}")
print(f"f_mean length: {len(f_mean)} f_std length: {len(f_std)}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")
ex_columns=["timestamp"]
future_feature_names, past_feature_names = classify_feature_names(main_trimmed, target_col="ontario_demand_mw", exclude_columns=ex_columns)
print(f"Past feature names: {past_feature_names} of length {len(past_feature_names)}")
print(f"Future feature names: {future_feature_names} of length {len(future_feature_names)}")

d_mean = f_mean["ontario_demand_mw"]
d_std = f_std["ontario_demand_mw"]
cp = torch.load("/Users/chloekentebe/enerstasis-electricity-forecaster/primary_model/lucky_optimized_final_checkpoints/final_trial21_h64_lr6.75e-04_bs32_do0.46_heads4_layers1_wd9.47e-05_gn1.0_epoch28.pt", device)
hs = 64
model = TemporalFusionTransformer(
    past_feature_names, future_feature_names,
    input_size=hs, hidden_size=hs,
    n_heads=cp["config"]["n_heads"], dropout=cp["config"]["dropout"], num_layers=1
).to(device)
model.load_state_dict(cp["model_state"])

scenarios_to_explore = {
    "AI Expansion": {"total_large_load_mw": 1.3},
    "EV Adoption": {"bev_registration": 1.15, "phev_registration": 1.05},
    "Solar Panel Deployment": {"embedded_solar_mw": 1.2}
}

scenario_engine = ScenarioEngine(model, f_mean, f_std, past_feature_names, future_feature_names, device)
keep = ["ontario_demand_mw", "season_fall", "phev_registration", "total_large_load_mw", "timestamp"]
df = main[top_feats + keep].copy()
# July 1 - Canada Day
index = df.index.get_loc(df[df["timestamp"] == "2026-07-01 00:00:00"].index[0])
# this interval is part of the test dataset since it contains df data from 2025 onwards
encoder_df, decoder_df = extract_windows(df, index, past_feature_names, future_feature_names)


results_load = scenario_engine.compare(
    encoder_df, decoder_df, scenarios_to_explore["AI Expansion"],
    test_set, "AI Expansion"
)

results_ev = scenario_engine.compare(
    encoder_df, decoder_df, scenarios_to_explore["EV Adoption"],
    test_set, "EV Adoption"
)

results_solar = scenario_engine.compare(
    encoder_df, decoder_df, scenarios_to_explore["Solar Panel Deployment"],
    test_set, "Solar Panel Deployment"
)

print("*******LOAD RESULTS*******")
if results_load['modification_warning']:
    print("WARNING", results_load['modification_warning'])
print(f"Peak Demand: {results_load['base_peak_mw']} MW --> {results_load['scenario_peak_mw']} MW "
      f"Peak Shift: {results_load['peak_shift_mw']} MW")
print(f"Peak Hour: {results_load['peak_hour_base']} --> {results_load['peak_hour_scenario']} "
      f"Average Shift: {results_load['average_shift_mw']} MW")
# DIFFERENCE BETWEEN UPPER AND LOWER QUANTILE
print(f"Band of Uncertainty: {results_load['bandwidth_base']} MW --> {results_load['bandwidth_scenario']} MW")
print()

print("*******EV RESULTS*******")
if results_ev['modification_warning']:
    print("WARNING", results_ev['modification_warning'])
print(f"Peak Demand: {results_ev['base_peak_mw']} MW --> {results_ev['scenario_peak_mw']} MW "
      f"Peak Shift: {results_ev['peak_shift_mw']} MW")
print(f"Peak Hour: {results_ev['peak_hour_base']} --> {results_ev['peak_hour_scenario']} "
      f"Average Shift: {results_ev['average_shift_mw']} MW")
# DIFFERENCE BETWEEN UPPER AND LOWER QUANTILE
print(f"Band of Uncertainty: {results_ev['bandwidth_base']} MW --> {results_ev['bandwidth_scenario']} MW")
print()

print("*******SOLAR RESULTS*******")
if results_solar['modification_warning']:
    print("WARNING", results_solar['modification_warning'])
print(f"Peak Demand: {results_solar['base_peak_mw']} MW --> {results_solar['scenario_peak_mw']} MW "
      f"Peak Shift: {results_solar['peak_shift_mw']} MW")
print(f"Peak Hour: {results_solar['peak_hour_base']} --> {results_solar['peak_hour_scenario']} "
      f"Average Shift: {results_solar['average_shift_mw']} MW")
# DIFFERENCE BETWEEN UPPER AND LOWER QUANTILE
print(f"Band of Uncertainty: {results_solar['bandwidth_base']} MW --> {results_solar['bandwidth_scenario']} MW")