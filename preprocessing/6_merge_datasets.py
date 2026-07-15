# process ALL csv files from processed_data
# pivot weather data into wide format
# use one hot encoding for categorical coumns (season)

from pathlib import Path
import pandas as pd
import numpy as np

directory = Path("processed_data")

file = directory / "calendar.csv"
calendar = pd.read_csv(file)

file = directory / "demand.csv"
demand = pd.read_csv(file)

file = directory / "se_infra.csv"
si = pd.read_csv(file)

file = directory / "weather.csv"
weather = pd.read_csv(file)

file = directory / "zonal_demand.csv"
zd = pd.read_csv(file)

# pivot weather df
weather_wide = weather.pivot(index="timestamp", columns="zone")
weather_wide.columns = [f"{zone.lower()}_{feature}" for feature, zone in weather_wide.columns]
print(weather_wide.head())
weather_wide = weather_wide.reset_index()

# consolidated demand
demand = demand.iloc[:-1]
zd = zd.drop(
    columns=[
        "year", "month", "day", "hour"
    ]
)

demand = demand.drop(
    columns=[
        "year", "month", "day", "hour"
    ]
)

cs_demand = demand.merge(
    zd,
    on=["timestamp", "ontario_demand_mw"],
    how="left",
)

zone_cols_inter = [
    "northwest_mw",
    "northeast_mw",
    "ottawa_mw",
    "east_mw",
    "toronto_mw",
    "essa_mw",
    "bruce_mw",
    "southwest_mw",
    "niagara_mw",
    "west_mw",
    "zone_total_mw",
    "ontario_zone_total_difference_mw",
    "northwest_ppd",
    "northwest_ppd",
    "ottawa_ppd",
    "east_ppd",
    "toronto_ppd",
    "essa_ppd",
    "bruce_ppd",
    "southwest_ppd",
    "niagara_ppd",
    "west_ppd",
]

zone_cols = [
    "northwest",
    "northeast",
    "ottawa",
    "east",
    "toronto",
    "essa",
    "bruce",
    "southwest",
    "niagara",
    "west",
]

for zc in zone_cols_inter:
    cs_demand[zc] = pd.to_numeric(cs_demand[zc], errors="coerce")
    cs_demand[zc] = cs_demand[zc].interpolate(method="linear")
    cs_demand[zc] = cs_demand[zc].ffill().bfill()

# recompute ppd after interpolation
for zone in zone_cols:
    cs_demand[f"{zone}_ppd"] = (cs_demand[f"{zone}_mw"] / cs_demand["ontario_demand_mw"])
cs_demand = cs_demand.drop(columns = ["ontario_zone_total_difference_mw", "zone_total_mw"])

main = cs_demand.merge(
    weather_wide,
    on="timestamp",
    how="left",
)

main = main.merge(
    calendar,
    on="timestamp",
    how="left",
)

main = main.merge(
    si,
    on="timestamp",
    how="left"
)
# cap dataset to weather length
weather_cap = weather_wide["timestamp"].max()
main = main[main["timestamp"] <= weather_cap]
main = main.reset_index(drop=True)

# one-hot encode cateogircal columns (season)
main = pd.get_dummies(
    main, columns=["season"], dtype=int,
)
print("main isna:", main.isna().sum())
#print("main shape:", main.shape)

print("num of duplication timestamps:", main["timestamp"].duplicated().sum())
main = main.sort_values("timestamp")
print(main.dtypes)
timestamp_check = pd.date_range(
    start=main["timestamp"].min(), end=main["timestamp"].max(), freq="h"
)
main["timestamp"] = pd.to_datetime(main["timestamp"])
print(len(timestamp_check), len(main))
missing = timestamp_check.difference(main["timestamp"])
print(len(missing))
print(missing) # because there is not demand data for 2025-01-01 00:00

# fill ev columns with 0 because those values are recorded starting from 2017-01-01
cols = [
    "bev_registration",
    "phev_registration"
]
main.loc[main["timestamp"] < "2017-01-01", cols] = 0

const_cols = [
    c for c in main.columns if c != "timestamp" and main[c].nunique() <= 1
]
print("const cols:", const_cols)
bool_cols = ["bruce_snow_indicator",
    "east_snow_indicator",
    "essa_snow_indicator", 
    "niagara_snow_indicator",
    "northeast_snow_indicator", 
    "northwest_snow_indicator", 
    "ottawa_snow_indicator", 
    "southwest_snow_indicator",
    "toronto_snow_indicator", 
    "west_snow_indicator"
]
for c in bool_cols:
    main[c] = main[c].astype(str).str.strip().str.lower()
    main[c] = main[c].map({'true': 1, 'false': 0})

main["monthly_peak_time"] = pd.to_datetime(main["monthly_peak_time"], errors="coerce")
main["monthly_min_time"] = pd.to_datetime(main["monthly_min_time"], errors="coerce")

monthly_peak_hour = main["monthly_peak_time"].dt.hour + (main["monthly_peak_time"].dt.minute / 60.0)
monthly_min_hour = main["monthly_min_time"].dt.hour + (main["monthly_min_time"].dt.minute / 60.0)

main["monthly_peak_hour_sin"] = np.sin(2*np.pi*monthly_peak_hour/24.0)
main["monthly_peak_hour_cos"] = np.cos(2*np.pi*monthly_peak_hour/24.0)

main["monthly_min_hour_sin"] = np.sin(2*np.pi*monthly_min_hour/24.0)
main["monthly_min_hour_cos"] = np.cos(2*np.pi*monthly_min_hour/24.0)

main = main.drop(columns=["monthly_peak_time", "monthly_min_time"])
main = main.drop(columns=["year", "month", "day", "hour", "day_of_week", "day_of_year"])

# 24-hour forecast profile
for hour in range(1, 25):
    main[f"demand_target_hour_{hour}"] = main["ontario_demand_mw"].shift(-hour)

main["demand_lag_1"] = main["ontario_demand_mw"].shift(1)
main["demand_lag_24"] = main["ontario_demand_mw"].shift(24)
main["demand_lag_168"] = main["ontario_demand_mw"].shift(168)
main["demand_rolling24_mean"] = main["ontario_demand_mw"].rolling(window=24).mean()
main["demand_rolling24_std"] = main["ontario_demand_mw"].rolling(window=24).std()
print(main.isna().sum())
main = main.dropna().reset_index(drop=True)
print(main.isna().sum())

print("main shape:", main.shape)
print(main.describe()) # statistics

output_file = directory / "main_dataset.csv"
main.to_csv(output_file, index=False)
main.describe().T.index.name = "Feature"
main.describe().T.to_csv(directory / "main_statistics.csv")