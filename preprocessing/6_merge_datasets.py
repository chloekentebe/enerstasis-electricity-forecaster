# process ALL csv files from processed_data
# pivot weather data into wide format
# use one hot encoding for categorical coumns (season)

from pathlib import Path
import pandas as pd

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
print("main shape:", main.shape)

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

output_file = directory / "main_dataset.csv"
main.to_csv(output_file, index=False)