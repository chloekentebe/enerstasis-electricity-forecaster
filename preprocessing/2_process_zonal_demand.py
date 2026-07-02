# creating an intermediate dataset that also includes
# zone percentage of provincial demand

from pathlib import Path
import pandas as pd

data_dir = Path("data/ieso_demand")
processed_dir=Path("processed_data")
processed_dir.mkdir(parents=True, exist_ok=True)

files = sorted(data_dir.glob("PUB_DemandZonal_*.csv"))

dataframes = []
for file in files:
    df = pd.read_csv(file, comment='\\')
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)

df = df.rename(columns={
    "Ontario Demand": "ontario_demand_mw",
    "Zone Total": "zone_total_mw",
    "Diff": "ontario_zone_total_difference_mw",
})

zone_columns = [
    "Northwest",
    "Northeast",
    "Ottawa",
    "East",
    "Toronto",
    "Essa",
    "Bruce",
    "Southwest",
    "Niagara",
    "West",
]

df = df.rename(
    columns={c: f"{c.lower()}_mw" for c in zone_columns}
)

df["Hour"] = df["Hour"] - 1
df["timestamp"] = (pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"], unit="h"))
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour

zone_mw_columns = [f"{zone.lower()}_mw" for zone in zone_columns]
for c in zone_mw_columns:
    # ppd = percentage of provincial demand
    ppd_column = c.replace("_mw", "_ppd")
    df[ppd_column] = (df[c] / df["ontario_demand_mw"])

df = df.drop(columns=["Date", "Hour"])
df.insert(0, "timestamp", df.pop("timestamp"))

output_file = processed_dir / "zonal_demand.csv"
df.to_csv(output_file, index=False)