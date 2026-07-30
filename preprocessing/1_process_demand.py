# create an intermediate dataset with columns
# timestamp,ontario_demand_mw,market_demand_mw,year,month,day,hour,monthly_peak_mw,monthly_peak_time,monthly_min_mw_monthly_min_time

from pathlib import Path # for extracting files in data/ieso_demand
import pandas as pd # for tabular data

data_dir = Path("data/ieso_demand")
processed_dir = Path("processed_data")
processed_dir.mkdir(parents=True, exist_ok=True)

files = sorted(data_dir.glob("PUB_Demand_*.csv")) # returns a list of all files

dataframes = []
for file in files:
    df = pd.read_csv(file, comment='\\')
    dataframes.append(df)

# stack dfs vertifcally
df = pd.concat(dataframes, ignore_index=True)

df = df.rename(columns={
    "Ontario Demand": "ontario_demand_mw",
    "Market Demand": "market_demand_mw"
})

df["Hour"] = df["Hour"] - 1 # want time ot begin at 00:00
df["timestamp"] = (pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"], unit="h"))

# create additional time-based columns
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour

# compute max ontario demand
monthly_peak = (
    df.groupby(["year", "month"])["ontario_demand_mw"]
    .max() .reset_index(name="monthly_peak_mw")
)
peak_idx = (
    df.groupby(["year", "month"])["ontario_demand_mw"]
    .idxmax() # row index of maximum demand mw
)
monthly_peak_times = (
    df.loc[peak_idx, ["year", "month", "timestamp"]]
    .rename(columns={"timestamp":"monthly_peak_time"})
)


# compute min ontario demand
monthly_min = (
    df.groupby(["year", "month"])["ontario_demand_mw"]
    .min() .reset_index(name="monthly_min_mw")
)
min_idx = (
    df.groupby(["year", "month"])["ontario_demand_mw"]
    .idxmin()
)
monthly_min_times = (
    df.loc[min_idx, ["year", "month", "timestamp"]]
    .rename(columns={"timestamp": "monthly_min_time"})
)

# merge main dataframe, monthly peaks, monthly mins, peak times, min times
# each mini dataframe shares year and month
df = df.merge(
    monthly_peak,
    on=["year", "month"],
    how="left"
)
df = df.merge(
    monthly_peak_times,
    on=["year", "month"],
    how="left"
)
df = df.merge(
    monthly_min,
    on=["year", "month"],
    how="left"
)
df = df.merge(
    monthly_min_times,
    on=["year","month"],
    how="left"
)

# drop date and hour because they're not needed and timestamps exist
df = df.drop(columns=["Date", "Hour"])
df = df[[
    "timestamp",
    "ontario_demand_mw",
    "market_demand_mw",
    "year",
    "month",
    "day",
    "hour",
    "monthly_peak_mw",
    "monthly_peak_time",
    "monthly_min_mw",
    "monthly_min_time",
]]

output_file = processed_dir / "demand.csv"
df.to_csv(output_file, index=False)