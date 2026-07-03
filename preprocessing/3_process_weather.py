# an intermediate dataset that includes engineered features
# relative humidity, wind speed, heating/cooling degree hours, snow indicator, rolling weather

from pathlib import Path
import pandas as pd
import numpy as np

BASE_T = 18.0

data_dir = Path("data/era5_weather")
processed_dir = Path("processed_data")
processed_dir.mkdir(parents=True, exist_ok=True)

files = sorted(data_dir.glob("*.csv"))

dataframes = []
for file in files:
    df = pd.read_csv(file, compression="zip")
    df.columns = df.columns.str.strip()
    zone = file.stem.split("_")[0]
    df["zone"] = zone
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)
df = df.rename(columns={
    "u10": "u_wind",
    "v10": "v_wind",
    "d2m": "dewpoint_temp",
    "t2m": "temp",
    "valid_time": "timestamp",
})

print(df.columns)
# output from printing
# Index(['valid_time', 'u10', 'v10', 'd2m', 't2m', 'ssrd', 'tp', 'latitude', 'longitude'], dtype='str')
df["total_precip_mm"] = df["tp"] * 1000
df["solar_rad_wm2"] = df["ssrd"] / 3600
df["temp"] = df["temp"] - 273.15
df["dewpoint_temp"] = df["dewpoint_temp"] - 273.15

temp = df["temp"]
temp_dew = df["dewpoint_temp"]

# compute engineered features
df["relative_humidity"] = (
    100 *
    np.exp((17.625*temp_dew)/(243.04+temp_dew)) /
    np.exp((17.625*temp)/(243.04+temp))
)
df["wind_speed"] = np.sqrt(df["u_wind"]**2 + df["v_wind"]**2)
df["snow_indicator"] = ((df["temp"] <= 0) & (df["total_precip_mm"] > 0).astype(int)) # boolean (1 or 0)
# 18 is the base temperature
df["heating_deg_hrs"] = ((BASE_T - df["temp"]).clip(lower=0))
df["cooling_deg_hrs"] = ((df["temp"] - BASE_T).clip(lower=0))
# rolling weather features
df["rolling_24temp"] = (df["temp"].rolling(window=24, min_periods=1).mean())
df["rolling_24rh"] = (df["relative_humidity"].rolling(window=24, min_periods=1).mean())
df["rolling_24ws"] = (df["wind_speed"].rolling(window=24, min_periods=1).mean())
df["rolling_24solar"] = (df["solar_rad_wm2"].rolling(window=24, min_periods=1).mean())
df["rolling_24tp"] = (df["total_precip_mm"].rolling(window=24, min_periods=1).sum())

df = df.drop(columns=["latitude", "longitude", "tp", "u_wind", "v_wind", "ssrd"])
df.insert(0, "zone", df.pop("zone"))
output_file = processed_dir / "weather.csv"
df.to_csv(output_file, index=False)