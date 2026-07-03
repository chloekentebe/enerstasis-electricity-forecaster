# process school breaks and disaster events and add hour, day, weekend, holidays, season

from pathlib import Path
import pandas as pd
import holidays
import numpy as np

data_dir = Path("data/calendar")
processed_dir = Path("processed_data")
processed_dir.mkdir(parents=True, exist_ok=True)

file = data_dir / "school_breaks.csv"
school_df = pd.read_csv(file, comment='\\')

# create master dataframe with the same time interval as era5 weather
calendar = pd.DataFrame({
    "timestamp": pd.date_range(
        start="2015-01-01 00:00",
        end="2026-06-17 23:00",
        freq="h"
    )
})

calendar["year"] = calendar["timestamp"].dt.year
calendar["month"] = calendar["timestamp"].dt.month
calendar["day"] = calendar["timestamp"].dt.day
calendar["hour"] = calendar["timestamp"].dt.hour
calendar["day_of_week"] = calendar["timestamp"].dt.dayofweek
calendar["day_of_year"] = calendar["timestamp"].dt.dayofyear

# weekend
calendar["is_weekend"] = (calendar["day_of_week"] >= 5).astype(int)

# holidays
holidays = holidays.CA(subdiv="ON", years=range(2015,2027))
calendar["is_holiday"] = (calendar["timestamp"].dt.date.isin(holidays).astype(int))

def get_season(month):
    if month in [12,1,2]:
        return "winter"
    elif month in [3,4,5]:
        return "spring"
    elif month in [6,7,9]:
        return "summer"
    else:
        return "fall"

calendar["season"] = calendar["month"].apply(get_season)
school_df["start_break"] = pd.to_datetime(school_df["start_break"])
school_df["end_break"] = pd.to_datetime(school_df["end_break"])
calendar["school_break"] = 0
calendar["break_type"] = None

for _,row in school_df.iterrows():
    mask = ((calendar["timestamp"] >= row["start_break"]) & (calendar["timestamp"] <= row["end_break"] + pd.Timedelta(days=1)))

    calendar.loc[mask, "school_break"] = 1
    calendar.loc[mask, "break_type"] = row["break"]
calendar=pd.get_dummies(
    calendar,
    columns=["break_type"],
    prefix="break",
    dtype=int
)

# DISASTER DATAFRAME
disaster_file = data_dir / "CanadianDisasterDatabase.xlsx"
disaster_df = pd.read_excel(disaster_file)
disaster_df.columns = disaster_df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

disaster_df = disaster_df [
    disaster_df["PROVINCES_AFFECTED / PROVINCES AFFECTÉES"].str.contains("ON", na=False)
]

disaster_df["EVENT_START_DATE"] = pd.to_datetime(
    disaster_df["EVENT_START_DATE"]
)
disaster_df["EVENT_END_DATE"] = pd.to_datetime(
    disaster_df["EVENT_END_DATE"]
)
calendar["extreme_weather"] = 0
calendar["geological_event"] = 0
calendar["technology_event"] = 0
calendar["conflict_event"] = 0

mask = (calendar["timestamp"].isin(disaster_df["EVENT_START_DATE"]))
event_mask = disaster_df["EVENT_TYPE_DESCRIPTION"].isin([
    "Storms and Severe Thunderstorms",
    "Flood",
    "Wildfire",
    "Drought",
    "Heat Event",
])
combined_mask = mask & event_mask
calendar.loc[combined_mask, "extreme_weather"] = 1

event_mask = disaster_df["EVENT_TYPE_DESCRIPTION"].isin([
    "Landslide",
])
combined = mask & event_mask
calendar.loc[combined_mask, "geological_event"] = 1

event_mask = disaster_df["EVENT_SUBGROUP_NAME"].isin([
    "Fire",
    "Explosion",
    "Infrastructure failure",
    "Transportation accident",
    "Hazardous Chemicals",
])
combined = mask & event_mask
calendar.loc[combined_mask, "technology_event"] = 1
event_mask = disaster_df["EVENT_SUBGROUP_NAME"].isin([
    "Civil Incident",
    "Terrorist",
    "Arson",
])
combined = mask & event_mask
calendar.loc[combined_mask, "conflict_event"] = 1

# CYCLICAL ENCODING
calendar["hour_sin"] = np.sin(2*np.pi*calendar["hour"]/24)
calendar["hour_cos"] = np.cos(2*np.pi*calendar["hour"]/24)
calendar["dow_sin"] = np.sin(2*np.pi*calendar["day_of_week"]/7)
calendar["dow_cos"] = np.cos(2*np.pi*calendar["day_of_week"]/7)
calendar["month_sin"] = np.sin(2*np.pi*calendar["month"]/12)
calendar["month_cos"] = np.cos(2*np.pi*calendar["month"]/12)
calendar["doy_sin"] = np.sin(2*np.pi*calendar["day_of_year"]/365.25)
calendar["doy_cos"] = np.cos(2*np.pi*calendar["day_of_year"]/365.25)

output_file = processed_dir / "calendar.csv"
calendar.to_csv(output_file, index=False)