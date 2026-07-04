# process all socioeconomic files and the infrastructure load file

from pathlib import Path
import pandas as pd

dir1 = Path("data/infrastructure_load")
dir2 = Path("data/socioeconomics")
processed_dir = Path("processed_data")
processed_dir.mkdir(parents=True, exist_ok=True)

file = dir1 / "loads.csv"
loads = pd.read_csv(file, comment='//', engine='python')

file = dir2 / "employment.csv"
employment = pd.read_csv(file, comment='//', engine='python')

file = dir2 / "ev_registration.csv"
ev = pd.read_csv(file, comment='//', engine='python')

file = dir2 / "gdp.csv"
gdp = pd.read_csv(file, comment='//', engine='python')

file = dir2 / "population_size.csv"
pop = pd.read_csv(file, comment='//', engine='python')

loads["timestamp"] = pd.to_datetime({
    "year": loads["year"],
    "month":1,
    "day": 1,
})
loads = loads.drop(columns=["year"])

employment["timestamp"] = pd.to_datetime({
    "year": employment["year"],
    "month":1,
    "day": 1,
})
employment = employment.drop(columns=["year"])


gdp["timestamp"] = pd.to_datetime({
    "year": gdp["year"],
    "month":1,
    "day": 1,
})
gdp = gdp.drop(columns=["year"])

q2m = {
    1: 1,
    2: 4,
    3: 7,
    4: 10,
}
ev["month"] = ev["quarter"].map(q2m)
ev["timestamp"] = pd.to_datetime({
    "year": ev["year"],
    "month": ev["month"],
    "day": 1
})
ev = ev.drop(columns=["year", "quarter", "month"])

pop["month"] = pop["quarter"].map(q2m)
pop["timestamp"] = pd.to_datetime({
    "year": pop["year"],
    "month": pop["month"],
    "day": 1
})
pop = pop.drop(columns=["year", "quarter", "month"])

print(loads.head())
print(employment.head())
print(ev.head())
print(gdp.head())
print(pop.head())

se_infra = loads \
            .merge(employment, on="timestamp", how="outer") \
            .merge(gdp, on="timestamp", how="outer") \
            .merge(ev, on="timestamp", how="outer") \
            .merge(pop, on="timestamp", how="outer")


timeline = pd.DataFrame({
    "timestamp": pd.date_range(
        "2015-01-01",
        "2026-06-17 23:00",
        freq="h"
    )
})

se_infra = timeline.merge(se_infra, on="timestamp", how="left")
se_infra = se_infra.ffill()
se_infra.insert(0, "timestamp", se_infra.pop("timestamp"))

output_file = processed_dir / "se_infra.csv"
se_infra.to_csv(output_file, index=False)