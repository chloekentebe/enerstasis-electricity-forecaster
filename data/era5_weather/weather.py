# Extracting weather data from ERA5 using the API request provided on the Climate Data Store

import cdsapi
from calendar import monthrange
from pathlib import Path

dataset = "reanalysis-era5-single-levels-timeseries"
request = {
    "variable": [
        "2m_dewpoint_temperature",
        "surface_solar_radiation_downwards",
        "2m_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind"
    ],
    "data_format": "csv"
}

client = cdsapi.Client()

IESO_ZONE_COORDS = {
    "Northwest": (48.3809, -89.2477), # thunder bay
    "Northeast": (46.4917, -80.9930), # sudbury
    "Ottawa": (45.4215, -75.6972), 
    "East": (44.2312, -76.4860), # kingston
    "Toronto": (43.6532, -79.3832),
    "Essa": (44.3160, -79.8830), # angus (essa township)
    "Bruce": (44.1730, -81.6360), # kincardine
    "Southwest": (42.9849, -81.2453), # london
    "Niagara": (43.1594, -79.2469), # st. catharines
    "West": (43.2557, -79.8711), # hamilton
}

dir = Path("data/era5_weather")
dir.mkdir(parents=True, exist_ok=True)

for zone, (lat, lon) in IESO_ZONE_COORDS.items():
    request["location"] = {"latitude": lat, "longitude": lon}

    for year in range(2015, 2027):
        last_month = 6 if year == 2025 else 12

        for month in range(1, last_month + 1):
            days = monthrange(year, month)[1]

            if year == 2026 and month == 6:
                days = 17 # limit from era5

            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{days:02d}"
            request["date"] = [f"{start_date}/{end_date}"]
            filename = dir / f"{zone}_{year}_{month:02d}.csv"
            
            client.retrieve(
                dataset,
                request).download(str(filename))