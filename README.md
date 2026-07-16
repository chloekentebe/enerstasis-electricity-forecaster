# enerstasis-electricity-forecaster

AI based electricty demand forecaster that uses a Temporal Fusion Transformer model with a scenario engine for future and capacity planning.

## Dataset

This project uses data from multiple sources:
- IESO
- ERA5
- Ontario Ministry of Education, Ministry of Finance, Ministry of Energy and Mines
- Canadian Disaster Database, Statistics Canada, Natural Resources Canada
- City of Toronto

## Repository Structure

```text
data/
processed_data/
preprocessing/
primry_model/
xgboost_model/
notebooks/
results/
docs/
