# WindFarm Profitability Analysis

This project analyzes the capital expenditure (CAPEX) and profitability of European offshore wind farms. It provides a data processing pipeline, a predictive model for CAPEX, and an interactive web-based UI for visualizing the results.

## Features

- **Data Preprocessing**: Cleans, enriches, and prepares raw data for analysis.
- **Profitability Calculation**: Calculates key financial metrics for wind farm projects.
- **Predictive Modeling**: Trains a model to predict CAPEX for new projects.
- **Interactive Dashboard**: A Streamlit-based user interface to explore the data and model predictions.
- **Profitability Map**: Visualizes the profitability of wind farms on a map of Europe.

## Installation & Usage

Install Python 3.12.0 and the required dependencies:

```bash
pip install -r src2/requirements.txt
```

To run the full pipeline, execute the `main.py` script:

```bash
py -m src2.main
```

This will launch the Streamlit application in [your web browser](http://localhost:8050/).

### Important

**Data Preprocessing** and **Profitability Calculation** stages run only if there are no `cleaned_european_offshore_wind_capex.csv` and `calculated_offshore_wind_data.csv` files in `/data` folder.
It prevents overwriting the existing data and allows to save time when you want to quickly test changes in the UI or model. If you want to re-run the preprocessing or calculations, simply delete the corresponding files from the `/data` folder.

## Configuration

The project's configuration is managed in the `src2/config.py` file. This file contains constants for data paths, model parameters, and other settings.

## To-Do

- [ ] If possible fetch water depth based on location (add in `enrichment.py`, fill `NaN`s in historical data, fetch based on location for new sample and remove user input for `water_depth_min_m` and `water_depth_max_m`)

- [ ] If possible calculate distance from shore based on location (fill `NaN`s in historical data, calculate based on location for new sample and remove user input for `distance_from_shore_min_km` and `distance_from_shore_max_km`)

- [ ] Analyse `foundation_type` column - should we merge some unique values into one category? if so implement is as a part of the preprocessing

- [ ] Check budget parsing
  - how should we handle cases before 1999 when EUR didn't exist (`parsing.py`)
  - decide what to do with values like `PLN 30 billion (combined 2+3)` (drop rows? split cost? ??)
  - 
- [ ] Implement better prediction models, evaluate and visualise results.

- [ ] Wykorzystanie danych o projektach z okresu starszego niż 5 lat. Przykładowe wykorzystanie danych: (1)
      porównanie oszacowania opłacalności projektów na podstawie pełnego zestawu danych z
      oszacowaniem wykonanym tylko na danych z ostatnich 5 lat. Wymagane przedstawienie oceny
      niepewności algorytmu; (2) porównanie oszacowania opłacalności projektów przy różnych indeksacjach
      kosztów i parametrów starszych projektów.

- [ ] Enhence logging.

- [ ] Prettify UI.

- [ ] Sensitivity analysis?

- [ ] Review what was already done in case there are any errors :)
