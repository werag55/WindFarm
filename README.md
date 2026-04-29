# WindFarm Profitability Analysis

This project analyzes the capital expenditure (CAPEX) and profitability of European offshore wind farms. It provides a data processing pipeline, a predictive model for CAPEX, and an interactive web-based UI for visualizing the results.

## Features

- **Data Preprocessing**: Cleans, enriches, and prepares raw data for analysis.
- **Profitability Calculation**: Calculates key financial metrics for wind farm projects.
- **Predictive Modeling**: Trains a model to predict CAPEX for new projects.
- **Interactive Dashboard**: A Streamlit-based user interface to explore the data and model predictions.
- **Profitability Map**: Visualizes the profitability of wind farms on a map of Europe.

## Installation & Usage

Install the required dependencies:

```bash
pip install -r src2/requirements.txt
```

To run the full pipeline, execute the `main.py` script:

```bash
py -m src2.main
```

This will launch the Streamlit application in [your web browser](http://localhost:8050/).

## Configuration

The project's configuration is managed in the `src2/config.py` file. This file contains constants for data paths, model parameters, and other settings.

## To-Do

- [ ] Fetch mean wave height based on location (`enrichment.py`)

- [ ] If possible fetch water depth based on location (add in `enrichment.py`, fill `NaN`s in historical data, fetch based on location for new sample and remove user input for `water_depth_min_m` and `water_depth_max_m`)

- [ ] If possible calculate distance from shore based on location (fill `NaN`s in historical data, calculate based on location for new sample and remove user input for `distance_from_shore_min_km` and `distance_from_shore_max_km`)

- [ ] Calculate distance from nearest port (`add_distance_from_port` in `enrichment.py`) based on the wind farm location and European Offshore Wind Construction Ports locations (see European Offshore Wind Construction Ports.pdf)
      Should we use it as feature? Or only to calculate OPEX? Adjust `config.py` if neccessary

- [ ] Split `turbine_model` into two features in `european_offshore_wind_capex.xlsx`: `turbine_producer` and `turbine_power_MW`

- [ ] Analyse `foundation_type` column - should we merge some unique values into one category? if so implement is as a part of the preprocessing

- [ ] Prepare mapping Country -> categorical columns based on Modele kosztów przyłączenia morskich farm wiatrowych w Europie (ostatnie ~20 lat).pdf. Ensure the data is filled for the training data and user input

- [ ] Check budget parsing
  - how should we handle cases before 1999 when EUR didn't exist (`parsing.py`)
  - decide what to do with values like `PLN 30 billion (combined 2+3)` (drop rows? split cost? ??)

- [ ] Adjust inflation indexation (`indexation.py`)

- [ ] Fill missing data in `area_sqkm` column.

- [ ] Fill missing data in `total_project_budget` column.

- [ ] Implement better prediction models, evaluate and visualise results.

- [ ] Wykorzystanie danych o projektach z okresu starszego niż 5 lat. Przykładowe wykorzystanie danych: (1)
      porównanie oszacowania opłacalności projektów na podstawie pełnego zestawu danych z
      oszacowaniem wykonanym tylko na danych z ostatnich 5 lat. Wymagane przedstawienie oceny
      niepewności algorytmu; (2) porównanie oszacowania opłacalności projektów przy różnych indeksacjach
      kosztów i parametrów starszych projektów.

- [ ] Enhence logging.

- [ ] Prettify UI.

- [ ] Sensitivity analysis?

- [ ] Clustering?

- [ ] Review what was already done in case there are any errors :)
