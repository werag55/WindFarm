# WindFarm Profitability Analysis

This project analyzes the profitability of offshore wind projects in various regions of the world. It provides a complete pipeline from raw data processing to profitability calculation, modeling, and visualization.

## Features

- **Data Cleaning and Preparation:** Cleans and prepares raw wind farm data, handling missing values and inconsistencies.
- **Data Enrichment:** Enriches the dataset with environmental data from the Global Wind Atlas.
- **Profitability Calculation:** Calculates key financial and operational metrics, including CAPEX, OPEX, and Levelized Cost of Energy (LCOE).
- **Predictive Modeling:** Trains a linear regression model to predict LCOE for new locations.
- **Interactive Visualizations:** Generates an interactive map to visualize wind farm profitability and a dashboard to analyze the model's performance.

## Pipeline

The project's main pipeline consists of the following steps:

1.  **Data Preparation:** The raw dataset is cleaned, enriched with environmental data, and indexed for inflation.
2.  **Calculations:** Financial and operational metrics are calculated for each project.
3.  **Profitability Map:** An interactive map is generated to visualize the LCOE of different wind farms.
4.  **Model Training and Evaluation:** A linear regression model is trained to predict LCOE, and its performance is evaluated.

## Installation & Usage

Install the required dependencies:

```bash
pip install -r src/requirements.txt
```

To run the full pipeline, execute the `main.py` script:

```bash
py -m src.main
```

This will perform all the steps from data preparation to model training and generate the output files in the `results` directory.

## Configuration

The project's configuration is managed in the `src/config.py` file. This file contains constants for data paths, model parameters, and other settings.

## To-Do

- [ ] Fill missing data (marked in red in `../data/european_offshore_wind_capex.xlsx`)
- [ ] Add more prediction models
- [ ] Perform sensitivity analysis
- [ ] Enhance logging throughout the pipeline
- [ ] Prepare country -> foundation scope map
- [ ] Implement actual wave height fetching
- [ ] Verify / Adjust inflation indexation
