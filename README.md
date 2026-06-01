# WindFarm Profitability Analysis

This project analyzes the capital expenditure (CAPEX) and profitability of European offshore wind farms. It provides a data processing pipeline, a predictive model for CAPEX / MW, a farms similarity analysis engine, and an interactive web-based UI for visualizing the results.

## Features

- **Data Preprocessing**: Cleans, enriches, and prepares raw data for analysis.
- **Profitability Calculation**: Calculates key financial metrics for wind farm projects.
- **Predictive Modeling**: Trains a model to predict CAPEX / MW for new projects.
- **Similarity Analysis**: Determines the similarity between farms.
- **Interactive Dashboard**: A Streamlit-based user interface to explore the data and model predictions.
- **Profitability Map**: Visualizes the profitability of wind farms on a map of Europe.
- **Similarity Table**: Summarizes the farms that are most similar to the one proposed.

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

