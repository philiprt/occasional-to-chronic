# The Global South is disproportionately vulnerable to rapid increases in coastal flooding
Thompson et al. (2024), *Submitted to Nature Climate Change*, doi:[????](https://doi.org/????)

This repository contains the code and notebooks used to produce the results and figures presented in the paper cited above. For best results, run this code in a virtual environment generated from the requirements.txt file in the root directory of this repository. A description of the contents of this repository follows:

## Tide-gauge sea-level data
The `data/tide_gauge_data` directory contains the hourly tide-gauge sea-level data used in this work. It was obtained from the [UH Sea Level Center](https://uhslc.soest.hawaii.edu) and can be updated if desired using one of the options [here](https://uhslc.soest.hawaii.edu/datainfo).

## Data assessment and cleaning
The quality of the tide-gauge data was assessed and determinations about how to handle questionable data were made using code in `quality_control_playground.ipynb`. Changes to the raw hourly time series can be gleaned from examining the code in `quality_control.py`, which is applied to the data in `station_analysis.py`. Executing `station_analysis.py` produces figures of the data from each tide-gauge record in `figures/quality_control/` showing any adjustments made.

## Location-specific analysis
The fundamental calculations were made by executing `station_analysis.py`, which loops over the individual tide-gauge records and performs the fundamental location-specific calculations utilizing the sea-level rise scenarios in `data/slr_scenarios/`. Output is exported to `output/global_analysis.csv`. 

## Geographic sampling ensemble
Executing the notebook `geographic_sampling_ensemble.ipynb` imports code from `geographic_sampling_ensemble.py` and performs the Monte Carlo simulations necessary to establish statistical significance of the differences between geographic and socioeconomic groupings of locations. Output is exported to `output/*.csv`. A figure for the manuscript is exported to `figures/manuscript/`.

## Global analysis and visualization
Executing the notebook `aggregate_and_visualize.ipynb` imports code from `aggregate_and_visualize.py` and performs global analyses of the output from `station_analysis.py`. Output includes a variety of figures for the manuscript exported to `figures/manuscript/`. Metadata and results for individual tide-gauge records are exported to a table in `output/supplementary_tables.xlsx`.