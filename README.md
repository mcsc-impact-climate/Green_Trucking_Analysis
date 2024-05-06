# Calibration and regional analysis of Green group trucking model

## Summary

This code updates the electricity costing and EV truck emissions+TCS analyses developed by the Green group with operating parameters obtained for the Tesla Semi obtained from PepsiCo's Tesla Semi pilot in California using data published [here](https://runonless.com/run-on-less-electric-depot-reports/) by NACFE from their 2023 Run On Less pilot. The code used to obtain the operating parameters for the Tesla semi can be found in [this repo](https://github.com/mcsc-impact-climate/PepsiCo_NACFE_Analysis).

Please see [these slides](https://docs.google.com/presentation/d/1l4Rhx-8UHH76ify1ockwKjT68-v6luahxVGwR-wyRJk/edit?usp=sharing) for a summary of methodology and results obtained with this code. 

## Install dependencies

To install python3 dependencies needed to run the code:

```bash
pip install -r requirements.txt
```

## Install input data files

```bash
cd data

# Monthly seasonally adjusted consumer price index, obtained from https://fred.stlouisfed.org/series/CPIAUCSL on May 5, 2024
wget "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CPIAUCSL&scale=left&cosd=1947-01-01&coed=2024-03-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-05-05&revision_date=2024-05-05&nd=1947-01-01" -O CPIAUCSL.csv

# Historical diesel prices, from https://www.eia.gov/petroleum/gasdiesel/
wget https://www.eia.gov/petroleum/gasdiesel/xls/psw18vwall.xls

# Shapefile containing state boundaries, from https://www.sciencebase.gov/catalog
wget "https://www.sciencebase.gov/catalog/file/get/52c78623e4b060b9ebca5be5?facet=tl_2012_us_state" -O state_boundaries.zip
unzip state_boundaries.zip -d state_boundaries
rm state_boundaries.zip

cd ..
```

## Check impact of neglecting road grade with uncalibrated model
Run [`road_grade_comparison.py`](./source/road_grade_comparison.py) to check the impact of neglecting road grade on results obtained with the original uncalibrated model:

```bash
python source/road_grade_comparison.py 
```

This will produce `plots/results_comparison_costing.png` and `plots/results_comparison_emissions.png` that compare the original model results with vs. without road grade information. 

## Check nominal payload distribution and impact of varying model parameters

The code in [`semi_parameter_scans.py`](./source/semi_parameter_scans.py) first evaluates and plots the distribution of best-fitting payloads for the truck model with drag coefficient and frontal area set to the Tesla Semi values of 0.22 and 10.7 m^2. For a single drivecycle (pepsi 1 truck, drivecycle 2), it then performs scans over various model parameters one at a time, evaluating the best-fitting payload at each scan value. The code takes ~15 minutes to run in full (you can comment out unneeded sections to shorten the run time). 

To run:

```bash
python source/semi_parameter_scans.py
```

This will produce the following:
* `plots/Evaluated_GVW_Distribution.png`: Box plot showing the distribution of gross vehicle weights (GVWs) corresponding to the fitted payloads for each truck with the scan parameters set to their default values
* `plots/matching_gvw_vs_max_motor_power.png`: Variation of best-fitting GVW for drivecycle 2 of pepsi 1 truck, with the max motor power allowed to vary. Max motor power is set to the Semi value of 942900 W for subsequent plots.
* `plots/matching_gvw_vs_combined_eff.png`: Variation of best-fitting GVW for drivecycle 2 of pepsi 1 truck, with the combined powertrain efficiency allowed to vary.
* `plots/matching_gvw_vs_battery_energy_density.png`: Variation of best-fitting GVW for drivecycle 2 of pepsi 1 truck, with the battery energy density allowed to vary.
* `plots/matching_gvw_vs_battery_roundtrip_efficiency.png`: Variation of best-fitting GVW for drivecycle 2 of pepsi 1 truck, with the rountrip battery efficiency allowed to vary.
* `plots/matching_gvw_vs_resistance_coef.png`: Variation of best-fitting GVW for drivecycle 2 of pepsi 1 truck, with the coefficient of rolling resistance allowed to vary.

You can also check out sample payload fits in `plots/truck_model_results_vs_payload_*_drivecycle_*.png`

## Scan full GVW distributions over combined powertrain efficiency

The code in [`semi_combined_efficiency_scans_full.py`](source/semi_combined_efficiency_scans_full.py) performs a scan over combined powertrain efficiencies, evaluating the fitted GVW over all 20 Tesla Semi drivecycles. By default the rolling resistance is set to 0.0044 for the scan, but it can be modified on [this line](source/semi_combined_efficiency_scans_full.py#L55). The code takes ~10 minutes to run.

To run:

```bash
python source/semi_combined_efficiency_scans_full.py
```

You can then produce box plots of the resulting GW distributions as a function of combined efficiency by running [`plot_combined_efficiency_scans_full.py`](source/plot_combined_efficiency_scans_full.py):

```bash
python source/plot_combined_efficiency_scans_full.py
```

The resulting plot can be found in `plots/gvw_dist_vs_combined_eff.png`.

## Evaluate straight line approximation of fuel economy as a function of payload

Using the Tesla Semi parameters established from the above analysis, the scripts [`evaluate_payload_vs_mileage.py`](source/evaluate_payload_vs_mileage.py) and [`make_payload_vs_mileage_function.py`](source/make_payload_vs_mileage_function.py) use the Green group's model to evaluate the best-fitting payload for each EV drivecycle, then perform a linear fit of fuel economy vs. payload to approximate a linear functional relationship between these two parameters. The code takes ~1 minute to run.

To run:

```bash
# Evaluate the best-fitting payload for each EV drivecycle
python source/evaluate_payload_vs_mileage.py

# Evaluate coefficients for the approximate linear relationship between payload and fuel economy (gal/mile) if the drivecycles were performed by a comparable diesel truck
python source/evaluate_payload_vs_mileage_diesel.py

# Evaluate the best fit relationships for fuel economy vs. payload
python source/make_payload_vs_mileage_function.py
python source/make_payload_vs_mileage_function_diesel.py
```

This will produce visualizations of the fits in `plots/payload_vs_mileage_function.png` and `plots/payload_vs_mileage_function_diesel.png`, along with csv files containing the best-fit linear parameters in `tables/payload_vs_mileage_best_fit_params.csv'` and `tables/payload_vs_mileage_linear_coefs_diesel.csv`.


## Lifecycle costing and emissions

### Evaluate average state diesel prices

Average diesel prices are evaluated for each state using the prior 5 years of historical diesel prices, adjusted by the consumer price index to account for inflation.

To run:

```bash
python source/get_diesel_prices_by_state.py
```

This will produce an output file `tables/average_diesel_price_by_state.csv`.

### Validation plots

Lifecycle costing and emissions is performed using the tools contained in [`costing_tools.py`](./costing_tools.py), [`emissions_tools.py`](./emissions_tools.py), and [`tco_emissions_tools.py`](./tco_emissions_tools.py). 

Run [`validate_costing_and_emissions_tools.py`](./source/validate_costing_and_emissions_tools.py) to produce validation plots for the costing and emissions code:

```bash
python source/validate_costing_and_emissions_tools.py
```

This will produce the following plots:

* `plots/VMT_distribution_average_*.png`: Distribution of VMT (annual vehicle miles traeled) with the average VMT specified [here](./source/validate_costing_and_emissions_tools.py#L85).
* `plots/payload_distribution_average_*lb.png`: Payload distribution, with the average payload specified [here](./source/validate_costing_and_emissions_tools.py#L81).
* `plots/electricity_unit_price.png`: Electricity unit cost for each year of the truck's life, broken down into its components.
* `plots/lifecycle_emissions_validation.png`: Validation plot showing the components of evaluated lifecycle emissions for the sample inputs defined in the main() function.
* `plots/lifecycle_costs_validation.png`: Same validation plot as above, but for lifecycle costs.
* `plots/grid_emission_intensity_projection_CAMX.png`: Comparison of grid emission intensity for the WECC California balancing authority compared with the average US EIA projection.

### Evaluating regional costs and emissions

The script [`evaluate_regional_costs_and_emissions.py`](./source/evaluate_regional_costs_and_emissions.py) uses emissions and costing code to evaluate:

* Lifecycle emissions for each US grid balancing authority, based on the authority's reported grid CO2e intensity. 
* Lifecycle costs for each US state, based on the state's commercial electricity price and average demand charge.

The code reads in geojson files containing the boundaries of these balancing authorities and states, along with data on the associated CO2e intensity, electricity price and demand charge. 

To download the input geojsons into the [`geojsons`](./geojsons) dir:

```bash
# From the top level of the repo:
wget https://mcsc-datahub-files.s3.us-west-2.amazonaws.com/geojsons_simplified/egrid2020_subregions_merged.geojson ./geojsons
wget https://mcsc-datahub-files.s3.us-west-2.amazonaws.com/geojsons_simplified/demand_charges_by_state.geojson ./geojsons
wget https://mcsc-datahub-files.s3.us-west-2.amazonaws.com/geojsons_simplified/electricity_rates_by_state_merged.geojson ./geojsons
```

To run the code:

```bash
python source/evaluate_regional_costs_and_emissions.py
```

This will produce two new files in the `geojsons` dir called `geojsons/costs_per_mile.geojson` and `geojsons/emissions_per_mile.geojson`, which contain the evaluated regional lifecycle costs and emissions per mile. It will also produce the following validation plots:
* `plots/emissions_per_mile.png`: Emissions per mile, broken down into its components, for a few sample balancing authorities
* `plots/costs_per_mile.png`: Same as above, but costs per mile for a few sample states.

