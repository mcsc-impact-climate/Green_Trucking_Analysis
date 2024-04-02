# Calibration and regional analysis of Green group trucking model

## Summary

This code updates the electricity costing and EV truck emissions+TCS analyses developed by the Green group with operating parameters obtained for the Tesla Semi obtained from PepsiCo's Tesla Semi pilot in California using data published [here](https://runonless.com/run-on-less-electric-depot-reports/) by NACFE from their 2023 Run On Less pilot. The code used to obtain the operating parameters for the Tesla semi can be found in [this repo](https://github.com/mcsc-impact-climate/PepsiCo_NACFE_Analysis).

Please see [these slides](https://docs.google.com/presentation/d/1l4Rhx-8UHH76ify1ockwKjT68-v6luahxVGwR-wyRJk/edit?usp=sharing) for a summary of methodology and results obtained with this code. 

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

Using the Tesla Semi parameters established from the above analysis, the scripts [`evaluate_payload_vs_energy_economy.py`](source/evaluate_payload_vs_energy_economy.py) and [`make_payload_vs_mileage_function.py`](source/make_payload_vs_mileage_function.py) evaluate the best-fitting payload for each drivecycle, then perform a linear fit of fuel economy vs. payload to approximate a linear functional relationship between these two parameters. The code takes ~1 minute to run.

To run:

```bash
# Evaluate the best-fitting payload for each drivecycle
python source/evaluate_payload_vs_energy_economy.py

# Evaluate the best fit line of fuel economy vs. payload
python source/make_payload_vs_mileage_function.py
```

This will produce a visualization of the linear fit in `plots/payload_vs_mileage_function.png` and a csv containing the best-fit linear parameters in `tables/payload_vs_mileage_best_fit_params.csv'`.

## Lifecycle costing and emissions

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

