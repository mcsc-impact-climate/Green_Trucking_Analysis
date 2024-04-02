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
* `plots/Evaluated_GVW_Distribution.png`: Box plot showing the fitted payload distribution for each truck with the scan parameters set to their default values
* `plots/matching_gvw_vs_max_motor_power.png`: Variation of best-fitting payload for drivecycle 2 of pepsi 1 truck, with the max motor power allowed to vary. Max motor power is set to the Semi value of 942900 W for subsequent plots.
* `plots/matching_gvw_vs_combined_eff.png`: Variation of best-fitting payload for drivecycle 2 of pepsi 1 truck, with the combined powertrain efficiency allowed to vary.
* `plots/matching_gvw_vs_battery_energy_density.png`: Variation of best-fitting payload for drivecycle 2 of pepsi 1 truck, with the battery energy density allowed to vary.
* `plots/matching_gvw_vs_battery_roundtrip_efficiency.png`: Variation of best-fitting payload for drivecycle 2 of pepsi 1 truck, with the rountrip battery efficiency allowed to vary.
* `plots/matching_gvw_vs_resistance_coef.png`: Variation of best-fitting payload for drivecycle 2 of pepsi 1 truck, with the coefficient of rolling resistance allowed to vary.

