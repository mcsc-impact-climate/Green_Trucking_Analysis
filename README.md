# Updating Green group trucking analysis with Tesla parameters

## Summary

This code updates the electricity costing and EV truck emissions+TCS analyses developed by the Green group with operating parameters obtained for the Tesla Semi obtained from PepsiCo's Tesla Semi pilot in California using data published [here](https://runonless.com/run-on-less-electric-depot-reports/) by NACFE from their 2023 Run On Less pilot. The code used to obtain the operating parameters for the Tesla semi can be found in [this repo](https://github.com/mcsc-impact-climate/PepsiCo_NACFE_Analysis).

## Electricity price for charging

The script [`bet_electricity_costing.py`](./source/bet_electricity_costing.py) calculates the electricity price for EV truck charging for both the nominal parameters used by the Green group or the parameters obtained for the Tesla semi:

To run:

```bash
python source/bet_electricity_costing.py
```

This should produce a csv file `data/electricity_costing_results.csv` which contains both 1) the inputs for the nominal analysis and Tesla semi, and 2) the respective output electricity price and its components. It also produces the following plots to visualize the results for the nominal and Tesla Semi inputs: `plots/electricity_prices_nominal.png` and `plots/electricity_prices_tesla.png`.

## Costing and emissions for electric long-haul

The script [`bet_emissions_and_costing.py`](source/bet_emissions_and_costing.py) calculates lifecycle emissions and total cost to society (TCS) for electric long-haul trucks with either LFP or NMC batteries, and compares it with diesel long-haul. Inputs are either nominal or specific to the Tesla Semi. 

To run:

```bash
python source/bet_emissions_and_costing.py
```

This should produce plots comparing lifecycle emissions and TCS for the four sets of inputs considered (NMC/LFP batteries with either noninal or Tesla semi operating parameters): `plots/wtw_emissions.png` and `plots/tcs.png`.

