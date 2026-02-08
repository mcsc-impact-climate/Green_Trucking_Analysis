"""
Date: 260201
Author: danikae
Purpose: Generate drivecycle comparison plots with option for original or optimized parameters
         Model results are saved per driving event to allow separation of computation from analysis
"""

import pandas as pd
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pickle
import truck_model_tools_messy as truck_model_tools_messy
import truck_model_tools_diesel_messy as truck_model_tools_diesel_messy
import retired.costing_tools_orig as costing_tools
import retired.emissions_tools_orig as emissions_tools
import data_collection_tools_messy
from costing_and_emissions_tools_messy import get_payload_distribution, get_payload_penalty, evaluate_emissions, evaluate_costs, evaluate_costs_diesel, calculate_replacements

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

KG_PER_TON = 1000
KG_PER_LB = 0.453592
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
M_PER_MILE = 1609.34
S_PER_H = 3600
G_PER_KG = 1000

BASE_DIR = Path(__file__).resolve().parent.parent
battery_caps = pd.read_csv(BASE_DIR / "messy_middle_results" / "battery_capacities_linear_summary.csv").set_index('Value')


def load_optimized_parameters(use_optimized=False):
	"""
	Load parameter optimization results if requested.
	
	Parameters:
	-----------
	use_optimized : bool
		If True, load optimized parameters from parameter_optimization_results.csv
		If False, return None (use original parameters)
	
	Returns:
	--------
	optimized_params : dict or None
		Dictionary mapping truck name to optimized parameters (cd, cr, eta_i*eta_m)
	"""
	if not use_optimized:
		return None
	
	try:
		opt_df = pd.read_csv(str(BASE_DIR / 'parameter_optimization_results.csv'))
		# Remove duplicates
		opt_df = opt_df.drop_duplicates(subset=['Truck'])
		
		optimized_params = {}
		for idx, row in opt_df.iterrows():
			truck_name = row['Truck']
			optimized_params[truck_name] = {
				'cd': row['cd_optimal'],
				'cr': row['cr_optimal'],
				'eta_combined': row['eta_optimal'],
			}
		
		print(f"\nLoaded optimized parameters for {len(optimized_params)} trucks")
		return optimized_params
	
	except FileNotFoundError:
		print("Warning: parameter_optimization_results.csv not found. Using original parameters.")
		return None


def apply_optimized_parameters(parameters, optimized_params, truck_name):
	"""
	Apply optimized parameters to a parameters object if available.
	
	Parameters:
	-----------
	parameters : truck_model_tools_messy.read_parameters or share_parameters
		Original parameters object
	optimized_params : dict or None
		Dictionary of optimized parameters
	truck_name : str
		Name of the truck
	
	Returns:
	--------
	parameters : truck_model_tools_messy.share_parameters
		Parameters object with optimized values if available
	"""
	if optimized_params is None or truck_name not in optimized_params:
		return parameters
	
	opt = optimized_params[truck_name]
	
	# Create new parameters object with optimized values
	# Split combined efficiency back to inverter and motor (use sqrt approximation)
	eta_opt = np.sqrt(opt['eta_combined'])
	
	# Store battery_chemistry if it exists, for later use
	battery_chemistry = getattr(parameters, 'battery_chemistry', None)
	
	optimized = truck_model_tools_messy.share_parameters(
		m_ave_payload=parameters.m_ave_payload,
		m_max=parameters.m_max,
		m_truck_no_bat=parameters.m_truck_no_bat,
		p_aux=parameters.p_aux,
		p_motor_max=parameters.p_motor_max,
		cd=opt['cd'],
		cr=opt['cr'],
		a_cabin=parameters.a_cabin,
		g=parameters.g,
		rho_air=parameters.rho_air,
		DoD=parameters.DoD,
		eta_i=eta_opt,
		eta_m=eta_opt,
		eta_gs=parameters.eta_gs,
		eta_rb=parameters.eta_rb,
		eta_grid_transmission=parameters.eta_grid_transmission,
		VMT=parameters.VMT,
		discountrate=parameters.discountrate,
	)
	
	# Copy over battery_chemistry if it exists
	if battery_chemistry is not None:
		optimized.battery_chemistry = battery_chemistry
	
	return optimized


def plot_drivecycle_comparison(drivecycle_df, model_df, model_fuel_consumption, model_DoD, summary_path, driving_event):
	"""Plot drivecycle signals and compare NACFE vs model metrics."""
	summary_df = pd.read_csv(summary_path, index_col="Driving event")
	if driving_event not in summary_df.index:
		raise KeyError(f"Driving event {driving_event} not found in {summary_path}")

	nacfe_fuel_consumption = summary_df.loc[driving_event, "Fuel economy (kWh/mile)"]
	nacfe_DoD_percent = summary_df.loc[driving_event, "Depth of Discharge (%)"]
	
	# Calculate standard error in the mean (SEM) for this driving event from instantaneous measurements
	# SEM = std(instantaneous_fuel_economy) / sqrt(n)
	# where n is the number of valid instantaneous measurements
	instantaneous_fuel = drivecycle_df['Instantaneous Energy (kWh/mile)'].dropna()
	if len(instantaneous_fuel) > 1:
		nacfe_fuel_sem = instantaneous_fuel.std() / np.sqrt(len(instantaneous_fuel))
		nacfe_fuel_sem_pct = (nacfe_fuel_sem / nacfe_fuel_consumption * 100) if nacfe_fuel_consumption > 0 else 0
	else:
		nacfe_fuel_sem_pct = 0
	
	# For DoD: The SOC decreases monotonically during the drive, so its std doesn't represent measurement uncertainty.
	# Instead, estimate uncertainty from the precision of SOC measurements (typically ±0.1-0.5% for BMS)
	# Using ±0.5% as a conservative estimate for SOC measurement precision
	# DoD uncertainty propagates from initial and final SOC: sqrt(σ_initial² + σ_final²)
	soc_measurement_precision = 0.5  # percent
	dod_uncertainty = np.sqrt(2) * soc_measurement_precision  # sqrt(2) because initial and final measurements
	nacfe_DoD_sem_pct = (dod_uncertainty / nacfe_DoD_percent * 100) if nacfe_DoD_percent > 0 else 0

	time_s = drivecycle_df["Time (s)"]
	speed_mps = drivecycle_df["Vehicle speed (m/s)"]
	road_grade_pct = drivecycle_df["Road Grade (%)"]

	delta_t_s = time_s.diff().replace(0, np.nan)
	instantaneous_power_w = -((drivecycle_df["Delta Battery Energy (kWh)"] * 3.6e6) / delta_t_s)

	model_DoD_percent = model_DoD * 100

	fuel_pct_diff = (model_fuel_consumption - nacfe_fuel_consumption) / nacfe_fuel_consumption * 100
	DoD_pct_diff = (model_DoD_percent - nacfe_DoD_percent) / nacfe_DoD_percent * 100

	fig = plt.figure(figsize=(12, 7))
	gs = fig.add_gridspec(3, 2, width_ratios=[4, 1], height_ratios=[1, 1, 1], wspace=0.3, hspace=0.2)

	ax_speed = fig.add_subplot(gs[0, 0])
	ax_grade = fig.add_subplot(gs[1, 0], sharex=ax_speed)
	ax_power = fig.add_subplot(gs[2, 0], sharex=ax_speed)
	text_ax = fig.add_subplot(gs[:, 1])

	ax_speed.plot(time_s, speed_mps, color="tab:blue")
	ax_speed.set_ylabel("Speed (m/s)")
	ax_speed.set_title(f"Drivecycle and Model Comparison (Event {driving_event})")

	ax_grade.plot(time_s, road_grade_pct, color="tab:green")
	ax_grade.set_ylabel("Road Grade (%)")

	ax_power.scatter(time_s, instantaneous_power_w, s=12, color="tab:gray", label="Drivecycle power (W)")
	ax_power.plot(time_s, model_df["Power request at battery (W)"], color="tab:red", label="Model power (W)")
	ax_power.set_ylabel("Power (W)")
	ax_power.set_xlabel("Time (s)")
	ax_power.legend(loc="upper right")

	text_ax.axis("off")
	text_ax.text(
		0.0,
		0.95,
		"NACFE vs Model\n",
		fontsize=12,
		fontweight="bold",
		va="top",
	)
	text_ax.text(
		0.0,
		0.85,
		(
			f"Fuel economy (kWh/mi)\n"
			f"  NACFE: {nacfe_fuel_consumption:.3f} ± {nacfe_fuel_sem_pct:.1f}%\n"
			f"  Model: {model_fuel_consumption:.3f}\n"
			f"  % diff: {fuel_pct_diff:+.2f}%\n\n"
			f"DoD (%)\n"
			f"  NACFE: {nacfe_DoD_percent:.3f} ± {nacfe_DoD_sem_pct:.1f}%\n"
			f"  Model: {model_DoD_percent:.3f}\n"
			f"  % diff: {DoD_pct_diff:+.2f}%"
		),
		fontsize=11,
		va="top",
	)

	return fig


def save_model_results(truck_name, event_id, power_request_w, fuel_consumption, dod, model_results_dir):
	"""
	Save model results for a single driving event.
	
	Parameters:
	-----------
	truck_name : str
		Name of the truck
	event_id : int
		Driving event ID
	power_request_w : pd.Series
		Power request at battery (W) over time
	fuel_consumption : float
		Modeled fuel consumption (kWh/mile)
	dod : float
		Modeled depth of discharge (0-1)
	model_results_dir : Path
		Directory to save model results
	"""
	# Create truck subdirectory
	truck_dir = model_results_dir / truck_name
	truck_dir.mkdir(parents=True, exist_ok=True)
	
	# Save results as pickle
	results = {
		'power_request_w': power_request_w.values,  # Save as numpy array for efficiency
		'fuel_consumption': fuel_consumption,
		'dod': dod,
	}
	
	pkl_path = truck_dir / f"event_{event_id}.pkl"
	with open(pkl_path, 'wb') as f:
		pickle.dump(results, f)


def load_model_results(truck_name, event_id, model_results_dir):
	"""
	Load model results for a single driving event.
	
	Parameters:
	-----------
	truck_name : str
		Name of the truck
	event_id : int
		Driving event ID
	model_results_dir : Path
		Directory containing model results
	
	Returns:
	--------
	results : dict or None
		Dictionary with 'power_request_w', 'fuel_consumption', 'dod'
		Returns None if file not found
	"""
	pkl_path = model_results_dir / truck_name / f"event_{event_id}.pkl"
	
	if not pkl_path.exists():
		return None
	
	with open(pkl_path, 'rb') as f:
		results = pickle.load(f)
	
	return results


def plot_truck_summary(truck_name, results, param_suffix, plots_dir):
	"""
	Create summary plots comparing model vs data for all driving events.
	
	Parameters:
	-----------
	truck_name : str
		Name of the truck
	results : list of dict
		List of dictionaries containing results for each driving event
	param_suffix : str
		Suffix indicating parameter type ('_original' or '_optimized')
	plots_dir : Path
		Directory to save plots
	"""
	if not results:
		return
	
	# Extract data
	events = [r['event'] for r in results]
	data_fuel = np.array([r['data_fuel'] for r in results])
	model_fuel = np.array([r['model_fuel'] for r in results])
	fuel_unc_pct = np.array([r['fuel_unc_pct'] for r in results])
	data_dod = np.array([r['data_dod'] for r in results])
	model_dod = np.array([r['model_dod'] for r in results])
	dod_unc_pct = np.array([r['dod_unc_pct'] for r in results])
	
	# Calculate absolute uncertainties for error bars
	fuel_unc = data_fuel * fuel_unc_pct / 100
	dod_unc = data_dod * dod_unc_pct / 100
	
	# Create figure with two subplots
	fig, axes = plt.subplots(2, 1, figsize=(12, 10))
	
	# Fuel Economy comparison
	ax1 = axes[0]
	x = np.arange(len(events))
	width = 0.35
	
	ax1.bar(x - width/2, data_fuel, width, label='NACFE Data', alpha=0.8, color='tab:blue')
	ax1.errorbar(x - width/2, data_fuel, yerr=fuel_unc, fmt='none', color='black', capsize=3, linewidth=1)
	ax1.bar(x + width/2, model_fuel, width, label='Model', alpha=0.8, color='tab:orange')
	
	ax1.set_ylabel('Fuel Economy (kWh/mile)')
	ax1.set_title(f'{truck_name} - Fuel Economy Comparison')
	ax1.set_xticks(x)
	ax1.set_xticklabels(events, rotation=45, ha='right')
	ax1.set_xlabel('Driving Event')
	ax1.legend()
	ax1.grid(True, alpha=0.3, axis='y')
	
	# Add uncertainty-weighted mean comparison as text
	def weighted_mean(values, weights):
		mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
		if not np.any(mask):
			return float(np.mean(values))
		return float(np.average(values[mask], weights=weights[mask]))

	fuel_weights = np.where(fuel_unc > 0, 1 / fuel_unc**2, np.nan)
	mean_data_fuel = weighted_mean(data_fuel, fuel_weights)
	mean_model_fuel = weighted_mean(model_fuel, fuel_weights)
	mean_fuel_diff_pct = (mean_model_fuel - mean_data_fuel) / mean_data_fuel * 100
	
	ax1.axhline(y=mean_data_fuel, color='tab:blue', linestyle='--', linewidth=2, alpha=0.5, label=f'Mean Data: {mean_data_fuel:.3f}')
	ax1.axhline(y=mean_model_fuel, color='tab:orange', linestyle='--', linewidth=2, alpha=0.5, label=f'Mean Model: {mean_model_fuel:.3f}')
	ax1.legend()
	
	ax1.text(0.02, 0.98, f'Weighted mean difference: {mean_fuel_diff_pct:+.2f}%', 
			transform=ax1.transAxes, verticalalignment='top',
			bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
	
	# DoD comparison
	ax2 = axes[1]
	ax2.bar(x - width/2, data_dod, width, label='NACFE Data', alpha=0.8, color='tab:blue')
	ax2.errorbar(x - width/2, data_dod, yerr=dod_unc, fmt='none', color='black', capsize=3, linewidth=1)
	ax2.bar(x + width/2, model_dod, width, label='Model', alpha=0.8, color='tab:orange')
	
	ax2.set_ylabel('Depth of Discharge (%)')
	ax2.set_title(f'{truck_name} - DoD Comparison')
	ax2.set_xticks(x)
	ax2.set_xticklabels(events, rotation=45, ha='right')
	ax2.set_xlabel('Driving Event')
	ax2.legend()
	ax2.grid(True, alpha=0.3, axis='y')
	
	# Add uncertainty-weighted mean comparison as text
	dod_weights = np.where(dod_unc > 0, 1 / dod_unc**2, np.nan)
	mean_data_dod = weighted_mean(data_dod, dod_weights)
	mean_model_dod = weighted_mean(model_dod, dod_weights)
	mean_dod_diff_pct = (mean_model_dod - mean_data_dod) / mean_data_dod * 100
	
	ax2.axhline(y=mean_data_dod, color='tab:blue', linestyle='--', linewidth=2, alpha=0.5, label=f'Mean Data: {mean_data_dod:.3f}')
	ax2.axhline(y=mean_model_dod, color='tab:orange', linestyle='--', linewidth=2, alpha=0.5, label=f'Mean Model: {mean_model_dod:.3f}')
	ax2.legend()
	
	ax2.text(0.02, 0.98, f'Weighted mean difference: {mean_dod_diff_pct:+.2f}%', 
			transform=ax2.transAxes, verticalalignment='top',
			bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
	
	plt.tight_layout()
	plt.savefig(plots_dir / f'{truck_name}_summary{param_suffix}.png', dpi=300, bbox_inches='tight')
	plt.close(fig)
	
	print(f"  Summary plot saved: {truck_name}_summary{param_suffix}.png")


def add_cost_emissions(
			datasets,
			plots_dir,
			param_suffix,
			average_vmt,
			optimized_params,
			battery_caps,
			m_truck_max_kg,
		):
	"""Add per-event costs/emissions columns to each results CSV."""
	energy_cost_emissions_data = pd.read_csv(str(BASE_DIR / "data_messy" / "energy_costs_emissions.csv"), index_col="Truck ")
	max_charging_power_data = pd.read_csv(str(BASE_DIR / "messy_middle_results" / "max_charging_powers.csv"), index_col="truck_name")
	for dataset in datasets:
		truck_name = dataset['name']
		results_csv = plots_dir / f"{truck_name}_results{param_suffix}.csv"
		if not results_csv.exists():
			print(f"Warning: Results CSV not found for {truck_name}, skipping summary plot.")
			continue
		
		# Read in saved results
		results_df = pd.read_csv(results_csv)
		
		# Get the electricity cost and emissions info for the given truck
		energy_lookup_name = dataset["truck_params"] if dataset["truck_params"] in energy_cost_emissions_data.index else truck_name
		electricity_rate = energy_cost_emissions_data.loc[energy_lookup_name, "Commercial electricity ($/kWh)"]
		demand_charge = energy_cost_emissions_data.loc[energy_lookup_name, "Demand charge ($/kW)"]
		present_grid_ci =  energy_cost_emissions_data.loc[energy_lookup_name, "Present Grid CI (g CO2e / kWh)"]

		# Collect relevant info for the truck
		parameters = data_collection_tools_messy.read_parameters(
			truck_params=dataset["truck_params"],
			vmt_params='daycab_vmt_vius_2021',
			run='messy_middle',
			truck_type='EV',
		)
		parameters = apply_optimized_parameters(parameters, optimized_params, dataset['name'])

		battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=parameters.battery_chemistry)
		e_density = battery_params_dict['Energy density (kWh/ton)']

		e_bat = battery_caps.loc['Mean', dataset["battery_col"]]
		m_bat_kg = e_bat / e_density * KG_PER_TON          # Battery mass, in kg
		m_bat_lb = m_bat_kg / KG_PER_LB

		m_truck_no_bat_kg = parameters.m_truck_no_bat
		m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB

		# Get the maximum charging power for the truck
		max_charging_power = max_charging_power_data.loc[truck_name, "99th_percentile_charging_power_kw"]

		# Loop through all results in the results_df and calculate cost and emissions for each driving event, then add to the dataframe
		expected_cost_cols = [
			"Total capital ($/mi)",
			"Total operating ($/mi)",
			"Total electricity ($/mi)",
			"Total labor ($/mi)",
			"Other OPEXs ($/mi)",
			"TCO ($/mi)",
		]
		expected_emissions_cols = [
			"GHGs manufacturing (gCO2/mi)",
			"GHGs grid (gCO2/mi)",
			"GHGs total (gCO2/mi)",
		]
		costs = []
		emissions = []
		for idx, row in results_df.iterrows():
			fuel_consumption_kwh_per_mile = row['model_fuel']
			driving_event = int(row['event'])
			
			# Load the drivecycle data for this event
			drivecycle_path = BASE_DIR / "messy_middle_results" / f"{truck_name}_drivecycle_{driving_event}_detailed.csv"
			drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))

			# Collect payload information for this drivecycle
			m_gvw_kg = drivecycle_data["GVW (kg)"].loc[0]
			m_gvw_lb = m_gvw_kg / KG_PER_LB
			m_payload_lb = m_gvw_lb - m_truck_no_bat_lb - m_bat_lb
			
			# Calculate cost and emissions
			cost_per_mile = evaluate_costs(mileage=fuel_consumption_kwh_per_mile, payload_lb=m_payload_lb, electricity_charge=electricity_rate, demand_charge=demand_charge, average_VMT=average_vmt, charging_power=max_charging_power, e_bat=e_bat, battery_chemistry=parameters.battery_chemistry, truck_name=dataset["truck_params"])
			emissions_per_mile = evaluate_emissions(mileage=fuel_consumption_kwh_per_mile, payload_lb=m_payload_lb, grid_emission_intensity=present_grid_ci, average_VMT=average_vmt, e_bat=e_bat, battery_chemistry=parameters.battery_chemistry, truck_name=dataset["truck_params"])
			
			costs.append(cost_per_mile)
			emissions.append(emissions_per_mile)

		# Save the costs and emissions back to the results CSV
		costs_df = pd.DataFrame(costs, columns=expected_cost_cols).add_prefix("cost_")
		emissions_df = pd.DataFrame(emissions, columns=expected_emissions_cols).add_prefix("emissions_")
		results_df = results_df.drop(columns=["cost_per_mile", "emissions_per_mile"], errors="ignore").reset_index(drop=True)
		results_df = pd.concat([results_df, costs_df, emissions_df], axis=1)
		results_df.to_csv(results_csv, index=False)
		print(f"Updated results CSV with costs and emissions: {results_csv}")

		# Warn if expected columns are empty after update
		missing_values = []
		for col in costs_df.columns.tolist() + emissions_df.columns.tolist():
			if col not in results_df.columns:
				missing_values.append(col)
				continue
			col_values = results_df.loc[:, col]
			if col_values.isna().all().all():
				missing_values.append(col)
		if missing_values:
			print(f"Warning: {truck_name} has missing values for columns: {missing_values}")


def generate_summary_plots(datasets, plots_dir, param_suffix):
	"""Generate summary plots for each truck from the updated results CSVs."""
	for dataset in datasets:
		truck_name = dataset['name']
		results_csv = plots_dir / f"{truck_name}_results{param_suffix}.csv"
		if not results_csv.exists():
			continue
		results_df = pd.read_csv(results_csv)
		plot_truck_summary(truck_name, results_df.to_dict('records'), param_suffix, plots_dir)


def plot_distribution_comparisons(datasets, plots_dir, param_suffix):
	"""Plot distribution comparisons for fuel economy, TCO, and emissions across trucks."""
	truck_data = {}
	for dataset in datasets:
		truck_name = dataset['name']
		results_csv = plots_dir / f"{truck_name}_results{param_suffix}.csv"
		if not results_csv.exists():
			print(f"Warning: Results CSV not found for {truck_name}, skipping distribution plot.")
			continue
		results_df = pd.read_csv(results_csv)
		truck_data[truck_name] = results_df

	if not truck_data:
		print("Warning: No results CSVs found for distribution plots.")
		return

	def plot_box(data_map, column, title, ylabel, output_name):
		labels = []
		values_list = []
		for truck_name, df in data_map.items():
			if column not in df.columns:
				continue
			values = df[column].dropna().values
			if values.size == 0:
				continue
			labels.append(truck_name)
			values_list.append(values)

		if not values_list:
			print(f"Warning: No data for {column}, skipping {output_name}.")
			return

		plt.figure(figsize=(10, 6))
		plt.boxplot(values_list, tick_labels=labels, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markeredgecolor='red'))
		plt.title(title)
		plt.ylabel(ylabel)
		plt.xticks(rotation=20, ha='right')
		plt.tight_layout()
		output_path = plots_dir / output_name
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		print(f"Saved plot: {output_path.absolute()}")
		plt.close()

	plot_box(
		truck_data,
		column="model_fuel",
		title="Fuel Economy Distribution (All Trucks)",
		ylabel="Fuel Economy (kWh/mile)",
		output_name=f"fuel_economy_distribution{param_suffix}.png",
	)

	plot_box(
		truck_data,
		column="cost_TCO ($/mi)",
		title="Total Cost per Mile Distribution (All Trucks)",
		ylabel="TCO ($/mile)",
		output_name=f"tco_distribution{param_suffix}.png",
	)

	plot_box(
		truck_data,
		column="emissions_GHGs total (gCO2/mi)",
		title="Total Emissions per Mile Distribution (All Trucks)",
		ylabel="Emissions (gCO2/mi)",
		output_name=f"emissions_distribution{param_suffix}.png",
	)


def process_drivecycles_and_save_results(
		datasets,
		regenerate_model_results,
		model_results_dir,
		optimized_params,
		battery_caps,
		m_truck_max_kg,
		plots_dir,
		param_suffix,
	):
	"""Process drivecycles for each dataset and write per-event results CSVs."""
	for dataset in datasets:
		print(f"\nProcessing {dataset['name']}...")
		
		# Skip parameter loading if not regenerating
		if not regenerate_model_results:
			parameters = None
			battery_params_dict = None
			e_bat = None
			m_bat_kg = None
			m_bat_lb = None
			m_truck_no_bat_kg = None
			m_truck_no_bat_lb = None
		else:
			parameters = data_collection_tools_messy.read_parameters(
				truck_params=dataset["truck_params"],
				vmt_params='daycab_vmt_vius_2021',
				run='messy_middle',
				truck_type='EV',
			)

			# Apply optimized parameters if requested
			parameters = apply_optimized_parameters(parameters, optimized_params, dataset['name'])

			battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=parameters.battery_chemistry)
			e_density = battery_params_dict['Energy density (kWh/ton)']

			e_bat = battery_caps.loc['Mean', dataset["battery_col"]]
			m_bat_kg = e_bat / e_density * KG_PER_TON          # Battery mass, in kg
			m_bat_lb = m_bat_kg / KG_PER_LB

			m_truck_no_bat_kg = parameters.m_truck_no_bat
			m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB

		drivecycle_files = sorted((BASE_DIR / "messy_middle_results").glob(dataset["drivecycle_glob"]))
		
		# Collect results for summary plot
		truck_results = []
		
		# Read summary data for uncertainties
		summary_df = pd.read_csv(str(BASE_DIR / dataset["summary_path"]), index_col="Driving event")

		for drivecycle_path in drivecycle_files:
			parts = drivecycle_path.stem.split("_")
			driving_event = int(parts[-2]) if parts[-1] == "detailed" else int(parts[-1])
			
			# Try to load model results first
			if not regenerate_model_results:
				model_results = load_model_results(dataset['name'], driving_event, model_results_dir)
				if model_results is None:
					print(f"  Warning: Model results not found for event {driving_event}, skipping...")
					continue
				
				fuel_consumption = model_results['fuel_consumption']
				DoD = model_results['dod']
				power_request_w = model_results['power_request_w']
			else:
				# Regenerate model results
				drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))

				# Get the GVW for the drivecycle and infer the payload
				m_gvw_kg = drivecycle_data["GVW (kg)"].loc[0]
				m_gvw_lb = m_gvw_kg / KG_PER_LB
				m_payload_lb = m_gvw_lb - m_truck_no_bat_lb - m_bat_lb

				# Scale the VIUS payload distribution to one with the same shape whose average is the given payload
				payload_distribution = get_payload_distribution(m_payload_lb)

				# Calculate the payload penalty factor
				payload_penalty_factor = get_payload_penalty(payload_distribution, m_bat_kg, parameters.m_truck_no_bat, m_truck_max_kg)

				df, fuel_consumption, DoD = truck_model_tools_messy.truck_model(parameters).get_power_requirement(
					drivecycle_data,
					m_gvw_kg,
					eta_battery=battery_params_dict['Roundtrip efficiency'],
					e_bat=e_bat,
				)

				print(f"  {drivecycle_path.name}: e_bat={e_bat:.1f}, fuel={fuel_consumption:.3f}, DoD={DoD:.3f}")

				# Save model results
				power_request_w = df["Power request at battery (W)"]
				save_model_results(dataset['name'], driving_event, power_request_w, fuel_consumption, DoD, model_results_dir)
			
			# Get data values and uncertainties
			if driving_event in summary_df.index:
				data_fuel_consumption = summary_df.loc[driving_event, "Fuel economy (kWh/mile)"]
				data_DoD_percent = summary_df.loc[driving_event, "Depth of Discharge (%)"]
				
				# Calculate uncertainties
				drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
				instantaneous_fuel = drivecycle_data['Instantaneous Energy (kWh/mile)'].dropna()
				if len(instantaneous_fuel) > 1:
					fuel_sem = instantaneous_fuel.std() / np.sqrt(len(instantaneous_fuel))
					fuel_sem_pct = (fuel_sem / data_fuel_consumption * 100) if data_fuel_consumption > 0 else 0
				else:
					fuel_sem_pct = 0
				
				soc_measurement_precision = 0.5
				dod_uncertainty = np.sqrt(2) * soc_measurement_precision
				dod_sem_pct = (dod_uncertainty / data_DoD_percent * 100) if data_DoD_percent > 0 else 0
				
				# Store results
				truck_results.append({
					'event': driving_event,
					'data_fuel': data_fuel_consumption,
					'model_fuel': fuel_consumption,
					'fuel_unc_pct': fuel_sem_pct,
					'data_dod': data_DoD_percent,
					'model_dod': DoD * 100,
					'dod_unc_pct': dod_sem_pct,
				})

				# Generate drivecycle comparison plot
				reconstructed_df = pd.DataFrame({'Power request at battery (W)': power_request_w})
				fig = plot_drivecycle_comparison(
					drivecycle_data,
					reconstructed_df,
					fuel_consumption,
					DoD,
					dataset["summary_path"],
					driving_event,
				)

				plot_path = plots_dir / f"{dataset['name']}_drivecycle_{driving_event}_comparison{param_suffix}.png"
				fig.savefig(
					plot_path,
					dpi=300,
					bbox_inches="tight",
				)
				print(f"    Saved plot: {plot_path}")
				plt.close(fig)
		
		# Save results to CSV for later summary plotting
		if truck_results:
			results_df = pd.DataFrame(truck_results)
			csv_path = plots_dir / f"{dataset['name']}_results{param_suffix}.csv"
			results_df.to_csv(csv_path, index=False)
			print(f"Saved results CSV: {csv_path}")


def evaluate_truck_on_all_drivecycles(
		datasets,
		optimized_params,
		battery_caps,
		m_truck_max_kg,
		results_dir,
		param_suffix,
	):
	"""
	For each truck, evaluate the truck model with optimized parameters on all drivecycles
	(from all trucks) and save results to CSV.
	
	Parameters:
	-----------
	datasets : list of dict
		List of dataset configurations
	optimized_params : dict or None
		Dictionary of optimized parameters per truck
	battery_caps : pd.DataFrame
		Battery capacity data
	m_truck_max_kg : float
		Maximum truck mass in kg
	results_dir : Path
		Directory to save results CSVs
	param_suffix : str
		Suffix for output files (e.g., '_optimized')
	"""
	# Get all drivecycle files regardless of truck
	all_drivecycle_files = sorted((BASE_DIR / "messy_middle_results").glob("*_drivecycle_*_detailed.csv"))
	
	# Group drivecycles by source truck
	drivecycles_by_source = {}
	for drivecycle_path in all_drivecycle_files:
		parts = drivecycle_path.stem.split("_")
		# Extract truck name from filename (e.g., "saia2_drivecycle_1_detailed" -> "saia2")
		if "drivecycle" in parts:
			drivecycle_idx = parts.index("drivecycle")
			source_truck = "_".join(parts[:drivecycle_idx])
			driving_event = int(parts[drivecycle_idx + 1])
		else:
			source_truck = parts[0]
			driving_event = int(parts[-2]) if parts[-1] == "detailed" else int(parts[-1])
		
		if source_truck not in drivecycles_by_source:
			drivecycles_by_source[source_truck] = []
		drivecycles_by_source[source_truck].append((drivecycle_path, driving_event))
	
	# For each evaluation truck, run model on all drivecycles
	for dataset in datasets:
		eval_truck_name = dataset['name']
		print(f"\nEvaluating {eval_truck_name} model on all drivecycles...")
		
		# Load parameters and battery info for evaluation truck
		parameters = data_collection_tools_messy.read_parameters(
			truck_params=dataset["truck_params"],
			vmt_params='daycab_vmt_vius_2021',
			run='messy_middle',
			truck_type='EV',
		)
		
		# Apply optimized parameters
		parameters = apply_optimized_parameters(parameters, optimized_params, eval_truck_name)
		
		battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=parameters.battery_chemistry)
		e_density = battery_params_dict['Energy density (kWh/ton)']
		
		e_bat = battery_caps.loc['Mean', dataset["battery_col"]]
		m_bat_kg = e_bat / e_density * KG_PER_TON
		m_bat_lb = m_bat_kg / KG_PER_LB
		
		m_truck_no_bat_kg = parameters.m_truck_no_bat
		m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB
		
		# Collect results for all drivecycles
		all_results = []
		
		# Iterate through all drivecycles from all source trucks
		for source_truck, drivecycle_list in drivecycles_by_source.items():
			print(f"  Processing {source_truck} drivecycles...")
			for drivecycle_path, driving_event in drivecycle_list:
				try:
					# Load drivecycle data
					drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
					
					# Get payload information
					m_gvw_kg = drivecycle_data["GVW (kg)"].loc[0]
					m_gvw_lb = m_gvw_kg / KG_PER_LB
					m_payload_lb = m_gvw_lb - m_truck_no_bat_lb - m_bat_lb
					
					# Get payload distribution and penalty factor
					payload_distribution = get_payload_distribution(m_payload_lb)
					payload_penalty_factor = get_payload_penalty(
						payload_distribution, m_bat_kg, parameters.m_truck_no_bat, m_truck_max_kg
					)
					
					# Run truck model
					df, fuel_consumption, DoD = truck_model_tools_messy.truck_model(parameters).get_power_requirement(
						drivecycle_data,
						m_gvw_kg,
						eta_battery=battery_params_dict['Roundtrip efficiency'],
						e_bat=e_bat,
					)
					
					# Store result
					all_results.append({
						'source_truck': source_truck,
						'drivecycle_number': driving_event,
						'fuel_economy_kWh_per_mile': fuel_consumption,
					})
					
				except Exception as e:
					print(f"    Warning: Error processing {drivecycle_path.name}: {e}")
					continue
		
		# Save results to CSV
		if all_results:
			results_df = pd.DataFrame(all_results)
			output_path = results_dir / f"{eval_truck_name}_all_drivecycles_results{param_suffix}.csv"
			results_df.to_csv(output_path, index=False)
			print(f"Saved {eval_truck_name} cross-evaluation results: {output_path}")
		else:
			print(f"Warning: No results collected for {eval_truck_name}")


def resolve_drivecycle_path(source_truck, driving_event):
	"""Resolve a drivecycle file path, with fallback for truncated truck names."""
	direct = BASE_DIR / "messy_middle_results" / f"{source_truck}_drivecycle_{driving_event}_detailed.csv"
	if direct.exists():
		return direct

	candidates = sorted((BASE_DIR / "messy_middle_results").glob(f"*drivecycle_{driving_event}_detailed.csv"))
	if not candidates:
		return direct

	for candidate in candidates:
		if source_truck in candidate.stem:
			return candidate

	return candidates[0]


def add_cost_emissions_all_drivecycles(
		datasets,
		plots_dir,
		param_suffix,
		average_vmt,
		optimized_params,
		battery_caps,
		m_truck_max_kg,
	):
	"""Add per-event costs/emissions columns to each all-drivecycles results CSV using US averages."""
	usa_data = pd.read_csv(str(BASE_DIR / "data_messy" / "energy_costs_emissions_usa.csv")).set_index("Parameter")
	electricity_rate = usa_data.loc["Commercial electricity ($/kWh)", "Value"]
	demand_charge = usa_data.loc["Demand charge ($/kW)", "Value"]
	present_grid_ci = usa_data.loc["Present Grid CI (g CO2e / kWh)", "Value"]

	max_charging_power_data = pd.read_csv(str(BASE_DIR / "messy_middle_results" / "max_charging_powers.csv"), index_col="truck_name")

	for dataset in datasets:
		truck_name = dataset["name"]
		results_csv = plots_dir / f"{truck_name}_all_drivecycles_results{param_suffix}.csv"
		if not results_csv.exists():
			print(f"Warning: All-drivecycles results CSV not found for {truck_name}, skipping.")
			continue

		results_df = pd.read_csv(results_csv)

		parameters = data_collection_tools_messy.read_parameters(
			truck_params=dataset["truck_params"],
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="EV",
		)
		parameters = apply_optimized_parameters(parameters, optimized_params, dataset["name"])

		battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=parameters.battery_chemistry)
		e_density = battery_params_dict["Energy density (kWh/ton)"]

		e_bat = battery_caps.loc["Mean", dataset["battery_col"]]
		m_bat_kg = e_bat / e_density * KG_PER_TON
		m_bat_lb = m_bat_kg / KG_PER_LB

		m_truck_no_bat_kg = parameters.m_truck_no_bat
		m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB

		max_charging_power = max_charging_power_data.loc[truck_name, "99th_percentile_charging_power_kw"]

		expected_cost_cols = [
			"Total capital ($/mi)",
			"Total operating ($/mi)",
			"Total electricity ($/mi)",
			"Total labor ($/mi)",
			"Other OPEXs ($/mi)",
			"TCO ($/mi)",
		]
		expected_emissions_cols = [
			"GHGs manufacturing (gCO2/mi)",
			"GHGs grid (gCO2/mi)",
			"GHGs total (gCO2/mi)",
		]

		costs = []
		emissions = []
		for _, row in results_df.iterrows():
			fuel_consumption_kwh_per_mile = row["fuel_economy_kWh_per_mile"]
			source_truck = row["source_truck"]
			driving_event = int(row["drivecycle_number"])

			drivecycle_path = resolve_drivecycle_path(source_truck, driving_event)
			drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))

			m_gvw_kg = drivecycle_data["GVW (kg)"].loc[0]
			m_gvw_lb = m_gvw_kg / KG_PER_LB
			m_payload_lb = m_gvw_lb - m_truck_no_bat_lb - m_bat_lb

			cost_per_mile = evaluate_costs(
				mileage=fuel_consumption_kwh_per_mile,
				payload_lb=m_payload_lb,
				electricity_charge=electricity_rate,
				demand_charge=demand_charge,
				average_VMT=average_vmt,
				charging_power=max_charging_power,
				e_bat=e_bat,
				battery_chemistry=parameters.battery_chemistry,
				truck_name=dataset["truck_params"],
			)
			emissions_per_mile = evaluate_emissions(
				mileage=fuel_consumption_kwh_per_mile,
				payload_lb=m_payload_lb,
				grid_emission_intensity=present_grid_ci,
				average_VMT=average_vmt,
				e_bat=e_bat,
				battery_chemistry=parameters.battery_chemistry,
				truck_name=dataset["truck_params"],
			)

			costs.append(cost_per_mile)
			emissions.append(emissions_per_mile)

		costs_df = pd.DataFrame(costs, columns=expected_cost_cols).add_prefix("cost_")
		emissions_df = pd.DataFrame(emissions, columns=expected_emissions_cols).add_prefix("emissions_")

		columns_to_drop = costs_df.columns.tolist() + emissions_df.columns.tolist()
		results_df = results_df.drop(columns=columns_to_drop, errors="ignore").reset_index(drop=True)
		results_df = pd.concat([results_df, costs_df, emissions_df], axis=1)
		results_df.to_csv(results_csv, index=False)
		print(f"Updated all-drivecycles results CSV with costs and emissions: {results_csv}")

		missing_values = []
		for col in columns_to_drop:
			if col not in results_df.columns:
				missing_values.append(col)
				continue
			col_values = results_df.loc[:, col]
			if col_values.isna().all().all():
				missing_values.append(col)
		if missing_values:
			print(f"Warning: {truck_name} has missing values for columns: {missing_values}")


def evaluate_diesel_on_all_drivecycles(
		datasets,
		optimized_params,
		results_dir,
		param_suffix,
	):
	"""Evaluate diesel truck model on all drivecycles using EV-optimized drag/resistance values."""
	all_drivecycle_files = sorted((BASE_DIR / "messy_middle_results").glob("*_drivecycle_*_detailed.csv"))

	drivecycles_by_source = {}
	for drivecycle_path in all_drivecycle_files:
		parts = drivecycle_path.stem.split("_")
		if "drivecycle" in parts:
			drivecycle_idx = parts.index("drivecycle")
			source_truck = "_".join(parts[:drivecycle_idx])
			driving_event = int(parts[drivecycle_idx + 1])
		else:
			source_truck = parts[0]
			driving_event = int(parts[-2]) if parts[-1] == "detailed" else int(parts[-1])

		if source_truck not in drivecycles_by_source:
			drivecycles_by_source[source_truck] = []
		drivecycles_by_source[source_truck].append((drivecycle_path, driving_event))

	for dataset in datasets:
		eval_truck_name = dataset["name"]
		print(f"\nEvaluating diesel model for {eval_truck_name} on all drivecycles...")

		ev_parameters = data_collection_tools_messy.read_parameters(
			truck_params=dataset["truck_params"],
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="EV",
		)
		ev_parameters = apply_optimized_parameters(ev_parameters, optimized_params, eval_truck_name)

		diesel_truck_params = f"{dataset['truck_params']}_diesel"
		diesel_parameters = data_collection_tools_messy.read_parameters(
			truck_params=diesel_truck_params,
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="diesel",
		)
		diesel_parameters.cd = ev_parameters.cd
		diesel_parameters.cr = ev_parameters.cr

		all_results = []
		for source_truck, drivecycle_list in drivecycles_by_source.items():
			print(f"  Processing {source_truck} drivecycles...")
			for drivecycle_path, driving_event in drivecycle_list:
				try:
					drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
					m_gvw_kg = drivecycle_data["GVW (kg)"].loc[0]

					_, fuel_consumption_kwh_per_mile, mpg = truck_model_tools_diesel_messy.truck_model(
						diesel_parameters
					).get_power_requirement(drivecycle_data, m_gvw_kg)

					all_results.append({
						"source_truck": source_truck,
						"drivecycle_number": driving_event,
						"fuel_economy_kWh_per_mile": fuel_consumption_kwh_per_mile,
						"fuel_economy_mpg": mpg,
					})
				except Exception as e:
					print(f"    Warning: Error processing {drivecycle_path.name}: {e}")
					continue

		if all_results:
			results_df = pd.DataFrame(all_results)
			output_path = results_dir / f"{eval_truck_name}_all_drivecycles_diesel_results{param_suffix}.csv"
			results_df.to_csv(output_path, index=False)
			print(f"Saved {eval_truck_name} diesel cross-evaluation results: {output_path}")
		else:
			print(f"Warning: No diesel results collected for {eval_truck_name}")


def add_costs_diesel_all_drivecycles(
		datasets,
		plots_dir,
		param_suffix,
		average_vmt,
	):
	"""Add cost columns to each all-drivecycles diesel results CSV using US averages."""
	usa_data = pd.read_csv(str(BASE_DIR / "data_messy" / "energy_costs_emissions_usa.csv")).set_index("Parameter")
	diesel_price = usa_data.loc["Diesel cost ($/gal)", "Value"]

	for dataset in datasets:
		truck_name = dataset["name"]
		results_csv = plots_dir / f"{truck_name}_all_drivecycles_diesel_results{param_suffix}.csv"
		if not results_csv.exists():
			print(f"Warning: All-drivecycles diesel results CSV not found for {truck_name}, skipping.")
			continue

		results_df = pd.read_csv(results_csv)

		diesel_truck_params = f"{dataset['truck_params']}_diesel"
		diesel_parameters = data_collection_tools_messy.read_parameters(
			truck_params=diesel_truck_params,
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="diesel",
		)

		expected_cost_cols = [
			"Total capital ($/mi)",
			"Total operating ($/mi)",
			"Total fuel ($/mi)",
			"Total labor ($/mi)",
			"Other OPEXs ($/mi)",
			"TCO ($/mi)",
		]

		costs = []
		for _, row in results_df.iterrows():
			mpg = row["fuel_economy_mpg"]
			source_truck = row["source_truck"]
			driving_event = int(row["drivecycle_number"])

			drivecycle_path = resolve_drivecycle_path(source_truck, driving_event)
			drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))

			m_gvw_kg = drivecycle_data["GVW (kg)"].loc[0]
			m_gvw_lb = m_gvw_kg / KG_PER_LB
			m_truck_lb = diesel_parameters.m_truck / KG_PER_LB
			m_payload_lb = m_gvw_lb - m_truck_lb

			cost_per_mile = evaluate_costs_diesel(
				mileage_mpg=mpg,
				payload_lb=m_payload_lb,
				diesel_price=diesel_price,
				average_VMT=average_vmt,
				truck_type=diesel_truck_params,
			)
			costs.append(cost_per_mile)

		costs_df = pd.DataFrame(costs, columns=expected_cost_cols).add_prefix("diesel_cost_")

		columns_to_drop = costs_df.columns.tolist()
		results_df = results_df.drop(columns=columns_to_drop, errors="ignore").reset_index(drop=True)
		results_df = pd.concat([results_df, costs_df], axis=1)
		results_df.to_csv(results_csv, index=False)
		print(f"Updated all-drivecycles diesel results CSV with costs: {results_csv}")

		missing_values = []
		for col in columns_to_drop:
			if col not in results_df.columns:
				missing_values.append(col)
				continue
			col_values = results_df.loc[:, col]
			if col_values.isna().all().all():
				missing_values.append(col)
		if missing_values:
			print(f"Warning: {truck_name} has missing values for columns: {missing_values}")


def plot_tco_premium_all_drivecycles(datasets, plots_dir, param_suffix):
	"""Plot box-plot distributions of EV TCO premium relative to diesel across all drivecycles."""
	labels = []
	values_list = []
	rows = []

	for dataset in datasets:
		truck_name = dataset["name"]
		ev_csv = plots_dir / f"{truck_name}_all_drivecycles_results{param_suffix}.csv"
		diesel_csv = plots_dir / f"{truck_name}_all_drivecycles_diesel_results{param_suffix}.csv"
		if not ev_csv.exists() or not diesel_csv.exists():
			print(f"Warning: Missing EV or diesel results for {truck_name}, skipping.")
			continue

		ev_df = pd.read_csv(ev_csv)
		diesel_df = pd.read_csv(diesel_csv)

		merged = ev_df.merge(
			diesel_df,
			on=["source_truck", "drivecycle_number"],
			how="inner",
			suffixes=("_ev", "_diesel"),
		)
		if merged.empty:
			print(f"Warning: No matching drivecycles for {truck_name}, skipping.")
			continue

		if "cost_TCO ($/mi)" not in merged.columns or "diesel_cost_TCO ($/mi)" not in merged.columns:
			print(f"Warning: Missing TCO columns for {truck_name}, skipping.")
			continue

		premium = ((merged["cost_TCO ($/mi)"] - merged["diesel_cost_TCO ($/mi)"]) / merged["diesel_cost_TCO ($/mi)"]) * 100
		premium = premium.replace([np.inf, -np.inf], np.nan).dropna().values
		if premium.size == 0:
			print(f"Warning: No premium data for {truck_name}, skipping.")
			continue

		labels.append(truck_name)
		values_list.append(premium)
		for value in premium:
			rows.append({"truck_name": truck_name, "tco_premium_%": value})

	if not values_list:
		print("Warning: No TCO premium data found for plotting.")
		return

	plt.figure(figsize=(10, 6))
	plt.boxplot(values_list, tick_labels=labels, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markeredgecolor='red'))
	plt.title("EV TCO Premium vs Diesel (All Drivecycles)")
	plt.ylabel("TCO Premium (%)")
	plt.xticks(rotation=20, ha='right')
	plt.tight_layout()
	output_path = plots_dir / f"tco_premium_all_drivecycles{param_suffix}.png"
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"Saved plot: {output_path.absolute()}")
	plt.close()

	if rows:
		premium_df = pd.DataFrame(rows)
		premium_csv = plots_dir / f"tco_premium_all_drivecycles{param_suffix}.csv"
		premium_df.to_csv(premium_csv, index=False)
		print(f"Saved TCO premium values: {premium_csv}")


def plot_distribution_comparisons_all_drivecycles(datasets, plots_dir, param_suffix):
	"""Plot distribution comparisons for all-drivecycles cost/emissions across trucks."""
	truck_data = {}
	for dataset in datasets:
		truck_name = dataset["name"]
		results_csv = plots_dir / f"{truck_name}_all_drivecycles_results{param_suffix}.csv"
		if not results_csv.exists():
			print(f"Warning: All-drivecycles results CSV not found for {truck_name}, skipping distribution plot.")
			continue
		results_df = pd.read_csv(results_csv)
		truck_data[truck_name] = results_df

	if not truck_data:
		print("Warning: No all-drivecycles results CSVs found for distribution plots.")
		return

	def plot_box(data_map, column, title, ylabel, output_name):
		labels = []
		values_list = []
		for truck_name, df in data_map.items():
			if column not in df.columns:
				continue
			values = df[column].dropna().values
			if values.size == 0:
				continue
			labels.append(truck_name)
			values_list.append(values)

		if not values_list:
			print(f"Warning: No data for {column}, skipping {output_name}.")
			return

		plt.figure(figsize=(10, 6))
		plt.boxplot(values_list, tick_labels=labels, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markeredgecolor='red'))
		plt.title(title)
		plt.ylabel(ylabel)
		plt.xticks(rotation=20, ha='right')
		plt.tight_layout()
		output_path = plots_dir / output_name
		plt.savefig(output_path, dpi=300, bbox_inches="tight")
		print(f"Saved plot: {output_path.absolute()}")
		plt.close()

	plot_box(
		truck_data,
		column="fuel_economy_kWh_per_mile",
		title="Fuel Economy Distribution (All Drivecycles)",
		ylabel="Fuel Economy (kWh/mile)",
		output_name=f"fuel_economy_distribution_all_drivecycles{param_suffix}.png",
	)

	plot_box(
		truck_data,
		column="cost_TCO ($/mi)",
		title="Total Cost per Mile Distribution (All Drivecycles)",
		ylabel="TCO ($/mile)",
		output_name=f"tco_distribution_all_drivecycles{param_suffix}.png",
	)

	plot_box(
		truck_data,
		column="emissions_GHGs total (gCO2/mi)",
		title="Total Emissions per Mile Distribution (All Drivecycles)",
		ylabel="Emissions (gCO2/mi)",
		output_name=f"emissions_distribution_all_drivecycles{param_suffix}.png",
	)


def main():
	"""Main routine for generating drivecycle comparison plots with optional parameter optimization."""
	
	# Configuration flags - set these to control behavior
	optimized = True  # Set to True to use optimized parameters
	regenerate_model_results = False  # Set to True to regenerate model results (slow), False to use cached results (fast)
	
	# Constants
	average_vmt = 100000
	m_truck_max_lb = 82000
	m_truck_max_kg = m_truck_max_lb * KG_PER_LB
	
	# Load optimized parameters if requested
	optimized_params = load_optimized_parameters(use_optimized=optimized)
	
	# Determine output directory and suffix
	param_suffix = "_optimized" if optimized else "_original"
	plots_dir = Path("plots_messy")
	plots_dir.mkdir(parents=True, exist_ok=True)

	# Make plots_dir absolute to reference files from root
	plots_dir = BASE_DIR / "plots_messy"
	
	model_results_dir = plots_dir / f"model_results{param_suffix}"
	if regenerate_model_results:
		model_results_dir.mkdir(parents=True, exist_ok=True)
	
	print(f"\nRunning analysis with {'optimized' if optimized else 'original'} parameters")
	print(f"Regenerate model results: {regenerate_model_results}")
	print(f"Output directory: {plots_dir.absolute()}")
	print(f"Model results cache: {model_results_dir.absolute()}")
	
	datasets = [
		{
			"name": "saia2",
			"truck_params": "saia",
			"battery_col": "saia2",
			"drivecycle_glob": "saia2_drivecycle_*_detailed.csv",
			"summary_path": "messy_middle_results/saia2_drivecycle_data.csv",
		},
		{
			"name": "4gen",
			"truck_params": "4gen",
			"battery_col": "4gen",
			"drivecycle_glob": "4gen_drivecycle_*_detailed.csv",
			"summary_path": "messy_middle_results/4gen_drivecycle_data.csv",
		},
		{
			"name": "joyride",
			"truck_params": "joyride",
			"battery_col": "joyride",
			"drivecycle_glob": "joyride_drivecycle_*_detailed.csv",
			"summary_path": "messy_middle_results/joyride_drivecycle_data.csv",
		},
		{
			"name": "nevoya_with_weight",
			"truck_params": "nevoya",
			"battery_col": "nevoya_with_weight",
			"drivecycle_glob": "nevoya_with_weight_drivecycle_*_detailed.csv",
			"summary_path": "messy_middle_results/nevoya_with_weight_drivecycle_data.csv",
		},
	]
	# 	optimized_params=optimized_params,
	# 	battery_caps=battery_caps,
	# 	m_truck_max_kg=m_truck_max_kg,
	# 	plots_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# )

	# # Read in saved results for each truck and calculate and plot distributions of the cost/mile and CO2e/mile for each driving event
	# add_cost_emissions(
	# 	datasets=datasets,
	# 	plots_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# 	average_vmt=average_vmt,
	# 	optimized_params=optimized_params,
	# 	battery_caps=battery_caps,
	# 	m_truck_max_kg=m_truck_max_kg,
	# )

	# # Generate summary plots after updating cost/emissions columns
	# for dataset in datasets:
	# 	truck_name = dataset['name']
	# 	results_csv = plots_dir / f"{truck_name}_results{param_suffix}.csv"
	# 	if not results_csv.exists():
	# 		continue
	# 	results_df = pd.read_csv(results_csv)
	# 	plot_truck_summary(truck_name, results_df.to_dict('records'), param_suffix, plots_dir)
	
	# # Generate distribution comparison plots across trucks
	# plot_distribution_comparisons(datasets=datasets, plots_dir=plots_dir, param_suffix=param_suffix)
	
	# # Evaluate each truck model on all drivecycles
	# evaluate_truck_on_all_drivecycles(
	# 	datasets=datasets,
	# 	optimized_params=optimized_params,
	# 	battery_caps=battery_caps,
	# 	m_truck_max_kg=m_truck_max_kg,
	# 	results_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# )

	# # Add US-average cost/emissions to all-drivecycles results and compare distributions
	# add_cost_emissions_all_drivecycles(
	# 	datasets=datasets,
	# 	plots_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# 	average_vmt=average_vmt,
	# 	optimized_params=optimized_params,
	# 	battery_caps=battery_caps,
	# 	m_truck_max_kg=m_truck_max_kg,
	# )
	# plot_distribution_comparisons_all_drivecycles(
	# 	datasets=datasets,
	# 	plots_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# )

	# # Evaluate diesel model on all drivecycles (using EV-optimized cd/cr), add diesel costs, and plot EV TCO premium
	# evaluate_diesel_on_all_drivecycles(
	# 	datasets=datasets,
	# 	optimized_params=optimized_params,
	# 	results_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# )
	# add_costs_diesel_all_drivecycles(
	# 	datasets=datasets,
	# 	plots_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# 	average_vmt=average_vmt,
	# )
	# plot_tco_premium_all_drivecycles(
	# 	datasets=datasets,
	# 	plots_dir=plots_dir,
	# 	param_suffix=param_suffix,
	# )
	
	# Generate detailed cost and emissions breakdown
	generate_cost_emissions_breakdown(
		datasets=datasets,
		plots_dir=plots_dir,
		optimized_params=optimized_params,
		param_suffix=param_suffix,
		average_vmt=average_vmt,
	)
	
	# Generate presentation-style step-by-step visuals
	generate_presentation_visuals(
		datasets=datasets,
		plots_dir=plots_dir,
		optimized_params=optimized_params,
		param_suffix=param_suffix,
		average_vmt=average_vmt,
	)


def generate_cost_emissions_breakdown(
	datasets,
	plots_dir,
	optimized_params,
	param_suffix="_optimized",
	average_vmt=100000,
):
	"""
	Generate a detailed side-by-side cost and emissions breakdown for EV vs Diesel trucks.
	
	Uses pre-computed cost values from the result CSVs to avoid double-counting.
	Breaks down costs into:
	- Capital costs (base truck, battery/fuel tank)
	- Operating costs (electricity/fuel, labor, maintenance, insurance, tolls, permits)
	- Emissions
	
	Creates detailed tables and visualizations comparing all components.
	"""
	
	breakdown_dir = plots_dir / f"cost_emissions_breakdown{param_suffix}"
	breakdown_dir.mkdir(parents=True, exist_ok=True)
	
	print("\n" + "="*80)
	print("COST & EMISSIONS BREAKDOWN: EV vs Diesel")
	print("="*80)
	
	breakdown_data = []
	
	for dataset in datasets:
		truck_name = dataset["name"]
		print(f"\n{'='*80}")
		print(f"Analyzing: {truck_name}")
		print(f"{'='*80}")
		
		# Load EV results CSV (pre-computed costs)
		ev_results_path = plots_dir / f"{truck_name}_all_drivecycles_results{param_suffix}.csv"
		if not ev_results_path.exists():
			print(f"Warning: EV results not found at {ev_results_path}")
			continue
		
		# Load diesel results CSV (pre-computed costs)
		diesel_results_path = plots_dir / f"{truck_name}_all_drivecycles_diesel_results{param_suffix}.csv"
		if not diesel_results_path.exists():
			print(f"Warning: Diesel results not found at {diesel_results_path}")
			continue
		
		# Load the CSVs
		ev_df = pd.read_csv(ev_results_path)
		diesel_df = pd.read_csv(diesel_results_path)
		
		# Extract average values from pre-computed cost columns
		# EV costs are in columns like 'cost_Total capital ($/mi)', 'cost_Total electricity ($/mi)', etc.
		ev_capital_per_mi = ev_df['cost_Total capital ($/mi)'].mean()
		ev_operating_per_mi = ev_df['cost_Total operating ($/mi)'].mean()
		ev_electricity_per_mi = ev_df['cost_Total electricity ($/mi)'].mean()
		ev_labor_per_mi = ev_df['cost_Total labor ($/mi)'].mean()
		ev_other_opex_per_mi = ev_df['cost_Other OPEXs ($/mi)'].mean()
		
		# Diesel costs are in columns like 'diesel_cost_Total capital ($/mi)', 'diesel_cost_Total fuel ($/mi)', etc.
		diesel_capital_per_mi = diesel_df['diesel_cost_Total capital ($/mi)'].mean()
		diesel_operating_per_mi = diesel_df['diesel_cost_Total operating ($/mi)'].mean()
		diesel_fuel_per_mi = diesel_df['diesel_cost_Total fuel ($/mi)'].mean()
		diesel_labor_per_mi = diesel_df['diesel_cost_Total labor ($/mi)'].mean()
		diesel_other_opex_per_mi = diesel_df['diesel_cost_Other OPEXs ($/mi)'].mean()
		
		# Calculate detailed breakdowns
		# Load parameters to get detailed cost components
		ev_params = data_collection_tools_messy.read_parameters(
			truck_params=dataset["truck_params"],
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="EV",
		)
		ev_params = apply_optimized_parameters(ev_params, optimized_params, truck_name)
		
		# Get truck cost data for detailed breakdown
		ev_truck_cost_data = data_collection_tools_messy.read_truck_cost_data(
			truck_type='EV',
			chemistry=ev_params.battery_chemistry,
		)
		diesel_truck_cost_data = data_collection_tools_messy.read_truck_cost_data(truck_type='diesel')
		
		# Get battery and charging info
		e_bat_baseline = battery_caps.loc['Mean', dataset["battery_col"]]
		max_charging_power_data = pd.read_csv(str(BASE_DIR / "messy_middle_results" / "max_charging_powers.csv"), index_col="truck_name")
		max_charging_power = max_charging_power_data.loc[truck_name, "99th_percentile_charging_power_kw"]
		
		# Load US average values for electricity breakdown
		usa_data = pd.read_csv(str(BASE_DIR / "data_messy" / "energy_costs_emissions_usa.csv")).set_index("Parameter")
		electricity_cost_baseline = usa_data.loc["Commercial electricity ($/kWh)", "Value"]
		demand_charge_baseline = usa_data.loc["Demand charge ($/kW)", "Value"]
		
		# Get average fuel economy from results
		ev_fuel_kwh_per_mi_avg = ev_df['model_fuel'].mean() if 'model_fuel' in ev_df.columns else 1.85
		
		# Calculate electricity cost breakdown by year
		from costing_and_emissions_tools_messy import get_electricity_cost_by_year
		electricity_unit_by_year = get_electricity_cost_by_year(
			ev_params,
			ev_fuel_kwh_per_mi_avg,
			demand_charge_baseline,
			electricity_cost_baseline,
			max_charging_power,
		)
		
		# Break down electricity costs proportionally
		# The electricity_unit_by_year contains normalized costs ($/kWh)
		# We use the pre-computed total and break it down by component proportions
		total_normalized = np.mean(electricity_unit_by_year['Total'])
		
		# Calculate proportions of each component
		energy_proportion = np.mean(electricity_unit_by_year['Normalized energy charge']) / total_normalized
		demand_proportion = np.mean(electricity_unit_by_year['Normalized demand charge']) / total_normalized
		capital_proportion = np.mean(electricity_unit_by_year['Normalized capital']) / total_normalized
		fixed_proportion = np.mean(electricity_unit_by_year['Normalized fixed']) / total_normalized
		
		# Apply proportions to pre-computed total electricity cost
		electricity_energy_charge = ev_electricity_per_mi * energy_proportion
		electricity_demand_charge = ev_electricity_per_mi * demand_proportion
		electricity_charger_capital = ev_electricity_per_mi * capital_proportion
		electricity_charger_fixed = ev_electricity_per_mi * fixed_proportion
		
		# Battery cost breakdown (initial + replacements)
		battery_unit_cost = ev_truck_cost_data['Battery Unit Cost ($/kWh)']
		battery_kwh = e_bat_baseline
		
		# Initial battery (purchased upfront, year 0)
		battery_initial_cost = battery_unit_cost * battery_kwh
		
		# Calculate replacements based on degradation
		lifetime_miles = ev_params.VMT['VMT (miles)'].sum()
		ev_fuel_kwh_per_mi_avg = ev_df['model_fuel'].mean() if 'model_fuel' in ev_df.columns else 1.85
		
		# Get discount factor for each year
		discountfactor = 1 / np.power(1 + ev_params.discountrate, np.arange(10))
		
		# Calculate replacement batteries needed
		battery_replacements = calculate_replacements(
			ev_params.VMT['VMT (miles)'],
			ev_fuel_kwh_per_mi_avg,
			e_bat=battery_kwh,
		)
		
		# Replacement cost occurs mid-life with discounting
		# Typically at year 5 (middle of 10-year period)
		replacement_discount_factor = discountfactor[5] if battery_replacements > 0 else 0
		battery_replacement_cost = battery_unit_cost * battery_kwh * battery_replacements * replacement_discount_factor
		
		# Convert to per-mile costs
		ev_battery_initial_per_mi = battery_initial_cost / lifetime_miles
		ev_battery_replacement_per_mi = battery_replacement_cost / lifetime_miles
		ev_battery_cost_per_mi = ev_battery_initial_per_mi + ev_battery_replacement_per_mi
		
		# EV component costs
		ev_glider = ev_truck_cost_data['Capital Costs']['glider ($)']
		ev_motor_inverter = ev_truck_cost_data['Capital Costs']['motor and inverter ($/kW)'] * max_charging_power
		ev_dcdc = ev_truck_cost_data['Capital Costs']['DC-DC converter ($/kW)'] * max_charging_power
		
		# Diesel component costs
		diesel_glider = diesel_truck_cost_data['Capital Costs']['glider ($)']
		diesel_engine = diesel_truck_cost_data['Capital Costs']['engine ($/kW)'] * max_charging_power  # Approx power
		diesel_trans = diesel_truck_cost_data['Capital Costs']['transmission ($)']
		diesel_aftertx = diesel_truck_cost_data['Capital Costs']['aftertreatment ($)']
		diesel_tank = diesel_truck_cost_data['Capital Costs']['fuel tank ($)']
		
		# Convert to per-mile costs
		ev_glider_per_mi = ev_glider / lifetime_miles
		ev_motor_inverter_per_mi = ev_motor_inverter / lifetime_miles
		ev_dcdc_per_mi = ev_dcdc / lifetime_miles
		
		diesel_glider_per_mi = diesel_glider / lifetime_miles
		diesel_engine_per_mi = diesel_engine / lifetime_miles
		diesel_trans_per_mi = diesel_trans / lifetime_miles
		diesel_aftertx_per_mi = diesel_aftertx / lifetime_miles
		diesel_tank_per_mi = diesel_tank / lifetime_miles
		
		# Break down Other OPEXs proportionally based on truck cost data
		# Get nominal rates from cost data
		ev_maint_nominal = ev_truck_cost_data['Operating Costs'].get('maintenance & repair ($/mi)', 0)
		ev_insurance_nominal = ev_truck_cost_data['Operating Costs'].get('insurance ($/mi-$)', 0) * ev_capital_per_mi
		ev_tolls_nominal = ev_truck_cost_data['Operating Costs'].get('tolls ($/mi)', 0)
		ev_permits_nominal = ev_truck_cost_data['Operating Costs'].get('permits and licenses ($/mi)', 0)
		ev_misc_nominal = ev_truck_cost_data['Operating Costs'].get('misc ($/mi)', 0)
		ev_total_nominal = ev_maint_nominal + ev_insurance_nominal + ev_tolls_nominal + ev_permits_nominal + ev_misc_nominal
		
		# Apply proportionally to match pre-computed total
		if ev_total_nominal > 0:
			ev_maintenance_per_mi = (ev_maint_nominal / ev_total_nominal) * ev_other_opex_per_mi
			ev_insurance_per_mi = (ev_insurance_nominal / ev_total_nominal) * ev_other_opex_per_mi
			ev_tolls_per_mi = (ev_tolls_nominal / ev_total_nominal) * ev_other_opex_per_mi
			ev_permits_per_mi = (ev_permits_nominal / ev_total_nominal) * ev_other_opex_per_mi
			ev_misc_per_mi = (ev_misc_nominal / ev_total_nominal) * ev_other_opex_per_mi
		else:
			ev_maintenance_per_mi = ev_insurance_per_mi = ev_tolls_per_mi = ev_permits_per_mi = ev_misc_per_mi = 0
		
		# Same for diesel
		diesel_maint_nominal = diesel_truck_cost_data['Operating Costs'].get('maintenance & repair ($/mi)', 0)
		diesel_insurance_nominal = diesel_truck_cost_data['Operating Costs'].get('insurance ($/mi-$)', 0) * diesel_capital_per_mi
		diesel_tolls_nominal = diesel_truck_cost_data['Operating Costs'].get('tolls ($/mi)', 0)
		diesel_permits_nominal = diesel_truck_cost_data['Operating Costs'].get('permits and licenses ($/mi)', 0)
		diesel_total_nominal = diesel_maint_nominal + diesel_insurance_nominal + diesel_tolls_nominal + diesel_permits_nominal
		
		if diesel_total_nominal > 0:
			diesel_maintenance_per_mi = (diesel_maint_nominal / diesel_total_nominal) * diesel_other_opex_per_mi
			diesel_insurance_per_mi = (diesel_insurance_nominal / diesel_total_nominal) * diesel_other_opex_per_mi
			diesel_tolls_per_mi = (diesel_tolls_nominal / diesel_total_nominal) * diesel_other_opex_per_mi
			diesel_permits_per_mi = (diesel_permits_nominal / diesel_total_nominal) * diesel_other_opex_per_mi
		else:
			diesel_maintenance_per_mi = diesel_insurance_per_mi = diesel_tolls_per_mi = diesel_permits_per_mi = 0
		
		# Create detailed breakdown table with component-level detail
		breakdown_df = pd.DataFrame({
			'Component': [
				'CAPITAL COSTS',
				'  Chassis (Glider)',
				'  Powertrain',
				'    Motor & Inverter',
				'    DC-DC Converter',
				'    Engine',
				'    Transmission',
				'    Aftertreatment',
				'    Fuel Tank',
				'  Battery',
				'    Initial Battery',
				'    Replacement Batteries',
				'Total Capital',
				'',
				'OPERATING COSTS',
				'  Energy/Fuel',
				'    Energy Charge',
				'    Demand Charge',
				'    Charger Capital',
				'    Charger Fixed Costs',
				'  Labor',
				'  Maintenance & Repair',
				'  Insurance',
				'  Tolls',
				'  Permits & Licenses',
				'  Misc',
				'Total Operating',
				'',
				'TOTAL TCO',
			],
			'EV ($/mi)': [
				'',  # Section header
				ev_glider_per_mi,
				'',  # Powertrain header
				ev_motor_inverter_per_mi,
				ev_dcdc_per_mi,
				0.0,  # No engine
				0.0,  # No transmission
				0.0,  # No aftertreatment
				0.0,  # No fuel tank
				'',  # Battery header
				ev_battery_initial_per_mi,
				ev_battery_replacement_per_mi,
				ev_capital_per_mi,
				'',  # Spacer
				'',  # Section header
				ev_electricity_per_mi,
				electricity_energy_charge,
				electricity_demand_charge,
				electricity_charger_capital,
				electricity_charger_fixed,
				ev_labor_per_mi,
				ev_maintenance_per_mi,
				ev_insurance_per_mi,
				ev_tolls_per_mi,
				ev_permits_per_mi,
				ev_misc_per_mi,
				ev_operating_per_mi,
				'',  # Spacer
				ev_capital_per_mi + ev_operating_per_mi,
			],
			'Diesel ($/mi)': [
				'',  # Section header
				diesel_glider_per_mi,
				'',  # Powertrain header
				0.0,  # No motor/inverter
				0.0,  # No DC-DC
				diesel_engine_per_mi,
				diesel_trans_per_mi,
				diesel_aftertx_per_mi,
				diesel_tank_per_mi,
				'',  # No battery
				0.0,  # No initial battery
				0.0,  # No replacement batteries
				diesel_capital_per_mi,
				'',  # Spacer
				'',  # Section header
				diesel_fuel_per_mi,
				diesel_fuel_per_mi,  # All fuel is energy charge
				0.0,  # No demand charge
				0.0,  # No charger capital
				0.0,  # No charger fixed costs
				diesel_labor_per_mi,
				diesel_maintenance_per_mi,
				diesel_insurance_per_mi,
				diesel_tolls_per_mi,
				diesel_permits_per_mi,
				0.0,  # No misc for diesel
				diesel_operating_per_mi,
				'',  # Spacer
				diesel_capital_per_mi + diesel_operating_per_mi,
			],
		})
		
		# Calculate differences (only for numeric rows)
		breakdown_df['EV - Diesel ($/mi)'] = pd.to_numeric(breakdown_df['EV ($/mi)'], errors='coerce') - pd.to_numeric(breakdown_df['Diesel ($/mi)'], errors='coerce')
		breakdown_df['% Difference'] = (breakdown_df['EV - Diesel ($/mi)'] / pd.to_numeric(breakdown_df['Diesel ($/mi)'], errors='coerce')) * 100
		
		# Save to CSV
		csv_path = breakdown_dir / f"{truck_name}_cost_breakdown.csv"
		breakdown_df.to_csv(csv_path, index=False)
		print(f"\nSaved cost breakdown: {csv_path}")
		print(breakdown_df.to_string())
		
		breakdown_data.append({
			'truck_name': truck_name,
			'df': breakdown_df,
		})
		
		# ===== CREATE VISUALIZATION =====
		fig, axes = plt.subplots(2, 3, figsize=(20, 12))
		
		# Extract component values for plotting
		def get_value(component_name):
			return pd.to_numeric(breakdown_df[breakdown_df['Component'] == component_name]['EV ($/mi)'].values[0], errors='coerce')
		
		def get_diesel_value(component_name):
			return pd.to_numeric(breakdown_df[breakdown_df['Component'] == component_name]['Diesel ($/mi)'].values[0], errors='coerce')
		
		# Plot 1: Overall Capital vs Operating Breakdown
		ax1 = axes[0, 0]
		x_pos = [0, 1]
		widths = [0.6, 0.6]
		
		ev_capital = get_value('Total Capital')
		ev_operating = get_value('Total Operating')
		diesel_capital = get_diesel_value('Total Capital')
		diesel_operating = get_diesel_value('Total Operating')
		
		ax1.bar(x_pos[0], ev_capital, widths[0], label='Capital', color='#1f77b4', alpha=0.8)
		ax1.bar(x_pos[0], ev_operating, widths[0], bottom=ev_capital, label='Operating', color='#ff7f0e', alpha=0.8)
		
		ax1.bar(x_pos[1], diesel_capital, widths[1], color='#1f77b4', alpha=0.8)
		ax1.bar(x_pos[1], diesel_operating, widths[1], bottom=diesel_capital, color='#ff7f0e', alpha=0.8)
		
		ax1.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=11)
		ax1.set_title(f'Total Cost Overview', fontweight='bold', fontsize=13)
		ax1.set_xticks(x_pos)
		ax1.set_xticklabels(['EV', 'Diesel'])
		ax1.legend(fontsize=9)
		ax1.grid(True, alpha=0.3, axis='y')
		
		ev_total = ev_capital + ev_operating
		diesel_total = diesel_capital + diesel_operating
		ax1.text(x_pos[0], ev_total + 0.05, f'${ev_total:.2f}/mi', ha='center', va='bottom', fontweight='bold', fontsize=10)
		ax1.text(x_pos[1], diesel_total + 0.05, f'${diesel_total:.2f}/mi', ha='center', va='bottom', fontweight='bold', fontsize=10)
		
		# Plot 2: Capital Cost Breakdown with Powertrain Details
		ax2 = axes[0, 1]
		
		# EV components
		ev_chassis = get_value('  Chassis (Glider)')
		ev_motor = get_value('    Motor & Inverter')
		ev_dcdc = get_value('    DC-DC Converter')
		ev_battery_initial = get_value('    Initial Battery')
		ev_battery_replacement = get_value('    Replacement Batteries')
		ev_battery = ev_battery_initial + ev_battery_replacement if ev_battery_initial is not None else 0
		
		# Diesel components
		diesel_chassis = get_diesel_value('  Chassis (Glider)')
		diesel_engine = get_diesel_value('    Engine')
		diesel_trans = get_diesel_value('    Transmission')
		diesel_aftertx = get_diesel_value('    Aftertreatment')
		diesel_tank = get_diesel_value('    Fuel Tank')
		
		# Stack EV components
		ev_bottom = ev_chassis
		ax2.bar(x_pos[0], ev_chassis, widths[0], label='Chassis', color='#8c564b', alpha=0.8)
		ax2.bar(x_pos[0], ev_motor, widths[0], bottom=ev_bottom, label='Motor & Inverter', color='#e377c2', alpha=0.8)
		ev_bottom += ev_motor
		ax2.bar(x_pos[0], ev_dcdc, widths[0], bottom=ev_bottom, label='DC-DC Converter', color='#7f7f7f', alpha=0.8)
		ev_bottom += ev_dcdc
		ax2.bar(x_pos[0], ev_battery, widths[0], bottom=ev_bottom, label='Battery', color='#9467bd', alpha=0.8)
		
		# Stack Diesel components
		diesel_bottom = diesel_chassis
		ax2.bar(x_pos[1], diesel_chassis, widths[1], color='#8c564b', alpha=0.8)
		ax2.bar(x_pos[1], diesel_engine, widths[1], bottom=diesel_bottom, label='Engine', color='#1f77b4', alpha=0.8)
		diesel_bottom += diesel_engine
		ax2.bar(x_pos[1], diesel_trans, widths[1], bottom=diesel_bottom, label='Transmission', color='#ff7f0e', alpha=0.8)
		diesel_bottom += diesel_trans
		ax2.bar(x_pos[1], diesel_aftertx, widths[1], bottom=diesel_bottom, label='Aftertreatment', color='#2ca02c', alpha=0.8)
		diesel_bottom += diesel_aftertx
		ax2.bar(x_pos[1], diesel_tank, widths[1], bottom=diesel_bottom, label='Fuel Tank', color='#d62728', alpha=0.8)
		
		ax2.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=11)
		ax2.set_title('Capital Cost Breakdown (with Components)', fontweight='bold', fontsize=13)
		ax2.set_xticks(x_pos)
		ax2.set_xticklabels(['EV', 'Diesel'])
		ax2.legend(fontsize=8, loc='upper left')
		ax2.grid(True, alpha=0.3, axis='y')
		ax2.text(x_pos[0], ev_capital + 0.01, f'${ev_capital:.2f}', ha='center', va='bottom', fontsize=9)
		ax2.text(x_pos[1], diesel_capital + 0.01, f'${diesel_capital:.2f}', ha='center', va='bottom', fontsize=9)
		
		# Plot 3: Electricity/Fuel Cost Detailed Breakdown
		ax3 = axes[0, 2]
		ev_energy_charge = get_value('    Energy Charge')
		ev_demand_charge = get_value('    Demand Charge')
		ev_charger_capital = get_value('    Charger Capital')
		ev_charger_fixed = get_value('    Charger Fixed Costs')
		diesel_fuel = get_diesel_value('    Energy Charge')
		
		# EV electricity breakdown
		ax3.bar(x_pos[0], ev_energy_charge, widths[0], label='Energy Charge', color='#8c564b', alpha=0.8)
		ax3.bar(x_pos[0], ev_demand_charge, widths[0], bottom=ev_energy_charge, label='Demand Charge', color='#e377c2', alpha=0.8)
		ax3.bar(x_pos[0], ev_charger_capital, widths[0], bottom=ev_energy_charge+ev_demand_charge, label='Charger Capital', color='#7f7f7f', alpha=0.8)
		ax3.bar(x_pos[0], ev_charger_fixed, widths[0], bottom=ev_energy_charge+ev_demand_charge+ev_charger_capital, label='Charger Fixed', color='#bcbd22', alpha=0.8)
		
		# Diesel fuel
		ax3.bar(x_pos[1], diesel_fuel, widths[1], label='Diesel Fuel', color='#17becf', alpha=0.8)
		
		ax3.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=11)
		ax3.set_title('Energy/Fuel Cost Breakdown', fontweight='bold', fontsize=13)
		ax3.set_xticks(x_pos)
		ax3.set_xticklabels(['EV', 'Diesel'])
		ax3.legend(fontsize=8, loc='upper left')
		ax3.grid(True, alpha=0.3, axis='y')
		
		ev_elec_total = ev_energy_charge + ev_demand_charge + ev_charger_capital + ev_charger_fixed
		ax3.text(x_pos[0], ev_elec_total + 0.02, f'${ev_elec_total:.2f}', ha='center', va='bottom', fontsize=9)
		ax3.text(x_pos[1], diesel_fuel + 0.02, f'${diesel_fuel:.2f}', ha='center', va='bottom', fontsize=9)
		
		# Plot 4: Other Operating Costs Breakdown
		ax4 = axes[1, 0]
		other_components = ['  Labor', '  Maintenance & Repair', '  Insurance', '  Tolls', '  Permits & Licenses']
		ev_other_vals = [get_value(c) for c in other_components]
		diesel_other_vals = [get_diesel_value(c) for c in other_components]
		
		x_pos_other = np.arange(len(other_components))
		width = 0.35
		
		ax4.bar(x_pos_other - width/2, ev_other_vals, width, label='EV', color='#2ca02c', alpha=0.8)
		ax4.bar(x_pos_other + width/2, diesel_other_vals, width, label='Diesel', color='#d62728', alpha=0.8)
		
		ax4.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=11)
		ax4.set_title('Other Operating Costs', fontweight='bold', fontsize=13)
		ax4.set_xticks(x_pos_other)
		component_labels = [c.strip() for c in other_components]
		ax4.set_xticklabels(component_labels, rotation=45, ha='right', fontsize=9)
		ax4.legend(fontsize=9)
		ax4.grid(True, alpha=0.3, axis='y')
		
		# Plot 5: EV Electricity Components Pie Chart
		ax5 = axes[1, 1]
		elec_components = ['Energy\nCharge', 'Demand\nCharge', 'Charger\nCapital', 'Charger\nFixed']
		elec_values = [ev_energy_charge, ev_demand_charge, ev_charger_capital, ev_charger_fixed]
		colors_pie = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
		
		# Create pie chart with percentages
		wedges, texts, autotexts = ax5.pie(elec_values, labels=elec_components, colors=colors_pie, 
										   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
		for autotext in autotexts:
			autotext.set_color('white')
			autotext.set_fontweight('bold')
		
		ax5.set_title(f'EV Electricity Cost\nComponents (${ev_elec_total:.2f}/mi)', fontweight='bold', fontsize=13)
		
		# Plot 6: All Operating Cost Components Stacked
		ax6 = axes[1, 2]
		
		# Stack all operating components
		ev_elec = get_value('  Energy/Fuel')
		ev_labor = get_value('  Labor')
		ev_maint = get_value('  Maintenance & Repair')
		ev_insurance = get_value('  Insurance')
		ev_tolls = get_value('  Tolls')
		ev_permits = get_value('  Permits & Licenses')
		
		diesel_fuel = get_diesel_value('  Energy/Fuel')
		diesel_labor = get_diesel_value('  Labor')
		diesel_maint = get_diesel_value('  Maintenance & Repair')
		diesel_insurance = get_diesel_value('  Insurance')
		diesel_tolls = get_diesel_value('  Tolls')
		diesel_permits = get_diesel_value('  Permits & Licenses')
		
		# Stacked bars
		ax6.bar(x_pos[0], ev_elec, widths[0], label='Energy/Fuel', color='#1f77b4', alpha=0.8)
		ax6.bar(x_pos[0], ev_labor, widths[0], bottom=ev_elec, label='Labor', color='#ff7f0e', alpha=0.8)
		ax6.bar(x_pos[0], ev_maint, widths[0], bottom=ev_elec+ev_labor, label='Maintenance', color='#2ca02c', alpha=0.8)
		ax6.bar(x_pos[0], ev_tolls+ev_permits+ev_insurance, widths[0], bottom=ev_elec+ev_labor+ev_maint, label='Other', color='#d62728', alpha=0.8)
		
		ax6.bar(x_pos[1], diesel_fuel, widths[1], color='#1f77b4', alpha=0.8)
		ax6.bar(x_pos[1], diesel_labor, widths[1], bottom=diesel_fuel, color='#ff7f0e', alpha=0.8)
		ax6.bar(x_pos[1], diesel_maint, widths[1], bottom=diesel_fuel+diesel_labor, color='#2ca02c', alpha=0.8)
		ax6.bar(x_pos[1], diesel_tolls+diesel_permits+diesel_insurance, widths[1], bottom=diesel_fuel+diesel_labor+diesel_maint, color='#d62728', alpha=0.8)
		
		ax6.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=11)
		ax6.set_title('Operating Costs Stacked', fontweight='bold', fontsize=13)
		ax6.set_xticks(x_pos)
		ax6.set_xticklabels(['EV', 'Diesel'])
		ax6.legend(fontsize=9)
		ax6.grid(True, alpha=0.3, axis='y')
		ax6.text(x_pos[0], ev_operating + 0.05, f'${ev_operating:.2f}', ha='center', va='bottom', fontsize=9)
		ax6.text(x_pos[1], diesel_operating + 0.05, f'${diesel_operating:.2f}', ha='center', va='bottom', fontsize=9)
		
		# Add overall title
		fig.suptitle(f'{truck_name} - Detailed Cost Breakdown Analysis', fontsize=16, fontweight='bold', y=0.995)
		
		plt.tight_layout()
		plot_path = breakdown_dir / f"{truck_name}_cost_breakdown{param_suffix}.png"
		plt.savefig(plot_path, dpi=300, bbox_inches='tight')
		print(f"Saved cost breakdown plot: {plot_path}")
		plt.close(fig)
	
	print("\n" + "="*80)
	print("COST & EMISSIONS BREAKDOWN COMPLETE")
	print("="*80)


def generate_presentation_visuals(
	datasets,
	plots_dir,
	optimized_params,
	param_suffix="_optimized",
	average_vmt=100000,
):
	"""
	Generate step-by-step presentation visualizations that build up the analysis.
	
	Creates 6 separate plots for each truck:
	1. Capital costs only
	2. Capital + Operating costs
	3. Capital cost breakdown (with components)
	4. Other Operating Costs breakdown
	5. Operating costs stacked with only energy/fuel
	6. Operating costs stacked with all components
	
	All plots keep consistent colors and y-axis ranges from the summary visualization,
	but exclude titles, legends, and text overlays for presentation use.
	"""
	
	presentation_dir = plots_dir / f"presentation_visuals{param_suffix}"
	presentation_dir.mkdir(parents=True, exist_ok=True)
	
	print("\n" + "="*80)
	print("GENERATING PRESENTATION VISUALS")
	print("="*80)
	
	breakdown_dir = plots_dir / f"cost_emissions_breakdown{param_suffix}"
	
	for dataset in datasets:
		truck_name = dataset["name"]
		print(f"\nGenerating presentation visuals for: {truck_name}")
		
		# Load the breakdown CSV
		csv_path = breakdown_dir / f"{truck_name}_cost_breakdown.csv"
		if not csv_path.exists():
			print(f"Warning: Breakdown CSV not found for {truck_name}, skipping.")
			continue
		
		breakdown_df = pd.read_csv(csv_path)
		
		# Helper function to get values from breakdown_df
		def get_value(component_name):
			try:
				return pd.to_numeric(breakdown_df[breakdown_df['Component'] == component_name]['EV ($/mi)'].values[0], errors='coerce')
			except:
				return 0
		
		def get_diesel_value(component_name):
			try:
				return pd.to_numeric(breakdown_df[breakdown_df['Component'] == component_name]['Diesel ($/mi)'].values[0], errors='coerce')
			except:
				return 0
		
		# Extract all values needed
		ev_capital = get_value('Total Capital')
		ev_operating = get_value('Total Operating')
		diesel_capital = get_diesel_value('Total Capital')
		diesel_operating = get_diesel_value('Total Operating')
		
		# Capital components
		ev_chassis = get_value('  Chassis (Glider)')
		ev_motor = get_value('    Motor & Inverter')
		ev_dcdc = get_value('    DC-DC Converter')
		ev_battery_initial = get_value('    Initial Battery')
		ev_battery_replacement = get_value('    Replacement Batteries')
		ev_battery = ev_battery_initial + ev_battery_replacement if ev_battery_initial is not None else 0
		
		diesel_chassis = get_diesel_value('  Chassis (Glider)')
		diesel_engine = get_diesel_value('    Engine')
		diesel_trans = get_diesel_value('    Transmission')
		diesel_aftertx = get_diesel_value('    Aftertreatment')
		diesel_tank = get_diesel_value('    Fuel Tank')
		
		# Operating components
		ev_energy_fuel = get_value('  Energy/Fuel')
		ev_labor = get_value('  Labor')
		ev_maintenance = get_value('  Maintenance & Repair')
		ev_insurance = get_value('  Insurance')
		ev_tolls = get_value('  Tolls')
		ev_permits = get_value('  Permits & Licenses')
		ev_misc = get_value('  Misc')
		
		diesel_fuel = get_diesel_value('  Energy/Fuel')
		diesel_labor = get_diesel_value('  Labor')
		diesel_maintenance = get_diesel_value('  Maintenance & Repair')
		diesel_insurance = get_diesel_value('  Insurance')
		diesel_tolls = get_diesel_value('  Tolls')
		diesel_permits = get_diesel_value('  Permits & Licenses')
		
		# Electricity/Fuel breakdown components
		ev_energy_charge = get_value('    Energy Charge')
		ev_demand_charge = get_value('    Demand Charge')
		ev_charger_capital = get_value('    Charger Capital')
		ev_charger_fixed = get_value('    Charger Fixed Costs')
		diesel_energy_fuel_breakdown = get_diesel_value('    Energy Charge')
		
		# Calculate max y-axis value for consistency
		ev_total = ev_capital + ev_operating
		diesel_total = diesel_capital + diesel_operating
		max_y = max(ev_total, diesel_total) * 1.15
		
		x_pos = [0, 1]
		widths = [0.6, 0.6]
		
		# ========== PLOT 1: Capital costs only ==========
		fig1, ax1 = plt.subplots(figsize=(8, 6))
		
		ax1.bar(x_pos[0], ev_capital, widths[0], color='#1f77b4', alpha=0.8)
		ax1.bar(x_pos[1], diesel_capital, widths[1], color='#1f77b4', alpha=0.8)
		
		ax1.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=24)
		ax1.set_xticks(x_pos)
		ax1.set_xticklabels(['EV', 'Diesel'], fontsize=21)
		ax1.tick_params(axis='y', labelsize=18)
		ax1.set_ylim(0, max_y)
		ax1.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		output_path = presentation_dir / f"{truck_name}_01_capital_only{param_suffix}.png"
		fig1.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close(fig1)
		
		# ========== PLOT 2: Capital + Operating costs ==========
		fig2, ax2 = plt.subplots(figsize=(8, 6))
		
		ax2.bar(x_pos[0], ev_capital, widths[0], color='#1f77b4', alpha=0.8)
		ax2.bar(x_pos[0], ev_operating, widths[0], bottom=ev_capital, color='#ff7f0e', alpha=0.8)
		
		ax2.bar(x_pos[1], diesel_capital, widths[1], color='#1f77b4', alpha=0.8)
		ax2.bar(x_pos[1], diesel_operating, widths[1], bottom=diesel_capital, color='#ff7f0e', alpha=0.8)
		
		ax2.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=24)
		ax2.set_xticks(x_pos)
		ax2.set_xticklabels(['EV', 'Diesel'], fontsize=21)
		ax2.tick_params(axis='y', labelsize=18)
		ax2.set_ylim(0, max_y)
		ax2.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		output_path = presentation_dir / f"{truck_name}_02_capital_and_operating{param_suffix}.png"
		fig2.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close(fig2)
		
		# ========== PLOT 3: Capital cost breakdown ==========
		fig3, ax3 = plt.subplots(figsize=(8, 6))
		
		# Stack EV components
		ev_bottom = ev_chassis
		ax3.bar(x_pos[0], ev_chassis, widths[0], color='#8c564b', alpha=0.8)
		ax3.bar(x_pos[0], ev_motor, widths[0], bottom=ev_bottom, color='#e377c2', alpha=0.8)
		ev_bottom += ev_motor
		ax3.bar(x_pos[0], ev_dcdc, widths[0], bottom=ev_bottom, color='#7f7f7f', alpha=0.8)
		ev_bottom += ev_dcdc
		ax3.bar(x_pos[0], ev_battery, widths[0], bottom=ev_bottom, color='#9467bd', alpha=0.8)
		
		# Stack Diesel components
		diesel_bottom = diesel_chassis
		ax3.bar(x_pos[1], diesel_chassis, widths[1], color='#8c564b', alpha=0.8)
		ax3.bar(x_pos[1], diesel_engine, widths[1], bottom=diesel_bottom, color='#1f77b4', alpha=0.8)
		diesel_bottom += diesel_engine
		ax3.bar(x_pos[1], diesel_trans, widths[1], bottom=diesel_bottom, color='#ff7f0e', alpha=0.8)
		diesel_bottom += diesel_trans
		ax3.bar(x_pos[1], diesel_aftertx, widths[1], bottom=diesel_bottom, color='#2ca02c', alpha=0.8)
		diesel_bottom += diesel_aftertx
		ax3.bar(x_pos[1], diesel_tank, widths[1], bottom=diesel_bottom, color='#d62728', alpha=0.8)
		
		ax3.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=24)
		ax3.set_xticks(x_pos)
		ax3.set_xticklabels(['EV', 'Diesel'], fontsize=21)
		ax3.tick_params(axis='y', labelsize=18)
		# Use a tighter y-axis range for capital breakdown with buffer at top
		max_capital_y = max(ev_capital, diesel_capital) * 1.25
		ax3.set_ylim(0, max_capital_y)
		ax3.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		output_path = presentation_dir / f"{truck_name}_03_capital_breakdown{param_suffix}.png"
		fig3.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close(fig3)
		
		# ========== PLOT 3b: Energy/Fuel Cost Breakdown ==========
		fig3b, ax3b = plt.subplots(figsize=(8, 6))
		
		# EV electricity breakdown (stacked)
		ax3b.bar(x_pos[0], ev_energy_charge, widths[0], color='#8c564b', alpha=0.8)
		ax3b.bar(x_pos[0], ev_demand_charge, widths[0], bottom=ev_energy_charge, color='#e377c2', alpha=0.8)
		ax3b.bar(x_pos[0], ev_charger_capital, widths[0], bottom=ev_energy_charge+ev_demand_charge, color='#7f7f7f', alpha=0.8)
		ax3b.bar(x_pos[0], ev_charger_fixed, widths[0], bottom=ev_energy_charge+ev_demand_charge+ev_charger_capital, color='#bcbd22', alpha=0.8)
		
		# Diesel fuel
		ax3b.bar(x_pos[1], diesel_energy_fuel_breakdown, widths[1], color='#17becf', alpha=0.8)
		
		ax3b.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=24)
		ax3b.set_xticks(x_pos)
		ax3b.set_xticklabels(['EV', 'Diesel'], fontsize=21)
		ax3b.tick_params(axis='y', labelsize=18)
		ax3b.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		output_path = presentation_dir / f"{truck_name}_03b_energy_fuel_breakdown{param_suffix}.png"
		fig3b.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close(fig3b)
		
		# ========== PLOT 4: Other Operating Costs ==========
		fig4, ax4 = plt.subplots(figsize=(8, 6))
		
		other_components = ['  Labor', '  Maintenance & Repair', '  Insurance', '  Tolls', '  Permits & Licenses']
		ev_other_vals = [get_value(c) for c in other_components]
		diesel_other_vals = [get_diesel_value(c) for c in other_components]
		
		x_pos_other = np.arange(len(other_components))
		width = 0.35
		
		ax4.bar(x_pos_other - width/2, ev_other_vals, width, color='#2ca02c', alpha=0.8)
		ax4.bar(x_pos_other + width/2, diesel_other_vals, width, color='#d62728', alpha=0.8)
		
		ax4.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=24)
		ax4.set_xticks(x_pos_other)
		component_labels = [c.strip() for c in other_components]
		ax4.set_xticklabels(component_labels, rotation=45, ha='right', fontsize=16)
		ax4.tick_params(axis='y', labelsize=18)
		ax4.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		output_path = presentation_dir / f"{truck_name}_04_other_operating{param_suffix}.png"
		fig4.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close(fig4)
		
		# ========== PLOT 5: Operating costs stacked with only energy/fuel ==========
		fig5, ax5 = plt.subplots(figsize=(8, 6))
		
		# Calculate max operating cost for consistent y-axis
		max_operating_y = max(ev_operating, diesel_operating) * 1.15
		
		ax5.bar(x_pos[0], ev_energy_fuel, widths[0], color='#1f77b4', alpha=0.8)
		ax5.bar(x_pos[1], diesel_fuel, widths[1], color='#1f77b4', alpha=0.8)
		
		ax5.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=24)
		ax5.set_xticks(x_pos)
		ax5.set_xticklabels(['EV', 'Diesel'], fontsize=21)
		ax5.tick_params(axis='y', labelsize=18)
		ax5.set_ylim(0, max_operating_y)
		ax5.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		output_path = presentation_dir / f"{truck_name}_05_operating_energy_only{param_suffix}.png"
		fig5.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close(fig5)
		
		# ========== PLOT 6: Operating costs stacked with all components ==========
		fig6, ax6 = plt.subplots(figsize=(8, 6))
		
		# Aggregate other costs (Insurance + Tolls + Permits)
		ev_other_costs = ev_insurance + ev_tolls + ev_permits
		diesel_other_costs = diesel_insurance + diesel_tolls + diesel_permits
		
		# Stack EV operating components (Energy/Fuel + Labor + Maintenance + Other)
		ev_op_bottom = ev_energy_fuel
		ax6.bar(x_pos[0], ev_energy_fuel, widths[0], color='#1f77b4', alpha=0.8)
		ax6.bar(x_pos[0], ev_labor, widths[0], bottom=ev_op_bottom, color='#ff7f0e', alpha=0.8)
		ev_op_bottom += ev_labor
		ax6.bar(x_pos[0], ev_maintenance, widths[0], bottom=ev_op_bottom, color='#2ca02c', alpha=0.8)
		ev_op_bottom += ev_maintenance
		ax6.bar(x_pos[0], ev_other_costs, widths[0], bottom=ev_op_bottom, color='#d62728', alpha=0.8)
		
		# Stack Diesel operating components (Fuel + Labor + Maintenance + Other)
		diesel_op_bottom = diesel_fuel
		ax6.bar(x_pos[1], diesel_fuel, widths[1], color='#1f77b4', alpha=0.8)
		ax6.bar(x_pos[1], diesel_labor, widths[1], bottom=diesel_op_bottom, color='#ff7f0e', alpha=0.8)
		diesel_op_bottom += diesel_labor
		ax6.bar(x_pos[1], diesel_maintenance, widths[1], bottom=diesel_op_bottom, color='#2ca02c', alpha=0.8)
		diesel_op_bottom += diesel_maintenance
		ax6.bar(x_pos[1], diesel_other_costs, widths[1], bottom=diesel_op_bottom, color='#d62728', alpha=0.8)
		
		ax6.set_ylabel('Cost ($/mi)', fontweight='bold', fontsize=24)
		ax6.set_xticks(x_pos)
		ax6.set_xticklabels(['EV', 'Diesel'], fontsize=21)
		ax6.tick_params(axis='y', labelsize=18)
		ax6.set_ylim(0, max_operating_y)
		ax6.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		output_path = presentation_dir / f"{truck_name}_06_operating_all_components{param_suffix}.png"
		fig6.savefig(output_path, dpi=300, bbox_inches="tight")
		plt.close(fig6)
		
		print(f"  ✓ Generated 7 presentation visuals for {truck_name}")
	
	print(f"\n✓ Presentation visuals saved to: {presentation_dir.absolute()}")


if __name__ == "__main__":
	main()
