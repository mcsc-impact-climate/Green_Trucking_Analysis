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
from costing_and_emissions_tools_messy import get_payload_distribution, get_payload_penalty, evaluate_emissions, evaluate_costs, evaluate_costs_diesel

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


def sensitivity_analysis(datasets, plots_dir, optimized_params, battery_caps, m_truck_max_kg, param_suffix):
	"""
	Perform sensitivity analysis of EV cost premium relative to diesel across parameter ranges.
	
	Parameters:
	-----------
	datasets : list of dict
		List of dataset configurations
	plots_dir : Path
		Directory to save results
	optimized_params : dict or None
		Dictionary of optimized parameters per truck
	battery_caps : pd.DataFrame
		Battery capacity data
	m_truck_max_kg : float
		Maximum truck mass in kg
	param_suffix : str
		Suffix for output files
	"""
	import time
	
	# Load parameter ranges
	param_ranges_df = pd.read_csv(BASE_DIR / "data_messy" / "parameter_sensitivity_ranges.csv").set_index("Parameter")
	param_ranges = param_ranges_df.to_dict("index")
	
	# Load US average values
	usa_data = pd.read_csv(str(BASE_DIR / "data_messy" / "energy_costs_emissions_usa.csv")).set_index("Parameter")
	electricity_cost_current = usa_data.loc["Commercial electricity ($/kWh)", "Value"]
	demand_charge_current = usa_data.loc["Demand charge ($/kW)", "Value"]
	diesel_cost_current = usa_data.loc["Diesel cost ($/gal)", "Value"]
	
	# Load charging power data
	max_charging_power_data = pd.read_csv(str(BASE_DIR / "messy_middle_results" / "max_charging_powers.csv"), index_col="truck_name")
	
	sensitivity_results = {}
	tornado_data = {}
	
	print("\n" + "="*80)
	print("SENSITIVITY ANALYSIS: EV TCO Premium vs Diesel")
	print("="*80)
	
	# For each truck
	for dataset in datasets:
		truck_name = dataset["name"]
		print(f"\n{'='*80}")
		print(f"Analyzing: {truck_name}")
		print(f"{'='*80}")
		
		# Load EV results for current baseline
		ev_csv = plots_dir / f"{truck_name}_all_drivecycles_results{param_suffix}.csv"
		diesel_csv = plots_dir / f"{truck_name}_all_drivecycles_diesel_results{param_suffix}.csv"
		
		if not ev_csv.exists() or not diesel_csv.exists():
			print(f"Warning: Missing baseline results for {truck_name}, skipping sensitivity analysis.")
			continue
		
		ev_df = pd.read_csv(ev_csv)
		diesel_df = pd.read_csv(diesel_csv)
		
		# Load truck parameters for current value
		parameters = data_collection_tools_messy.read_parameters(
			truck_params=dataset["truck_params"],
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="EV",
		)
		parameters = apply_optimized_parameters(parameters, optimized_params, truck_name)
		
		battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=parameters.battery_chemistry)
		e_density = battery_params_dict['Energy density (kWh/ton)']
		
		e_bat_current = battery_caps.loc['Mean', dataset["battery_col"]]
		m_bat_kg_current = e_bat_current / e_density * KG_PER_TON
		m_bat_lb_current = m_bat_kg_current / KG_PER_LB
		
		m_truck_no_bat_kg = parameters.m_truck_no_bat
		m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB
		
		vmt_current = parameters.VMT
		max_charging_power = max_charging_power_data.loc[truck_name, "99th_percentile_charging_power_kw"]
		
		# Baseline premium
		merged = ev_df.merge(diesel_df, on=["source_truck", "drivecycle_number"], how="inner", suffixes=("_ev", "_diesel"))
		baseline_premium_pct = (
			((merged["cost_TCO ($/mi)"] - merged["diesel_cost_TCO ($/mi)"]) / merged["diesel_cost_TCO ($/mi)"]) * 100
		).mean()
		print(f"\nBaseline EV TCO Premium: {baseline_premium_pct:.2f}%")
		
		# Dictionary to store sensitivity results
		truck_sensitivity = {}
		tornado_impacts = {}
		
		# Define sensitivity parameters and their scanning logic
		sensitivity_params = {
			"VMT (miles/year)": {
				"current": vmt_current,
				"range": param_ranges["VMT (miles/year)"],
				"n_points": 5,
			},
			"Electricity Cost ($/kWh)": {
				"current": electricity_cost_current,
				"range": param_ranges["Electricity Cost ($/kWh)"],
				"n_points": 5,
			},
			"Demand Charge ($/kW)": {
				"current": demand_charge_current,
				"range": param_ranges["Demand Charge ($/kW)"],
				"n_points": 5,
			},
			"Diesel Cost ($/gal)": {
				"current": diesel_cost_current,
				"range": param_ranges["Diesel Cost ($/gal)"],
				"n_points": 5,
			},
			"Battery Capacity (kWh)": {
				"current": e_bat_current,
				"range": param_ranges["Battery Capacity (kWh)"],
				"n_points": 5,
			},
			"Battery Cost ($/kWh)": {
				"current": None,  # Will be extracted from costing function
				"range": param_ranges["Battery Cost ($/kWh)"],
				"n_points": 5,
			},
			"Payload (lb)": {
				"current": 0,  # We'll vary average payload
				"range": param_ranges["Payload (lb)"],
				"n_points": 5,
			},
		}
		
		# Scan each parameter
		total_params = len(sensitivity_params)
		for param_idx, (param_name, param_config) in enumerate(sensitivity_params.items(), 1):
			start_time = time.time()
			print(f"\n[{param_idx}/{total_params}] Scanning: {param_name}")
			
			range_min = param_config["range"]["Min"]
			range_max = param_config["range"]["Max"]
			n_points = param_config["n_points"]
			current_val = param_config["current"]
			
			# Generate equally spaced values
			scan_values = np.linspace(range_min, range_max, n_points)
			print(f"  Range: {range_min} to {range_max}")
			print(f"  Current value: {current_val:.4f}")
			print(f"  Scan points: {', '.join([f'{v:.4f}' for v in scan_values])}")
			
			param_results = []
			premiums = []
			
			for point_idx, scan_value in enumerate(scan_values, 1):
				# Recalculate TCO with varied parameter
				try:
					if param_name == "VMT (miles/year)":
						# Vary VMT
						ev_costs = []
						diesel_costs = []
						for _, row in merged.iterrows():
							source_truck = row["source_truck"]
							drivecycle_num = int(row["drivecycle_number"])
							drivecycle_path = resolve_drivecycle_path(source_truck, drivecycle_num)
							drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
							m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
							m_gvwr_lb = m_gvwr_kg / KG_PER_LB
							m_payload_lb = m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_current
							
							ev_cost = evaluate_costs(
								mileage=row["fuel_economy_kWh_per_mile_ev"],
								payload_lb=m_payload_lb,
								electricity_charge=electricity_cost_current,
								demand_charge=demand_charge_current,
								average_VMT=scan_value,
								charging_power=max_charging_power,
								e_bat=e_bat_current,
								battery_chemistry=parameters.battery_chemistry,
								truck_name=dataset["truck_params"],
							)
							ev_costs.append(ev_cost["TCO ($/mi)"])
							
							diesel_cost = evaluate_costs_diesel(
								mileage_mpg=row["fuel_economy_mpg"],
								payload_lb=m_payload_lb,
								diesel_price=diesel_cost_current,
								average_VMT=scan_value,
								truck_type=f"{dataset['truck_params']}_diesel",
							)
							diesel_costs.append(diesel_cost["TCO ($/mi)"])
						
						avg_premium = ((np.mean(ev_costs) - np.mean(diesel_costs)) / np.mean(diesel_costs)) * 100
						
					elif param_name == "Electricity Cost ($/kWh)":
						ev_costs = []
						diesel_costs = []
						for _, row in merged.iterrows():
							source_truck = row["source_truck"]
							drivecycle_num = int(row["drivecycle_number"])
							drivecycle_path = resolve_drivecycle_path(source_truck, drivecycle_num)
							drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
							m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
							m_gvwr_lb = m_gvwr_kg / KG_PER_LB
							m_payload_lb = m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_current
							
							ev_cost = evaluate_costs(
								mileage=row["fuel_economy_kWh_per_mile_ev"],
								payload_lb=m_payload_lb,
								electricity_charge=scan_value,
								demand_charge=demand_charge_current,
								average_VMT=vmt_current,
								charging_power=max_charging_power,
								e_bat=e_bat_current,
								battery_chemistry=parameters.battery_chemistry,
								truck_name=dataset["truck_params"],
							)
							ev_costs.append(ev_cost["TCO ($/mi)"])
							diesel_costs.append(row["diesel_cost_TCO ($/mi)"])
						
						avg_premium = ((np.mean(ev_costs) - np.mean(diesel_costs)) / np.mean(diesel_costs)) * 100
						
					elif param_name == "Demand Charge ($/kW)":
						ev_costs = []
						diesel_costs = []
						for _, row in merged.iterrows():
							source_truck = row["source_truck"]
							drivecycle_num = int(row["drivecycle_number"])
							drivecycle_path = resolve_drivecycle_path(source_truck, drivecycle_num)
							drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
							m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
							m_gvwr_lb = m_gvwr_kg / KG_PER_LB
							m_payload_lb = m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_current
							
							ev_cost = evaluate_costs(
								mileage=row["fuel_economy_kWh_per_mile_ev"],
								payload_lb=m_payload_lb,
								electricity_charge=electricity_cost_current,
								demand_charge=scan_value,
								average_VMT=vmt_current,
								charging_power=max_charging_power,
								e_bat=e_bat_current,
								battery_chemistry=parameters.battery_chemistry,
								truck_name=dataset["truck_params"],
							)
							ev_costs.append(ev_cost["TCO ($/mi)"])
							diesel_costs.append(row["diesel_cost_TCO ($/mi)"])
						
						avg_premium = ((np.mean(ev_costs) - np.mean(diesel_costs)) / np.mean(diesel_costs)) * 100
						
					elif param_name == "Diesel Cost ($/gal)":
						ev_costs = []
						diesel_costs = []
						for _, row in merged.iterrows():
							source_truck = row["source_truck"]
							drivecycle_num = int(row["drivecycle_number"])
							drivecycle_path = resolve_drivecycle_path(source_truck, drivecycle_num)
							drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
							m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
							m_gvwr_lb = m_gvwr_kg / KG_PER_LB
							m_payload_lb = m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_current
							
							ev_costs.append(row["cost_TCO ($/mi)"])
							
							diesel_cost = evaluate_costs_diesel(
								mileage_mpg=row["fuel_economy_mpg"],
								payload_lb=m_payload_lb,
								diesel_price=scan_value,
								average_VMT=vmt_current,
								truck_type=f"{dataset['truck_params']}_diesel",
							)
							diesel_costs.append(diesel_cost["TCO ($/mi)"])
						
						avg_premium = ((np.mean(ev_costs) - np.mean(diesel_costs)) / np.mean(diesel_costs)) * 100
						
					elif param_name == "Battery Capacity (kWh)":
						# For battery capacity, we would need to recalculate efficiency and fuel economy
						# For now, use linear approximation (cost scales roughly linearly with capacity for TCO)
						m_bat_lb_scan = (scan_value / e_density * KG_PER_TON) / KG_PER_LB
						payload_diff = m_bat_lb_current - m_bat_lb_scan
						
						ev_costs = []
						diesel_costs = []
						for _, row in merged.iterrows():
							source_truck = row["source_truck"]
							drivecycle_num = int(row["drivecycle_number"])
							drivecycle_path = resolve_drivecycle_path(source_truck, drivecycle_num)
							drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
							m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
							m_gvwr_lb = m_gvwr_kg / KG_PER_LB
							m_payload_lb = m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_scan
							
							ev_cost = evaluate_costs(
								mileage=row["fuel_economy_kWh_per_mile_ev"],
								payload_lb=m_payload_lb,
								electricity_charge=electricity_cost_current,
								demand_charge=demand_charge_current,
								average_VMT=vmt_current,
								charging_power=max_charging_power,
								e_bat=scan_value,
								battery_chemistry=parameters.battery_chemistry,
								truck_name=dataset["truck_params"],
							)
							ev_costs.append(ev_cost["TCO ($/mi)"])
							diesel_costs.append(row["diesel_cost_TCO ($/mi)"])
						
						avg_premium = ((np.mean(ev_costs) - np.mean(diesel_costs)) / np.mean(diesel_costs)) * 100
						
					elif param_name == "Battery Cost ($/kWh)":
						# Battery cost: We approximate this by scaling the capital cost component
						# This is a proxy - actual implementation would require modifying evaluate_costs
						avg_premium = baseline_premium_pct  # Placeholder for now
						
					elif param_name == "Payload (lb)":
						# Payload variation
						ev_costs = []
						diesel_costs = []
						for _, row in merged.iterrows():
							source_truck = row["source_truck"]
							drivecycle_num = int(row["drivecycle_number"])
							drivecycle_path = resolve_drivecycle_path(source_truck, drivecycle_num)
							drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
							m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
							m_gvwr_lb = m_gvwr_kg / KG_PER_LB
							# Use baseline payload but scale to vary it
							m_baseline_payload = m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_current
							m_payload_lb = m_baseline_payload * (scan_value / 35000) if scan_value > 0 else 1000
							
							ev_cost = evaluate_costs(
								mileage=row["fuel_economy_kWh_per_mile_ev"],
								payload_lb=m_payload_lb,
								electricity_charge=electricity_cost_current,
								demand_charge=demand_charge_current,
								average_VMT=vmt_current,
								charging_power=max_charging_power,
								e_bat=e_bat_current,
								battery_chemistry=parameters.battery_chemistry,
								truck_name=dataset["truck_params"],
							)
							ev_costs.append(ev_cost["TCO ($/mi)"])
							
							diesel_cost = evaluate_costs_diesel(
								mileage_mpg=row["fuel_economy_mpg"],
								payload_lb=m_payload_lb,
								diesel_price=diesel_cost_current,
								average_VMT=vmt_current,
								truck_type=f"{dataset['truck_params']}_diesel",
							)
							diesel_costs.append(diesel_cost["TCO ($/mi)"])
						
						avg_premium = ((np.mean(ev_costs) - np.mean(diesel_costs)) / np.mean(diesel_costs)) * 100
					
					premiums.append(avg_premium)
					param_results.append({
						"parameter_value": scan_value,
						"tco_premium_%": avg_premium,
					})
					
					elapsed = time.time() - start_time
					avg_time_per_point = elapsed / point_idx
					eta = avg_time_per_point * (n_points - point_idx)
					print(f"    [{point_idx}/{n_points}] Value: {scan_value:.4f} → Premium: {avg_premium:.2f}% (Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)")
					
				except Exception as e:
					print(f"    [{point_idx}/{n_points}] Value: {scan_value:.4f} → ERROR: {e}")
					premiums.append(np.nan)
					param_results.append({
						"parameter_value": scan_value,
						"tco_premium_%": np.nan,
					})
			
			# Calculate impact (max - min)
			valid_premiums = [p for p in premiums if not np.isnan(p)]
			if valid_premiums:
				impact = max(valid_premiums) - min(valid_premiums)
				tornado_impacts[param_name] = impact
			else:
				tornado_impacts[param_name] = 0
			
			# Save parameter results
			truck_sensitivity[param_name] = pd.DataFrame(param_results)
			
			elapsed_total = time.time() - start_time
			print(f"  ✓ Completed in {elapsed_total:.1f}s, Impact: {tornado_impacts[param_name]:.2f}%")
		
		sensitivity_results[truck_name] = truck_sensitivity
		tornado_data[truck_name] = tornado_impacts
		
		# Save sensitivity results to CSVs
		sensitivity_dir = plots_dir / f"sensitivity_analysis{param_suffix}"
		sensitivity_dir.mkdir(parents=True, exist_ok=True)
		
		for param_name, results_df in truck_sensitivity.items():
			param_filename = param_name.replace(" ", "_").replace("(", "").replace(")", "").replace("$", "").replace("/", "_")
			csv_path = sensitivity_dir / f"{truck_name}_{param_filename}_sensitivity.csv"
			results_df.to_csv(csv_path, index=False)
		
		# Create sensitivity plots for this truck
		fig, axes = plt.subplots(2, 4, figsize=(16, 10))
		fig.suptitle(f"{truck_name} - TCO Premium Sensitivity Analysis", fontsize=16, fontweight='bold')
		axes_flat = axes.flatten()
		
		for ax_idx, (param_name, results_df) in enumerate(truck_sensitivity.items()):
			if ax_idx >= len(axes_flat):
				break
			
			ax = axes_flat[ax_idx]
			x_values = results_df["parameter_value"].values
			y_values = results_df["tco_premium_%"].values
			
			# Plot line
			ax.plot(x_values, y_values, 'b-o', linewidth=2, markersize=6, label="Sensitivity scan")
			
			# Mark current value
			current_val = sensitivity_params[param_name]["current"]
			if current_val is not None:
				# Find closest point to current value
				closest_idx = np.argmin(np.abs(x_values - current_val))
				ax.axvline(x=current_val, color='red', linestyle='--', linewidth=2, label="Current value")
				ax.plot(x_values[closest_idx], y_values[closest_idx], 'ro', markersize=8)
			
			ax.set_xlabel(param_name)
			ax.set_ylabel("EV TCO Premium (%)")
			ax.grid(True, alpha=0.3)
			ax.legend(fontsize=8)
		
		# Hide unused subplots
		for ax_idx in range(len(truck_sensitivity), len(axes_flat)):
			axes_flat[ax_idx].set_visible(False)
		
		plt.tight_layout()
		plot_path = sensitivity_dir / f"{truck_name}_sensitivity_scans{param_suffix}.png"
		plt.savefig(plot_path, dpi=300, bbox_inches='tight')
		print(f"\nSaved sensitivity scan plots: {plot_path}")
		plt.close(fig)
	
	# Create tornado plot across all trucks
	if tornado_data:
		fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 8))
		if len(datasets) == 1:
			axes = [axes]
		
		for ax_idx, dataset in enumerate(datasets):
			truck_name = dataset["name"]
			if truck_name not in tornado_data:
				continue
			
			impacts = tornado_data[truck_name]
			sorted_params = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
			param_names = [p[0] for p in sorted_params]
			impact_values = [p[1] for p in sorted_params]
			
			ax = axes[ax_idx]
			y_pos = np.arange(len(param_names))
			
			# Color bars by magnitude
			colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(param_names)))
			ax.barh(y_pos, impact_values, color=colors)
			
			ax.set_yticks(y_pos)
			ax.set_yticklabels(param_names, fontsize=10)
			ax.set_xlabel("TCO Premium Impact (%)", fontsize=11)
			ax.set_title(f"{truck_name}", fontsize=12, fontweight='bold')
			ax.grid(True, alpha=0.3, axis='x')
			
			# Add value labels on bars
			for i, (param, impact) in enumerate(zip(param_names, impact_values)):
				ax.text(impact, i, f" {impact:.1f}%", va='center', fontsize=9)
		
		plt.suptitle("Tornado Plot: Parameter Impact on EV TCO Premium", fontsize=14, fontweight='bold')
		plt.tight_layout()
		plot_path = plots_dir / f"tornado_plot{param_suffix}.png"
		plt.savefig(plot_path, dpi=300, bbox_inches='tight')
		print(f"\nSaved tornado plot: {plot_path}")
		plt.close(fig)
		
		# Save tornado data to CSV
		tornado_csv_path = plots_dir / f"tornado_data{param_suffix}.csv"
		tornado_df = pd.DataFrame(tornado_data).T
		tornado_df.to_csv(tornado_csv_path)
		print(f"Saved tornado data: {tornado_csv_path}")


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
	
	# process_drivecycles_and_save_results(
	# 	datasets=datasets,
	# 	regenerate_model_results=regenerate_model_results,
	# 	model_results_dir=model_results_dir,
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

	# Evaluate diesel model on all drivecycles (using EV-optimized cd/cr), add diesel costs, and plot EV TCO premium
	evaluate_diesel_on_all_drivecycles(
		datasets=datasets,
		optimized_params=optimized_params,
		results_dir=plots_dir,
		param_suffix=param_suffix,
	)
	add_costs_diesel_all_drivecycles(
		datasets=datasets,
		plots_dir=plots_dir,
		param_suffix=param_suffix,
		average_vmt=average_vmt,
	)
	plot_tco_premium_all_drivecycles(
		datasets=datasets,
		plots_dir=plots_dir,
		param_suffix=param_suffix,
	)
	
	# Perform sensitivity analysis
	print("\n" + "="*80)
	print("STARTING SENSITIVITY ANALYSIS")
	print("="*80)
	sensitivity_analysis(
		datasets=datasets,
		plots_dir=plots_dir,
		optimized_params=optimized_params,
		battery_caps=battery_caps,
		m_truck_max_kg=m_truck_max_kg,
		param_suffix=param_suffix,
	)
	print("\n" + "="*80)
	print("SENSITIVITY ANALYSIS COMPLETE")
	print("="*80)

if __name__ == "__main__":
	main()
