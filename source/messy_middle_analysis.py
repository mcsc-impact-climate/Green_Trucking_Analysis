"""
Date: 260201
Author: danikae
Purpose: Generate drivecycle comparison plots with option for original or optimized parameters
"""

import pandas as pd
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import truck_model_tools_messy as truck_model_tools_messy
import retired.costing_tools_orig as costing_tools
import retired.emissions_tools_orig as emissions_tools
import data_collection_tools_messy
from costing_and_emissions_tools import get_payload_distribution, get_payload_penalty

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

battery_caps = pd.read_csv("messy_middle_results/battery_capacities_linear_summary.csv").set_index('Value')


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
		opt_df = pd.read_csv('parameter_optimization_results.csv')
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
			f"  NACFE: {nacfe_fuel_consumption:.3f}\n"
			f"  Model: {model_fuel_consumption:.3f}\n"
			f"  % diff: {fuel_pct_diff:+.2f}%\n\n"
			f"DoD (%)\n"
			f"  NACFE: {nacfe_DoD_percent:.3f}\n"
			f"  Model: {model_DoD_percent:.3f}\n"
			f"  % diff: {DoD_pct_diff:+.2f}%"
		),
		fontsize=11,
		va="top",
	)

	return fig


######################################### Obtain model parameters #########################################
average_vmt = 50000

m_truck_max_lb=82000
m_truck_max_kg = m_truck_max_lb * KG_PER_LB

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate drivecycle comparison plots')
parser.add_argument('--optimized', action='store_true', 
					help='Use optimized parameters from parameter_optimization_results.csv')
args = parser.parse_args()

# Load optimized parameters if requested
optimized_params = load_optimized_parameters(use_optimized=args.optimized)

# Determine output directory and suffix
param_suffix = "_optimized" if args.optimized else "_original"
plots_dir = Path("plots_messy")
plots_dir.mkdir(parents=True, exist_ok=True)

print(f"\nRunning analysis with {'optimized' if args.optimized else 'original'} parameters")

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

for dataset in datasets:
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

	drivecycle_files = sorted(Path("messy_middle_results").glob(dataset["drivecycle_glob"]))

	for drivecycle_path in drivecycle_files:
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

		print(drivecycle_path.name, e_bat, fuel_consumption, DoD)

		parts = drivecycle_path.stem.split("_")
		driving_event = int(parts[-2]) if parts[-1] == "detailed" else int(parts[-1])

		fig = plot_drivecycle_comparison(
			drivecycle_data,
			df,
			fuel_consumption,
			DoD,
			dataset["summary_path"],
			driving_event,
		)

		fig.savefig(
			plots_dir / f"{dataset['name']}_drivecycle_{driving_event}_comparison{param_suffix}.png",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close(fig)


