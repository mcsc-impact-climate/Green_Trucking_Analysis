"""
Standalone sensitivity analysis for EV TCO premium
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

# Add source to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import data_collection_tools_messy
import truck_model_tools_messy
import truck_model_tools_diesel_messy
import costing_tools
from costing_and_emissions_tools_messy import (
	evaluate_costs,
	evaluate_costs_diesel,
	get_payload_distribution,
	get_payload_penalty,
	get_vehicle_model_results,
	get_electricity_cost_by_year,
	calculate_replacements,
)

BASE_DIR = Path(__file__).resolve().parent.parent
KG_PER_TON = 1000
KG_PER_LB = 0.453592

def apply_optimized_parameters(parameters, optimized_params, truck_name):
	"""Apply optimized parameters to a parameters object if available."""
	if optimized_params is None or truck_name not in optimized_params:
		return parameters
	
	opt = optimized_params[truck_name]
	eta_opt = np.sqrt(opt['eta_combined'])
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
	
	if battery_chemistry is not None:
		optimized.battery_chemistry = battery_chemistry
	
	return optimized


def load_optimized_parameters(use_optimized=False):
	"""Load parameter optimization results if requested."""
	if not use_optimized:
		return None
	
	try:
		opt_df = pd.read_csv(str(BASE_DIR / 'parameter_optimization_results.csv'))
		opt_df = opt_df.drop_duplicates(subset=['Truck'])
		
		optimized_params = {}
		for idx, row in opt_df.iterrows():
			truck_name = row['Truck']
			optimized_params[truck_name] = {
				'cd': row['cd_optimal'],
				'cr': row['cr_optimal'],
				'eta_combined': row['eta_optimal'],
			}
		
		print(f"Loaded optimized parameters for {len(optimized_params)} trucks")
		return optimized_params
	except FileNotFoundError:
		print("Warning: parameter_optimization_results.csv not found. Using original parameters.")
		return None


def resolve_drivecycle_path(source_truck, driving_event):
	"""Resolve a drivecycle file path with fallback for truncated truck names."""
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


def evaluate_costs_with_battery_cost(
		mileage,
		payload_lb,
		electricity_charge,
		demand_charge,
		average_VMT,
		charging_power,
		e_bat,
		battery_chemistry='NMC',
		truck_name=None,
		battery_unit_cost_override=None,
	):
	"""Evaluate EV costs with optional battery unit cost override."""
	parameters, vehicle_model_results_dict = get_vehicle_model_results(
		mileage,
		payload_lb,
		average_VMT,
		truck_name=truck_name,
		battery_chemistry=battery_chemistry,
		e_bat=e_bat,
	)

	truck_cost_data = data_collection_tools_messy.read_truck_cost_data(truck_type='EV')
	if battery_unit_cost_override is not None:
		truck_cost_data['Battery Unit Cost ($/kWh)'] = float(battery_unit_cost_override)

	battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=battery_chemistry)
	battery_params_dict['Replacements'] = calculate_replacements(
		parameters.VMT['VMT (miles)'],
		vehicle_model_results_dict['Fuel economy (kWh/mi)'],
		e_bat=e_bat,
	)

	electricity_cost_df = get_electricity_cost_by_year(
		parameters,
		vehicle_model_results_dict['Fuel economy (kWh/mi)'],
		demand_charge,
		electricity_charge,
		charging_power,
	)

	return costing_tools.cost(parameters).get_TCO(
		vehicle_model_results_dict,
		truck_cost_data['Capital Costs'],
		truck_cost_data['Battery Unit Cost ($/kWh)'],
		truck_cost_data['Operating Costs'],
		electricity_cost_df['Total'],
		battery_params_dict['Replacements'],
		vehicle_purchase_price=None,
		e_bat=e_bat,
	)


def sensitivity_analysis_simple(
		datasets,
		plots_dir,
		optimized_params,
		battery_caps,
		param_suffix="_optimized",
		average_vmt=100000,
	):
	"""Sensitivity analysis: compute TCO premium from scratch for each parameter variation."""
	
	# Load parameter ranges
	param_ranges_df = pd.read_csv(BASE_DIR / "data_messy" / "parameter_sensitivity_ranges.csv").set_index("Parameter")
	param_ranges = param_ranges_df.to_dict("index")
	
	# Load US average values
	usa_data = pd.read_csv(str(BASE_DIR / "data_messy" / "energy_costs_emissions_usa.csv")).set_index("Parameter")
	electricity_cost_baseline = usa_data.loc["Commercial electricity ($/kWh)", "Value"]
	demand_charge_baseline = usa_data.loc["Demand charge ($/kW)", "Value"]
	diesel_cost_baseline = usa_data.loc["Diesel cost ($/gal)", "Value"]
	
	print(f"\nBaseline parameters:")
	print(f"  Electricity cost: ${electricity_cost_baseline:.4f}/kWh")
	print(f"  Demand charge: ${demand_charge_baseline:.2f}/kW")
	print(f"  Diesel cost: ${diesel_cost_baseline:.2f}/gal")
	sys.stdout.flush()
	
	# Load charging power data
	max_charging_power_data = pd.read_csv(str(BASE_DIR / "messy_middle_results" / "max_charging_powers.csv"), index_col="truck_name")
	
	# Load battery caps
	battery_caps = pd.read_csv(BASE_DIR / "messy_middle_results" / "battery_capacities_linear_summary.csv").set_index('Value')

	tornado_data = {}
	sensitivity_dir = plots_dir / f"sensitivity_analysis{param_suffix}"
	sensitivity_dir.mkdir(parents=True, exist_ok=True)
	
	print("\n" + "="*80)
	print("SENSITIVITY ANALYSIS: EV TCO Premium vs Diesel (From Scratch)")
	print("="*80)
	sys.stdout.flush()
	
	for dataset in datasets:
		truck_name = dataset["name"]
		print(f"\n{'='*80}")
		print(f"Analyzing: {truck_name}")
		print(f"{'='*80}")
		sys.stdout.flush()
		
		# Load truck parameters
		parameters = data_collection_tools_messy.read_parameters(
			truck_params=dataset["truck_params"],
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="EV",
		)
		parameters = apply_optimized_parameters(parameters, optimized_params, truck_name)
		
		battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=parameters.battery_chemistry)
		e_density = battery_params_dict['Energy density (kWh/ton)']
		
		e_bat_baseline = battery_caps.loc['Mean', dataset["battery_col"]]
		m_bat_kg_baseline = e_bat_baseline / e_density * KG_PER_TON
		m_bat_lb_baseline = m_bat_kg_baseline / KG_PER_LB
		
		m_truck_no_bat_kg = parameters.m_truck_no_bat
		m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB
		
		# Use provided average VMT for baseline (requested 100,000)
		vmt_baseline = float(average_vmt)
		
		max_charging_power = max_charging_power_data.loc[truck_name, "99th_percentile_charging_power_kw"]
		
		diesel_truck_params = f"{dataset['truck_params']}_diesel"
		
		# Load diesel parameters once (baseline) - IMPORTANT: do this outside the loop
		diesel_parameters = data_collection_tools_messy.read_parameters(
			truck_params=diesel_truck_params,
			vmt_params="daycab_vmt_vius_2021",
			run="messy_middle",
			truck_type="diesel",
		)
		diesel_parameters.cd = parameters.cd
		diesel_parameters.cr = parameters.cr
		
		print(f"  EV battery: {e_bat_baseline:.1f} kWh")
		print(f"  VMT baseline: {vmt_baseline:.0f} miles/year")
		print(f"  Max charging power: {max_charging_power:.1f} kW")
		sys.stdout.flush()
		
		# Load all-drivecycles results to use correct fuel economy baselines
		ev_csv = plots_dir / f"{truck_name}_all_drivecycles_results{param_suffix}.csv"
		diesel_csv = plots_dir / f"{truck_name}_all_drivecycles_diesel_results{param_suffix}.csv"
		if not ev_csv.exists() or not diesel_csv.exists():
			print(f"Warning: Missing all-drivecycles results for {truck_name}, skipping.")
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

		# Baseline battery unit cost and payload
		truck_cost_data = data_collection_tools_messy.read_truck_cost_data(
			truck_type='EV',
			chemistry=parameters.battery_chemistry,
		)
		battery_cost_baseline = float(truck_cost_data['Battery Unit Cost ($/kWh)'])

		payload_values = []
		for _, row in merged.iterrows():
			source_truck = row["source_truck"]
			driving_event = int(row["drivecycle_number"])
			drivecycle_path = resolve_drivecycle_path(source_truck, driving_event)
			drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
			m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
			m_gvwr_lb = m_gvwr_kg / KG_PER_LB
			payload_values.append(m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_baseline)

		payload_baseline = float(np.nanmean(payload_values)) if payload_values else 0.0

		print(f"  Battery unit cost baseline: ${battery_cost_baseline:.2f}/kWh")
		print(f"  Payload baseline: {payload_baseline:.0f} lb")
		sys.stdout.flush()

		# Define parameters to scan
		params_to_scan = [
			("Electricity Cost ($/kWh)", electricity_cost_baseline, param_ranges["Electricity Cost ($/kWh)"]),
			("Demand Charge ($/kW)", demand_charge_baseline, param_ranges["Demand Charge ($/kW)"]),
			("Diesel Cost ($/gal)", diesel_cost_baseline, param_ranges["Diesel Cost ($/gal)"]),
			("VMT (miles/year)", vmt_baseline, param_ranges["VMT (miles/year)"]),
			("Battery Capacity (kWh)", e_bat_baseline, param_ranges["Battery Capacity (kWh)"]),
			("Battery Cost ($/kWh)", battery_cost_baseline, param_ranges["Battery Cost ($/kWh)"]),
			("Payload (lb)", payload_baseline, param_ranges["Payload (lb)"]),
		]
		
		tornado_impacts = {}
		sensitivity_results = {}
		
		# Baseline premium (using current costs in merged results)
		print("\nCalculating baseline premium (from merged results)...")
		baseline_premium = (
			(merged["cost_TCO ($/mi)"] - merged["diesel_cost_TCO ($/mi)"]) / merged["diesel_cost_TCO ($/mi)"] * 100
		).replace([np.inf, -np.inf], np.nan).dropna().mean()
		print(f"  Baseline EV TCO Premium: {baseline_premium:.2f}%")
		sys.stdout.flush()
		
		# Scan each parameter
		total_params = len(params_to_scan)
		for param_idx, (param_name, current_val, param_range) in enumerate(params_to_scan, 1):
			start_time = time.time()
			print(f"\n[{param_idx}/{total_params}] Scanning: {param_name}")
			
			range_min = param_range["Min"]
			range_max = param_range["Max"]
			n_points = 2  # Min and max only
			
			# Generate equally spaced values
			scan_values = np.linspace(range_min, range_max, n_points)
			print(f"  Range: {range_min} to {range_max}")
			print(f"  Current value: {current_val:.6f}")
			print(f"  Scan points: {', '.join([f'{v:.4f}' for v in scan_values])}")
			sys.stdout.flush()
			
			param_results = []
			premiums = []
			
			for point_idx, scan_value in enumerate(scan_values, 1):
				try:
					# Calculate average premium with this parameter value
					avg_premium = 0
					count = 0
					
					# Use all drivecycles for accurate averages
					for _, row in merged.iterrows():
						source_truck = row["source_truck"]
						driving_event = int(row["drivecycle_number"])
						drivecycle_path = resolve_drivecycle_path(source_truck, driving_event)
						drivecycle_data = truck_model_tools_messy.extract_drivecycle_data(str(drivecycle_path))
						m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
						m_gvwr_lb = m_gvwr_kg / KG_PER_LB

						# Vary only the scanned parameter, keep others at baseline
						if param_name == "Electricity Cost ($/kWh)":
							elec_cost = scan_value
							demand_ch = demand_charge_baseline
							diesel_p = diesel_cost_baseline
							vmt = vmt_baseline
							e_bat = e_bat_baseline
							battery_cost_override = None
							payload_override = None
						elif param_name == "Demand Charge ($/kW)":
							elec_cost = electricity_cost_baseline
							demand_ch = scan_value
							diesel_p = diesel_cost_baseline
							vmt = vmt_baseline
							e_bat = e_bat_baseline
							battery_cost_override = None
							payload_override = None
						elif param_name == "Diesel Cost ($/gal)":
							elec_cost = electricity_cost_baseline
							demand_ch = demand_charge_baseline
							diesel_p = scan_value
							vmt = vmt_baseline
							e_bat = e_bat_baseline
							battery_cost_override = None
							payload_override = None
						elif param_name == "VMT (miles/year)":
							elec_cost = electricity_cost_baseline
							demand_ch = demand_charge_baseline
							diesel_p = diesel_cost_baseline
							vmt = scan_value
							e_bat = e_bat_baseline
							battery_cost_override = None
							payload_override = None
						elif param_name == "Battery Capacity (kWh)":
							elec_cost = electricity_cost_baseline
							demand_ch = demand_charge_baseline
							diesel_p = diesel_cost_baseline
							vmt = vmt_baseline
							e_bat = scan_value
							battery_cost_override = None
							payload_override = None
						elif param_name == "Battery Cost ($/kWh)":
							elec_cost = electricity_cost_baseline
							demand_ch = demand_charge_baseline
							diesel_p = diesel_cost_baseline
							vmt = vmt_baseline
							e_bat = e_bat_baseline
							battery_cost_override = scan_value
							payload_override = None
						elif param_name == "Payload (lb)":
							elec_cost = electricity_cost_baseline
							demand_ch = demand_charge_baseline
							diesel_p = diesel_cost_baseline
							vmt = vmt_baseline
							e_bat = e_bat_baseline
							battery_cost_override = None
							payload_override = scan_value

						# Compute payload based on scanned battery size (EV) and diesel truck mass
						m_bat_lb_scan = (e_bat / e_density * KG_PER_TON) / KG_PER_LB
						m_payload_lb = m_gvwr_lb - m_truck_no_bat_lb - m_bat_lb_scan
						m_truck_lb_diesel = diesel_parameters.m_truck / KG_PER_LB
						m_payload_lb_diesel = m_gvwr_lb - m_truck_lb_diesel

						if payload_override is not None:
							m_payload_lb = payload_override
							m_payload_lb_diesel = payload_override

						# EV cost
						ev_fuel_kwh_per_mi = row.get("fuel_economy_kWh_per_mile_ev", row.get("fuel_economy_kWh_per_mile"))
						ev_cost_dict = evaluate_costs_with_battery_cost(
							mileage=ev_fuel_kwh_per_mi,
							payload_lb=m_payload_lb,
							electricity_charge=elec_cost,
							demand_charge=demand_ch,
							average_VMT=vmt,
							charging_power=max_charging_power,
							e_bat=e_bat,
							battery_chemistry=parameters.battery_chemistry,
							truck_name=dataset["truck_params"],
							battery_unit_cost_override=battery_cost_override,
						)
						ev_tco = ev_cost_dict["TCO ($/mi)"]
						
						# Diesel cost
						diesel_mpg = row["fuel_economy_mpg"]
						diesel_cost_dict = evaluate_costs_diesel(
							mileage_mpg=diesel_mpg,
							payload_lb=m_payload_lb_diesel,
							diesel_price=diesel_p,
							average_VMT=vmt,
							truck_type=diesel_truck_params,
						)
						diesel_tco = diesel_cost_dict["TCO ($/mi)"]
						
						premium = ((ev_tco - diesel_tco) / diesel_tco) * 100
						avg_premium += premium
						count += 1
					
					if count > 0:
						avg_premium /= count
					
					premiums.append(avg_premium)
					param_results.append({
						"parameter_value": scan_value,
						"tco_premium_%": avg_premium,
					})
					
					elapsed = time.time() - start_time
					print(f"    [{point_idx}/{n_points}] Value: {scan_value:.4f} → Premium: {avg_premium:.2f}%")
					sys.stdout.flush()
					
				except Exception as e:
					print(f"    [{point_idx}/{n_points}] Value: {scan_value:.4f} → ERROR: {str(e)[:80]}")
					sys.stdout.flush()
					premiums.append(np.nan)
					param_results.append({
						"parameter_value": scan_value,
						"tco_premium_%": np.nan,
					})
			
			# Calculate impact
			valid_premiums = [p for p in premiums if not np.isnan(p)]
			if valid_premiums:
				impact = max(valid_premiums) - min(valid_premiums)
				tornado_impacts[param_name] = impact
			else:
				tornado_impacts[param_name] = 0
			
			# Save results
			sensitivity_results[param_name] = pd.DataFrame(param_results)
			param_filename = param_name.replace(" ", "_").replace("(", "").replace(")", "").replace("$", "").replace("/", "_")
			csv_path = sensitivity_dir / f"{truck_name}_{param_filename}_sensitivity.csv"
			pd.DataFrame(param_results).to_csv(csv_path, index=False)
			
			print(f"  ✓ Completed, Impact: {tornado_impacts[param_name]:.2f}%")
			sys.stdout.flush()
		
		tornado_data[truck_name] = tornado_impacts
		
		# Combined parameter scan (all parameters varying together)
		print(f"\n[{len(params_to_scan)+1}/{len(params_to_scan)+1}] Combined Parameter Scan (all parameters together)")
		sys.stdout.flush()
		
		combined_results = []
		combined_premiums = []
		scan_configs = [
			{"label": "All Min", "values": {}},
			{"label": "All Max", "values": {}},
		]
		
		# Set up min/max values for all parameters
		# For "All Min" (lowest EV premium): cheap electricity, no demand charges, expensive diesel, high VMT, small battery, cheap battery cost, low payload
		# For "All Max" (highest EV premium): expensive electricity, high demand charges, cheap diesel, low VMT, large battery, expensive battery cost, high payload
		for param_name, _, param_range in params_to_scan:
			if param_name == "Diesel Cost ($/gal)":
				# Reverse: expensive diesel (max) = lower EV premium (better for EV)
				scan_configs[0]["values"][param_name] = param_range["Max"]  # All Min premium uses Max diesel cost
				scan_configs[1]["values"][param_name] = param_range["Min"]  # All Max premium uses Min diesel cost
			elif param_name == "VMT (miles/year)":
				# Reverse: high VMT (max) = lower EV premium (fixed costs spread over more miles)
				scan_configs[0]["values"][param_name] = param_range["Max"]  # All Min premium uses Max VMT
				scan_configs[1]["values"][param_name] = param_range["Min"]  # All Max premium uses Min VMT
			else:
				# Normal direction
				scan_configs[0]["values"][param_name] = param_range["Min"]
				scan_configs[1]["values"][param_name] = param_range["Max"]
		
		for config_idx, config in enumerate(scan_configs, 1):
			try:
				print(f"  [{config_idx}/{len(scan_configs)}] Evaluating: {config['label']}")
				sys.stdout.flush()
				
				# Calculate average premium with ALL parameters at specified values
				avg_premium = 0
				count = 0
				
				for _, row in merged.iterrows():
					source_truck = row["source_truck"]
					
					# Use all parameter values from config
					elec_cost = config["values"]["Electricity Cost ($/kWh)"]
					demand_ch = config["values"]["Demand Charge ($/kW)"]
					diesel_p = config["values"]["Diesel Cost ($/gal)"]
					vmt = config["values"]["VMT (miles/year)"]
					e_bat = config["values"]["Battery Capacity (kWh)"]
					battery_cost_override = config["values"]["Battery Cost ($/kWh)"]
					payload_override = config["values"]["Payload (lb)"]
					
					# Use payload override if specified
					if payload_override is not None:
						m_payload_lb = payload_override
					else:
						m_payload_lb = parameters.m_payload_kg / KG_PER_LB
					
					# EV cost
					ev_fuel_kwh_per_mi = row.get("fuel_economy_kWh_per_mile_ev", row.get("fuel_economy_kWh_per_mile"))
					ev_cost_dict = evaluate_costs_with_battery_cost(
						mileage=ev_fuel_kwh_per_mi,
						payload_lb=m_payload_lb,
						electricity_charge=elec_cost,
						demand_charge=demand_ch,
						average_VMT=vmt,
						charging_power=max_charging_power,
						e_bat=e_bat,
						battery_chemistry=parameters.battery_chemistry,
						truck_name=dataset["truck_params"],
						battery_unit_cost_override=battery_cost_override,
					)
					ev_tco = ev_cost_dict["TCO ($/mi)"]
					
					# Diesel cost
					diesel_mpg = row["fuel_economy_mpg"]
					diesel_cost_dict = evaluate_costs_diesel(
						mileage_mpg=diesel_mpg,
						payload_lb=m_payload_lb,
						diesel_price=diesel_p,
						average_VMT=vmt,
						truck_type=diesel_truck_params,
					)
					diesel_tco = diesel_cost_dict["TCO ($/mi)"]
					
					if diesel_tco > 0:
						premium = ((ev_tco - diesel_tco) / diesel_tco) * 100
						avg_premium += premium
						count += 1
				
				if count > 0:
					avg_premium /= count
				
				combined_premiums.append(avg_premium)
				combined_results.append({
					"configuration": config["label"],
					"tco_premium_%": avg_premium,
				})
				
				print(f"    {config['label']} → Premium: {avg_premium:.2f}%")
				sys.stdout.flush()
				
			except Exception as e:
				print(f"    {config['label']} → ERROR: {str(e)[:80]}")
				sys.stdout.flush()
				combined_premiums.append(np.nan)
				combined_results.append({
					"configuration": config["label"],
					"tco_premium_%": np.nan,
				})
		
		# Calculate combined impact
		valid_combined_premiums = [p for p in combined_premiums if not np.isnan(p)]
		if valid_combined_premiums and len(valid_combined_premiums) >= 2:
			combined_impact = max(valid_combined_premiums) - min(valid_combined_premiums)
			print(f"  ✓ Completed, Combined Impact: {combined_impact:.2f}%")
			print(f"    Range: {min(valid_combined_premiums):.2f}% to {max(valid_combined_premiums):.2f}%")
		else:
			combined_impact = 0
			print(f"  ✓ Completed (insufficient data for impact calculation)")
		sys.stdout.flush()
		
		# Save combined results
		combined_df = pd.DataFrame(combined_results)
		combined_csv_path = sensitivity_dir / f"{truck_name}_combined_all_parameters_sensitivity.csv"
		combined_df.to_csv(combined_csv_path, index=False)
		
		# Create combined parameter plot
		print(f"Creating combined parameter plot for {truck_name}...")
		sys.stdout.flush()
		fig_combined, ax_combined = plt.subplots(figsize=(8, 6))
		
		x_positions = np.arange(len(combined_results))
		y_values = [r["tco_premium_%"] for r in combined_results]
		labels = [r["configuration"] for r in combined_results]
		
		bars = ax_combined.bar(x_positions, y_values, color=['#2196F3', '#FF5722'], alpha=0.7, edgecolor='black')
		ax_combined.axhline(y=baseline_premium, color='green', linestyle='--', linewidth=2, label=f"Baseline ({baseline_premium:.1f}%)")
		
		ax_combined.set_xlabel("Configuration", fontweight='bold', fontsize=12)
		ax_combined.set_ylabel("EV TCO Premium (%)", fontweight='bold', fontsize=12)
		ax_combined.set_title(f"{truck_name} - Combined Parameter Sensitivity\n(All Parameters Varying Together)", 
		                      fontweight='bold', fontsize=14)
		ax_combined.set_xticks(x_positions)
		ax_combined.set_xticklabels(labels)
		ax_combined.grid(True, alpha=0.3, axis='y')
		ax_combined.legend()
		
		# Add value labels on bars
		for i, (bar, val) in enumerate(zip(bars, y_values)):
			height = bar.get_height()
			ax_combined.text(bar.get_x() + bar.get_width()/2., height,
			                f'{val:.1f}%',
			                ha='center', va='bottom', fontweight='bold', fontsize=10)
		
		plt.tight_layout()
		combined_plot_path = sensitivity_dir / f"{truck_name}_combined_all_parameters{param_suffix}.png"
		plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
		print(f"Saved combined parameter plot: {combined_plot_path}")
		plt.close(fig_combined)
		sys.stdout.flush()
		
		# Create sensitivity plots
		print(f"\nCreating sensitivity scan plots for {truck_name}...")
		sys.stdout.flush()
		fig, axes = plt.subplots(2, 3, figsize=(15, 10))
		fig.suptitle(f"{truck_name} - TCO Premium Sensitivity Analysis", fontsize=16, fontweight='bold')
		axes_flat = axes.flatten()
		
		for ax_idx, (param_name, results_df) in enumerate(sensitivity_results.items()):
			if ax_idx >= len(axes_flat):
				break
			
			ax = axes_flat[ax_idx]
			x_values = results_df["parameter_value"].values
			y_values = results_df["tco_premium_%"].values
			
			ax.plot(x_values, y_values, 'b-o', linewidth=2, markersize=6, label="Sensitivity scan")
			
			# Mark current value
			current_param_val = [v for k, v, _ in params_to_scan if k == param_name]
			if current_param_val:
				current_param_val = current_param_val[0]
				ax.axvline(x=current_param_val, color='red', linestyle='--', linewidth=2, label="Current value")
			
			ax.set_xlabel(param_name)
			ax.set_ylabel("EV TCO Premium (%)")
			ax.grid(True, alpha=0.3)
			ax.legend(fontsize=8)
		
		for ax_idx in range(len(sensitivity_results), len(axes_flat)):
			axes_flat[ax_idx].set_visible(False)
		
		plt.tight_layout()
		plot_path = sensitivity_dir / f"{truck_name}_sensitivity_scans{param_suffix}.png"
		plt.savefig(plot_path, dpi=300, bbox_inches='tight')
		print(f"Saved sensitivity scan plots: {plot_path}")
		plt.close(fig)
		sys.stdout.flush()
	
	# Tornado plot disabled during debugging (requested)
	print("\nTornado plot generation is disabled for debugging.")
	sys.stdout.flush()
	
	print("\n" + "="*80)
	print("SENSITIVITY ANALYSIS COMPLETE")
	print("="*80)


if __name__ == "__main__":
	plots_dir = Path("plots_messy")
	plots_dir.mkdir(parents=True, exist_ok=True)
	
	optimized = True
	optimized_params = load_optimized_parameters(use_optimized=optimized)
	
	battery_caps = pd.read_csv(BASE_DIR / "messy_middle_results" / "battery_capacities_linear_summary.csv").set_index('Value')
	
	datasets = [
		{"name": "saia2", "truck_params": "saia", "battery_col": "saia1"},
		{"name": "4gen", "truck_params": "4gen", "battery_col": "4gen"},
		{"name": "joyride", "truck_params": "joyride", "battery_col": "joyride"},
		{"name": "nevoya_with_weight", "truck_params": "nevoya", "battery_col": "nevoya_with_weight"},
	]
	
	param_suffix = "_optimized" if optimized else "_original"
	
	average_vmt = 100000
	sensitivity_analysis_simple(
		datasets,
		plots_dir,
		optimized_params,
		battery_caps,
		param_suffix,
		average_vmt=average_vmt,
	)
