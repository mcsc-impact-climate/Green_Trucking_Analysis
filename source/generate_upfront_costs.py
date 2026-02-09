import sys
sys.path.insert(0, 'source')
import pandas as pd
from pathlib import Path
from data_collection_tools_messy import read_parameters, read_battery_params, read_truck_cost_data

BASE_DIR = Path('.')

# Load battery capacities
battery_caps = pd.read_csv(
    BASE_DIR / 'messy_middle_results' / 'battery_capacities_linear_summary.csv'
).set_index('Value')

# Define datasets
datasets = [
    {'name': 'saia2', 'truck_params': 'saia', 'battery_col': 'saia1'},
    {'name': '4gen', 'truck_params': '4gen', 'battery_col': '4gen'},
    {'name': 'joyride', 'truck_params': 'joyride', 'battery_col': 'joyride'},
    {'name': 'nevoya_with_weight', 'truck_params': 'nevoya', 'battery_col': 'nevoya_with_weight'},
]

# Chemistry mapping
chemistry_map = {
    'saia2': 'NMC',
    '4gen': 'NCA',
    'joyride': 'LFP',
    'nevoya_with_weight': 'NMC'
}

results = []

for dataset in datasets:
    truck_name = dataset['name']
    truck_params_name = dataset['truck_params']
    battery_col = dataset['battery_col']
    chemistry = chemistry_map[truck_name]
    
    # Get battery capacity
    battery_capacity = battery_caps.loc['Mean', battery_col]
    
    # Load parameters for EV
    ev_parameters = read_parameters(
        truck_params_name, 
        economy_params='default', 
        vmt_params='daycab_vmt_vius_2021', 
        truck_type='EV', 
        run='messy_middle'
    )
    
    # Load parameters for Diesel (using diesel-specific params file)
    diesel_parameters = read_parameters(
        f"{truck_params_name}_diesel", 
        economy_params='default', 
        vmt_params='daycab_vmt_vius_2021', 
        truck_type='diesel', 
        run='messy_middle'
    )
    
    # Load cost data
    cost_data_ev = read_truck_cost_data('class_8_daycab', 'EV', chemistry)
    cost_data_diesel = read_truck_cost_data('class_8_daycab', 'diesel', chemistry)
    
    # Extract capital costs
    capital_ev = cost_data_ev['Capital Costs']
    capital_diesel = cost_data_diesel['Capital Costs']
    
    # Calculate component costs for EV
    glider_cost = capital_ev['glider ($)']
    motor_inverter_cost = capital_ev['motor and inverter ($/kW)'] * ev_parameters.p_motor_max / 1000  # Convert W to kW
    dcdc_cost = capital_ev['DC-DC converter ($/kW)'] * ev_parameters.p_motor_max / 1000
    battery_cost = cost_data_ev['Battery Unit Cost ($/kWh)'] * battery_capacity
    
    # Total EV cost
    ev_total_cost = glider_cost + motor_inverter_cost + dcdc_cost + battery_cost
    
    # Calculate component costs for Diesel
    engine_cost = capital_diesel['engine ($/kW)'] * diesel_parameters.p_motor_max / 1000
    transmission_cost = capital_diesel['transmission ($)']
    aftertreatment_cost = capital_diesel['aftertreatment ($)']
    fuel_tank_cost = capital_diesel['fuel tank ($)']
    
    # Total Diesel cost
    diesel_total_cost = glider_cost + engine_cost + transmission_cost + aftertreatment_cost + fuel_tank_cost
    
    results.append({
        'Truck': truck_name,
        'Battery Chemistry': chemistry,
        'Battery Capacity (kWh)': battery_capacity,
        'EV Glider ($)': glider_cost,
        'EV Motor/Inverter ($)': motor_inverter_cost,
        'EV DC-DC Converter ($)': dcdc_cost,
        'EV Battery ($)': battery_cost,
        'EV Total Cost ($)': ev_total_cost,
        'Diesel Glider ($)': glider_cost,
        'Diesel Engine ($)': engine_cost,
        'Diesel Transmission ($)': transmission_cost,
        'Diesel Aftertreatment ($)': aftertreatment_cost,
        'Diesel Fuel Tank ($)': fuel_tank_cost,
        'Diesel Total Cost ($)': diesel_total_cost,
    })

# Create DataFrame and save
df = pd.DataFrame(results)
output_path = BASE_DIR / 'tables_messy' / 'upfront_truck_costs.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Updated upfront truck costs table with all components:")
print(df.to_string(index=False))
print(f"\n✓ Saved to: {output_path}")
