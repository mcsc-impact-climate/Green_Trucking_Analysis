"""
Date: March 26, 2024
Purpose: Collect input data for truck simulation and lifecycle cost and emissions assessment
Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader
"""

import pandas as pd
import truck_model_tools
import truck_model_tools_diesel

# Function to read in non-battery parameters for the truck model
def read_parameters(truck_params='default', economy_params='default', vmt_params='default', truck_type='EV'):
    truck_params_str = f'data/{truck_params}_truck_params.csv'
    economy_params_str = f'data/{economy_params}_economy_params.csv'
    vmt_params_str = f'data/{vmt_params}_vmt.csv'
    if truck_type == 'EV':
        parameters = truck_model_tools.read_parameters(truck_params_str, economy_params_str, 'data/constants.csv', 'data/default_vmt.csv')
    else:
        parameters = truck_model_tools_diesel.read_parameters(truck_params_str, economy_params_str, 'data/constants.csv', 'data/default_vmt.csv')
    return parameters

# Function to read in battery parameters for the truck model
def read_battery_params(battery_params='default', chemistry='NMC'):
    battery_params_str = f'data/default_battery_params.csv'
    df_battery_data = pd.read_csv(battery_params_str, index_col=0)
    
    battery_params_dict = {
        'Energy density (kWh/ton)': df_battery_data['Value'].loc[f'{chemistry} battery energy density'],
        'Roundtrip efficiency': df_battery_data['Value'].loc[f'{chemistry} roundtrip efficiency'],
        'Manufacturing emissions (CO2/kWh)': df_battery_data['Value'].loc[f'{chemistry} manufacturing emissions'],
        'Replacements': df_battery_data['Value'].loc[f'{chemistry} replacements']
        }
    
    return battery_params_dict
    
def read_truck_cost_data(truck = 'class_8_daycab', truck_type = 'EV', chemistry='NMC'):
    df_truck_cost = pd.read_csv(f'data/{truck}_cost_data.csv', index_col=0)
    
    truck_cost_data_dict = {}
    
    if truck_type == 'EV':
        truck_cost_data_dict['Capital Costs'] = {
            'glider ($)': float(df_truck_cost['Value'].loc['Glider']),
            'motor and inverter ($/kW)': float(df_truck_cost['Value'].loc['Motor and inverter']),
            'DC-DC converter ($/kW)': float(df_truck_cost['Value'].loc['DC-DC converter'])
            }
        truck_cost_data_dict['Battery Unit Cost ($/kWh)'] = float(df_truck_cost['Value'].loc[f'{chemistry} battery'])
    elif truck_type == 'diesel' or truck_type == 'Diesel':
        truck_cost_data_dict['Capital Costs'] = {
            'glider ($)': float(df_truck_cost['Value'].loc['Glider']),
            'aftertreatment ($)': float(df_truck_cost['Value'].loc['Aftertreatment']),
            'engine ($/kW)': float(df_truck_cost['Value'].loc['Engine']),
            'transmission ($)': float(df_truck_cost['Value'].loc['Transmission']),
            'fuel tank ($)': float(df_truck_cost['Value'].loc['Fuel tank']),
            'WHR system ($)': float(df_truck_cost['Value'].loc['WHR system']),
            }
    else:
        print(f'Error: No info for truck type {truck_type}')
    
    truck_cost_data_dict['Operating Costs'] = {
        'maintenance & repair ($/mi)': float(df_truck_cost['Value'].loc['Maintenance and repair']),
        'labor ($/mi)': float(df_truck_cost['Value'].loc['Labor']),
        'insurance ($/mi-$)': float(df_truck_cost['Value'].loc['Insurance']),
        'tolls ($/mi)': float(df_truck_cost['Value'].loc['Tolls']),
        'permits and licenses ($/mi)': float(df_truck_cost['Value'].loc['Permits and licenses'])
        }
    
    return truck_cost_data_dict
    
    
# Basic code to test the functions defined above
#print(read_truck_cost_data(truck_type='diesel'))
    
