"""
Date: March 26, 2024
Purpose: Collect input data for truck simulation and lifecycle cost and emissions assessment
Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader
"""

import pandas as pd
import truck_model_tools

# Function to read in non-battery parameters for the truck model
def read_parameters(truck_params='default', economy_params='default', vmt_params='default'):
    truck_params_str = f'data/{truck_params}_truck_params.csv'
    economy_params_str = f'data/{economy_params}_economy_params.csv'
    vmt_params_str = f'data/{vmt_params}_vmt.csv'
    parameters = truck_model_tools.read_parameters(truck_params_str, economy_params_str, 'data/constants.csv', 'data/default_vmt.csv')
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
    
def read_scenario_data(scenario='Present', chemistry='NMC'):
    df_scenarios = pd.read_csv('data/scenario_data.csv', index_col=0)
    scenario_data_dict = {}
    scenario_data_dict['Energy Density (kWh/ton)'] = float(df_scenarios[f'{chemistry} battery energy density'].loc[scenario])

    scenario_data_dict['Capital Costs ($/kW)'] = {
        'glider ($)': float(df_scenarios['Cost of glider'].loc[scenario]),
        'motor and inverter ($/kW)': float(df_scenarios['Cost of motor and inverter'].loc[scenario]),
        'DC-DC converter ($/kW)': float(df_scenarios['Cost of DC-DC converter'].loc[scenario])
    }

    scenario_data_dict['Operating Costs ($/mi)'] = {
        'maintenance & repair ($/mi)': float(df_scenarios['Maintenance and repair cost'].loc[scenario]),
        'labor ($/mi)': float(df_scenarios['Labor cost'].loc[scenario]),
        'insurance ($/mi-$)': float(df_scenarios['Insurance cost'].loc[scenario]),
        'misc ($/mi)': float(df_scenarios[' Miscellaneous costs'].loc[scenario])
    }
    
    scenario_data_dict['Battery Unit Cost ($/kWh)'] = float(df_scenarios[f'{chemistry} battery energy density'].loc[scenario])
    
    return scenario_data_dict
