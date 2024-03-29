"""
Date: March 26, 2024
Purpose: Collect input data for truck simulation and lifecycle cost and emissions assessment
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
    
def get_scenario_data(scenario='Present', chemistry='NMC'):
    df_scenarios = pd.read_csv('data/scenario_data.csv', index_col=0)
    scenario_data_dict = {}
    scenario_data_dict['Energy Density (kWh/ton)'] = float(df_scenarios['NMC battery energy density'].loc[scenario])

    scenario_data_dict['Capital Costs ($/kW)'] = pd.DataFrame({'glider ($)': [float(df_scenarios['Cost of glider'].iloc[0])], 'motor and inverter ($/kW)': [float(df_scenarios['Cost of motor and inverter'].iloc[0])], 'DC-DC converter ($/kW)': [float(df_scenarios['Cost of DC-DC converter'].iloc[0])]})

    scenario_data_dict['Operating Costs ($/mi)'] = pd.DataFrame({'maintenance & repair ($/mi)': [float(df_scenarios['Maintenance and repair cost'].iloc[0])], 'labor ($/mi)': [float(df_scenarios['Labor cost'].iloc[0])], 'insurance ($/mi)': [float(df_scenarios['Insurance cost'].iloc[0])], 'misc ($/mi)': [float(df_scenarios[' Miscellaneous costs'].iloc[0])]})
