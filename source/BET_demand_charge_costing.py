#!/usr/bin/env python
# coding: utf-8

"""
This code was originally written by Sayandeep Biswas, and contains modifications by Danika MacDonell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

costing_data_df = pd.read_csv('data/electricity_costing.csv')

def electric_price_calculator(charging_energy_per_month, scenario, demand_charge, energy_charge, charging_power):
    lifetime = 15 #Years
    discount = 7
    
    # Convert charging energy per month to kWh
    charging_energy_per_month = charging_energy_per_month*1000

    installation_cost_dict = {'optimistic':18577, 'pessimistic':47781}
    hardware_cost_dict = {'optimistic':75000, 'pessimistic':100000}
    fixed_monthly_dict = {'optimistic':20, 'pessimistic':100}
    EVSE_effeciency = 0.92
    
    lifetime_energy_sold = (charging_energy_per_month*12*lifetime)
    capital_cost = ((hardware_cost_dict[scenario] + installation_cost_dict[scenario]))*(1+discount/100.)**lifetime
    norm_cap_cost = capital_cost/lifetime_energy_sold
    norm_demand_charge = charging_power*demand_charge/charging_energy_per_month
    norm_energy_charge = energy_charge/EVSE_effeciency
    norm_fixed_monthly = fixed_monthly_dict[scenario]*12*lifetime/lifetime_energy_sold
    total_charge = norm_cap_cost + (norm_demand_charge + norm_energy_charge + norm_fixed_monthly)
    
    return np.array([norm_cap_cost,norm_fixed_monthly,norm_energy_charge,norm_demand_charge])

costing_data_df['Capital cost [baseline]'] = ''
costing_data_df['Capital cost [optimistic]'] = ''
costing_data_df['Capital cost [pessimistic]'] = ''
costing_data_df['Fixed charge [baseline]'] = ''
costing_data_df['Fixed charge [optimistic]'] = ''
costing_data_df['Fixed charge [pessimistic]'] = ''
costing_data_df['Energy charge [baseline]'] = ''
costing_data_df['Energy charge [optimistic]'] = ''
costing_data_df['Energy charge [pessimistic]'] = ''
costing_data_df['Demand charge [baseline]'] = ''
costing_data_df['Demand charge [optimistic]'] = ''
costing_data_df['Demand charge [pessimistic]'] = ''

for index, row in costing_data_df.iterrows():
    electricity_prices_optimistic = electric_price_calculator(row['Energy per Month (MWh) [optimistic]'],
                                                                'optimistic',
                                                                row['Demand charge ($/kW) [optimistic]'],
                                                                row['Energy charge ($/kWh) [optimistic]'],
                                                                row['Charging Power (kW)'])
                                                                
    electricity_prices_pessimistic = electric_price_calculator(row['Energy per Month (MWh) [pessimistic]'],
                                                                'pessimistic',
                                                                row['Demand charge ($/kW) [pessimistic]'],
                                                                row['Energy charge ($/kWh) [pessimistic]'],
                                                                row['Charging Power (kW)'])
                                                                
    electricity_prices_baseline = (electricity_prices_optimistic + electricity_prices_pessimistic) / 2.
                                                                
    costing_data_df.at[index, 'Capital cost [baseline]'] = electricity_prices_baseline[0]
    costing_data_df.at[index, 'Capital cost [optimistic]'] = electricity_prices_optimistic[0]
    costing_data_df.at[index, 'Capital cost [pessimistic]'] = electricity_prices_pessimistic[0]
    costing_data_df.at[index, 'Fixed charge [baseline]'] = electricity_prices_baseline[1]
    costing_data_df.at[index, 'Fixed charge [optimistic]'] = electricity_prices_optimistic[1]
    costing_data_df.at[index, 'Fixed charge [pessimistic]'] = electricity_prices_pessimistic[1]
    costing_data_df.at[index, 'Energy charge [baseline]'] = electricity_prices_baseline[2]
    costing_data_df.at[index, 'Energy charge [optimistic]'] = electricity_prices_optimistic[2]
    costing_data_df.at[index, 'Energy charge [pessimistic]'] = electricity_prices_pessimistic[2]
    costing_data_df.at[index, 'Demand charge [baseline]'] = electricity_prices_baseline[3]
    costing_data_df.at[index, 'Demand charge [optimistic]'] = electricity_prices_optimistic[3]
    costing_data_df.at[index, 'Demand charge [pessimistic]'] = electricity_prices_pessimistic[3]

def plot_electricity_data(title, name_save, costing_data_df, costing_data_df_up=None, costing_data_df_down=None):
    x = ['Optimistic', 'Baseline', 'Pessimistic']
    cost_types = ['Capital cost', 'Fixed charge', 'Energy charge', 'Demand charge']
    s = np.array([0,0,0])
    s_up = np.array([0,0,0])
    s_down = np.array([0,0,0])
    
    for cost_type in cost_types:
        y_plot = [float(costing_data_df[f'{cost_type} [optimistic]']), float(costing_data_df[f'{cost_type} [baseline]']), float(costing_data_df[f'{cost_type} [pessimistic]'])]
        
        if not (costing_data_df_up is None):
            y_plot_up = [float(costing_data_df_up[f'{cost_type} [optimistic]']), float(costing_data_df_up[f'{cost_type} [baseline]']), float(costing_data_df_up[f'{cost_type} [pessimistic]'])]
            s_up = s_up + y_plot_up
            
        if not (costing_data_df_down is None):
            y_plot_down = [float(costing_data_df_down[f'{cost_type} [optimistic]']), float(costing_data_df_down[f'{cost_type} [baseline]']), float(costing_data_df_down[f'{cost_type} [pessimistic]'])]
            s_down = s_down + y_plot_down
            
        plt.bar(x, y_plot, bottom=s, label = cost_type)
        s = s + y_plot
        
    if not (costing_data_df_up is None):
        plt.errorbar(x, s, yerr=[s_up - s, s - s_down], marker='', linestyle='', capsize=5, color='black')
        
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.ylabel('Cost of electricity ($/kWh)', fontsize=16)
    plt.title(title, fontsize=18)
    plt.ylim(0,1)
    plt.savefig(f'plots/electricity_prices_{name_save}.png')
    plt.close()

plot_electricity_data('Nominal', 'nominal', costing_data_df[costing_data_df['Name'] == 'Nominal'], None, None)

plot_electricity_data('PepsiCo Tesla Semi', 'pepsico', costing_data_df[costing_data_df['Name'] == 'PepsiCo (central)'], costing_data_df[costing_data_df['Name'] == 'PepsiCo (up)'], costing_data_df[costing_data_df['Name'] == 'PepsiCo (down)'])

costing_data_df.to_csv('data/electricity_costing_results.csv')
