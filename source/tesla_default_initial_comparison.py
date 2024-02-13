# -*- coding: utf-8 -*-
"""
This code was originally written by Kariana Moreno Sader, and contains modifications by Danika MacDonell.

Original file is located at
    https://colab.research.google.com/drive/1r_BqbeE98uhmCH-SI06JDAE97AihVnQu

December 3, 2023
### Kariana Moreno Sader: **Workflow for the analysis of long-haul truck electrification**



The workflow consists of 3 main steps: energy consumption, emissions analysis and costing.

1. **Energy consumption**: using real-world drive cycle data and a physics-based  vehicle model, estimate fuel economy and battery energy capacity
2. **Payload penalty**: Calculate payload losses of battery electric vehicles given the payload distribution of conventional diesel (dataset from VIUS 2002).
The output is a payload penalty factor that is equivalent to the number of additional electric trucks needed to deliver the same amount of goods.
3. **Charging infrastructure and electricity pricing**:
4. **Emissions analysis**: Estimate Well-to-Wheel (WTW) emissions from fuel economy and US. grid carbon intensity
5. **Cost analysis**: Calculate of purchasing and operating a battery electric truck

Note: The defualt values used for this analysis are for a class 8 long-haul truck driving an average of 600 miles per day over flexible routes.
"""

# Import packages
import pandas as pd
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import truck_model_tools
import costing_tools
import emissions_tools

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

KG_PER_TON = 1000
KG_PER_LB = 0.453592


inputs_df = pd.read_csv('data/emission_costing_inputs.csv')


# DMM: Can get short-haul VMT from the same paper
VMT_nominal=np.array(pd.read_csv('data/default_vmt.csv')['VMT (miles)']) #Vehicle miles traveled per year. Source: Burnham, A et al. (2021)
VMT_norm_dist = VMT_nominal / VMT_nominal[0]

# DMM: Assume the PepsiCo VMT distribution is the same as nominal, and scales according to the miles traveled in the first year
VMT_Tesla = VMT_norm_dist * float(inputs_df[inputs_df['Value'] == 'Year 1 VMT (central) [miles/year]']['PepsiCo'])

#extract vehicle speed, and road grade from drive cycle
#Drive cycle. Source: Jones, R et al. (2023).Developing and Benchmarking a US Long-haul Drive Cycle forVehicle Simulations, Costing and Emissions Analysis
#https://docs.google.com/spreadsheets/d/1Q2uO-JHfwvGxir_PU8IO5zmo0vs4ooC_/edit?usp=sharing&ouid=102742490305620802920&rtpof=true&sd=true
df = truck_model_tools.extract_drivecycle_data('data/drivecycle.xlsx') #drive cycle as a dataframe

#extract payload distribution from excel file
#Note: Dataset from VIUS 2002, filtered and cleaned by authors for this analysis. Source: 2002 Economic Census: Vehicle Inventory and Use Survey
#https://docs.google.com/spreadsheets/d/1Oe_jBIUb-kJ5yy9vkwaPgldVe4cloAtG/edit?usp=sharing&ouid=102742490305620802920&rtpof=true&sd=true
payload_distribution = pd.read_excel('data/payloaddistribution.xlsx')
payload_distribution['Payload (kg)'] = payload_distribution['Payload (lb)']*KG_PER_LB #payload distribution in kgs

#parameters = truck_model_tools.share_parameters(m_ave_payload, m_max, m_truck_no_bat, m_guess, p_aux, p_motor_max, cd, cr, a_cabin, g, rho_air, DoD, eta_i, eta_m, eta_gs, eta_rb, eta_grid_transmission, VMT_nominal, discountrate)
parameters = truck_model_tools.read_parameters('data/default_truck_params.csv', 'data/default_economy_params.csv', 'data/constants.csv', 'data/default_vmt.csv')
parameters_Tesla = truck_model_tools.share_parameters(parameters.m_ave_payload, parameters.m_max, parameters.m_truck_no_bat, parameters.m_guess, parameters.p_aux, parameters.p_motor_max, parameters.cd, parameters.cr, parameters.a_cabin, parameters.g, parameters.rho_air, parameters.DoD, parameters.eta_i, parameters.eta_m, parameters.eta_gs, parameters.eta_rb, parameters.eta_grid_transmission, VMT_Tesla, parameters.discountrate)

#Define variables for case analysis:
df_battery_data = pd.read_csv('data/default_battery_params.csv', index_col=0)
df_scenarios = pd.read_csv('data/scenario_data.csv', index_col=0)

e_density_list_LFP = np.array(df_scenarios['LFP battery energy density'].iloc[0:3].astype(float)) #energy density (kWh/ton)
eta_battery_LFP = df_battery_data['Value'].loc['LFP roundtrip efficiency'] #efficiency for LFP. DMM: Roundtrip battery efficiency
alpha=1 #for payload penalty factor calculations (alpha = 1 for base case, alpha = 2: complete dependency in payload measurements)

#######################################################################################################

vehicle_model_results_LFP = pd.DataFrame(columns = ['Energy battery (kWh)', 'Battery mass (lbs)', 'Fuel economy (kWh/mi)', 'Payload penalty factor', 'Total vehicle mass (lbs)'])

for e_density in e_density_list_LFP:
  m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(df, eta_battery_LFP, e_density)
  print(f'Calculated battery capacity for LFP battery with density {e_density} kWh/ton: {e_bat} kWh')
  print(f'Calculated mileage for LFP battery with density {e_density} kWh/ton: {mileage} kWh/mile')
  payload_penalty_factor=truck_model_tools.payload(parameters).get_penalty(payload_distribution, m_bat, alpha)
  vehicle_model_results_LFP.loc[len(vehicle_model_results_LFP)] = [e_bat, m_bat/KG_PER_LB, mileage, payload_penalty_factor, m/KG_PER_LB]

#print(vehicle_model_results_LFP.head())

#Define variables for case analysis:

e_density_list_NMC = np.array(df_scenarios['NMC battery energy density'].iloc[0:3].astype(float)) #energy density (kWh/ton)
eta_battery_NMC = df_battery_data['Value'].loc['NMC roundtrip efficiency'] #efficiency for LFP
alpha=1; #for payload penalty factor calculations (alpha = 1 for base case, alpha = 2: complete dependency in payload measurements)

#######################################################################################################

vehicle_model_results_NMC = pd.DataFrame(columns = ['Energy battery (kWh)', 'Battery mass (lbs)', 'Fuel economy (kWh/mi)', 'Payload penalty factor', 'Total vehicle mass (lbs)'])

for e_density in e_density_list_NMC:
  m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(df, eta_battery_NMC, e_density)
  print(f'Calculated battery capacity for NMC battery with density {e_density} kWh/ton: {e_bat} kWh')
  print(f'Calculated mileage for NMC battery with density {e_density} kWh/ton: {mileage} kWh/mile')
  payload_penalty_factor = truck_model_tools.payload(parameters).get_penalty(payload_distribution, m_bat, alpha)
  vehicle_model_results_NMC.loc[len(vehicle_model_results_NMC)] = [e_bat, m_bat/KG_PER_LB, mileage, payload_penalty_factor, m/KG_PER_LB]

#print(vehicle_model_results_NMC.head())

#######################################################################################################

columns = ['Energy battery (kWh)', 'Battery mass (lbs)', 'Fuel economy (kWh/mi)', 'Payload penalty factor', 'Total vehicle mass (lbs)']
vehicle_model_results_Tesla_LFP = pd.DataFrame({col: [None, None, None] for col in columns})
vehicle_model_results_Tesla_NMC = pd.DataFrame({col: [None, None, None] for col in columns})

for i in range(3):
    vehicle_model_results_Tesla_LFP.loc[i]['Energy battery (kWh)'] = float(inputs_df[inputs_df['Value'] == 'Battery Capacity (central) [kWh]']['PepsiCo'])
#    vehicle_model_results_Tesla_LFP.loc[i]['Energy battery up (kWh)'] = inputs_df['Battery Capacity (up) [kWh]']
#    vehicle_model_results_Tesla_LFP.loc[i]['Energy battery down (kWh)'] = inputs_df['Battery Capacity (down) [kWh]']
    vehicle_model_results_Tesla_NMC.loc[i]['Energy battery (kWh)'] = float(inputs_df[inputs_df['Value'] == 'Battery Capacity (central) [kWh]']['PepsiCo'])
    
    vehicle_model_results_Tesla_LFP.loc[i]['Fuel economy (kWh/mi)'] = float(inputs_df[inputs_df['Value'] == 'Electricity per mile (up) [kWh/mile]']['PepsiCo'])
    vehicle_model_results_Tesla_NMC.loc[i]['Fuel economy (kWh/mi)'] = float(inputs_df[inputs_df['Value'] == 'Electricity per mile (up) [kWh/mile]']['PepsiCo'])
    
    m_bat_LFP = float(inputs_df[inputs_df['Value'] == 'Battery Capacity (central) [kWh]']['PepsiCo']) / (e_density_list_LFP[i] / KG_PER_TON)
    m_bat_NMC = float(inputs_df[inputs_df['Value'] == 'Battery Capacity (central) [kWh]']['PepsiCo']) / (e_density_list_NMC[i] / KG_PER_TON)

    vehicle_model_results_Tesla_LFP.loc[i]['Battery mass (lbs)'] = m_bat_LFP / KG_PER_LB
    vehicle_model_results_Tesla_NMC.loc[i]['Battery mass (lbs)'] = m_bat_NMC / KG_PER_LB
    
    payload_penalty_factor_LFP = truck_model_tools.payload(parameters_Tesla).get_penalty(payload_distribution, m_bat_LFP, alpha)
    payload_penalty_factor_NMC = truck_model_tools.payload(parameters_Tesla).get_penalty(payload_distribution, m_bat_NMC, alpha)
    
    vehicle_model_results_Tesla_LFP.loc[i]['Payload penalty factor'] = payload_penalty_factor_LFP
    vehicle_model_results_Tesla_NMC.loc[i]['Payload penalty factor'] = payload_penalty_factor_NMC
    
    vehicle_model_results_Tesla_LFP.loc[i]['Total vehicle mass (lbs)'] = (m_bat_LFP + parameters_Tesla.m_ave_payload + parameters_Tesla.m_truck_no_bat) / KG_PER_LB
    vehicle_model_results_Tesla_NMC.loc[i]['Total vehicle mass (lbs)'] = (m_bat_NMC + parameters_Tesla.m_ave_payload + parameters_Tesla.m_truck_no_bat) / KG_PER_LB
    
#print(vehicle_model_results_Tesla_LFP.head())

#Define variables for case analysis:

GHG_bat_unit_LFP = df_battery_data['Value'].loc['LFP manufacturing emissions'] #g CO2/kWh
replacements_LFP = df_battery_data['Value'].loc['LFP replacements']

#######################################################################################################
GHG_emissions_LFP = emissions_tools.emission(parameters).get_WTW(vehicle_model_results_LFP, GHG_bat_unit_LFP,  replacements_LFP)
GHG_emissions_Tesla_LFP = emissions_tools.emission(parameters_Tesla).get_WTW(vehicle_model_results_Tesla_LFP, GHG_bat_unit_LFP,  replacements_LFP)

GHG_emissions_LFP.to_csv('tables/GHG_emissions_LFP.csv')
GHG_emissions_Tesla_LFP.to_csv('tables/GHG_emissions_Tesla_LFP.csv')

#Define variables for case analysis:

GHG_bat_unit_NMC = df_battery_data['Value'].loc['NMC manufacturing emissions'] #g CO2/kWh
replacements_NMC = df_battery_data['Value'].loc['NMC replacements']

#######################################################################################################

GHG_emissions_NMC = emissions_tools.emission(parameters).get_WTW(vehicle_model_results_NMC, GHG_bat_unit_NMC,  replacements_NMC)
GHG_emissions_Tesla_NMC = emissions_tools.emission(parameters_Tesla).get_WTW(vehicle_model_results_Tesla_NMC, GHG_bat_unit_NMC,  replacements_NMC)

GHG_emissions_NMC.to_csv('tables/GHG_emissions_NMC.csv')
GHG_emissions_Tesla_NMC.to_csv('tables/GHG_emissions_Tesla_NMC.csv')


############# Plot results ##########################
GHG_emissions_LFP = pd.read_csv('tables/GHG_emissions_LFP.csv')
GHG_emissions_Tesla_LFP = pd.read_csv('tables/GHG_emissions_Tesla_LFP.csv')

GHG_emissions_NMC = pd.read_csv('tables/GHG_emissions_NMC.csv')
GHG_emissions_Tesla_NMC = pd.read_csv('tables/GHG_emissions_Tesla_NMC.csv')

##GHGs emissions for diesel baseline:
GHG_emissions_diesel = pd.DataFrame({'Diesel Baseline (gCO2/mi)': [1507, 1180, 1081]}) # in gCO2/mi Source: Jones, R et al. (2023).Developing and Benchmarking a US Long-haul Drive Cycle for Vehicle Simulations, Costing and Emissions Analysis.
GHG_emissions_diesel.head()


#plots
fig, ax = plt.subplots(figsize=(14, 5))
bar_width = 0.15


bar1 = np.arange(len(GHG_emissions_LFP.index))*1.2
bar2 = [x + 1.2*bar_width for x in bar1]
bar3 = [x + 1.2*bar_width for x in bar2]
bar4 = [x + 1.2*bar_width for x in bar3]
bar5 = [x + 1.2*bar_width for x in bar4]


ax.bar(bar1,[0,0,0], width=bar_width, color='darkorange')
ax.bar(bar2, GHG_emissions_LFP['GHGs grid (gCO2/mi)'], label='Battery manufacturing', width=bar_width, color='darkorange')
ax.bar(bar3, GHG_emissions_NMC['GHGs grid (gCO2/mi)'], width=bar_width, color='darkorange')
ax.bar(bar4, GHG_emissions_Tesla_LFP['GHGs grid (gCO2/mi)'], width=bar_width, color='darkorange')
ax.bar(bar5, GHG_emissions_Tesla_NMC['GHGs grid (gCO2/mi)'], width=bar_width, color='darkorange')


plt1=ax.bar(bar1, GHG_emissions_diesel['Diesel Baseline (gCO2/mi)'], width=bar_width, bottom=[0,0,0], color='gray')
plt2=ax.bar(bar2, GHG_emissions_LFP['GHGs manufacturing (gCO2/mi)'], width=bar_width, label='Electricity generation',  bottom= GHG_emissions_LFP['GHGs grid (gCO2/mi)'], color='#F7D57D')
plt3=ax.bar(bar3, GHG_emissions_NMC['GHGs manufacturing (gCO2/mi)'], width=bar_width, bottom= GHG_emissions_NMC['GHGs grid (gCO2/mi)'], color='#F7D57D')
plt4=ax.bar(bar4, GHG_emissions_Tesla_LFP['GHGs manufacturing (gCO2/mi)'], width=bar_width, bottom= GHG_emissions_Tesla_LFP['GHGs grid (gCO2/mi)'], color='#F7D57D')
plt5=ax.bar(bar5, GHG_emissions_Tesla_NMC['GHGs manufacturing (gCO2/mi)'], width=bar_width, bottom= GHG_emissions_Tesla_NMC['GHGs grid (gCO2/mi)'], color='#F7D57D')

plt.text(bar1[1], 1.1*GHG_emissions_diesel['Diesel Baseline (gCO2/mi)'][1], 'DIESEL', ha='center', va='top')
plt.text(bar2[1], 1.1*(GHG_emissions_LFP['GHGs grid (gCO2/mi)'] + GHG_emissions_LFP['GHGs manufacturing (gCO2/mi)'])[1], 'LFP', ha='center', va='top')
plt.text(bar3[1], 1.1*(GHG_emissions_NMC['GHGs grid (gCO2/mi)'] + GHG_emissions_NMC['GHGs manufacturing (gCO2/mi)'])[1], 'NMC', ha='center', va='top')
plt.text(bar4[1], 1.4*(GHG_emissions_Tesla_LFP['GHGs grid (gCO2/mi)'] + GHG_emissions_Tesla_LFP['GHGs manufacturing (gCO2/mi)'])[1], 'Tesla\nLFP', ha='center', va='top')
plt.text(bar5[1], 1.4*(GHG_emissions_Tesla_NMC['GHGs grid (gCO2/mi)'] + GHG_emissions_Tesla_NMC['GHGs manufacturing (gCO2/mi)'])[1], 'Tesla\nNMC', ha='center', va='top')

ax.set_ylabel('Well-to-Wheel Emissions (gCO2/mi)', weight='bold')
ax.set_xticks([r + bar_width*2 for r in np.arange(len(GHG_emissions_LFP.index))*1.2])
ax.set_xticklabels(['Present', 'Mid term', 'Long term'],weight='bold')
ax.legend(loc='upper right')

plt.savefig('plots/wtw_emissions.png')

# **Costing analysis**

####****Input parameters for capital and operating unit costs****####

# Motor and inverter cost is given per unit of drivetrain power rating (Motor peak power)
# DC-DC converter cost is given per unit of auxiliary power rating  (Auxiliary loads)
# Insurance cost is per unit of capital cost of a single BET (no payload penalty included). We computed from reported insurance cost (0.1969 $/mi) for a BET vehicle cost (0.9933 $/mi). Source: https://publications.anl.gov/anlpubs/2021/05/167399.pdf
# Glider cost from Jones, R et al. (2023). Developing and Benchmarking a US Long-haul Drive Cycle forVehicle Simulations, Costing and Emissions Analysis

# DMM: Update these numbers for short-haul trucks in https://publications.anl.gov/anlpubs/2021/05/167399.pdf
# DMM: Insurance increases with cost of vehicle
capital_cost_unit = pd.DataFrame({'glider ($)': np.array(df_scenarios['Cost of glider'].iloc[0:3].astype(float)), 'motor and inverter ($/kW)': np.array(df_scenarios['Cost of motor and inverter'].iloc[0:3].astype(float)), 'DC-DC converter ($/kW)': np.array(df_scenarios['Cost of DC-DC converter'].iloc[0:3].astype(float))})

operating_cost_unit = pd.DataFrame({'maintenance & repair ($/mi)': np.array(df_scenarios['Maintenance and repair cost'].iloc[0:3].astype(float)), 'labor ($/mi)': np.array(df_scenarios['Labor cost'].iloc[0:3].astype(float)), 'insurance ($/mi)': np.array(df_scenarios['Insurance cost'].iloc[0:3].astype(float)), 'misc ($/mi)': np.array(df_scenarios[' Miscellaneous costs'].iloc[0:3].astype(float))})

####****Input parameters for case analysis****####
electricity_unit = np.array(df_scenarios['Electricity price'].iloc[0:3].astype(float))    # DMM: Electricity price will vary by region. The value here includes demand charges, etc. (retail price)
electricity_unit_Tesla = float(inputs_df[inputs_df['Value'] == 'Electricity rate (central) [$/kWh]']['PepsiCo']) * electricity_unit / electricity_unit[0]    # Assume the time projection of Tesla electricity price follows the same shape as nominal

SCC = np.array(df_scenarios['Social cost of carbon'].iloc[0:3].astype(float)) #social cost of carbon in $/ton CO2. Source: https://www.whitehouse.gov/wp-content/uploads/2021/02/TechnicalSupportDocument_SocialCostofCarbonMethaneNitrousOxide.pdf

battery_unit_cost_LFP = np.array(df_scenarios['LFP battery unit cost'].iloc[0:3].astype(float)) #LFP unit cost in $/kWh
TCS_LFP = costing_tools.cost(parameters).get_TCS(vehicle_model_results_LFP, capital_cost_unit, battery_unit_cost_LFP, operating_cost_unit, electricity_unit, replacements_LFP, GHG_emissions_LFP, SCC)
TCS_Tesla_LFP = costing_tools.cost(parameters_Tesla).get_TCS(vehicle_model_results_Tesla_LFP, capital_cost_unit, battery_unit_cost_LFP, operating_cost_unit, electricity_unit_Tesla, replacements_LFP, GHG_emissions_Tesla_LFP, SCC)

#print(TCS_LFP.head())
#print(TCS_Tesla_LFP.head())
TCS_LFP.to_csv('tables/TCS_LFP.csv')
TCS_Tesla_LFP.to_csv('tables/TCS_Tesla_LFP.csv')

battery_unit_cost_NMC = np.array(df_scenarios['NMC battery unit cost'].iloc[0:3].astype(float)) #NMC unit cost in $/kWh
TCS_NMC=costing_tools.cost(parameters).get_TCS(vehicle_model_results_NMC, capital_cost_unit, battery_unit_cost_NMC, operating_cost_unit, electricity_unit, replacements_NMC, GHG_emissions_NMC, SCC)
TCS_Tesla_NMC = costing_tools.cost(parameters_Tesla).get_TCS(vehicle_model_results_Tesla_NMC, capital_cost_unit, battery_unit_cost_NMC, operating_cost_unit, electricity_unit_Tesla, replacements_NMC, GHG_emissions_Tesla_NMC, SCC)

print(TCS_NMC.head()) #TCS in $/mi
print(TCS_Tesla_NMC.head())
TCS_NMC.to_csv('tables/TCS_NMC.csv')
TCS_Tesla_NMC.to_csv('tables/TCS_Tesla_NMC.csv')

#plot results

TCS_LFP = pd.read_csv('tables/TCS_LFP.csv')
TCS_Tesla_LFP = pd.read_csv('tables/TCS_Tesla_LFP.csv')

TCS_NMC = pd.read_csv('tables/TCS_NMC.csv')
TCS_Tesla_NMC = pd.read_csv('tables/TCS_Tesla_NMC.csv')

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.2


bar1 = np.arange(len(TCS_NMC.index))*1.2
bar2 = [x + 1.2*bar_width for x in bar1]
bar3 = [x + 1.2*bar_width for x in bar2]
bar4 = [x + 1.2*bar_width for x in bar3]

ax.bar(bar1, TCS_LFP['Total electricity ($/mi)'], label='Electricity', width=bar_width, color='#AA6127')
ax.bar(bar2, TCS_NMC['Total electricity ($/mi)'], width=bar_width, color='#AA6127')
ax.bar(bar3, TCS_Tesla_LFP['Total electricity ($/mi)'], width=bar_width, color='#AA6127')
ax.bar(bar4, TCS_Tesla_NMC['Total electricity ($/mi)'], width=bar_width, color='#AA6127')


ax.bar(bar1, TCS_LFP['Total labor ($/mi)'], width=bar_width, bottom=TCS_LFP['Total electricity ($/mi)'], color='#E48825')
ax.bar(bar2, TCS_NMC['Total labor ($/mi)'], width=bar_width, label='Labor',  bottom= TCS_NMC['Total electricity ($/mi)'], color='#E48825')
ax.bar(bar3, TCS_Tesla_LFP['Total labor ($/mi)'], width=bar_width, bottom=TCS_Tesla_LFP['Total electricity ($/mi)'], color='#E48825')
ax.bar(bar4, TCS_Tesla_NMC['Total labor ($/mi)'], width=bar_width, bottom=TCS_Tesla_NMC['Total electricity ($/mi)'], color='#E48825')


ax.bar(bar1, TCS_LFP['Other OPEXs ($/mi)'], width=bar_width, bottom=TCS_LFP['Total labor ($/mi)']+TCS_LFP['Total electricity ($/mi)'], color='#F0B05A')
ax.bar(bar2, TCS_NMC['Other OPEXs ($/mi)'], width=bar_width, label='Other OPEX',  bottom= TCS_NMC['Total labor ($/mi)']+TCS_NMC['Total electricity ($/mi)'], color='#F0B05A')
ax.bar(bar3, TCS_Tesla_LFP['Other OPEXs ($/mi)'], width=bar_width, bottom=TCS_Tesla_LFP['Total labor ($/mi)']+TCS_Tesla_LFP['Total electricity ($/mi)'], color='#F0B05A')
ax.bar(bar4, TCS_Tesla_NMC['Other OPEXs ($/mi)'], width=bar_width, bottom=TCS_Tesla_NMC['Total labor ($/mi)']+TCS_Tesla_NMC['Total electricity ($/mi)'], color='#F0B05A')

ax.bar(bar1, TCS_LFP['Total capital ($/mi)'], width=bar_width, bottom=TCS_LFP['Other OPEXs ($/mi)']+TCS_LFP['Total labor ($/mi)']+TCS_LFP['Total electricity ($/mi)'], color='#F7D57C')
ax.bar(bar2, TCS_NMC['Total capital ($/mi)'], width=bar_width, label='Capital',  bottom= TCS_NMC['Other OPEXs ($/mi)']+TCS_NMC['Total labor ($/mi)']+TCS_NMC['Total electricity ($/mi)'], color='#F7D57C')
ax.bar(bar3, TCS_Tesla_LFP['Total capital ($/mi)'], width=bar_width, bottom=TCS_Tesla_LFP['Other OPEXs ($/mi)']+TCS_Tesla_LFP['Total labor ($/mi)']+TCS_Tesla_LFP['Total electricity ($/mi)'], color='#F7D57C')
ax.bar(bar4, TCS_Tesla_NMC['Total capital ($/mi)'], width=bar_width, bottom=TCS_Tesla_NMC['Other OPEXs ($/mi)']+TCS_Tesla_NMC['Total labor ($/mi)']+TCS_Tesla_NMC['Total electricity ($/mi)'], color='#F7D57C')

plt1=ax.bar(bar1, TCS_LFP['GHGs emissions penalty ($/mi)'], width=bar_width, bottom=TCS_LFP['Total capital ($/mi)']+TCS_LFP['Other OPEXs ($/mi)']+TCS_LFP['Total labor ($/mi)']+TCS_LFP['Total electricity ($/mi)'], color='#7EC071')
plt2=ax.bar(bar2, TCS_NMC['GHGs emissions penalty ($/mi)'], width=bar_width, label='Carbon',  bottom= TCS_NMC['Total capital ($/mi)']+TCS_NMC['Other OPEXs ($/mi)']+TCS_NMC['Total labor ($/mi)']+TCS_NMC['Total electricity ($/mi)'], color='#7EC071')
plt3=ax.bar(bar3, TCS_Tesla_LFP['GHGs emissions penalty ($/mi)'], width=bar_width, bottom=TCS_Tesla_LFP['Total capital ($/mi)']+TCS_Tesla_LFP['Other OPEXs ($/mi)']+TCS_Tesla_LFP['Total labor ($/mi)']+TCS_Tesla_LFP['Total electricity ($/mi)'], color='#7EC071')
plt4=ax.bar(bar4, TCS_Tesla_NMC['GHGs emissions penalty ($/mi)'], width=bar_width, bottom=TCS_Tesla_NMC['Total capital ($/mi)']+TCS_Tesla_NMC['Other OPEXs ($/mi)']+TCS_Tesla_NMC['Total labor ($/mi)']+TCS_Tesla_NMC['Total electricity ($/mi)'], color='#7EC071')

plt.text(bar1[1], 1.1*TCS_LFP['TCS ($/mi)'][1], 'LFP', ha='center', va='top')
plt.text(bar2[1], 1.1*TCS_NMC['TCS ($/mi)'][1], 'NMC', ha='center', va='top')
plt.text(bar3[1], 1.2*TCS_Tesla_LFP['TCS ($/mi)'][1], 'Tesla\nLFP', ha='center', va='top')
plt.text(bar4[1], 1.2*TCS_Tesla_NMC['TCS ($/mi)'][1], 'Tesla\nNMC', ha='center', va='top')


ax.set_ylabel('TCS ($/mi)', weight='bold')
ax.set_xticks([(r + bar_width)*1.3 for r in np.arange(len(TCS_LFP.index))])
ax.set_xticklabels(['Present', 'Mid term', 'Long term'],weight='bold')
ax.legend(loc='upper right')
plt.savefig('plots/tcs.png')


