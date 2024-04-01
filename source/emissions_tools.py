# **Emissions analysis**
# Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

##Evolution of carbon intensities for US electric grid.
##Carbon intensities from EIA for every five years and fitted to a exponential decay curve
carbon_intensity_EIA=np.array([370, 388, 314, 183, 163, 153, 146, 135])
timeline=np.array([2020, 2021, 2025, 2030, 2035, 2040, 2045, 2050])

def mono_exp(x,a,b,c): #exponential decay curve
  return a*np.exp(-b*x)+c

params,cov=curve_fit(mono_exp, timeline, carbon_intensity_EIA, [1000000,0.003, 0.1],maxfev=50000000) #fit data to exponential decay curve

def CI_grid_cal(timeline):
  carbon_intensity_calc= mono_exp(timeline,params[0],params[1],params[2])
  return carbon_intensity_calc

#Carbon emissions intensity (g CO2 / kWh) of the US grid over time from various models and policy scenarios, and fitted curve from EIA data
data_CI=[
 {2020:352, 2021: 365, 2025: np.nan,2030: 168, 2035:90, 2040:60, 2045:44, 2050:34}, ##IEA STEPS
 {2020:352, 2021: 365, 2025: np.nan, 2030: 107, 2035: 15, 2040: -3, 2045:-6, 2050:-7}, ##IEA APS
 {2020:395, 2021: np.nan, 2025: 304, 2030:243, 2035:199, 2040:193, 2045:192, 2050:192}, #EPPA Paris 2C scenario
 {2020:395, 2021: np.nan, 2025:234, 2030:168, 2035:130, 2040:70, 2045:50, 2050:36}, ##EPPA Accelerated actions
  {2020: 370, 2021: 388, 2025: 314, 2030: 183, 2035:163, 2040: 153, 2045:146, 2050:135} ##EIA
          ]
CI=pd.DataFrame(data_CI, index=['IEA STEPS', 'IEA APS','EPPA Paris 2C', 'EPPA Accelerated', 'EIA'])
carbon_intensity_calc=CI_grid_cal(timeline)

####****Emissions analysis****####
class emission:
  def __init__(self, parameters):
    self.parameters = parameters

##Inputs: GHG battery manufacturing (GHG_bat_unit, g CO2/kWh), number of replacements (replacements), vehicle model results
##Output: GHGs emissions (gCO2/mi), Well to Wheel. We did not consider other emissions like PM2.5

  def get_CI_grid_projection(self, scenario='Present', grid_intensity_start=None, start_year=None):
  
    # Establish the timeline to consider for exponentially decaying grid carbon intenstiy
    timeline_start_year = 2020
    if start_year:
        timeline_start_year = start_year
    if scenario == 'Present':
        timeline = range(start_year, start_year+10)
    elif scenario == 'Mid term':
        timeline = range(2030, 2040)
    elif scenario == 'Long term':
        timeline = range(2050, 2060)
        
    # Get EIA projection for grid CI over the vehicle life for the entire US
    VMT_grid_CI_df = self.parameters.VMT.copy()
    CI_grid_projection = CI_grid_cal(timeline)
    
    VMT_grid_CI_df['US Average Grid CI (g CO2 / kWh)'] = CI_grid_projection
    
    # Scale to the grid CI in the first year for the given region
    if grid_intensity_start:
        CI_grid_projection = CI_grid_projection * grid_intensity_start / CI_grid_projection[0]
    
    VMT_grid_CI_df['Grid CI (g CO2 / kWh)'] = CI_grid_projection
    
    return VMT_grid_CI_df

  # Function to visualize the projected grid emissions intensity for a sample input region
  def plot_CI_grid_projection(self, scenario='Present', grid_intensity_start=None, start_year=None, label='', label_save=''):
    
    VMT_grid_CI_df = self.get_CI_grid_projection(scenario, grid_intensity_start, start_year)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Grid Emission Intensity (g CO2e / kWh)', fontsize=18)
    ax.set_xticks(VMT_grid_CI_df['Year'])
    ax.plot(VMT_grid_CI_df['Year'], VMT_grid_CI_df['US Average Grid CI (g CO2 / kWh)'], label='US Average (EIA)')
    ax.plot(VMT_grid_CI_df['Year'], VMT_grid_CI_df['Grid CI (g CO2 / kWh)'], label=label)
    ax.legend(fontsize=16)
    plt.savefig(f'plots/grid_emission_intensity_projection_{label_save}.png')

  def get_WTW(self, vehicle_model_results, GHG_bat_unit, replacements, scenario='Present', grid_intensity_start=None, start_year=None):
            
    # Emissions are broken down into battery manufacturing and electricity production on the grid
    GHG_emissions = {}
    
    GHG_emissions['GHGs manufacturing (gCO2/mi)'] = (vehicle_model_results['Payload penalty factor'] * (1 + replacements) * vehicle_model_results['Battery capacity (kWh)'] * GHG_bat_unit) / (self.parameters.VMT['VMT (miles)']).sum()
    
    VMT_grid_CI_df = self.get_CI_grid_projection(scenario, grid_intensity_start, start_year)

    average_grid_intensity = (VMT_grid_CI_df['VMT (miles)'] * VMT_grid_CI_df['Grid CI (g CO2 / kWh)']).sum() / VMT_grid_CI_df['VMT (miles)'].sum()

    # Multiply by the fuel economy to get the grid emissions per mile (accounting for payload penalty and grid transmission losses)
    GHG_emissions['GHGs grid (gCO2/mi)'] = vehicle_model_results['Payload penalty factor'] * vehicle_model_results['Fuel economy (kWh/mi)'] * average_grid_intensity
    
    GHG_emissions['GHGs total (gCO2/mi)'] = GHG_emissions['GHGs manufacturing (gCO2/mi)'] + GHG_emissions['GHGs grid (gCO2/mi)'] # WTW emissions
    
    return GHG_emissions # in gCO2/mi
