# **Emissions analysis**
# Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

##Evolution of carbon intensities for US electric grid.
##Carbon intensities from EIA for every five years and fitted to a exponential decay curve
carbon_intensity_EIA=np.array([370, 388, 314, 183, 163, 153, 146, 135])
timeline_fit=np.array([2020, 2021, 2025, 2030, 2035, 2040, 2045, 2050])

def mono_exp(x,a,b,c): #exponential decay curve
  return a*np.exp(-b*x)+c

def get_exp_params(timeline_fit, carbon_intensity=carbon_intensity_EIA):
    params, cov = curve_fit(mono_exp, timeline_fit, carbon_intensity, [1000000, 0.003, 0.1], maxfev=50000000) #fit carbon intensity data to exponential decay curve
    return params, cov

def CI_grid_cal(params, timeline):
  carbon_intensity_calc = mono_exp(timeline, params[0], params[1], params[2])
  return carbon_intensity_calc

####****Emissions analysis****####
class emission:
  def __init__(self, parameters):
    self.parameters = parameters

##Inputs: GHG battery manufacturing (GHG_bat_unit, g CO2/kWh), number of replacements (replacements), vehicle model results
##Output: GHGs emissions (gCO2/mi), Well to Wheel. We did not consider other emissions like PM2.5

  def get_CI_grid_projection(self, scenario='Present', grid_intensity_start=None, grid_intensity_start_year=None, start_year_truck_life=2024):
  
    # Establish the timeline to consider for exponentially decaying grid carbon intenstiy
    if scenario == 'Present':
        timeline = range(start_year_truck_life, start_year_truck_life+10)
    elif scenario == 'Mid term':
        timeline = range(2030, 2040)
    elif scenario == 'Long term':
        timeline = range(2050, 2060)
        
    # Get EIA projection for grid CI over the vehicle life for the entire US
    
    # Fit the EIA projection for emission intensity of the US grid to a decaying exponential
    params, cov = get_exp_params(timeline_fit, carbon_intensity_EIA)
    
    # Use the fit results to estimate the US grid emissions over the specified timeline
    CI_grid_projection = CI_grid_cal(params, timeline)
    
    # Mirror the structure of the VMT over all years of the truck's lifetime to populate the projected carbon intensity of the grid over those years
    VMT_grid_CI_df = self.parameters.VMT.copy()
    VMT_grid_CI_df['US Average Grid CI (g CO2 / kWh)'] = CI_grid_projection
    
    # Scale to the grid CI in the first year for the given region
    if grid_intensity_start:
        CI_grid_projection = CI_grid_projection * grid_intensity_start / mono_exp(grid_intensity_start_year, params[0], params[1], params[2])
    
    VMT_grid_CI_df['Grid CI (g CO2 / kWh)'] = CI_grid_projection
    VMT_grid_CI_df['Date Year'] = timeline
    
    return VMT_grid_CI_df

  # Function to visualize the projected grid emissions intensity for a sample input region
  def plot_CI_grid_projection(self, scenario='Present', grid_intensity_start=None, grid_intensity_start_year=None, start_year=None, label='', label_save=''):
    
    VMT_grid_CI_df = self.get_CI_grid_projection(scenario, grid_intensity_start, grid_intensity_start_year, start_year)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('Year', fontsize=22)
    ax.set_ylabel('Emission Intensity (g CO2e / kWh)', fontsize=21)
    ax.set_xticks(VMT_grid_CI_df['Year'])
    ax.plot(VMT_grid_CI_df['Year'], VMT_grid_CI_df['US Average Grid CI (g CO2 / kWh)'], label='US Average (EIA)', linewidth=2)
    ax.plot(VMT_grid_CI_df['Year'], VMT_grid_CI_df['Grid CI (g CO2 / kWh)'], label=label, linewidth=2)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'plots/grid_emission_intensity_projection_{label_save}.png')
    plt.savefig(f'plots/grid_emission_intensity_projection_{label_save}.pdf')

  def get_WTW(self, vehicle_model_results, GHG_bat_unit, replacements, scenario='Present', grid_intensity_start=None, grid_intensity_start_year=None, start_year=None):
            
    # Emissions are broken down into battery manufacturing and electricity production on the grid
    GHG_emissions = {}
    
    GHG_emissions['GHGs manufacturing (gCO2/mi)'] = (vehicle_model_results['Payload penalty factor'] * (1 + replacements) * vehicle_model_results['Battery capacity (kWh)'] * GHG_bat_unit) / (self.parameters.VMT['VMT (miles)']).sum()
    
    VMT_grid_CI_df = self.get_CI_grid_projection(scenario, grid_intensity_start, grid_intensity_start_year, start_year)

    # Average grid intensity, weighted by VMT for each year of the truck's lifetime
    average_grid_intensity = (VMT_grid_CI_df['VMT (miles)'] * VMT_grid_CI_df['Grid CI (g CO2 / kWh)']).sum() / VMT_grid_CI_df['VMT (miles)'].sum()

    # Multiply by the fuel economy to get the grid emissions per mile (accounting for payload penalty and grid transmission losses)
    GHG_emissions['GHGs grid (gCO2/mi)'] = vehicle_model_results['Payload penalty factor'] * vehicle_model_results['Fuel economy (kWh/mi)'] * average_grid_intensity
    
    # Sum battery manufacturing and grid emissions to obtain lifecycle emissions
    GHG_emissions['GHGs total (gCO2/mi)'] = GHG_emissions['GHGs manufacturing (gCO2/mi)'] + GHG_emissions['GHGs grid (gCO2/mi)']
    
    return GHG_emissions # in gCO2/mi
