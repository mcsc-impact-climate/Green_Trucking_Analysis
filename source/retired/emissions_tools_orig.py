# **Emissions analysis**

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

#Carbon emissions intensity of the US grid over time from various models and policy scenarios, and fitted curve from EIA data

data_CI=[
 {2020:352, 2021: 365, 2025: np.nan,2030: 168, 2035:90, 2040:60, 2045:44, 2050:34}, ##IEA STEPS
 {2020:352, 2021: 365, 2025: np.nan, 2030: 107, 2035: 15, 2040: -3, 2045:-6, 2050:-7}, ##IEA APS
 {2020:395, 2021: np.nan, 2025: 304, 2030:243, 2035:199, 2040:193, 2045:192, 2050:192}, #EPPA Paris 2C scenario
 {2020:395, 2021: np.nan, 2025:234, 2030:168, 2035:130, 2040:70, 2045:50, 2050:36}, ##EPPA Accelerated actions
  {2020: 370, 2021: 388, 2025: 314, 2030: 183, 2035:163, 2040: 153, 2045:146, 2050:135} ##EIA
          ]
CI=pd.DataFrame(data_CI, index=['IEA STEPS', 'IEA APS','EPPA Paris 2C', 'EPPA Accelerated', 'EIA'])
carbon_intensity_calc=CI_grid_cal(timeline)

###plots
#plt.plot(timeline, carbon_intensity_calc, ':',linewidth=3, color='#D37416', label='Fitted curve to EIA'), plt.xlabel('Years'), plt.ylabel('Grid carbon intensity ($gCO_2/kWh$)')
#plt.scatter(*zip(*sorted(data_CI[0].items())), color='#A1AE6F', marker="o", label='IEA STEPS')
#plt.scatter(*zip(*sorted(data_CI[1].items())), color='#75794E', marker="o", label='IEA APS')
#plt.scatter(*zip(*sorted(data_CI[2].items())), color='#678E47', marker="o", label='EPPA Paris $2^o$C')
#plt.scatter(*zip(*sorted(data_CI[3].items())), color='#4E8B78', marker="o", label='EPPA accelerated actions')
#plt.scatter(*zip(*sorted(data_CI[4].items())), color='#FFBC4A', marker="o", edgecolor='#D37416', linewidth=2, label='EIA')
#
#plt.fill_between(timeline, carbon_intensity_calc-CI.std(), carbon_intensity_calc+CI.std(),alpha=0.1, facecolor='#DACC75')
#plt.legend()
#plt.savefig('plots/Grid_carbon_intensity.png')

####****Emissions analysis****####
class emission:
  def __init__(self, parameters):
    self.parameters = parameters

##Inputs: GHG battery manufacturing (GHG_bat_unit, g CO2/kWh), number of replacements (replacements), vehicle model results
##Output: GHGs emissions (gCO2/mi), Well to Wheel. We did not consider other emissions like PM2.5

  def get_WTW(self, vehicle_model_results, GHG_bat_unit, replacements):
    GHG_emissions = pd.DataFrame(columns = ['GHGs manufacturing (gCO2/mi)', 'GHGs grid (gCO2/mi)', 'GHGs total (gCO2/mi)'])
    GHG_emissions['GHGs manufacturing (gCO2/mi)'] = (vehicle_model_results['Payload penalty factor']*(1+ replacements)*vehicle_model_results['Energy battery (kWh)']*GHG_bat_unit)/self.parameters.VMT.sum()
    
    CI_grid_miles = pd.DataFrame({'CI_grid_miles':[sum(xi * yi for xi, yi in zip(self.parameters.VMT, y)) for y in [CI_grid_cal(range(2020,2030)), CI_grid_cal(range(2030,2040)), CI_grid_cal(range(2050,2060))]]})
    GHG_emissions['GHGs grid (gCO2/mi)']=vehicle_model_results['Payload penalty factor']*vehicle_model_results['Fuel economy (kWh/mi)']*CI_grid_miles['CI_grid_miles']/(self.parameters.eta_grid_transmission*self.parameters.VMT.sum())
    GHG_emissions['GHGs total (gCO2/mi)'] =  GHG_emissions['GHGs manufacturing (gCO2/mi)'] + GHG_emissions['GHGs grid (gCO2/mi)'] # WTW emissions
    return GHG_emissions # in gCO2/mi
