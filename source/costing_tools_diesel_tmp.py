# Import packages
import pandas as pd
import numpy as np
import scipy as scipy
from scipy import integrate
import os, io
import matplotlib.pyplot as plt

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)
from google.colab import files
from scipy.optimize import curve_fit
from operator import add

VMT=np.array([108000,120000,114000,105000,92000,81000,74000,67000,59000,52000]) #Vehicle miles traveled per year. Source: Burnham, A et al. (2021)
discountrate=0.07 #discount rate

parameters = share_parameters(VMT, discountrate)


####***Passing input parameters to all classes****####

class share_parameters:
  def __init__(self,VMT, discountrate):
    self.VMT = VMT
    self.discountrate=discountrate

class cost:
  def __init__(self, parameters):
    self.parameters = parameters

  def get_capital(self, capital_cost_unit, discountfactor):
    capital = capital_cost_unit['glider ($)'] + capital_cost_unit['aftertreatment ($)'] + capital_cost_unit['engine ($)'] + capital_cost_unit['transmission ($)'] + capital_cost_unit['fuel tank ($)'] + capital_cost_unit['WHR ($)']
    total_CAPEX= capital/self.parameters.VMT.sum()
    return total_CAPEX #in $ per mile

  def get_operating(self, fuel_economy, operating_cost_unit, diesel_unit, total_CAPEX,discountfactor):
    diesel = (diesel_unit/fuel_economy['Fuel economy (mpg)'])*np.sum(self.parameters.VMT* discountfactor)/self.parameters.VMT.sum()
    labor =  operating_cost_unit['labor ($/mi)']*np.sum(self.parameters.VMT* discountfactor)/self.parameters.VMT.sum()
    others_opex = (operating_cost_unit['maintenance & repair ($/mi)']+ operating_cost_unit['misc ($/mi)'] + operating_cost_unit['insurance ($/mi)'])*np.sum(self.parameters.VMT* discountfactor)/self.parameters.VMT.sum()
    total_OPEX=  diesel + labor + others_opex
    return total_OPEX, diesel, labor, others_opex  #in $ per mile

  def get_penalty_emissions(self, fuel_economy, GHG_emissions, SCC):
    #Given that future SCC are quite uncertain and that SCC are already in present value, we did not apply discounted cash flow for this cost component
    GHG_emissions['GHGs total (gCO2/mi)']= GHG_emissions['GHGs PTW (gCO2/gal)']/fuel_economy['Fuel economy (mpg)']+GHG_emissions['GHGs WTP (gCO2/mi)']
    penalty_emissions= GHG_emissions['GHGs total (gCO2/mi)']*SCC/1000000 #convert ton CO2
    print(GHG_emissions['GHGs total (gCO2/mi)'])
    return penalty_emissions #in $ per mile

  def get_TCS(self, capital_cost_unit,  fuel_economy, operating_cost_unit, diesel_unit, GHG_emissions, SCC):
    costs_total = pd.DataFrame(columns = ['Total capital ($/mi)', 'Total operating ($/mi)', 'Total fuel ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)', 'GHGs emissions penalty ($/mi)', 'TCS ($/mi)'])

    discountfactor=1/np.power(1+self.parameters.discountrate,np.linspace(0,9,10)) #life time of trucks is 10 years
    costs_total['Total capital ($/mi)']= cost(parameters).get_capital(capital_cost_unit,discountfactor)
    costs_total[ 'Total operating ($/mi)'], costs_total['Total fuel ($/mi)'], costs_total['Total labor ($/mi)'], costs_total['Other OPEXs ($/mi)']= cost(parameters).get_operating(fuel_economy, operating_cost_unit, diesel_unit, costs_total['Total capital ($/mi)'], discountfactor)
    costs_total['TCS ($/mi)'] = costs_total['Total capital ($/mi)'] + costs_total[ 'Total operating ($/mi)']
    return costs_total #in $ per mile


####****Input parameters for capital and operating unit costs****####


capital_cost_unit= pd.DataFrame({'glider ($)': [95000, 95000, 95000], 'aftertreatment ($)': [5782.4, 7296.95, 9117.98], 'engine ($)': [12117.16, 16244.6, 17981.16],
                                 'transmission ($)': [10250, 10250, 10250], 'fuel tank ($)': [2120, 2120, 2120], 'WHR ($)': [0, 5900, 5900]})

operating_cost_unit = pd.DataFrame({'maintenance & repair ($/mi)': [0.143, 0.143, 0.143], 'labor ($/mi)': [0.693, 0.693, 0.693],
                                    'insurance ($/mi)': [0.068, 0.068,0.068], 'misc ($/mi)': [0.057,0.057,0.057]})

diesel_unit=[2.5, 2.73, 3.13]; #USD/gallon
SCC=[51, 62, 85] #social cost of carbon in $/ton CO2. Source: https://www.whitehouse.gov/wp-content/uploads/2021/02/TechnicalSupportDocument_SocialCostofCarbonMethaneNitrousOxide.pdf
fuel_economy = pd.DataFrame({'Fuel economy (mpg)': [7.5100, 9.5919, 10.4639]})
GHG_emissions =  pd.DataFrame({'GHGs WTP (gCO2/mi)':[149.7, 117.2, 107.4], 'GHGs PTW (gCO2/gal)': [10190, 10190, 10190]})
TCS=cost(parameters).get_TCS(capital_cost_unit,  fuel_economy, operating_cost_unit, diesel_unit, GHG_emissions, SCC)
TCS.head() #TCS in $/mi



