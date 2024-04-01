####****Cost analysis****####
#TCSs (in $ per vehicle mile travelled) are given in the corresponding present values of each scenario today, 2030 and 2050

#Inputs: Vehicle model results, number of replacements, capital and operating unit costs, GHG emissions, social cost of carbon and discount dactor
#Output: TCS, total operating costs per mile, total capital costs per mile, GHGs emissions penalty per mile

import pandas as pd
import numpy as np

class cost:
  def __init__(self, parameters):
    self.parameters = parameters

  def get_capital(self, vehicle_model_results, replacements, capital_cost_unit,battery_unit_cost,discountfactor):
    #We consider replacement of NMC battery in the 5th year of truck's lifetime
    # DMM: Costs are all per unit power rating, but if you find absolute cost that's fine too
    print()
    capital = capital_cost_unit['glider ($)'] + (capital_cost_unit['motor and inverter ($/kW)']*self.parameters.p_motor_max/1000) + (capital_cost_unit['DC-DC converter ($/kW)']*self.parameters.p_aux/1000) + ((1+(replacements*discountfactor[5]))*(battery_unit_cost*vehicle_model_results['Energy battery (kWh)']))
    total_CAPEX = vehicle_model_results['Payload penalty factor']*capital/self.parameters.VMT.sum()
    return total_CAPEX #in $ per mile

  def get_operating(self, vehicle_model_results, replacements, operating_cost_unit, electricity_unit, total_CAPEX,discountfactor):
    electricity = vehicle_model_results['Payload penalty factor']* electricity_unit*vehicle_model_results['Fuel economy (kWh/mi)']*np.sum(self.parameters.VMT* discountfactor)/self.parameters.VMT.sum()
    labor = vehicle_model_results['Payload penalty factor'] * operating_cost_unit['labor ($/mi)']*np.sum(self.parameters.VMT* discountfactor)/self.parameters.VMT.sum()
    # DMM: Maintenance cost may be less for EV truck (opportunity to narrow this down better relative to the source)
    others_opex = vehicle_model_results['Payload penalty factor']*(operating_cost_unit['maintenance & repair ($/mi)']+ operating_cost_unit['misc ($/mi)'] + (operating_cost_unit['insurance ($/mi)']*total_CAPEX/vehicle_model_results['Payload penalty factor']))*np.sum(self.parameters.VMT* discountfactor)/self.parameters.VMT.sum()
    total_OPEX=  electricity + labor + others_opex
    return total_OPEX, electricity, labor, others_opex  #in $ per mile

  def get_penalty_emissions(self, GHG_emissions, SCC):
    #Given that future SCC are quite uncertain and that SCC are already in present value, we did not apply discounted cash flow for this cost component
    penalty_emissions= GHG_emissions*SCC/1000000
    return penalty_emissions #in $ per mile

  def get_TCS(self, vehicle_model_results, capital_cost_unit, battery_unit_cost, operating_cost_unit, electricity_unit, replacements, GHG_emissions, SCC):
    costs_total = pd.DataFrame(columns = ['Total capital ($/mi)', 'Total operating ($/mi)', 'Total electricity ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)', 'GHGs emissions penalty ($/mi)', 'TCS ($/mi)'])

    discountfactor=1/np.power(1+self.parameters.discountrate,np.linspace(0,9,10)) #life time of trucks is 10 years
    costs_total['Total capital ($/mi)'] = cost(self.parameters).get_capital(vehicle_model_results, replacements, capital_cost_unit, battery_unit_cost, discountfactor)
    costs_total[ 'Total operating ($/mi)'], costs_total['Total electricity ($/mi)'], costs_total['Total labor ($/mi)'], costs_total['Other OPEXs ($/mi)']= cost(self.parameters).get_operating(vehicle_model_results, replacements, operating_cost_unit, electricity_unit, costs_total['Total capital ($/mi)'],discountfactor)
    costs_total['GHGs emissions penalty ($/mi)'] = cost(self.parameters).get_penalty_emissions(GHG_emissions['GHGs total (gCO2/mi)'], SCC)
    costs_total['TCS ($/mi)'] = costs_total['Total capital ($/mi)'] + costs_total[ 'Total operating ($/mi)'] + costs_total['GHGs emissions penalty ($/mi)']
    return costs_total #in $ per mile
