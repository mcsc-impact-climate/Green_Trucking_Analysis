

####****Cost analysis****####
#TCSs (in $ per vehicle mile travelled) are given in the corresponding present values of each scenario today, 2030 and 2050
# Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader

#Inputs: Vehicle model results, number of replacements, capital and operating unit costs, GHG emissions, social cost of carbon and discount dactor
#Output: TCS, total operating costs per mile, total capital costs per mile, GHGs emissions penalty per mile

import pandas as pd
import numpy as np

class cost:
  def __init__(self, parameters):
    self.parameters = parameters

  def get_capital(self, vehicle_model_results, replacements, capital_cost_unit, battery_unit_cost, discountfactor, vehicle_purchase_price=False):
    # DMM: Costs are all per unit power rating, but if you find absolute cost that's fine too
    # DMM: The motor and inverter cost might be underestimated for Tesla Semi because there are actually 3 smaller motors rather than 1 large motor
    if vehicle_purchase_price:
        capital = vehicle_purchase_price
    else:
        capital = capital_cost_unit['glider ($)'] + \
              (capital_cost_unit['motor and inverter ($/kW)'] * self.parameters.p_motor_max / 1000.) +  \
              (capital_cost_unit['DC-DC converter ($/kW)'] * self.parameters.p_aux / 1000.)
    
    # Cost to replace the battery
    battery_cost = (1 + replacements * discountfactor[5]) * battery_unit_cost * vehicle_model_results['Battery capacity (kWh)']
    capital = capital + battery_cost
    
    total_CAPEX = vehicle_model_results['Payload penalty factor'] * capital / self.parameters.VMT['VMT (miles)'].sum()
    return total_CAPEX #in $ per mile

  def get_operating(self, vehicle_model_results, operating_cost_unit, electricity_unit_by_year, total_CAPEX, discountfactor):
  
    # Calculate the average discounted electricity rate over the 10-year lifetime of the vehicle
    average_discounted_electricity_rate = np.sum(self.parameters.VMT['VMT (miles)'] * electricity_unit_by_year * discountfactor) / self.parameters.VMT['VMT (miles)'].sum()
    
    # Calculate the average cost of electricity per mile traveled, accounting for payload penalty
    electricity = vehicle_model_results['Payload penalty factor'] * vehicle_model_results['Fuel economy (kWh/mi)'] * average_discounted_electricity_rate
    
    # Calculate the average cost of labor per mile
    labor = vehicle_model_results['Payload penalty factor'] * operating_cost_unit['labor ($/mi)'] * np.sum(self.parameters.VMT['VMT (miles)'] * discountfactor)/self.parameters.VMT['VMT (miles)'].sum()
        
    # DMM: Maintenance cost may be less for EV truck (opportunity to narrow this down better relative to the source)
    # Calculate total operating costs per mile
    opex_cost_per_mile = vehicle_model_results['Payload penalty factor'] * ( \
                         operating_cost_unit['maintenance & repair ($/mi)'] + \
                         operating_cost_unit['tolls ($/mi)'] + \
                         operating_cost_unit['permits and licenses ($/mi)'] ) + \
                         operating_cost_unit['insurance ($/mi-$)'] * total_CAPEX
    others_opex = opex_cost_per_mile * np.sum(self.parameters.VMT['VMT (miles)'] * discountfactor) / self.parameters.VMT['VMT (miles)'].sum()
    total_OPEX = electricity + labor + others_opex
    return total_OPEX, electricity, labor, others_opex  #in $ per mile

  def get_TCO(self, vehicle_model_results, capital_cost_unit, battery_unit_cost, operating_cost_unit, electricity_unit_by_year, replacements, vehicle_purchase_price = None):
    costs_total = {} #'Total capital ($/mi)', 'Total operating ($/mi)', 'Total electricity ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)', 'GHGs emissions penalty ($/mi)', 'TCS ($/mi)'

    discountfactor = 1 / np.power(1 + self.parameters.discountrate, np.arange(10)) #life time of trucks is 10 years
    costs_total['Total capital ($/mi)'] = cost(self.parameters).get_capital(vehicle_model_results, replacements, capital_cost_unit, battery_unit_cost, discountfactor, vehicle_purchase_price)
    
    costs_total['Total operating ($/mi)'], costs_total['Total electricity ($/mi)'], costs_total['Total labor ($/mi)'], costs_total['Other OPEXs ($/mi)'] = cost(self.parameters).get_operating(vehicle_model_results, operating_cost_unit, electricity_unit_by_year, costs_total['Total capital ($/mi)'], discountfactor)
    
    costs_total['TCO ($/mi)'] = costs_total['Total capital ($/mi)'] + costs_total[ 'Total operating ($/mi)']
    
    return costs_total #in $ per mile


## Some basic code to test the functions defined above, should be commented out when not testing
#import data_collection_tools
#import costing_and_emissions_tools
#parameters, vehicle_model_results_dict = costing_and_emissions_tools.get_vehicle_model_results(m_payload_lb=50000, average_VMT=190000)
#truck_cost_data = data_collection_tools.read_truck_cost_data(truck_type='EV')
#discountfactor = 1 / np.power(1 + parameters.discountrate, np.arange(10)) #life time of trucks is 10 years
#total_CAPEX = cost(parameters).get_capital(vehicle_model_results_dict, 0, truck_cost_data['Capital Costs'], truck_cost_data['Battery Unit Cost ($/kWh)'], discountfactor)
#print(total_CAPEX)
#print(cost(parameters).get_operating(vehicle_model_results_dict, truck_cost_data['Operating Costs'], 0.24, total_CAPEX, discountfactor))
