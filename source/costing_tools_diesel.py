####****Cost analysis****####
#TCSs (in $ per vehicle mile travelled) are given in the corresponding present values of each scenario today, 2030 and 2050
# Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader

#Inputs: Vehicle model results, number of replacements, capital and operating unit costs, GHG emissions, social cost of carbon and discount dactor
#Output: TCS, total operating costs per mile, total capital costs per mile, GHGs emissions penalty per mile

W_PER_KW = 1000

import pandas as pd
import numpy as np

class cost:
  def __init__(self, parameters):
    self.parameters = parameters

  def get_capital(self, vehicle_model_results, capital_cost_unit, discountfactor, vehicle_purchase_price=None):
    if vehicle_purchase_price:
        capital = vehicle_purchase_price
    else:
        capital = capital_cost_unit['glider ($)'] + \
              (capital_cost_unit['aftertreatment ($)']) +  \
              (capital_cost_unit['transmission ($)']) +  \
              (capital_cost_unit['fuel tank ($)']) +  \
              (capital_cost_unit['WHR system ($)']) +  \
              (capital_cost_unit['engine ($/kW)'] * self.parameters.p_motor_max / W_PER_KW)
                  
    total_CAPEX = capital / self.parameters.VMT['VMT (miles)'].sum()
    return total_CAPEX #in $ per mile

  def get_operating(self, vehicle_model_results, operating_cost_unit, diesel_price, total_CAPEX, discountfactor):
  
    # Calculate the average discounted diesel price over the 10-year lifetime of the vehicle
    discounted_diesel_price = diesel_price * np.sum(self.parameters.VMT['VMT (miles)'] * discountfactor) / self.parameters.VMT['VMT (miles)'].sum()
    
    # Calculate the average cost of diesel per mile traveled
    fuel = discounted_diesel_price / vehicle_model_results['Fuel economy (miles/gal)']
    
    # Calculate the average cost of labor per mile
    labor = operating_cost_unit['labor ($/mi)'] * np.sum(self.parameters.VMT['VMT (miles)'] * discountfactor)/self.parameters.VMT['VMT (miles)'].sum()
        
    # DMM: Maintenance cost may be less for EV truck (opportunity to narrow this down better relative to the source)
    # Calculate total operating costs per mile
    opex_cost_per_mile = (operating_cost_unit['maintenance & repair ($/mi)']) + \
                         (operating_cost_unit['tolls ($/mi)']) + \
                         (operating_cost_unit['permits and licenses ($/mi)']) + \
                         (operating_cost_unit['insurance ($/mi-$)'] * total_CAPEX)
    others_opex = opex_cost_per_mile * np.sum(self.parameters.VMT['VMT (miles)'] * discountfactor) / self.parameters.VMT['VMT (miles)'].sum()
    total_OPEX = fuel + labor + others_opex
    return total_OPEX, fuel, labor, others_opex  #in $ per mile

  def get_TCO(self, vehicle_model_results, capital_cost_unit, operating_cost_unit, diesel_price, vehicle_purchase_price = None):
    costs_total = {} #'Total capital ($/mi)', 'Total operating ($/mi)', 'Total electricity ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)', 'GHGs emissions penalty ($/mi)', 'TCS ($/mi)'

    discountfactor = 1 / np.power(1 + self.parameters.discountrate, np.arange(10)) #life time of trucks is 10 years
    costs_total['Total capital ($/mi)'] = cost(self.parameters).get_capital(vehicle_model_results, capital_cost_unit, discountfactor, vehicle_purchase_price)
    
    costs_total['Total operating ($/mi)'], costs_total['Total fuel ($/mi)'], costs_total['Total labor ($/mi)'], costs_total['Other OPEXs ($/mi)'] = cost(self.parameters).get_operating(vehicle_model_results, operating_cost_unit, diesel_price, costs_total['Total capital ($/mi)'], discountfactor)
    
    costs_total['TCO ($/mi)'] = costs_total['Total capital ($/mi)'] + costs_total[ 'Total operating ($/mi)']
    
    return costs_total #in $ per mile


## Some basic code to test the functions defined above, should be commented out when not testing
#import data_collection_tools
#import costing_and_emissions_tools
#parameters, vehicle_model_results_dict = costing_and_emissions_tools.get_vehicle_model_results_diesel(m_payload_lb=50000, average_VMT=85000)
#truck_cost_data = data_collection_tools.read_truck_cost_data(truck_type='diesel')
#discountfactor = 1 / np.power(1 + parameters.discountrate, np.arange(10)) #life time of trucks is 10 years
#total_CAPEX = cost(parameters).get_capital(vehicle_model_results_dict, truck_cost_data['Capital Costs'], discountfactor)
#print(total_CAPEX)
#print(cost(parameters).get_operating(vehicle_model_results_dict, truck_cost_data['Operating Costs'], 4, total_CAPEX, discountfactor))
#print(cost(parameters).get_TCO(vehicle_model_results_dict, truck_cost_data['Capital Costs'], truck_cost_data['Operating Costs'], 4))
