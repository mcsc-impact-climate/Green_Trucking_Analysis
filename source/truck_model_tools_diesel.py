# Tools for truck model simulation
# Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader

import numpy as np
import scipy as scipy
from scipy import integrate
import pandas as pd

KG_PER_TON = 1000
KG_PER_LB = 0.453592
S_PER_H = 3600.
M_PER_MILE = 1609.34
W_PER_KW = 1000
DIESEL_BTU_PER_GAL = 128488 # From https://afdc.energy.gov/fuels/properties
BTU_PER_KWH = 3412.14
LB_PER_KG = 2.20462


class read_parameters:
  def __init__(self, f_truck_params, f_economy_params, f_constants, f_vmt):
    df_truck_params = pd.read_csv(f_truck_params, index_col=0)
    df_economy_params = pd.read_csv(f_economy_params, index_col=0)
    df_constants = pd.read_csv(f_constants, index_col=0)
    self.VMT = pd.read_csv(f_vmt, usecols=['Year', 'VMT (miles)'])
      
    # Weights and payloads
    self.m_ave_payload = float(df_truck_params['Value'].loc['Average payload'])
    self.m_max = float(df_truck_params['Value'].loc['Max gross vehicle weight'])
    self.m_truck = float(df_truck_params['Value'].loc['Diesel tractor weight']) * KG_PER_LB
    self.m = self.m_ave_payload + self.m_truck

    # Power consumption
    self.p_aux = float(df_truck_params['Value'].loc['Auxiliary power'])
    self.p_motor_max = float(df_truck_params['Value'].loc['Max motor power'])
    
    # Efficiencies
    self.eta_e = float(df_truck_params['Value'].loc['Engine efficiency'])
    self.eta_d = float(df_truck_params['Value'].loc['Drivetrain efficiency'])
    
    # Drag and resistance
    self.cd = float(df_truck_params['Value'].loc['Drag coefficient'])
    self.cr = float(df_truck_params['Value'].loc['Resistance coefficient'])
    self.a_cabin = float(df_truck_params['Value'].loc['Frontal cabin area'])
    
    # Constants
    self.g = float(df_constants['Value'].loc['Gravitational acceleration'])
    self.rho_air = float(df_constants['Value'].loc['Air density'])
    
    # Economy parameters
    self.eta_grid_transmission = float(df_economy_params['Value'].loc['Grid transmission efficiency'])
    self.discountrate = float(df_economy_params['Value'].loc['Discount rate'])

class share_parameters:
  def __init__(self,m_ave_payload, m_max, m_truck, m_guess, p_aux, p_motor_max, cd, cr, a_cabin, g, rho_air, eta_e, eta_d, eta_grid_transmission, VMT, discountrate):

    self.m_ave_payload=m_ave_payload
    self.m_max = m_max
    self.m_truck = m_truck
    self.p_aux=p_aux
    self.p_motor_max=p_motor_max

    self.cd = cd
    self.cr = cr
    self.a_cabin = a_cabin
    self.g = g
    self.rho_air = rho_air

    self.eta_e = eta_e
    self.eta_d = eta_d
    self.eta_grid_transmission = eta_grid_transmission

    self.VMT = VMT
    self.discountrate=discountrate
    
####**** Vehicle model and battery size****####

class truck_model:
  def __init__(self, parameters):
    self.parameters = parameters
    
  # Function to convert fuel consumption from kWh/mile to miles per gallon of diesel
  def kWh_per_mile_to_mpg(self, fuel_consumption):
    miles_per_kWh = 1/fuel_consumption
    miles_per_gallon = miles_per_kWh * DIESEL_BTU_PER_GAL / BTU_PER_KWH
    return miles_per_gallon

  ##Inputs: dataframe df with drive cycle data, m---> total truck mass (kg)
  ##outputs: fuel_consumption--->fuel consumption in kWh/mi, df ---> updated dataframe with the new variables (e.g. simulated vehicle speed)
  def get_power_requirement(self, df):
    # Set the GVW m assuming we're carrying the VIUS average payload
    m = self.parameters.m_truck + self.parameters.m_ave_payload
    
    v_drive_cycle = df['Vehicle speed (m/s)'].shift(-1)
    road_angle = df['Road angle']
    delta_t = df['Time (s)'].diff().fillna(0) #calculate time steps (delta time= 1 seconds for the US long haul drive cycle used). first data point Na is filled with zero
    simulated_vehicle_speeds = [0] #initialize variables for simulated vehicle speed as motor is limited
    
    power_request_motor_slopes, power_request_motor_intercepts = [],[]   # Slope and y_intercept of linear function fuel_consumption = slope * m + intercept

    for i in range(len(v_drive_cycle)-1):
      target_acceleration = v_drive_cycle[i] - simulated_vehicle_speeds[i] #required acceleration to match drive cycle in terms of vehicle speed\
      
      # Keeping this original code for reference because it shows the actual calculation of power request at the motor more clearly
#      fr = m*self.parameters.g*self.parameters.cr*np.cos(road_angle[i]) #force from rolling resistance in N
#      fg = m*self.parameters.g*np.sin(road_angle[i]) #force from gravitational in N
#      fd = self.parameters.rho_air*self.parameters.a_cabin*self.parameters.cd*np.power(simulated_vehicle_speeds[i],2)/2 #force from aerodynamic drag in N
#      maximum_acceleration = ((self.parameters.p_motor_max*self.parameters.eta_e*self.parameters.eta_d/simulated_vehicle_speeds[i]) - fr - fg - fd)/m if simulated_vehicle_speeds[i] > 0 else 1e9
#
#      a=min(target_acceleration,maximum_acceleration) #minimum acceleration between target acceleration to follow drive cycle versus maximum acceleration of truck at Pmax
#      fa=m*a
#
#      power_request_wheels = (fr + fg + fd + fa) * simulated_vehicle_speeds[i] #total power request at the wheels in W
#      power_request_motors.append(power_request_wheels/(self.parameters.eta_e*self.parameters.eta_d) if power_request_wheels > 0 else 0) #total power request at the motor in W
      
      ar = self.parameters.g*self.parameters.cr*np.cos(road_angle[i]) #acceleration from rolling resistance in N
      ag = self.parameters.g*np.sin(road_angle[i]) #acceleration from rolling resistance in N
      fd = self.parameters.rho_air*self.parameters.a_cabin*self.parameters.cd*np.power(simulated_vehicle_speeds[i], 2) / 2 #force from aerodynamic drag in N
      
      maximum_acceleration = ((self.parameters.p_motor_max*self.parameters.eta_e*self.parameters.eta_d/simulated_vehicle_speeds[i]) - ar*m - ag*m - fd)/m if simulated_vehicle_speeds[i] > 0 else 1e9
      a=min(target_acceleration,maximum_acceleration) #minimum acceleration between target acceleration to follow drive cycle versus maximum acceleration of truck at Pmax
      
      acc_request_wheels = (ar + ag + fd/m + a) * simulated_vehicle_speeds[i] #total acceleration request at the wheels in W
    
      # Find the slope and y-intercept for the approximate linear relationship power_request_motor = power_request_motor_slope * m + power_request_motor_intercept
      power_request_motor_slopes.append((ar + ag + a) * simulated_vehicle_speeds[i] / (self.parameters.eta_e*self.parameters.eta_d) if acc_request_wheels > 0 else 0)
      power_request_motor_intercepts.append(fd * simulated_vehicle_speeds[i] / (self.parameters.eta_e*self.parameters.eta_d) if acc_request_wheels > 0 else 0)
      
      simulated_vehicle_speeds.append(simulated_vehicle_speeds[i] + a * delta_t[i]) #update vehicle speed for next iteration

    power_request_motor_slopes.append(0)
    power_request_motor_intercepts.append(0)
    df['Simulated vehicle speed (m/s)'] = simulated_vehicle_speeds

    #fuel_consumption = np.trapz(df['Power request at the motor (W)'], df['Time (s)'])*2.7778*np.float_power(10,-7)/(np.trapz(df['Simulated vehicle speed (m/s)'], df['Time (s)'])/(1.609344*1000)) #energy consumption in kWh/mile
    
    slope = ( np.trapz(power_request_motor_slopes, df['Time (s)']) / np.trapz(df['Simulated vehicle speed (m/s)'], df['Time (s)']) ) * M_PER_MILE / (W_PER_KW * S_PER_H)  # Slope of linear motor power as a function of GVW, in kWh/kg
    
    y_intercept = ( np.trapz(power_request_motor_intercepts, df['Time (s)']) / np.trapz(df['Simulated vehicle speed (m/s)'], df['Time (s)']) ) * M_PER_MILE / (W_PER_KW * S_PER_H) # y-intercept of linear motor power as a function of GVW, in kWh
    
    #fuel_consumption = m * slope + y_intercept   # Energy economy, in kWh/mile
    
    # Integrate the tractor mass into the y_intercept to get fuel consumption = payload * slope + y_intercept_payload
    y_intercept_payload = y_intercept + self.parameters.m_truck * slope
    
    # Evaluate fuel consumption in kwh per mile for the average payload
    kwh_per_mile_ave_payload = self.parameters.m_ave_payload * slope + y_intercept_payload
    
    # Evaluate linear parameters in gal/mile
    slope_gpm = slope * BTU_PER_KWH / DIESEL_BTU_PER_GAL
    y_intercept_payload_gpm = y_intercept_payload * BTU_PER_KWH / DIESEL_BTU_PER_GAL
    
    # Now, evaluate the fuel consumption for the average payload in miles per gallon
    miles_per_gallon_ave_payload = 1 / (slope_gpm * self.parameters.m_ave_payload + y_intercept_payload_gpm)

    return slope_gpm, y_intercept_payload_gpm, miles_per_gallon_ave_payload
    
def extract_drivecycle_data(f_drivecycle):
    if f_drivecycle.endswith('.xlsx'):
        df = pd.read_excel(f_drivecycle) #drive cycle as a dataframe
    elif f_drivecycle.endswith('.csv'):
        df = pd.read_csv(f_drivecycle) #drive cycle as a dataframe

    else:
        extension = f_drivecycle.split('.')[-1]
        print(f'Cannot process drivecycle data for file ending in {extension}')
        return None
    df['Vehicle speed (m/s)'] = df['Vehicle speed (km/h)']*1000/3600 #vehicle speed converted from km/h to m/s
    df = df.drop(['Vehicle speed (km/h)'],axis=1) #remove column with vehicle speed in km/h
    df['Acceleration (m/s2)'] = df['Vehicle speed (m/s)'].diff()/df['Time (s)'].diff() #calculate acceleration in m/s2
    df['Acceleration (m/s2)'] = df['Acceleration (m/s2)'].fillna(0) #first data point as NaN, we replace with zero
    df['Road angle'] = df['Road grade (%)'].apply(lambda x: np.arctan(x / 100)) #convert road grade to road angle RG=100 tan(road angle)
    df['Cumulative distance (m)']= integrate.cumtrapz(df['Vehicle speed (m/s)'],df['Time (s)'], initial=0)
    
    return df

# Code to test the truck_model class
#parameters = read_parameters('data/diesel_daycab_truck_params.csv', 'data/default_economy_params.csv', 'data/constants.csv', 'data/default_vmt.csv')
#df_drivecycle = extract_drivecycle_data(f'data/pepsi_1_drive_cycle_27.csv')
#
#slope, y_intercept, fuel_consumption_ave_payload = truck_model(parameters).get_power_requirement(df_drivecycle)
#print(fuel_consumption_ave_payload)
