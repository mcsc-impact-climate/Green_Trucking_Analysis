# Tools for truck model simulation
# Note: Code adapted by Danika MacDonell from a colab notebook written by Kariana Moreno Sader

import numpy as np
import scipy as scipy
from scipy import integrate
import pandas as pd

KG_PER_TON = 1000
KG_PER_LB = 0.453592
M_PER_MILE = 1609.34
W_PER_KW = 1000
S_PER_H = 3600.

class read_parameters:
  def __init__(self, f_truck_params, f_economy_params, f_constants, f_vmt):
    df_truck_params = pd.read_csv(f_truck_params, index_col=0)
    df_economy_params = pd.read_csv(f_economy_params, index_col=0)
    df_constants = pd.read_csv(f_constants, index_col=0)
    self.VMT = pd.read_csv(f_vmt, usecols=['Year', 'VMT (miles)'])
      
    # Weights and payloads
    self.m_ave_payload = float(df_truck_params['Value'].loc['Average payload'])
    self.m_max = float(df_truck_params['Value'].loc['Max gross vehicle weight'])
    self.m_truck_no_bat = ( float(df_truck_params['Value'].loc['Diesel tractor weight']) + float(df_truck_params['Value'].loc['Weight of motor, inverter, electronics']) - float(df_truck_params['Value'].loc['Engine weight']) - float(df_truck_params['Value'].loc['Emission control system weight']) - float(df_truck_params['Value'].loc['Fuel system weight']) ) * KG_PER_LB

    # Power consumption
    self.p_aux = float(df_truck_params['Value'].loc['Auxiliary power'])
    self.p_motor_max = float(df_truck_params['Value'].loc['Max motor power'])
    
    # Efficiencies
    self.eta_i = float(df_truck_params['Value'].loc['Inverter efficiency'])
    self.eta_m = float(df_truck_params['Value'].loc['Motor efficiency'])
    self.eta_gs = float(df_truck_params['Value'].loc['Gear efficiency'])
    self.eta_rb = float(df_truck_params['Value'].loc['Regenerative braking efficiency'])
    
    # Depth of discharge
    self.DoD = float(df_truck_params['Value'].loc['Depth of discharge'])
    
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
  def __init__(self,m_ave_payload, m_max, m_truck_no_bat, p_aux, p_motor_max, cd, cr, a_cabin, g, rho_air, DoD, eta_i, eta_m, eta_gs, eta_rb, eta_grid_transmission, VMT, discountrate):

    self.m_ave_payload=m_ave_payload
    self.m_max = m_max
    self.m_truck_no_bat=m_truck_no_bat
    self.p_aux=p_aux
    self.p_motor_max=p_motor_max

    self.cd = cd
    self.cr = cr
    self.a_cabin = a_cabin
    self.g = g
    self.rho_air = rho_air
    self.DoD = DoD

    self.eta_i = eta_i
    self.eta_m = eta_m
    self.eta_gs = eta_gs
    self.eta_rb = eta_rb
    self.eta_grid_transmission = eta_grid_transmission

    self.VMT = VMT
    self.discountrate=discountrate
    
####**** Vehicle model and battery size****####

class truck_model:
  def __init__(self, parameters):
    self.parameters = parameters

  def get_linear_energy_economy_coefs(self, df, eta_battery, e_density, e_bat):
    """
    Uses the physics-based truck model to evaluate linear coefficients of energy economy as a function of payload for a Semi EV truck.
    
    Parameters
    ----------
    df (Pandas DataFrame): Dataframe containing the drivecycle.

    Returns
    -------
    slope (float): Slope of gallons per mile vs. payload
    y_intercept (float): Y-intercept of gallons per mile vs. payload
    kwh_per_mile_ave_payload: Miles per gallon evaluated for the average payload.
    """
    m_bat = e_bat*KG_PER_TON/e_density
    m = self.parameters.m_truck_no_bat + m_bat + self.parameters.m_ave_payload
    v_drive_cycle = df['Vehicle speed (m/s)'].shift(-1)
    road_angle = df['Road angle']
    delta_t = df['Time (s)'].diff().fillna(0) #calculate time steps (delta time= 1 seconds for the US long haul drive cycle used). first data point Na is filled with zero
    simulated_vehicle_speeds = [0] #initialize variables for simulated vehicle speed as motor is limited
    power_request_motor_slopes, power_request_motor_intercepts, power_request_motors = [],[],[]   # Slope and y_intercept of linear function fuel_consumption = slope * m + intercept

    for i in range(len(v_drive_cycle)-1):
      target_acceleration = v_drive_cycle[i] - simulated_vehicle_speeds[i] #required acceleration to match drive cycle in terms of vehicle speed
      ar = self.parameters.g*self.parameters.cr*np.cos(road_angle[i]) #force from rolling resistance in N
      ag = self.parameters.g*np.sin(road_angle[i]) #force from gravitational in N
      fd = self.parameters.rho_air*self.parameters.a_cabin*self.parameters.cd*np.power(simulated_vehicle_speeds[i], 2) / 2 #force from aerodynamic drag in N
      maximum_acceleration = ((self.parameters.p_motor_max*self.parameters.eta_i*self.parameters.eta_m*self.parameters.eta_gs/simulated_vehicle_speeds[i]) - ar*m - ag*m - fd)/m if simulated_vehicle_speeds[i] >0 else 1e9

      a=min(target_acceleration,maximum_acceleration) #minimum acceleration between target acceleration to follow drive cycle versus maximum acceleration of truck at Pmax
      
      acc_request_wheels = (ar + ag + fd/m + a) * simulated_vehicle_speeds[i] #total acceleration request at the wheels in W
      
      # Find the slope and y-intercept for the approximate linear relationship power_request_motor = power_request_motor_slope * m + power_request_motor_intercept
      power_request_motor_slopes.append((ar + ag + a) * simulated_vehicle_speeds[i] / (self.parameters.eta_i*self.parameters.eta_m*self.parameters.eta_gs*eta_battery) if acc_request_wheels > 0 else (ar + ag + a) * simulated_vehicle_speeds[i] * self.parameters.eta_rb)
      power_request_motor_intercepts.append(fd * simulated_vehicle_speeds[i] / (self.parameters.eta_i*self.parameters.eta_m*self.parameters.eta_gs*eta_battery) + self.parameters.p_aux/eta_battery if acc_request_wheels > 0 else fd * simulated_vehicle_speeds[i] * self.parameters.eta_rb + self.parameters.p_aux/eta_battery)
      
      power_request_motors.append(power_request_motor_slopes[-1]*m + power_request_motor_intercepts[-1])
      #print(f"power request to battery from linear coefs: {power_request_motors[-1]}")
      
      simulated_vehicle_speeds.append(simulated_vehicle_speeds[i]+a*delta_t[i]) #update vehicle speed for next iteration
    
    power_request_motor_slopes.append(0)
    power_request_motor_slopes = np.asarray(power_request_motor_slopes)
    power_request_motor_intercepts.append(0)
    power_request_motor_intercepts = np.asarray(power_request_motor_intercepts)
    power_request_motors.append(0)
    power_request_motors = np.asarray(power_request_motors)
    
    df['Simulated vehicle speed (m/s)'] = simulated_vehicle_speeds
    cSpeed = (df['Simulated vehicle speed (m/s)'] < 30) & (df['Simulated vehicle speed (m/s)'] >= 0)  # Exclude events with unphysical speeds
    cMotorMax = power_request_motors > -self.parameters.p_motor_max    # Exclude events for which the regenerative braking power exceeds the motor max
    slope = ( np.trapz(power_request_motor_slopes[cSpeed&cMotorMax], df['Time (s)'][cSpeed&cMotorMax]) / np.trapz(df['Simulated vehicle speed (m/s)'][cSpeed&cMotorMax], df['Time (s)'][cSpeed&cMotorMax]) ) * M_PER_MILE / (W_PER_KW * S_PER_H)  # Slope of linear motor power as a function of GVW, in kWh/kg
    
    y_intercept = ( np.trapz(power_request_motor_intercepts[cSpeed&cMotorMax], df['Time (s)'][cSpeed&cMotorMax]) / np.trapz(df['Simulated vehicle speed (m/s)'][cSpeed&cMotorMax], df['Time (s)'][cSpeed&cMotorMax]) ) * M_PER_MILE / (W_PER_KW * S_PER_H) # y-intercept of linear motor power as a function of GVW, in kWh
    
    # Integrate the tractor mass into the y_intercept to get fuel consumption = payload * slope + y_intercept_payload
    y_intercept_payload = y_intercept + (self.parameters.m_truck_no_bat + m_bat) * slope
    
    # Evaluate fuel consumption in kwh per mile for the average payload
    kwh_per_mile_ave_payload = self.parameters.m_ave_payload * slope + y_intercept_payload

    return slope, y_intercept_payload, kwh_per_mile_ave_payload

    
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
    df['Cumulative distance (m)']= integrate.cumulative_trapezoid(df['Vehicle speed (m/s)'],df['Time (s)'], initial=0)
    
    return df
    
####****Input parameters for payload penalty analysis****####
class payload:
  def __init__(self, parameters):
    self.parameters = parameters

##Inputs: m_bat--> battery mass in kg from truck model analysis, payload_distribution ---> database of payload distributions in the US (VIUS data 2002), self--> parameters, alpha --> parameter range from 0 to 2, base case =1
##Output: payload_Penalty --> penalty factor (number of additional trucks to be equivalent with diesel)

  def get_penalty(self, payload_distribution, m_bat, alpha):
    payload_max = self.parameters.m_max - m_bat - self.parameters.m_truck_no_bat #payload+trailer
    payload_distribution['Payload loss (kg)'] = payload_distribution['Payload (kg)'].apply(lambda x: np.maximum(x-payload_max,0))
    payload_penalty = 1 + (alpha*payload_distribution['Payload loss (kg)'].mean())/payload_max
    return payload_penalty
