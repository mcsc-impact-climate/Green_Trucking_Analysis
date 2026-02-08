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
    self.eta_t = float(df_truck_params['Value'].loc['Transmission efficiency'])
    
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
  def __init__(self,m_ave_payload, m_max, m_truck, m_guess, p_aux, p_motor_max, cd, cr, a_cabin, g, rho_air, eta_e, eta_t, eta_grid_transmission, VMT, discountrate):

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
    self.eta_t = eta_t
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
  ##outputs: fuel_consumption--->fuel consumption in kWh/mi, mpg ---> miles per gallon, df ---> updated dataframe with the new variables (e.g. simulated vehicle speed)
  def get_power_requirement(self, df, m):
    """
    Uses the physics-based truck model to evaluate fuel economy for a diesel truck
    on a specific drivecycle and gross vehicle mass.
    
    Parameters
    ----------
    df (Pandas DataFrame): Dataframe containing the drivecycle.
    m (float): Total truck mass (kg).

    Returns
    -------
    df (Pandas DataFrame): Updated dataframe with simulated signals.
    fuel_consumption (float): Fuel consumption in kWh/mile.
    mpg (float): Fuel economy in miles per gallon.
    """
    v_drive_cycle = df['Vehicle speed (m/s)'].shift(-1)
    road_angle = df['Road angle']
    delta_t = df['Time (s)'].diff().fillna(0)
    simulated_vehicle_speed, power_request_motor = [0], []

    for i in range(len(v_drive_cycle) - 1):
      dt = delta_t.iloc[i]
      target_speed = v_drive_cycle.iloc[i]
      target_acceleration = (target_speed - simulated_vehicle_speed[i]) / dt if dt > 0 else 0.0

      fr = m * self.parameters.g * self.parameters.cr * np.cos(road_angle.iloc[i])
      fg = m * self.parameters.g * np.sin(road_angle.iloc[i])
      fd = self.parameters.rho_air * self.parameters.a_cabin * self.parameters.cd * np.power(simulated_vehicle_speed[i], 2) / 2

      speed_for_power = simulated_vehicle_speed[i] if simulated_vehicle_speed[i] > 0 else 0.1
      maximum_acceleration = ((self.parameters.p_motor_max * self.parameters.eta_e * self.parameters.eta_t / speed_for_power) - fr - fg - fd) / m
      a = min(target_acceleration, maximum_acceleration)

      simulated_vehicle_speed.append(simulated_vehicle_speed[i] + a * dt)

      fa = m * a
      power_request_wheels = (fr + fg + fd + fa) * simulated_vehicle_speed[i]
      power_request_motor.append(power_request_wheels / (self.parameters.eta_e * self.parameters.eta_t) if power_request_wheels > 0 else 0)

    power_request_motor.append(0)
    df['Simulated vehicle speed (m/s)'] = simulated_vehicle_speed
    df['Power request at the motor (W)'] = power_request_motor

    energy_used = np.trapezoid(df['Power request at the motor (W)'], df['Time (s)']) * 2.7778e-7
    distance_miles = np.trapezoid(df['Simulated vehicle speed (m/s)'], df['Time (s)']) / (1.609344 * 1000)
    fuel_consumption = energy_used / distance_miles if distance_miles > 0 else np.nan
    mpg = self.kWh_per_mile_to_mpg(fuel_consumption) if fuel_consumption > 0 else np.nan

    return df, fuel_consumption, mpg
    
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

# Code to test the truck_model class
#parameters = read_parameters('data/diesel_daycab_truck_params.csv', 'data/default_economy_params.csv', 'data/constants.csv', 'data/default_vmt.csv')
#df_drivecycle = extract_drivecycle_data(f'data/pepsi_1_drive_cycle_27.csv')
#
#slope, y_intercept, fuel_consumption_ave_payload = truck_model(parameters).get_power_requirement(df_drivecycle)
#print(fuel_consumption_ave_payload)
