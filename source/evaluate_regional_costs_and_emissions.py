import tco_emissions_tools
import json
import copy
from shapely.geometry import shape, mapping
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

G_PER_LB = 453.592
DEFAULT_PAYLOAD = 50000     # Default payload, in lb
KWH_PER_MWH = 1000
CENTS_PER_DOLLAR = 100

# Emissions
with open('geojsons/egrid2020_subregions_merged.geojson', mode='r') as geojson_file:
    geojson_data = json.load(geojson_file)

def get_gCO2e_per_mile(lbCO2e_per_kWh, average_payload = DEFAULT_PAYLOAD):
    gCO2e_per_kWh = lbCO2e_per_kWh * G_PER_LB / KWH_PER_MWH
    gCO2e_per_mi_df = tco_emissions_tools.evaluate_emissions(average_payload, gCO2e_per_kWh)
    return gCO2e_per_mi_df

for feature in geojson_data['features']:
    # Check if the 'SRC2ERTA' field exists in the properties
    if 'SRC2ERTA' in feature['properties']:
        gCO2e_per_mi_df = get_gCO2e_per_mile(feature['properties']['SRC2ERTA'])
        feature['properties']['C_mi_man'] = gCO2e_per_mi_df['GHGs manufacturing (gCO2/mi)']
        feature['properties']['C_mi_grid'] = gCO2e_per_mi_df['GHGs grid (gCO2/mi)']
        feature['properties']['C_mi_tot'] = gCO2e_per_mi_df['GHGs total (gCO2/mi)']
        del feature['properties']['SRC2ERTA']

with open('geojsons/emissions_per_mile.geojson', mode='w') as emissions_geojson:
    json.dump(geojson_data, emissions_geojson, indent=4)

# Costs
def get_costs_per_mile(electricity_rate_cents, demand_charge, average_payload = DEFAULT_PAYLOAD):
    electricity_rate_dollars = electricity_rate_cents / CENTS_PER_DOLLAR
    cost_per_mi_df = tco_emissions_tools.evaluate_costs(average_payload, electricity_rate_dollars, demand_charge)
    return cost_per_mi_df

with open('geojsons/electricity_rates_by_state_merged.geojson', mode='r') as geojson_file:
    electricity_rates_geojson = json.load(geojson_file)

with open('geojsons/demand_charges_by_state.geojson', mode='r') as geojson_file:
    demand_charges_geojson = json.load(geojson_file)
    
costs_per_mi_geojson = copy.deepcopy(electricity_rates_geojson)

# Ensure both GeoJSON files have the same number of features
assert len(electricity_rates_geojson['features']) == len(demand_charges_geojson['features']), 'GeoJSON files have different numbers of features.'

# Loop through all states (note: the state features in the electricity rate and demand charge geojsons are in the same order because they're both derived from the same base shapefile)

for electricity_rate_feature, demand_charge_feature, cost_per_mi_feature in zip(electricity_rates_geojson['features'], demand_charges_geojson['features'], costs_per_mi_geojson['features']):
    # Check if the 'STUSPS' field (state abbreviation) exists in the properties
    if 'Cents_kWh' in electricity_rate_feature['properties'] and 'Average Ma' in demand_charge_feature['properties']:
    
        del cost_per_mi_feature['properties']['Cents_kWh']
    
        if electricity_rate_feature['properties']['Cents_kWh'] is None or demand_charge_feature['properties']['Average Ma'] is None:
            cost_per_mi_feature['properties']['$_mi_tot'] = None
            cost_per_mi_feature['properties']['$_mi_cap'] = None
            cost_per_mi_feature['properties']['$_mi_el'] = None
            cost_per_mi_feature['properties']['$_mi_lab'] = None
            cost_per_mi_feature['properties']['$_mi_op'] = None
        else:
            costs_per_mile = get_costs_per_mile(electricity_rate_feature['properties']['Cents_kWh'], demand_charge_feature['properties']['Average Ma'])
            cost_per_mi_feature['properties']['$_mi_tot'] = costs_per_mile['TCO ($/mi)']
            cost_per_mi_feature['properties']['$_mi_cap'] = costs_per_mile['Total capital ($/mi)']
            cost_per_mi_feature['properties']['$_mi_el'] = costs_per_mile['Total electricity ($/mi)']
            cost_per_mi_feature['properties']['$_mi_lab'] = costs_per_mile['Total labor ($/mi)']
            cost_per_mi_feature['properties']['$_mi_op'] = costs_per_mile['Other OPEXs ($/mi)']
#        print(electricity_rate_feature['properties']['STUSPS'])
#        print(cost_per_mi_feature['properties']['$_mi_tot'])
        
with open('geojsons/costs_per_mile.geojson', mode='w') as cost_geojson:
    json.dump(costs_per_mi_geojson, cost_geojson, indent=4)
