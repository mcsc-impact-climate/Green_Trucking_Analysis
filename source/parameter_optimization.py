"""
Date: 260204
Author: danikae
Purpose: Optimize drag coefficient, rolling resistance, and inverter*motor efficiency
         for each truck by minimizing chi-squared over all driving events
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import truck_model_tools_messy
import data_collection_tools_messy
from costing_and_emissions_tools import get_payload_distribution, get_payload_penalty

KG_PER_TON = 1000
KG_PER_LB = 0.453592
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
M_PER_MILE = 1609.34
S_PER_H = 3600
G_PER_KG = 1000

battery_caps = pd.read_csv("messy_middle_results/battery_capacities_linear_summary.csv").set_index('Value')

m_truck_max_lb = 82000
m_truck_max_kg = m_truck_max_lb * KG_PER_LB

datasets = [
    {
        "name": "saia2",
        "truck_params": "saia",
        "battery_col": "saia1",
        "truck_title": "Saia",
        "drivecycle_glob": "saia2_drivecycle_*_detailed.csv",
        "summary_path": "messy_middle_results/saia2_drivecycle_data.csv",
    },
    {
        "name": "4gen",
        "truck_params": "4gen",
        "battery_col": "4gen",
        "truck_title": "4Gen",
        "drivecycle_glob": "4gen_drivecycle_*_detailed.csv",
        "summary_path": "messy_middle_results/4gen_drivecycle_data.csv",
    },
    {
        "name": "joyride",
        "truck_params": "joyride",
        "battery_col": "joyride",
        "truck_title": "Joyride",
        "drivecycle_glob": "joyride_drivecycle_*_detailed.csv",
        "summary_path": "messy_middle_results/joyride_drivecycle_data.csv",
    },
    {
        "name": "nevoya_with_weight",
        "truck_params": "nevoya",
        "battery_col": "nevoya_with_weight",
        "truck_title": "Nevoya",
        "drivecycle_glob": "nevoya_with_weight_drivecycle_*_detailed.csv",
        "summary_path": "messy_middle_results/nevoya_with_weight_drivecycle_data.csv",
    },
]

# Parameter ranges for optimization
CD_RANGE = (0.25, 0.6)
CR_RANGE = (0.003, 0.006)
ETA_RANGE = (0.85, 0.97)


class ParameterOptimizer:
    """Optimize truck parameters to fit observed driving cycle data."""
    
    def __init__(self, dataset, parameters, battery_params_dict, e_bat, m_bat_kg, m_truck_no_bat_kg):
        """Initialize optimizer with truck and battery parameters."""
        self.dataset = dataset
        self.parameters = parameters
        self.battery_params_dict = battery_params_dict
        self.e_bat = e_bat
        self.m_bat_kg = m_bat_kg
        self.m_truck_no_bat_kg = m_truck_no_bat_kg
        self.m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB
        self.summary_df = pd.read_csv(dataset["summary_path"], index_col="Driving event")
        
        # Store initial parameter values
        self.cd_init = parameters.cd
        self.cr_init = parameters.cr
        self.eta_init = parameters.eta_i * parameters.eta_m
        
        self.drivecycle_cache = {}
        self.chi_squared_history = []
    
    def load_drivecycle(self, drivecycle_path):
        """Load and extract drivecycle data, with caching."""
        drivecycle_path_str = str(drivecycle_path)
        if drivecycle_path_str not in self.drivecycle_cache:
            self.drivecycle_cache[drivecycle_path_str] = truck_model_tools_messy.extract_drivecycle_data(drivecycle_path_str)
        return self.drivecycle_cache[drivecycle_path_str]
    
    def evaluate_single_event(self, cd, cr, eta_combined, drivecycle_data, m_gvwr_kg, driving_event):
        """
        Evaluate model for a single driving event.
        
        Parameters:
        -----------
        cd : float
            Drag coefficient
        cr : float
            Rolling resistance coefficient
        eta_combined : float
            Product of inverter efficiency * motor efficiency
        drivecycle_data : pd.DataFrame
            Driving cycle data
        m_gvwr_kg : float
            Gross vehicle weight rating in kg
        driving_event : int
            Driving event number
        
        Returns:
        --------
        chi_squared : float
            Chi-squared value for this event
        """
        try:
            # Create a modified parameters object with new parameter values
            params_modified = truck_model_tools_messy.share_parameters(
                m_ave_payload=self.parameters.m_ave_payload,
                m_max=self.parameters.m_max,
                m_truck_no_bat=self.parameters.m_truck_no_bat,
                p_aux=self.parameters.p_aux,
                p_motor_max=self.parameters.p_motor_max,
                cd=cd,
                cr=cr,
                a_cabin=self.parameters.a_cabin,
                g=self.parameters.g,
                rho_air=self.parameters.rho_air,
                DoD=self.parameters.DoD,
                eta_i=np.sqrt(eta_combined),  # Approximate split; could be adjusted
                eta_m=np.sqrt(eta_combined),
                eta_gs=self.parameters.eta_gs,
                eta_rb=self.parameters.eta_rb,
                eta_grid_transmission=self.parameters.eta_grid_transmission,
                VMT=self.parameters.VMT,
                discountrate=self.parameters.discountrate,
            )
            
            # Get modeled fuel consumption
            model = truck_model_tools_messy.truck_model(params_modified)
            df, model_fuel_consumption, model_DoD = model.get_power_requirement(
                drivecycle_data.copy(),
                m_gvwr_kg,
                eta_battery=self.battery_params_dict['Roundtrip efficiency'],
                e_bat=self.e_bat,
            )
            
            # Get observed fuel consumption
            if driving_event not in self.summary_df.index:
                return 1e10, 0  # Large penalty if event not found
            
            obs_fuel_consumption = self.summary_df.loc[driving_event, "Fuel economy (kWh/mile)"]
            
            # Calculate signed percentage difference (allows +/- to cancel)
            pct_diff = (model_fuel_consumption - obs_fuel_consumption) / obs_fuel_consumption * 100
            
            # Calculate uncertainty for this event
            instantaneous_fuel = drivecycle_data['Instantaneous Energy (kWh/mile)'].dropna()
            if len(instantaneous_fuel) > 1:
                fuel_sem = instantaneous_fuel.std() / np.sqrt(len(instantaneous_fuel))
                fuel_unc_pct = (fuel_sem / obs_fuel_consumption * 100) if obs_fuel_consumption > 0 else 0
            else:
                fuel_unc_pct = 0
            
            return pct_diff, fuel_unc_pct
        
        except Exception as e:
            print(f"Error evaluating event {driving_event}: {e}")
            return 1e10, 0
    
    def objective_function(self, x):
        """
        Objective function to minimize: weighted mean signed percentage error in fuel economy over all driving events.
        Allows +/- errors to cancel, penalizing systematic bias.
        Weights are inversely proportional to squared uncertainty for each event.
        
        Parameters:
        -----------
        x : array
            [cd, cr, eta_combined]
        
        Returns:
        --------
        abs_weighted_mean_signed_pct_error : float
            Absolute value of weighted mean signed percentage error over all driving events
        """
        cd, cr, eta_combined = x
        
        # Bounds checking
        if not (CD_RANGE[0] <= cd <= CD_RANGE[1]):
            return 1e10
        if not (CR_RANGE[0] <= cr <= CR_RANGE[1]):
            return 1e10
        if not (ETA_RANGE[0] <= eta_combined <= ETA_RANGE[1]):
            return 1e10
        
        pct_errors = []
        weights = []
        
        drivecycle_files = sorted(Path("messy_middle_results").glob(self.dataset["drivecycle_glob"]))
        
        for drivecycle_path in drivecycle_files:
            drivecycle_data = self.load_drivecycle(drivecycle_path)
            
            # Extract driving event number
            parts = drivecycle_path.stem.split("_")
            driving_event = int(parts[-2]) if parts[-1] == "detailed" else int(parts[-1])
            
            # Get GVW
            m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
            
            # Evaluate this event
            pct_diff, fuel_unc_pct = self.evaluate_single_event(cd, cr, eta_combined, drivecycle_data, m_gvwr_kg, driving_event)
            
            pct_errors.append(pct_diff)
            
            # Calculate weight inversely proportional to uncertainty squared
            weight = 1.0 / (fuel_unc_pct**2 + 1e-6)  # Add small epsilon to avoid division by zero
            weights.append(weight)
        
        # Return absolute value of weighted mean signed percentage error (to optimize for bias close to 0)
        if weights and pct_errors:
            weighted_mean_error = np.average(pct_errors, weights=weights)
            return abs(weighted_mean_error)
        else:
            return 1e10
    
    def optimize(self):
        """Perform optimization to find best-fit parameters."""
        # Initial guess: current parameter values
        x0 = np.array([self.cd_init, self.cr_init, self.eta_init])
        
        # Bounds
        bounds = [CD_RANGE, CR_RANGE, ETA_RANGE]
        
        print(f"\n{'='*70}")
        print(f"Optimizing parameters for {self.dataset['name']}")
        print(f"{'='*70}")
        print(f"Initial guess: cd={x0[0]:.4f}, cr={x0[1]:.6f}, eta={x0[2]:.4f}")
        print(f"Initial objective value: {self.objective_function(x0):.4f}")
        
        # Optimize using L-BFGS-B (handles bounds well)
        result = minimize(
            self.objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 100}
        )
        
        print(f"\nOptimization result:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Final objective value: {result.fun:.4f}")
        print(f"  Optimal parameters:")
        print(f"    cd = {result.x[0]:.4f} (initial: {x0[0]:.4f})")
        print(f"    cr = {result.x[1]:.6f} (initial: {x0[1]:.6f})")
        print(f"    eta_combined = {result.x[2]:.4f} (initial: {x0[2]:.4f})")
        print(f"    Improvement: {((x0[0] - result.fun)/x0[0]*100):.2f}%")
        
        return result


def get_model_predictions(optimizer, parameters, cd, cr, eta_combined):
    """
    Get model predictions for all driving events using specified parameters.
    
    Returns:
    --------
    predictions : dict
        Dictionary with keys 'driving_event', 'observed', 'predicted', 'chi_squared'
    """
    predictions = {
        'driving_event': [],
        'observed': [],
        'predicted': [],
        'chi_squared': [],
    }
    
    drivecycle_files = sorted(Path("messy_middle_results").glob(optimizer.dataset["drivecycle_glob"]))
    
    for drivecycle_path in drivecycle_files:
        drivecycle_data = optimizer.load_drivecycle(drivecycle_path)
        
        # Extract driving event number
        parts = drivecycle_path.stem.split("_")
        driving_event = int(parts[-2]) if parts[-1] == "detailed" else int(parts[-1])
        
        # Get GVW
        m_gvwr_kg = drivecycle_data["GVW (kg)"].loc[0]
        
        # Skip if event not in summary
        if driving_event not in optimizer.summary_df.index:
            continue
        
        obs_fuel = optimizer.summary_df.loc[driving_event, "Fuel economy (kWh/mile)"]
        
        # Get predicted fuel consumption
        try:
            params_modified = truck_model_tools_messy.share_parameters(
                m_ave_payload=optimizer.parameters.m_ave_payload,
                m_max=optimizer.parameters.m_max,
                m_truck_no_bat=optimizer.parameters.m_truck_no_bat,
                p_aux=optimizer.parameters.p_aux,
                p_motor_max=optimizer.parameters.p_motor_max,
                cd=cd,
                cr=cr,
                a_cabin=optimizer.parameters.a_cabin,
                g=optimizer.parameters.g,
                rho_air=optimizer.parameters.rho_air,
                DoD=optimizer.parameters.DoD,
                eta_i=np.sqrt(eta_combined),
                eta_m=np.sqrt(eta_combined),
                eta_gs=optimizer.parameters.eta_gs,
                eta_rb=optimizer.parameters.eta_rb,
                eta_grid_transmission=optimizer.parameters.eta_grid_transmission,
                VMT=optimizer.parameters.VMT,
                discountrate=optimizer.parameters.discountrate,
            )
            
            model = truck_model_tools_messy.truck_model(params_modified)
            df, pred_fuel, _ = model.get_power_requirement(
                drivecycle_data.copy(),
                m_gvwr_kg,
                eta_battery=optimizer.battery_params_dict['Roundtrip efficiency'],
                e_bat=optimizer.e_bat,
            )
            
            pct_diff = (pred_fuel - obs_fuel) / obs_fuel * 100
            
            predictions['driving_event'].append(driving_event)
            predictions['observed'].append(obs_fuel)
            predictions['predicted'].append(pred_fuel)
            predictions['chi_squared'].append(pct_diff)
        
        except Exception as e:
            print(f"Error predicting event {driving_event}: {e}")
            continue
    
    return predictions


def plot_validation_results(results_df):
    """
    Create comprehensive validation plots for optimization results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Optimization results dataframe
    """
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parameter change plots - stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    trucks = results_df['Truck_title'].values if 'Truck_title' in results_df.columns else results_df['Truck'].values
    
    # Drag coefficient
    axes[0].bar(np.arange(len(trucks)) - 0.2, results_df['cd_initial'], width=0.4, label='Initial', alpha=0.8)
    axes[0].bar(np.arange(len(trucks)) + 0.2, results_df['cd_optimal'], width=0.4, label='Calibrated', alpha=0.8)
    axes[0].axhline(y=0.25, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Min')
    axes[0].axhline(y=0.6, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Max')
    axes[0].set_ylabel('Drag Coefficient', fontsize=14)
    axes[0].set_xticks(np.arange(len(trucks)))
    axes[0].set_xticklabels([])  # No labels on top plot
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2, mode='expand', borderaxespad=0, fontsize=18, frameon=False)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='y', labelsize=12)
    
    # Rolling resistance
    axes[1].bar(np.arange(len(trucks)) - 0.2, results_df['cr_initial'], width=0.4, label='Initial', alpha=0.8)
    axes[1].bar(np.arange(len(trucks)) + 0.2, results_df['cr_optimal'], width=0.4, label='Calibrated', alpha=0.8)
    axes[1].axhline(y=0.003, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Min')
    axes[1].axhline(y=0.006, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Max')
    axes[1].set_ylabel('Rolling Resistance Coefficient', fontsize=14)
    axes[1].set_xticks(np.arange(len(trucks)))
    axes[1].set_xticklabels([])  # No labels on middle plot
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='y', labelsize=12)
    
    # Efficiency
    axes[2].bar(np.arange(len(trucks)) - 0.2, results_df['eta_initial'], width=0.4, label='Initial', alpha=0.8)
    axes[2].bar(np.arange(len(trucks)) + 0.2, results_df['eta_optimal'], width=0.4, label='Calibrated', alpha=0.8)
    axes[2].axhline(y=0.85, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Min')
    axes[2].axhline(y=0.97, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Max')
    axes[2].set_ylabel('Inverter × Motor Efficiency', fontsize=14)
    axes[2].set_xticks(np.arange(len(trucks)))
    axes[2].set_xticklabels(trucks, fontsize=20)  # Labels only on bottom plot, no rotation
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'parameter_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean signed percentage error improvement plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(trucks))
    width = 0.35
    
    # Plot absolute values for comparison
    ax.bar(x - width/2, results_df['mean_pct_error_initial'].abs(), width, label='Initial', alpha=0.8)
    ax.bar(x + width/2, results_df['mean_pct_error_optimal'].abs(), width, label='Calibrated', alpha=0.8)
    
    ax.set_ylabel('|Mean Signed % Error| in Fuel Economy', fontsize=14)
    ax.set_xlabel('Truck', fontsize=14)
    ax.set_title('Optimization Improvement: Mean Error Bias Reduction', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(trucks, fontsize=20)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2, mode='expand', borderaxespad=0, fontsize=18, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement labels
    for i, truck in enumerate(trucks):
        initial_abs = abs(results_df.iloc[i]['mean_pct_error_initial'])
        optimal_abs = abs(results_df.iloc[i]['mean_pct_error_optimal'])
        improvement = ((initial_abs - optimal_abs) / initial_abs * 100) if initial_abs > 0 else 0
        ax.text(i, max(initial_abs, optimal_abs) * 1.05,
               f'{improvement:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'mean_signed_pct_error_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plots saved to {plots_dir}/")


def plot_observed_vs_predicted(optimizer, results_entry):
    """
    Create observed vs predicted fuel economy plot for a single truck.
    
    Parameters:
    -----------
    optimizer : ParameterOptimizer
        The optimizer instance
    results_entry : dict
        Single row from results dataframe
    """
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    truck_name = results_entry['Truck']
    
    # Get predictions for both initial and optimal parameters
    pred_initial = get_model_predictions(
        optimizer,
        optimizer.parameters,
        results_entry['cd_initial'],
        results_entry['cr_initial'],
        results_entry['eta_initial']
    )
    
    pred_optimal = get_model_predictions(
        optimizer,
        optimizer.parameters,
        results_entry['cd_optimal'],
        results_entry['cr_optimal'],
        results_entry['eta_optimal']
    )
    
    if not pred_initial['driving_event']:
        print(f"No valid predictions for {truck_name}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Initial parameters
    obs_initial = np.array(pred_initial['observed'])
    pred_init_vals = np.array(pred_initial['predicted'])
    
    axes[0].scatter(obs_initial, pred_init_vals, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
    min_val = min(obs_initial.min(), pred_init_vals.min())
    max_val = max(obs_initial.max(), pred_init_vals.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    axes[0].set_xlabel('Observed Fuel Economy (kWh/mile)')
    axes[0].set_ylabel('Predicted Fuel Economy (kWh/mile)')
    axes[0].set_title(f'{truck_name} - Initial Parameters')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Calculate R-squared for initial
    ss_res_init = np.sum((obs_initial - pred_init_vals) ** 2)
    ss_tot_init = np.sum((obs_initial - obs_initial.mean()) ** 2)
    r2_init = 1 - (ss_res_init / ss_tot_init) if ss_tot_init > 0 else 0
    rmse_init = np.sqrt(np.mean((obs_initial - pred_init_vals) ** 2))
    
    axes[0].text(0.05, 0.95, f'R² = {r2_init:.4f}\nRMSE = {rmse_init:.4f}',
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Optimal parameters
    obs_optimal = np.array(pred_optimal['observed'])
    pred_opt_vals = np.array(pred_optimal['predicted'])
    
    axes[1].scatter(obs_optimal, pred_opt_vals, alpha=0.6, s=100, edgecolors='k', linewidth=0.5, color='green')
    min_val = min(obs_optimal.min(), pred_opt_vals.min())
    max_val = max(obs_optimal.max(), pred_opt_vals.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    axes[1].set_xlabel('Observed Fuel Economy (kWh/mile)')
    axes[1].set_ylabel('Predicted Fuel Economy (kWh/mile)')
    axes[1].set_title(f'{truck_name} - Optimized Parameters')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Calculate R-squared for optimal
    ss_res_opt = np.sum((obs_optimal - pred_opt_vals) ** 2)
    ss_tot_opt = np.sum((obs_optimal - obs_optimal.mean()) ** 2)
    r2_opt = 1 - (ss_res_opt / ss_tot_opt) if ss_tot_opt > 0 else 0
    rmse_opt = np.sqrt(np.mean((obs_optimal - pred_opt_vals) ** 2))
    
    axes[1].text(0.05, 0.95, f'R² = {r2_opt:.4f}\nRMSE = {rmse_opt:.4f}',
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{truck_name}_observed_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(optimizer, results_entry):
    """
    Create residual plots for a single truck.
    
    Parameters:
    -----------
    optimizer : ParameterOptimizer
        The optimizer instance
    results_entry : dict
        Single row from results dataframe
    """
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    truck_name = results_entry.get('Truck_title', results_entry['Truck'])
    
    # Get predictions for both initial and optimal parameters
    pred_initial = get_model_predictions(
        optimizer,
        optimizer.parameters,
        results_entry['cd_initial'],
        results_entry['cr_initial'],
        results_entry['eta_initial']
    )
    
    pred_optimal = get_model_predictions(
        optimizer,
        optimizer.parameters,
        results_entry['cd_optimal'],
        results_entry['cr_optimal'],
        results_entry['eta_optimal']
    )
    
    if not pred_initial['driving_event']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals vs observed (initial)
    obs_initial = np.array(pred_initial['observed'])
    pred_init_vals = np.array(pred_initial['predicted'])
    residuals_init = obs_initial - pred_init_vals
    
    axes[0, 0].scatter(obs_initial, residuals_init, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Observed Fuel Economy (kWh/mile)')
    axes[0, 0].set_ylabel('Residual (observed - predicted)')
    axes[0, 0].set_title(f'{truck_name} - Residuals (Initial Parameters)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs observed (optimal)
    obs_optimal = np.array(pred_optimal['observed'])
    pred_opt_vals = np.array(pred_optimal['predicted'])
    residuals_opt = obs_optimal - pred_opt_vals
    
    axes[0, 1].scatter(obs_optimal, residuals_opt, alpha=0.6, s=100, edgecolors='k', linewidth=0.5, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Observed Fuel Economy (kWh/mile)')
    axes[0, 1].set_ylabel('Residual (observed - predicted)')
    axes[0, 1].set_title(f'{truck_name} - Residuals (Optimized Parameters)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of residuals (initial)
    axes[1, 0].hist(residuals_init, bins=10, alpha=0.7, edgecolor='k')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Residual Distribution (Initial)\nMean: {residuals_init.mean():.6f}, Std: {residuals_init.std():.6f}')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Histogram of residuals (optimal)
    axes[1, 1].hist(residuals_opt, bins=10, alpha=0.7, color='green', edgecolor='k')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Residual Distribution (Optimized)\nMean: {residuals_opt.mean():.6f}, Std: {residuals_opt.std():.6f}')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{truck_name}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main optimization routine."""
    results_list = []
    optimizers = []
    
    # for dataset in datasets:
    #     # Read parameters
    #     parameters = data_collection_tools_messy.read_parameters(
    #         truck_params=dataset["truck_params"],
    #         vmt_params='daycab_vmt_vius_2021',
    #         run='messy_middle',
    #         truck_type='EV',
    #     )
        
    #     battery_params_dict = data_collection_tools_messy.read_battery_params(chemistry=parameters.battery_chemistry)
    #     e_density = battery_params_dict['Energy density (kWh/ton)']
        
    #     e_bat = battery_caps.loc['Mean', dataset["battery_col"]]
    #     m_bat_kg = e_bat / e_density * KG_PER_TON
    #     m_truck_no_bat_kg = parameters.m_truck_no_bat
        
    #     # Create optimizer
    #     optimizer = ParameterOptimizer(
    #         dataset,
    #         parameters,
    #         battery_params_dict,
    #         e_bat,
    #         m_bat_kg,
    #         m_truck_no_bat_kg
    #     )
    #     optimizers.append(optimizer)
        
    #     # Perform optimization
    #     result = optimizer.optimize()
        
    #     # Store results
    #     result_entry = {
    #         'Truck': dataset['name'],
    #         'Truck_title': dataset['truck_title'],
    #         'cd_initial': parameters.cd,
    #         'cd_optimal': result.x[0],
    #         'cr_initial': parameters.cr,
    #         'cr_optimal': result.x[1],
    #         'eta_initial': parameters.eta_i * parameters.eta_m,
    #         'eta_optimal': result.x[2],
    #         'mean_pct_error_initial': optimizer.objective_function(np.array([parameters.cd, parameters.cr, parameters.eta_i * parameters.eta_m])),
    #         'mean_pct_error_optimal': result.fun,
    #         'optimization_success': result.success,
    #     }
    #     results_list.append(result_entry)
    
    # # Write results to CSV
    # results_df = pd.DataFrame(results_list)
    # results_df.to_csv('parameter_optimization_results.csv', index=False)
    # print(f"\n{'='*70}")
    # print("Results saved to parameter_optimization_results.csv")
    # print(f"{'='*70}")
    
    # Read results from CSV for plotting (allows plot generation without rerunning optimization)
    results_df = pd.read_csv('parameter_optimization_results.csv')
    
    # Add Truck_title if not present (for backwards compatibility with old CSVs)
    if 'Truck_title' not in results_df.columns:
        truck_title_map = {ds['name']: ds['truck_title'] for ds in datasets}
        results_df['Truck_title'] = results_df['Truck'].map(truck_title_map)
    
    print(results_df.to_string())
    
    # Generate validation plots
    print(f"\n{'='*70}")
    print("Generating validation plots...")
    print(f"{'='*70}")
    
    # Summary plots
    plot_validation_results(results_df)
    
    # Per-truck plots
    for optimizer, results_entry in zip(optimizers, results_list):
        plot_observed_vs_predicted(optimizer, results_entry)
        plot_residuals(optimizer, results_entry)
    
    print(f"\n{'='*70}")
    print("All plots generated successfully!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
