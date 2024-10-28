import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from datetime import datetime
from utils.database_utils import fetch_data_from_db
import matplotlib.dates as mdates
from matplotlib.colors import Normalize

def fetch_baseline_data(vessel_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch baseline data for a given vessel and split into laden and ballast conditions.
    """
    query = f"""
    SELECT speed_kts, me_consumption_mt, load_type
    FROM vessel_performance_model_data
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    baseline_data = fetch_data_from_db(query)
    
    laden_baseline = baseline_data[baseline_data['load_type'].isin(['Scantling', 'Design'])]
    ballast_baseline = baseline_data[baseline_data['load_type'] == 'Ballast']
    
    return laden_baseline, ballast_baseline

def fetch_ops_data(vessel_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch operational data for a given vessel, apply filters, and split into laden and ballast conditions.
    """
    query = f"""
    SELECT observed_speed, beaufort_scale, load_type, reportdate, normalised_me_consumption
    FROM vessel_performance_summary
    WHERE UPPER(vessel_name) = '{vessel_name.upper()}'
    """
    ops_data = fetch_data_from_db(query)
    
    # Apply filters
    ops_data = ops_data[
        (ops_data['beaufort_scale'] < 4) & 
        (ops_data['normalised_me_consumption'] > 5)
    ]
    
    # Convert reportdate to datetime and handle potential errors
    ops_data['reportdate'] = pd.to_datetime(ops_data['reportdate'], errors='coerce')
    ops_data = ops_data.dropna(subset=['reportdate'])  # Remove rows with invalid dates
    
    # Split data
    laden_ops = ops_data[ops_data['load_type'] == 'Laden']
    ballast_ops = ops_data[ops_data['load_type'] == 'Ballast']
    
    return laden_ops, ballast_ops

def add_baseline_points(ops_data: pd.DataFrame, baseline_data: pd.DataFrame, speeds: List[float] = [8, 10, 14]) -> pd.DataFrame:
    """
    Add baseline points for specified speeds to the operational data.
    """
    baseline_points = baseline_data[baseline_data['speed_kts'].isin(speeds)]
    return pd.concat([ops_data, baseline_points.rename(columns={'speed_kts': 'observed_speed', 'me_consumption_mt': 'normalised_me_consumption'})], ignore_index=True)

def plot_speed_consumption(vessel_name: str, laden_ops: pd.DataFrame, ballast_ops: pd.DataFrame, 
                           laden_baseline: pd.DataFrame, ballast_baseline: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Plot speed consumption data for both laden and ballast conditions with background and color gradients.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#000C20')
    
    def plot_condition(ax, ops_data, baseline_data, condition):
        if ops_data.empty:
            ax.text(0.5, 0.5, f"No {condition} data available", ha='center', va='center', color='white')
            return

        # Sort ops_data by date
        ops_data = ops_data.sort_values('reportdate')
        
        # Create a color map based on the date
        date_vals = mdates.date2num(ops_data['reportdate'])
        norm = Normalize(date_vals.min(), date_vals.max())
        # Use a brighter colormap
        colors = plt.cm.plasma(norm(date_vals))

        # Plot ops data with bright color gradient
        scatter = ax.scatter(ops_data['observed_speed'], ops_data['normalised_me_consumption'],
                             c=date_vals, cmap='plasma', s=80, alpha=0.9, edgecolor='w', linewidth=0.5)
        
        # Create a custom colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Date', color='white')
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        cbar.ax.yaxis.set_tick_params(labelcolor='white')
        
        # Plot baseline data
        ax.scatter(baseline_data['speed_kts'], baseline_data['me_consumption_mt'], 
                   color='red', s=100, label='Baseline', alpha=0.8)
        
        # Fit exponential curves
        def fit_exp(x, y):
            return np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
        
        ops_fit = fit_exp(ops_data['observed_speed'], ops_data['normalised_me_consumption'])
        base_fit = fit_exp(baseline_data['speed_kts'], baseline_data['me_consumption_mt'])
        
        x_range = np.linspace(min(ops_data['observed_speed'].min(), baseline_data['speed_kts'].min()),
                              max(ops_data['observed_speed'].max(), baseline_data['speed_kts'].max()), 100)
        
        ax.plot(x_range, np.exp(ops_fit[1]) * np.exp(ops_fit[0] * x_range), 'lime', label='Ops Fit', linestyle='--', linewidth=2)
        ax.plot(x_range, np.exp(base_fit[1]) * np.exp(base_fit[0] * x_range), 'cyan', label='Baseline Fit', linestyle='--', linewidth=2)
        
        # Customize axis and legend colors
        ax.set_title(f'{condition} Condition', color='white')
        ax.set_xlabel('Speed (knots)', color='white')
        ax.set_ylabel('ME Consumption (mt/day)', color='white')
        ax.legend(facecolor='#000C20', edgecolor='white', fontsize=12, labelcolor='white')
        ax.set_facecolor('#000C20')
        ax.tick_params(colors='white')
    
    plot_condition(ax1, laden_ops, laden_baseline, 'Laden')
    plot_condition(ax2, ballast_ops, ballast_baseline, 'Ballast')
    
    fig.suptitle(f'Speed vs Consumption - {vessel_name}', fontsize=16, color='white')
    plt.tight_layout()
    return fig

def analyze_speed_consumption(vessel_name: str) -> Tuple[str, Optional[plt.Figure]]:
    """
    Analyze speed consumption for a given vessel and generate a plot.
    """
    try:
        laden_baseline, ballast_baseline = fetch_baseline_data(vessel_name)
        laden_ops, ballast_ops = fetch_ops_data(vessel_name)
        
        laden_ops = add_baseline_points(laden_ops, laden_baseline)
        ballast_ops = add_baseline_points(ballast_ops, ballast_baseline)
        
        if laden_ops.empty and ballast_ops.empty:
            return f"No operational data available for {vessel_name}", None
        
        fig = plot_speed_consumption(vessel_name, laden_ops, ballast_ops, laden_baseline, ballast_baseline)
        
        return f"Speed consumption analysis completed for {vessel_name}", fig
    except Exception as e:
        return f"Error in speed consumption analysis for {vessel_name}: {str(e)}", None

# Example usage (for testing)
if __name__ == "__main__":
    vessel_name = "Example Vessel"
    message, figure = analyze_speed_consumption(vessel_name)
    print(message)
    if figure:
        plt.show()
    else:
        print("No figure generated.")
