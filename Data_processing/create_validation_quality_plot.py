#Author: YashVardhan Singh Shaktawat
#Affiliation: IIT Patna
#Date: February 22,2026

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from ultimate_simulation2_trainer import UltimateSimulation2CCTTrainer
from simulation2_data_processor import Simulation2DataProcessor


class ValidationQualityPlotter:
  	def __init__(self):
        self.trainer = UltimateSimulation2CCTTrainer()
        self.processor = Simulation2DataProcessor()
        
    def create_validation_plot(self, model_name='hybrid_model', parameter_name='ki'):
        print("="*80)
        print("CREATING VALIDATION AND QUALITY PLOT")
        print("="*80)
        
        # Load models
        self.trainer.load_models('simulation2_models')
        
        # Load data
        if model_name == 'sg_model':
            df = pd.read_csv('simulation2_sg_data.csv')
            title = 'Validation of CCT Prediction Model - SG System (Kd Variation)'
        else:
            df = pd.read_csv('simulation2_hybrid_train.csv')
            title = 'Validation of CCT Prediction Model - Hybrid System (Ki Variation)'
        
        print(f"Loaded {len(df)} samples")
        
        # Make predictions
        predictions = []
        actuals = []
        param_values = []
        fault_locations = []
        
        for idx, row in df.iterrows():
            pred = self.trainer.predict(
                model_name=model_name,
                parameter_value=row[parameter_name],
                fault_location=int(row['fault_location'])
            )
            
            predictions.append(pred)
            actuals.append(row['cct'])
            param_values.append(row[parameter_name])
            fault_locations.append(row['fault_location'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        param_values = np.array(param_values)
        fault_locations = np.array(fault_locations)
        
        # Calculate errors
        errors = actuals - predictions
        abs_errors = np.abs(errors)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
        
        # Top plot: Stacked area chart showing prediction quality
        ax1 = fig.add_subplot(gs[0])
        self._create_stacked_quality_plot(ax1, param_values, actuals, predictions, 
                                          errors, parameter_name)
        
        # Bottom plot: Performance metrics over parameter range
        ax2 = fig.add_subplot(gs[1])
        self._create_performance_metrics_plot(ax2, param_values, abs_errors, 
                                              actuals, predictions, parameter_name)
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        filename = f'validation_quality_{model_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Validation plot saved: {filename}")
        
        plt.show()
        
        return predictions, actuals
    
    def _create_stacked_quality_plot(self, ax, param_values, actuals, predictions, 
                                     errors, parameter_name):
    
        
        # Sort by parameter value
        sort_idx = np.argsort(param_values)
        param_sorted = param_values[sort_idx]
        actuals_sorted = actuals[sort_idx]
        predictions_sorted = predictions[sort_idx]
        errors_sorted = errors[sort_idx]
        
        # Create bins for parameter values
        unique_params = np.unique(param_sorted)
        
        # For each parameter value, calculate statistics
        x_vals = []
        cct_actual = []
        cct_pred = []
        error_pos = []
        error_neg = []
        
        for param in unique_params:
            mask = param_sorted == param
            x_vals.append(param)
            cct_actual.append(np.mean(actuals_sorted[mask]))
            cct_pred.append(np.mean(predictions_sorted[mask]))
            
            # Positive and negative errors
            errors_param = errors_sorted[mask]
            error_pos.append(np.mean(errors_param[errors_param > 0]) if np.any(errors_param > 0) else 0)
            error_neg.append(np.mean(errors_param[errors_param < 0]) if np.any(errors_param < 0) else 0)
        
        x_vals = np.array(x_vals)
        cct_actual = np.array(cct_actual)
        cct_pred = np.array(cct_pred)
        error_pos = np.array(error_pos)
        error_neg = np.array(error_neg)
        
        # Create stacked areas - DON'T stack from 0, show actual CCT range
        # Base: Predicted CCT as a band
        y_lower = cct_pred + error_neg  # negative error goes below
        y_upper = cct_pred + error_pos  # positive error goes above
        
        # Fill the prediction band
        ax.fill_between(x_vals, y_lower, y_upper,
                       color='#87CEEB', alpha=0.4, label='Prediction Range', 
                       edgecolor='black', linewidth=0.5)
        
        # Predicted CCT line
        ax.plot(x_vals, cct_pred, 'b-', linewidth=2.5, label='Predicted CCT', zorder=10)
        
        # Actual CCT line
        ax.plot(x_vals, cct_actual, 'r-', linewidth=2.5, label='Actual CCT', zorder=11)
        
        # Styling
        ax.set_xlabel(f'{parameter_name.upper()} Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('CCT (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Quality Across Parameter Range', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis limits to actual data range
        y_min = min(y_lower.min(), cct_actual.min()) * 0.95
        y_max = max(y_upper.max(), cct_actual.max()) * 1.05
        ax.set_ylim([y_min, y_max])
        
        if parameter_name == 'ki':
            ax.set_xscale('log')
            ax.invert_xaxis()
        
        # Add text box with statistics
        mae = np.mean(np.abs(actuals - predictions))
        r2 = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - actuals.mean())**2)
        
        textstr = f'MAE: {mae:.4f}\nR²: {r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props, family='monospace')
    
    def _create_performance_metrics_plot(self, ax, param_values, abs_errors,
                                        actuals, predictions, parameter_name):
        
        # Sort by parameter value
        unique_params = np.sort(np.unique(param_values))
        
        # Calculate metrics for each parameter value
        mae_vals = []
        rmse_vals = []
        r2_vals = []
        
        for param in unique_params:
            mask = param_values == param
            
            # MAE
            mae = np.mean(abs_errors[mask])
            mae_vals.append(mae)
            
            # RMSE
            rmse = np.sqrt(np.mean((actuals[mask] - predictions[mask])**2))
            rmse_vals.append(rmse)
            
            # R²
            ss_res = np.sum((actuals[mask] - predictions[mask])**2)
            ss_tot = np.sum((actuals[mask] - actuals[mask].mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_vals.append(r2)
        
        # Create twin axis for R²
        ax2 = ax.twinx()
        
        # Plot MAE and RMSE on left axis
        line1 = ax.plot(unique_params, mae_vals, 'o-', linewidth=2.5, markersize=8,
                       color='#E74C3C', label='MAE', zorder=5)
        line2 = ax.plot(unique_params, rmse_vals, 's-', linewidth=2.5, markersize=8,
                       color='#3498DB', label='RMSE', zorder=5)
        
        # Plot R² on right axis
        line3 = ax2.plot(unique_params, r2_vals, '^-', linewidth=2.5, markersize=8,
                        color='#2ECC71', label='R²', zorder=5)
        
        # Styling
        ax.set_xlabel(f'{parameter_name.upper()} Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Metrics (MAE, RMSE)', fontsize=12, fontweight='bold', color='#E74C3C')
        ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold', color='#2ECC71')
        
        ax.set_title('Model Performance Metrics', fontsize=13, fontweight='bold', pad=10)
        
        # Color y-axis labels
        ax.tick_params(axis='y', labelcolor='#E74C3C')
        ax2.tick_params(axis='y', labelcolor='#2ECC71')
        
        # Set R² limits
        ax2.set_ylim([0.98, 1.001])
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if parameter_name == 'ki':
            ax.set_xscale('log')
            ax.invert_xaxis()
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)
        
        # Add performance annotation
        overall_mae = np.mean(abs_errors)
        overall_r2 = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - actuals.mean())**2)
        
        textstr = f'Overall:\nMAE={overall_mae:.4f}\nR²={overall_r2:.4f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=props, family='monospace')


def main():
    print("="*80)
    print("VALIDATION AND QUALITY VISUALIZATION")
    print("="*80)
    
    plotter = ValidationQualityPlotter()
    
    # Create validation plot for Hybrid model (Task 3)
    print("\n1. Creating validation plot for Hybrid Model (Task 3)...")
    predictions_hybrid, actuals_hybrid = plotter.create_validation_plot(
        model_name='hybrid_model',
        parameter_name='ki'
    )
    
    # Create validation plot for SG model (Task 2)
    print("\n2. Creating validation plot for SG Model (Task 2)...")
    predictions_sg, actuals_sg = plotter.create_validation_plot(
        model_name='sg_model',
        parameter_name='kd'
    )
    
    print("\n" + "="*80)
    print("VALIDATION PLOTS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. validation_quality_hybrid_model.png")
    print("  2. validation_quality_sg_model.png")
    print("\nThese plots show:")
    print("  - Prediction quality across parameter range (stacked area)")
    print("  - Performance metrics (MAE, RMSE, R²) variation")
    print("  - Model validation similar to PLL model validation")


if __name__ == '__main__':
    main()
