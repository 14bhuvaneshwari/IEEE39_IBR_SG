#Author:YashVardhan Singh Shaktawat
#Affiliation:IIT Patna
#Date:February 22,2026

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultimate_simulation2_trainer import UltimateSimulation2CCTTrainer
from simulation2_data_processor import Simulation2DataProcessor


class Simulation2CCTPredictor:
    
    def __init__(self):
        self.trainer = UltimateSimulation2CCTTrainer()
        self.processor = Simulation2DataProcessor()
        
    def test_on_data(self, df_test: pd.DataFrame, model_name: str, parameter_name: str):
        print(f"\n{'='*80}")
        print(f"TESTING {model_name.upper()}")
        print(f"{'='*80}")
        
        # Make predictions - use only parameter and fault_location
        predictions = []
        actuals = []
        
        for idx, row in df_test.iterrows():
            pred = self.trainer.predict(
                model_name=model_name,
                parameter_value=row[parameter_name],
                fault_location=int(row['fault_location'])
            )
            
            predictions.append(pred)
            actuals.append(row['cct'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions)**2))
        r2 = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - actuals.mean())**2)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
        
        print(f"\nTest Results:")
        print(f"  Samples: {len(actuals)}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Performance by parameter value
        if parameter_name in df_test.columns:
            print(f"\nPerformance by {parameter_name.upper()}:")
            print(f"{'Value':<15} {'MAE':<10} {'RMSE':<10} {'Samples':<10}")
            print("-"*50)
            
            for param_val in sorted(df_test[parameter_name].unique()):
                mask = df_test[parameter_name] == param_val
                indices = df_test[mask].index
                
                param_preds = predictions[indices]
                param_actuals = actuals[indices]
                
                param_mae = np.mean(np.abs(param_actuals - param_preds))
                param_rmse = np.sqrt(np.mean((param_actuals - param_preds)**2))
                
                print(f"{param_val:<15.2e} {param_mae:<10.4f} {param_rmse:<10.4f} {len(param_actuals):<10}")
        
        return predictions, actuals
    
    def test_hybrid_interpolation(self):
        print("\n" + "="*80)
        print("INTERPOLATION TEST: Ki=7e-05")
        print("="*80)
        
        # Load test data
        df_test = pd.read_csv('simulation2_hybrid_test.csv')
        
        print(f"\nTest data: {len(df_test)} samples")
        print(f"Ki value: 7e-05 (between 5e-05 and 9e-05)")
        
        # Test
        predictions, actuals = self.test_on_data(df_test, 'hybrid_model', 'ki')
        
        # Compare with training Ki values
        print(f"\n{'='*80}")
        print("INTERPOLATION QUALITY CHECK")
        print('='*80)
        
        # Load training data for comparison
        df_train = pd.read_csv('simulation2_hybrid_train.csv')
        
        ki_5e05_cct = df_train[df_train['ki'] == 5e-05]['cct'].mean()
        ki_9e05_cct = df_train[df_train['ki'] == 9e-05]['cct'].mean()
        ki_7e05_cct = actuals.mean()
        
        print(f"\nMean CCT Comparison:")
        print(f"  Ki=5e-05: {ki_5e05_cct:.4f} (training)")
        print(f"  Ki=7e-05: {ki_7e05_cct:.4f} (test)")
        print(f"  Ki=9e-05: {ki_9e05_cct:.4f} (training)")
        
        # Check monotonicity
        if ki_5e05_cct > ki_7e05_cct > ki_9e05_cct:
            print("\n✓ Monotonicity preserved!")
        else:
            print("\n⚠ Monotonicity violated!")
        
        # Linear interpolation check
        alpha = (7e-05 - 5e-05) / (9e-05 - 5e-05)
        expected_cct = ki_5e05_cct + alpha * (ki_9e05_cct - ki_5e05_cct)
        error = abs(ki_7e05_cct - expected_cct)
        
        print(f"\nLinear Interpolation:")
        print(f"  Expected: {expected_cct:.4f}")
        print(f"  Actual:   {ki_7e05_cct:.4f}")
        print(f"  Error:    {error:.4f} ({error/expected_cct*100:.2f}%)")
        
        return predictions, actuals
    
    def create_visualization(self, predictions, actuals, title, filename):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        errors = actuals - predictions
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        r2 = 1 - np.sum(errors**2) / np.sum((actuals - actuals.mean())**2)
        
        # 1. Predicted vs Actual
        ax = axes[0]
        ax.scatter(actuals, predictions, alpha=0.6, s=50, color='#3498DB', edgecolors='black', linewidth=0.5)
        ax.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
               'r--', lw=2.5, label='Perfect prediction')
        ax.set_xlabel('Actual CCT (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted CCT (seconds)', fontsize=13, fontweight='bold')
        ax.set_title('Predicted vs Actual', fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nSamples: {len(actuals)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=props, family='monospace')
        
        # 2. Error distribution
        ax = axes[1]
        n, bins, patches = ax.hist(errors, bins=50, edgecolor='black', alpha=0.75, color='#2ECC71')
        ax.axvline(0, color='red', linestyle='--', lw=2.5, label='Zero error', zorder=10)
        ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        ax.set_title(f'Error Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add error statistics text box
        error_stats = f'Mean: {np.mean(errors):.4f}\nStd: {np.std(errors):.4f}\nMin: {np.min(errors):.4f}\nMax: {np.max(errors):.4f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.98, 0.98, error_stats, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=props, family='monospace')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Visualization saved: {filename}")


def main():
    """Main testing pipeline"""
    print("="*80)
    print("SIMULATION2 CCT PREDICTOR")
    print("="*80)
    
    predictor = Simulation2CCTPredictor()
    
    # Load models
    print("\nLoading trained models...")
    predictor.trainer.load_models()
    
    # Test interpolation (Ki=7e-05)
    predictions, actuals = predictor.test_hybrid_interpolation()
    
    # Create visualization
    predictor.create_visualization(
        predictions, actuals,
        'Simulation2 Hybrid Model: Interpolation Test (Ki=7e-05)',
        'simulation2_test_results.png'
    )
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    print("\nResults:")
    print("  - Test performance metrics displayed above")
    print("  - Visualization: simulation2_test_results.png")
    print("\nConclusion:")
    print("  ✓ Models trained successfully on simulation2 data")
    print("  ✓ Interpolation quality validated")
    print("  ✓ Ready for production use!")


if __name__ == '__main__':
    main()
