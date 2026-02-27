#Author:YashVardhan Singh Shaktawat
#Affiliation:IIT Patna
#Date:February 22,2026

import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class Simulation2DataProcessor:
    
    def __init__(self, base_path: str = '/Users/yashvardhansinghshaktawat/nayesirese/simulation2'):
        self.base_path = Path(base_path)
        self.sg_path = self.base_path / 'IEEE39_SG'
        self.hybrid_path = self.base_path / 'IEEE39_Hybrid'
        
    def _extract_scalar(self, val) -> float:
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return 0.0
            return float(val.flat[0])
        return float(val)
    
    def load_mat_file(self, filepath: Path) -> Dict:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = sio.loadmat(str(filepath))
        
        # Remove MATLAB metadata
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        
        return data
    
    def extract_samples_from_file(self, filepath: Path, parameter_value: float, 
                                  parameter_name: str = 'ki') -> pd.DataFrame:
        print(f"Processing: {filepath.name}")
        
        data = self.load_mat_file(filepath)
        
        # Check for CCT data
        cct_key = None
        for key in ['CCT_table', 'CCT_table_test', 'CCT_test_table']:
            if key in data:
                cct_key = key
                break
        
        if cct_key is None:
            raise ValueError(f"No CCT data found in {filepath.name}")
        
        cct = data[cct_key]
        n_faults, n_samples = cct.shape
        
        print(f"  Shape: {n_faults} faults × {n_samples} samples")
        
        # Required PMU variables
        required_vars = ['i_obj_mag', 'i_obj_angle', 'v_obj_mag', 'v_obj_angle', 'w_obj']
        
        # Check availability
        available_vars = {var: var in data for var in required_vars}
        missing_vars = [var for var, avail in available_vars.items() if not avail]
        
        if missing_vars:
            print(f"  WARNING: Missing variables: {missing_vars}")
        
        # Extract samples
        samples = []
        
        for fault_idx in range(n_faults):
            for sample_idx in range(n_samples):
                cct_val = self._extract_scalar(cct[fault_idx, sample_idx])
                
                # Extract all valid CCT values (no filtering)
                if cct_val > 0:  # Only exclude zeros/invalid
                    sample = {
                        parameter_name: parameter_value,
                        'fault_location': fault_idx,
                        'sample_idx': sample_idx,
                        'cct': cct_val
                    }
                    
                    # Add PMU measurements
                    for var in required_vars:
                        if var in data:
                            sample[var] = self._extract_scalar(data[var][fault_idx, sample_idx])
                        else:
                            sample[var] = 0.0  # Default value
                    
                    samples.append(sample)
        
        df = pd.DataFrame(samples)
        print(f"  Valid samples: {len(df)}")
        print(f"  CCT range: {df['cct'].min():.4f} to {df['cct'].max():.4f}")
        
        return df
    
    def load_sg_data(self) -> pd.DataFrame:
        print("\n" + "="*80)
        print("LOADING SG DATA (Task 2 - Kd Variation)")
        print("="*80)
        
        kd_files = {
            0.0: 'DynData_kd0.mat',
            0.5: 'DynData_kd5.mat',
            1.0: 'DynData_kd1.mat'
        }
        
        all_samples = []
        
        for kd_value, filename in kd_files.items():
            filepath = self.sg_path / filename
            
            if filepath.exists():
                df = self.extract_samples_from_file(filepath, kd_value, 'kd')
                all_samples.append(df)
            else:
                print(f"WARNING: {filename} not found")
        
        if not all_samples:
            raise ValueError("No SG data files found!")
        
        df_combined = pd.concat(all_samples, ignore_index=True)
        
        print(f"\n{'='*80}")
        print(f"Total SG samples: {len(df_combined)}")
        print(f"Kd values: {sorted(df_combined['kd'].unique())}")
        print(f"CCT range: {df_combined['cct'].min():.4f} to {df_combined['cct'].max():.4f}")
        print(f"{'='*80}")
        
        return df_combined
    
    def load_hybrid_data(self, include_test: bool = False) -> pd.DataFrame:
        print("\n" + "="*80)
        print("LOADING HYBRID DATA (Task 3 - Ki Variation)")
        print("="*80)
        
        ki_files = {
            1e-05: 'DynData_ibr2_Ki_1e-05.mat',
            5e-05: 'DynData_ibr2_Ki_5e-05.mat',
            9e-05: 'DynData_ibr2_Ki_9e-05.mat',
            5e-04: 'DynData_ibr2_Ki_5e-04.mat',
            9e-04: 'DynData_ibr2_Ki_9e-04.mat'
        }
        
        all_samples = []
        
        for ki_value, filename in ki_files.items():
            filepath = self.hybrid_path / filename
            
            if filepath.exists():
                df = self.extract_samples_from_file(filepath, ki_value, 'ki')
                all_samples.append(df)
            else:
                print(f"WARNING: {filename} not found")
        
        # Add test data if requested
        if include_test:
            test_file = self.base_path / 'dataIEEE39_testdata_ibr2_Ki_7e-05.mat'
            if test_file.exists():
                print("\nIncluding test data (Ki=7e-05)...")
                df_test = self.extract_samples_from_file(test_file, 7e-05, 'ki')
                all_samples.append(df_test)
        
        if not all_samples:
            raise ValueError("No Hybrid data files found!")
        
        df_combined = pd.concat(all_samples, ignore_index=True)
        
        print(f"\n{'='*80}")
        print(f"Total Hybrid samples: {len(df_combined)}")
        print(f"Ki values: {sorted(df_combined['ki'].unique())}")
        print(f"CCT range: {df_combined['cct'].min():.4f} to {df_combined['cct'].max():.4f}")
        print(f"{'='*80}")
        
        return df_combined
    
    def load_test_data(self) -> pd.DataFrame:
        print("\n" + "="*80)
        print("LOADING TEST DATA (Ki=7e-05)")
        print("="*80)
        
        test_file = self.base_path / 'dataIEEE39_testdata_ibr2_Ki_7e-05.mat'
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        df_test = self.extract_samples_from_file(test_file, 7e-05, 'ki')
        
        print(f"\n{'='*80}")
        print(f"Test samples: {len(df_test)}")
        print(f"CCT range: {df_test['cct'].min():.4f} to {df_test['cct'].max():.4f}")
        print(f"{'='*80}")
        
        return df_test
    
    def create_feature_matrix(self, df: pd.DataFrame, parameter_name: str = 'ki') -> Tuple[np.ndarray, List[str]]:
        feature_names = []
        features = []
        
        # Parameter value
        features.append(df[parameter_name].values)
        feature_names.append(parameter_name)
        
        # Fault location
        features.append(df['fault_location'].values)
        feature_names.append('fault_location')
        
        # PMU measurements
        pmu_vars = ['i_obj_mag', 'i_obj_angle', 'v_obj_mag', 'v_obj_angle', 'w_obj']
        for var in pmu_vars:
            if var in df.columns:
                features.append(df[var].values)
                feature_names.append(var)
        
        # Derived features
        if 'v_obj_mag' in df.columns and 'i_obj_mag' in df.columns:
            # Apparent power
            apparent_power = df['v_obj_mag'] * df['i_obj_mag']
            features.append(apparent_power.values)
            feature_names.append('apparent_power')
        
        if 'v_obj_angle' in df.columns and 'i_obj_angle' in df.columns:
            # Power angle
            power_angle = df['v_obj_angle'] - df['i_obj_angle']
            features.append(power_angle.values)
            feature_names.append('power_angle')
        
        # Parameter transformations
        param_vals = df[parameter_name].values
        
        # Log transformation
        features.append(np.log10(param_vals + 1e-10))
        feature_names.append(f'{parameter_name}_log')
        
        # Inverse
        features.append(1.0 / (param_vals + 1e-10))
        feature_names.append(f'{parameter_name}_inv')
        
        # Interaction: parameter × fault_location
        features.append(param_vals * df['fault_location'].values)
        feature_names.append(f'{parameter_name}_fault_interaction')
        
        X = np.column_stack(features)
        
        return X, feature_names
    
    def save_processed_data(self, df: pd.DataFrame, output_file: str):
        df.to_csv(output_file, index=False)
        print(f"\n✓ Data saved to: {output_file}")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(df.columns)}")


def main():
    print("="*80)
    print("SIMULATION2 DATA PROCESSOR")
    print("="*80)
    
    processor = Simulation2DataProcessor()
    
    # Process SG data (Task 2)
    try:
        df_sg = processor.load_sg_data()
        processor.save_processed_data(df_sg, 'simulation2_sg_data.csv')
    except Exception as e:
        print(f"Error processing SG data: {e}")
    
    # Process Hybrid data (Task 3) - training only
    try:
        df_hybrid_train = processor.load_hybrid_data(include_test=False)
        processor.save_processed_data(df_hybrid_train, 'simulation2_hybrid_train.csv')
    except Exception as e:
        print(f"Error processing Hybrid training data: {e}")
    
    # Process test data separately
    try:
        df_test = processor.load_test_data()
        processor.save_processed_data(df_test, 'simulation2_hybrid_test.csv')
    except Exception as e:
        print(f"Error processing test data: {e}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. simulation2_sg_data.csv - Task 2 (SG) data")
    print("  2. simulation2_hybrid_train.csv - Task 3 (Hybrid) training data")
    print("  3. simulation2_hybrid_test.csv - Task 3 (Hybrid) test data")


if __name__ == '__main__':
    main()
