#Author:YashVardhan Singh Shaktawat
#Affiliation:IIT Patna
#Date:February 22,2026

import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path


def extract_cct_data(filepath, parameter_value, parameter_name='ki'):
    
    print(f"\nProcessing: {Path(filepath).name}")
    data = sio.loadmat(filepath)
    
    # Try different CCT key names
    cct_key = None
    for key in ['CCT_table', 'CCT_table_test', 'CCT_test_table']:
        if key in data:
            cct_key = key
            break
    
    if cct_key is None:
        print(f"  ERROR: No CCT data found!")
        return pd.DataFrame()
    
    cct_table = data[cct_key]
    print(f"  CCT table shape: {cct_table.shape} (key: {cct_key})")
    
    # Skip row 0 (threshold values) and column 0 (fault location numbers)
    # Rows 1-46 = fault locations 1-46
    # Columns 1-19 = 19 samples
    
    samples = []
    
    for fault_idx in range(1, cct_table.shape[0]):  # Skip row 0
        for sample_idx in range(1, cct_table.shape[1]):  # Skip column 0
            cct_val = cct_table[fault_idx, sample_idx]
            
            # Only include valid CCT values
            if cct_val > 0 and cct_val < 2.0:  # Reasonable CCT range
                samples.append({
                    parameter_name: parameter_value,
                    'fault_location': fault_idx,
                    'sample_idx': sample_idx,
                    'cct': cct_val
                })
    
    df = pd.DataFrame(samples)
    print(f"  Valid samples: {len(df)}")
    if len(df) > 0:
        print(f"  CCT range: {df['cct'].min():.4f} to {df['cct'].max():.4f}")
    
    return df


def main():
    
    print("="*80)
    print("CORRECT SIMULATION2 DATA EXTRACTION")
    print("="*80)
    
    base_path = Path('/Users/yashvardhansinghshaktawat/nayesirese/simulation2')
    
    # Task 2: SG data
    print("\n1. TASK 2 - SG SYSTEM (Kd variation)")
    print("="*80)
    
    sg_files = {
        0.0: base_path / 'IEEE39_SG' / 'DynData_kd0.mat',
        0.5: base_path / 'IEEE39_SG' / 'DynData_kd5.mat',
        1.0: base_path / 'IEEE39_SG' / 'DynData_kd1.mat'
    }
    
    sg_data = []
    for kd, filepath in sg_files.items():
        if filepath.exists():
            df = extract_cct_data(filepath, kd, 'kd')
            sg_data.append(df)
    
    df_sg = pd.concat(sg_data, ignore_index=True)
    df_sg.to_csv('simulation2_sg_data_correct.csv', index=False)
    
    print(f"\nSG Data Summary:")
    print(f"  Total samples: {len(df_sg)}")
    print(f"  Kd values: {sorted(df_sg['kd'].unique())}")
    print(f"  CCT range: {df_sg['cct'].min():.4f} to {df_sg['cct'].max():.4f}")
    print(f"  Saved to: simulation2_sg_data_correct.csv")
    
    # Task 3: Hybrid data
    print("\n2. TASK 3 - HYBRID SYSTEM (Ki variation)")
    print("="*80)
    
    hybrid_files = {
        1e-05: base_path / 'IEEE39_Hybrid' / 'DynData_ibr2_Ki_1e-05.mat',
        5e-05: base_path / 'IEEE39_Hybrid' / 'DynData_ibr2_Ki_5e-05.mat',
        9e-05: base_path / 'IEEE39_Hybrid' / 'DynData_ibr2_Ki_9e-05.mat',
        5e-04: base_path / 'IEEE39_Hybrid' / 'DynData_ibr2_Ki_5e-04.mat',
        9e-04: base_path / 'IEEE39_Hybrid' / 'DynData_ibr2_Ki_9e-04.mat'
    }
    
    hybrid_data = []
    for ki, filepath in hybrid_files.items():
        if filepath.exists():
            df = extract_cct_data(filepath, ki, 'ki')
            hybrid_data.append(df)
    
    df_hybrid = pd.concat(hybrid_data, ignore_index=True)
    df_hybrid.to_csv('simulation2_hybrid_train_correct.csv', index=False)
    
    print(f"\nHybrid Training Data Summary:")
    print(f"  Total samples: {len(df_hybrid)}")
    print(f"  Ki values: {sorted(df_hybrid['ki'].unique())}")
    print(f"  CCT range: {df_hybrid['cct'].min():.4f} to {df_hybrid['cct'].max():.4f}")
    print(f"  Saved to: simulation2_hybrid_train_correct.csv")
    
    # Test data
    print("\n3. TEST DATA (Ki=7e-05)")
    print("="*80)
    
    test_file = base_path / 'dataIEEE39_testdata_ibr2_Ki_7e-05.mat'
    if test_file.exists():
        df_test = extract_cct_data(test_file, 7e-05, 'ki')
        df_test.to_csv('simulation2_hybrid_test_correct.csv', index=False)
        
        print(f"\nTest Data Summary:")
        print(f"  Total samples: {len(df_test)}")
        print(f"  CCT range: {df_test['cct'].min():.4f} to {df_test['cct'].max():.4f}")
        print(f"  Saved to: simulation2_hybrid_test_correct.csv")
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
