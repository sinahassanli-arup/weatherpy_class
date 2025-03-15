import logging
import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Add the weatherpy_class directory to the path
sys.path.insert(0, r'C:\Users\Administrator\Documents\weatherpy_class')

import weatherpy as wp
from weatherpy.data.wd_base import WeatherData
from weatherpy.data.wd_importer import BOMWeatherDataImporter
from weatherpy.data.wd_cleaner import BOMDataCleaner
from weatherpy.data.wd_unifier import WeatherDataUnifier

# Import the legacy weatherpy
import weatherpy_legacy as wpl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_dataframes(df1, df2, name="DataFrames", detailed=False, ignore_columns=None, values_only=False):
    """
    Compare two DataFrames and print differences in a concise way.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame to compare (legacy)
    df2 : pandas.DataFrame
        Second DataFrame to compare (class-based)
    name : str, optional
        Name to use in output messages, by default "DataFrames"
    detailed : bool, optional
        Whether to provide detailed output, by default False
    ignore_columns : list, optional
        List of columns to ignore in the comparison, by default None
    values_only : bool, optional
        If True, consider DataFrames identical if common columns have identical values,
        even if shapes or column sets differ, by default False
        
    Returns
    -------
    bool
        True if DataFrames are considered identical based on criteria, False otherwise
    """
    if df1 is None or df2 is None:
        print(f"Cannot compare {name}: one or both are None")
        return False
    
    # Create copies to avoid modifying the original DataFrames
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    # Remove columns to ignore from comparison
    if ignore_columns:
        for col in ignore_columns:
            if col in df1_copy.columns:
                df1_copy = df1_copy.drop(columns=[col])
            if col in df2_copy.columns:
                df2_copy = df2_copy.drop(columns=[col])
    
    # Check for exact equality first
    if df1_copy.equals(df2_copy):
        print(f"✓ {name} are identical (ignoring specified columns).")
        return True
    
    # If not exactly equal, print differences
    print(f"✗ {name} differ.")
    
    # Compare column differences
    cols1 = set(df1_copy.columns)
    cols2 = set(df2_copy.columns)
    col_diff1 = cols1 - cols2
    col_diff2 = cols2 - cols1
    
    if col_diff1 or col_diff2:
        print("  Column differences:")
        if col_diff1:
            print(f"  - Only in legacy DataFrame: {', '.join(sorted(col_diff1))}")
        if col_diff2:
            print(f"  - Only in class-based DataFrame: {', '.join(sorted(col_diff2))}")
    
    # Compare shapes
    if df1_copy.shape != df2_copy.shape:
        print(f"  Shape difference: {df1_copy.shape} vs {df2_copy.shape}")
    
    # Check for NaN values (only for common columns)
    common_cols = list(cols1 & cols2)
    nan_differences = False
    if common_cols:
        try:
            if not df1_copy[common_cols].isna().equals(df2_copy[common_cols].isna()):
                print("  NaN value differences detected in common columns.")
                nan_differences = True
                if detailed:
                    for col in common_cols:
                        nan_count1 = df1_copy[col].isna().sum()
                        nan_count2 = df2_copy[col].isna().sum()
                        if nan_count1 != nan_count2:
                            print(f"    Column '{col}': NaN count {nan_count1} vs {nan_count2}")
        except:
            print("  Could not compare NaN values due to different indices.")
    
    # Check for data type consistency
    dtype_differences = False
    try:
        if not df1_copy[common_cols].dtypes.equals(df2_copy[common_cols].dtypes):
            print("  Data type differences detected in common columns:")
            dtype_differences = True
            dtype_diff = {}
            
            for col in common_cols:
                if df1_copy[col].dtype != df2_copy[col].dtype:
                    dtype_diff[col] = {'legacy': df1_copy[col].dtype, 'class-based': df2_copy[col].dtype}
            
            if dtype_diff:
                print("  Common columns with different dtypes:")
                for col, types in dtype_diff.items():
                    print(f"    {col}: {types['legacy']} vs {types['class-based']}")
    except:
        print("  Could not compare data types due to different indices.")
    
    # Check for index consistency
    index_differences = False
    if not df1_copy.index.equals(df2_copy.index):
        print("  Index differences detected.")
        index_differences = True
        print(f"  Legacy index range: {df1_copy.index.min()} to {df1_copy.index.max()}")
        print(f"  Class-based index range: {df2_copy.index.min()} to {df2_copy.index.max()}")
        print(f"  Legacy index size: {len(df1_copy.index)}, Class-based index size: {len(df2_copy.index)}")
    
    # For common columns, compare values for common indices
    value_differences = False
    if common_cols:
        try:
            # Get common indices
            common_indices = df1_copy.index.intersection(df2_copy.index)
            if len(common_indices) > 0:
                print(f"  Found {len(common_indices)} common indices for comparison")
                
                # Compare values for common indices and columns
                diff_cols = []
                for col in common_cols:
                    if not df1_copy.loc[common_indices, col].equals(df2_copy.loc[common_indices, col]):
                        diff_cols.append(col)
                
                if diff_cols:
                    print(f"  Value differences in {len(diff_cols)} columns: {', '.join(sorted(diff_cols))}")
                    value_differences = True
                    
                    # Show a sample of differences for each differing column
                    for sample_col in diff_cols[:3]:  # Limit to first 3 columns
                        print(f"\n  Sample differences in column '{sample_col}':")
                        # Find rows where values differ
                        df1_sample = df1_copy.loc[common_indices, sample_col]
                        df2_sample = df2_copy.loc[common_indices, sample_col]
                        
                        # Check if values are close (for floating point)
                        if df1_sample.dtype in [np.float64, np.float32] and df2_sample.dtype in [np.float64, np.float32]:
                            close_mask = np.isclose(df1_sample, df2_sample, rtol=1e-5, atol=1e-8)
                            if close_mask.all():
                                print(f"    All values are numerically close (within tolerance)")
                                continue
                        
                        mask = df1_sample != df2_sample
                        if mask.any():
                            # Get first 5 differing rows
                            diff_indices = mask[mask].index[:5]
                            for idx in diff_indices:
                                print(f"    Row {idx}: {df1_copy.loc[idx, sample_col]} vs {df2_copy.loc[idx, sample_col]}")
                            
                            # Show statistics for the differences
                            if df1_sample.dtype in [np.float64, np.float32] and df2_sample.dtype in [np.float64, np.float32]:
                                diff = df1_sample - df2_sample
                                print(f"    Mean difference: {diff.mean()}")
                                print(f"    Max difference: {diff.abs().max()}")
                                print(f"    Min difference: {diff.abs().min()}")
                else:
                    print("  No value differences found in common columns")
            else:
                print("  No common indices found for value comparison")
        except Exception as e:
            print(f"  Error comparing values: {e}")
    
    # If values_only is True and there are no value differences in common columns,
    # consider the DataFrames identical for our purposes
    if values_only and not value_differences and not nan_differences and not index_differences:
        print(f"✓ {name} are considered identical for common columns (values only).")
        return True
    
    return False

def compare_full_processing():
    """
    Compare the full data processing pipeline between legacy and class-based approaches.
    """
    # Input data from compare_data.py
    input_data = {
        'stationID': '066037',  # WYNDHAM AERO
        'dataType': 'BOM',
        'timeZone': 'UTC',
        'yearStart': 2018,
        'yearEnd': 2020,
        'interval': 60
    }
    
    # Cleaning parameters from windrose_recipe.py
    cleaning_params = {
        'clean_invalid': True,
        'col2valid': ['WindGust', 'WindDirection'],
        'clean_threshold': True,
        'thresholds': {'WindSpeed': (0, 50), 'PrePostRatio': (5, 30)},
        'clean_calms': False
    }
    
    print("\n=== Comparing Full Data Processing Pipeline ===")
    print("=" * 50)
    
    # Step 1: Import Data
    print("\n--- Step 1: Importing Data ---")
    
    # Legacy approach
    print("Importing data using legacy approach...")
    try:
        data_legacy_raw, _, _ = wpl.import_data(
            input_data['stationID'],
            input_data['dataType'],
            input_data['timeZone'],
            input_data['yearStart'],
            input_data['yearEnd'],
            input_data['interval'],
            save_raw=False
        )
        print(f"Legacy import successful. Shape: {data_legacy_raw.shape}")
    except Exception as e:
        print(f"Error importing legacy data: {e}")
        import traceback
        traceback.print_exc()
        data_legacy_raw = None
    
    # Class-based approach
    print("\nImporting data using class-based approach...")
    try:
        importer = BOMWeatherDataImporter(
            station_id=input_data['stationID'],
            data_type=input_data['dataType'],
            time_zone=input_data['timeZone'],
            year_start=input_data['yearStart'],
            year_end=input_data['yearEnd'],
            interval=input_data['interval']
        )
        weather_data = importer.import_data()
        data_class_raw = weather_data.data
        print(f"Class-based import successful. Shape: {data_class_raw.shape}")
    except Exception as e:
        print(f"Error importing class-based data: {e}")
        import traceback
        traceback.print_exc()
        data_class_raw = None
    
    # Compare raw imported data
    print("\nComparing raw imported data:")
    raw_identical = compare_dataframes(data_legacy_raw, data_class_raw, "Raw imported data")
    
    # Step 2: Unify Data
    print("\n--- Step 2: Unifying Data ---")
    
    # Legacy approach
    print("Unifying data using legacy approach...")
    try:
        data_legacy_unified = wpl.unify_datatype(
            data_legacy_raw,
            input_data['dataType']
        )
        print(f"Legacy unification successful. Shape: {data_legacy_unified.shape}")
    except Exception as e:
        print(f"Error unifying legacy data: {e}")
        import traceback
        traceback.print_exc()
        data_legacy_unified = None
    
    # Class-based approach
    print("\nUnifying data using class-based approach...")
    try:
        unifier = WeatherDataUnifier()
        weather_data_unified = unifier.unify(weather_data)
        data_class_unified = weather_data_unified.data
        print(f"Class-based unification successful. Shape: {data_class_unified.shape}")
    except Exception as e:
        print(f"Error unifying class-based data: {e}")
        import traceback
        traceback.print_exc()
        data_class_unified = None
    
    # Compare unified data
    print("\nComparing unified data:")
    # Ignore the 'UTC' column when comparing unified data and consider identical if common column values match
    unified_identical = compare_dataframes(data_legacy_unified, data_class_unified, "Unified data", detailed=True, ignore_columns=['UTC'], values_only=True)
    
    # Step 3: Clean Data
    print("\n--- Step 3: Cleaning Data ---")
    
    # Legacy approach
    print("Cleaning data using legacy approach...")
    try:
        data_legacy_cleaned, _, _ = wpl.clean_data(
            data_legacy_unified,
            dataType=input_data['dataType'],
            clean_invalid=cleaning_params['clean_invalid'],
            clean_threshold=cleaning_params['clean_threshold'],
            col2valid=cleaning_params['col2valid'],
            thresholds=cleaning_params['thresholds'],
            clean_calms=cleaning_params['clean_calms']
        )
        print(f"Legacy cleaning successful. Shape: {data_legacy_cleaned.shape}")
    except Exception as e:
        print(f"Error cleaning legacy data: {e}")
        import traceback
        traceback.print_exc()
        data_legacy_cleaned = None
    
    # Class-based approach
    print("\nCleaning data using class-based approach...")
    try:
        cleaner = BOMDataCleaner(data_class_unified)
        
        # First apply the standard BOM data cleaning steps
        # Fix wind directions between 0° and 5° to 360°
        cleaner.fix_wind_direction_0_to_5()
        
        # Round wind direction to nearest 10°
        cleaner.round_wind_direction()
        
        # Set wind direction to 0° for calm conditions
        cleaner.zero_calm_direction()
        
        # Apply cleaning operations based on parameters
        if cleaning_params['clean_invalid']:
            cleaner.clean_invalid(cleaning_params['col2valid'])
        
        if cleaning_params['clean_threshold']:
            cleaner.clean_threshold(cleaning_params['thresholds'])
        
        if cleaning_params['clean_calms']:
            cleaner.clean_calms()
            
        # Get the cleaned data
        data_class_cleaned = cleaner.data
        print(f"Class-based cleaning successful. Shape: {data_class_cleaned.shape}")
    except Exception as e:
        print(f"Error cleaning class-based data: {e}")
        import traceback
        traceback.print_exc()
        data_class_cleaned = None
    
    # Compare cleaned data
    print("\nComparing cleaned data:")
    # Ignore the 'UTC' column when comparing cleaned data and consider identical if common column values match
    cleaned_identical = compare_dataframes(data_legacy_cleaned, data_class_cleaned, "Cleaned data", detailed=True, ignore_columns=['UTC'], values_only=True)
    
    # Summary
    print("\n=== Processing Comparison Summary ===")
    print(f"Raw data identical: {raw_identical}")
    print(f"Unified data identical: {unified_identical}")
    print(f"Cleaned data identical: {cleaned_identical}")
    
    # Identify key differences in cleaning
    if not cleaned_identical and data_legacy_cleaned is not None and data_class_cleaned is not None:
        print("\n=== Analyzing Cleaning Differences ===")
        
        # Check if the class-based cleaner is removing invalid values
        if cleaning_params['clean_invalid']:
            print("\nChecking if class-based cleaner is properly removing invalid values...")
            
            # Count NaN values in key columns
            for col in cleaning_params['col2valid']:
                if col in data_class_unified.columns and col in data_class_cleaned.columns:
                    before_nan = data_class_unified[col].isna().sum()
                    after_nan = data_class_cleaned[col].isna().sum()
                    print(f"  Column '{col}': NaN values before cleaning: {before_nan}, after cleaning: {after_nan}")
                    print(f"  Difference: {after_nan - before_nan} new NaN values")
            
            # Check if the class-based cleaner is properly applying thresholds
            if cleaning_params['clean_threshold']:
                print("\nChecking if class-based cleaner is properly applying thresholds...")
                
                for col, (min_val, max_val) in cleaning_params['thresholds'].items():
                    if col in data_class_unified.columns and col in data_class_cleaned.columns:
                        before_outside = ((data_class_unified[col] < min_val) | (data_class_unified[col] > max_val)).sum()
                        after_outside = ((data_class_cleaned[col] < min_val) | (data_class_cleaned[col] > max_val)).sum()
                        print(f"  Column '{col}': Values outside [{min_val}, {max_val}] before cleaning: {before_outside}, after cleaning: {after_outside}")
        
        # Check wind direction rounding
        print("\nChecking wind direction rounding...")
        if 'WindDirection' in data_legacy_cleaned.columns and 'WindDirection' in data_class_cleaned.columns:
            # Check if all wind directions in legacy data are multiples of 10
            legacy_remainder = data_legacy_cleaned['WindDirection'] % 10
            legacy_rounded = (legacy_remainder == 0).all()
            print(f"  Legacy wind directions are all multiples of 10: {legacy_rounded}")
            
            # Check if all wind directions in class-based data are multiples of 10
            class_remainder = data_class_cleaned['WindDirection'] % 10
            class_rounded = (class_remainder == 0).all()
            print(f"  Class-based wind directions are all multiples of 10: {class_rounded}")
            
            if not class_rounded:
                # Show some examples of non-rounded values
                non_rounded = data_class_cleaned.loc[class_remainder != 0, 'WindDirection']
                if not non_rounded.empty:
                    print(f"  Examples of non-rounded wind directions in class-based data:")
                    for idx, val in non_rounded.head().items():
                        print(f"    {idx}: {val}")
    
    return {
        'raw': (data_legacy_raw, data_class_raw, raw_identical),
        'unified': (data_legacy_unified, data_class_unified, unified_identical),
        'cleaned': (data_legacy_cleaned, data_class_cleaned, cleaned_identical)
    }

if __name__ == "__main__":
    results = compare_full_processing() 