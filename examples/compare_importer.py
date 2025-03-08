"""
Compare the legacy import_data function with the new class-based WeatherDataImporter implementation.
"""

import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import logging
import traceback

# Add the parent directory to the path to import weatherpy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import legacy weatherpy
import weatherpy_legacy.data.initialization as wp_legacy

# Import new weatherpy
from weatherpy.data.wd_importer import BOMWeatherDataImporter, NOAAWeatherDataImporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_dataframes(df1, df2, name1="Legacy", name2="Class-based"):
    """Compare two DataFrames and print differences."""
    if df1.equals(df2):
        print(f"Data outputs are identical between {name1} and {name2}.")
        return True
    
    print(f"Data outputs differ between {name1} and {name2}.")
    
    # Compare column differences
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    print("\nColumn differences:")
    print(f"Only in {name1}: {cols1 - cols2}")
    print(f"Only in {name2}: {cols2 - cols1}")
    print(f"Common columns: {cols1 & cols2}")
    
    # Compare shapes
    print("\nShape comparison:")
    print(f"{name1} DataFrame shape: {df1.shape}")
    print(f"{name2} DataFrame shape: {df2.shape}")
    
    # For common columns, compare values
    common_cols = cols1 & cols2
    if common_cols:
        print("\nDetailed value comparison for common columns:")
        differences_found = False
        for col in common_cols:
            if not df1[col].equals(df2[col]):
                differences_found = True
                print(f"\nDifferences in {col}:")
                print(f"{name1} sample values:")
                print(df1[col].head())
                print(f"{name2} sample values:")
                print(df2[col].head())
                
                # Calculate statistics for numeric columns
                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    diff = df1[col] - df2[col]
                    print(f"Difference statistics:")
                    print(f"Mean difference: {diff.mean()}")
                    print(f"Max difference: {diff.max()}")
                    print(f"Min difference: {diff.min()}")
                    print(f"Standard deviation of difference: {diff.std()}")
        
        if not differences_found:
            print("All common columns have identical values.")
    
    return False

def compare_data_import(stationID, dataType, timeZone, yearStart, yearEnd, interval, verbose=True):
    """
    Compare data import between legacy and class-based methods.
    
    Parameters
    ----------
    stationID : str
        Station ID to import data for
    dataType : str
        Type of data to import ('BOM' or 'NOAA')
    timeZone : str
        Timezone for data ('LocalTime' or 'UTC')
    yearStart : int
        Start year for data import
    yearEnd : int
        End year for data import
    interval : int
        Data interval in minutes
    verbose : bool, optional
        Whether to print detailed output, by default True
        
    Returns
    -------
    tuple
        (legacy_data, class_data, is_identical)
    """
    if verbose:
        print(f'Starting data import comparison for {dataType} station {stationID}...')
        print(f'Parameters: timeZone={timeZone}, yearStart={yearStart}, yearEnd={yearEnd}, interval={interval}')

    try:
        # Time the legacy method
        start_time = time.time()
        # Legacy method
        data_legacy, yearStart_legacy, yearEnd_legacy = wp_legacy.import_data(
            stationID,
            dataType,
            timeZone,
            yearStart,
            yearEnd,
            interval
        )
        legacy_time = time.time() - start_time
        
        if verbose:
            print(f"\nLegacy method completed in {legacy_time:.2f} seconds")
            print(f"Legacy data shape: {data_legacy.shape}")
            print(f"Legacy data years: {yearStart_legacy} to {yearEnd_legacy}")
            print(f"Legacy data columns: {data_legacy.columns.tolist()}")
        
        # Time the class-based method
        start_time = time.time()
        # Class-based method
        if dataType == 'BOM':
            importer = BOMWeatherDataImporter(
                station_id=stationID,
                time_zone=timeZone,
                year_start=yearStart,
                year_end=yearEnd,
                interval=interval
            )
        elif dataType == 'NOAA':
            importer = NOAAWeatherDataImporter(
                station_id=stationID,
                time_zone=timeZone,
                year_start=yearStart,
                year_end=yearEnd,
                interval=interval
            )
        else:
            raise ValueError(f"Unsupported data type: {dataType}")
            
        data_class, yearStart_class, yearEnd_class = importer.import_data()
        class_time = time.time() - start_time
        
        if verbose:
            print(f"\nClass-based method completed in {class_time:.2f} seconds")
            print(f"Class-based data shape: {data_class.shape}")
            print(f"Class-based data years: {yearStart_class} to {yearEnd_class}")
            print(f"Class-based data columns: {data_class.columns.tolist()}")
            
            # Compare performance
            print(f"\nPerformance comparison:")
            print(f"Legacy method: {legacy_time:.2f} seconds")
            print(f"Class-based method: {class_time:.2f} seconds")
            if legacy_time > 0:
                print(f"Speedup: {legacy_time/class_time:.2f}x")
        
        # Compare data
        print("\nComparing data outputs:")
        is_identical = compare_dataframes(data_legacy, data_class, "Legacy", "Class-based")
        
        return data_legacy, data_class, is_identical
                    
    except Exception as e:
        logging.error('An error occurred during import comparison: %s', str(e))
        traceback.print_exc()
        return None, None, False

def run_comparison(test_cases):
    """
    Run comparison for multiple test cases.
    
    Parameters
    ----------
    test_cases : list
        List of dictionaries with test parameters
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"Running test case {i+1}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}")
        
        data_legacy, data_class, is_identical = compare_data_import(**test_case['params'])
        
        results.append({
            'name': test_case['name'],
            'is_identical': is_identical,
            'legacy_shape': data_legacy.shape if data_legacy is not None else None,
            'class_shape': data_class.shape if data_class is not None else None
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        status = "✅ IDENTICAL" if result['is_identical'] else "❌ DIFFERENT"
        print(f"{result['name']}: {status}")
        if result['legacy_shape'] and result['class_shape']:
            print(f"  Legacy shape: {result['legacy_shape']}, Class shape: {result['class_shape']}")
    
    # Overall assessment
    all_identical = all(result['is_identical'] for result in results)
    if all_identical:
        print("\n✅ All test cases produced identical results!")
    else:
        print("\n❌ Some test cases produced different results.")

if __name__ == "__main__":
    # Define test cases
    test_cases = [
        {
            'name': 'BOM Sydney Airport (2010-2020)',
            'params': {
                'stationID': '066037',
                'dataType': 'BOM',
                'timeZone': 'LocalTime',
                'yearStart': 2010,
                'yearEnd': 2020,
                'interval': 60,
                'verbose': True
            }
        },
        {
            'name': 'BOM Sydney Airport (2010-2020) UTC',
            'params': {
                'stationID': '066037',
                'dataType': 'BOM',
                'timeZone': 'UTC',
                'yearStart': 2010,
                'yearEnd': 2020,
                'interval': 60,
                'verbose': True
            }
        },
        {
            'name': 'NOAA New York (2010-2020)',
            'params': {
                'stationID': '72503014732',
                'dataType': 'NOAA',
                'timeZone': 'LocalTime',
                'yearStart': 2010,
                'yearEnd': 2020,
                'interval': 60,
                'verbose': True
            }
        }
    ]
    
    # Run comparison
    run_comparison(test_cases) 