import logging
import sys
import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime
import pytz

# Add the weatherpy_class directory to the path
sys.path.insert(0, r'C:\Users\Administrator\Documents\weatherpy_class')

import weatherpy as wp
from weatherpy.data.wd_base import WeatherData
from weatherpy.data.wd_importer import BOMWeatherDataImporter, NOAAWeatherDataImporter
from weatherpy.data.wd_stations import WeatherStationDatabase

# Import the legacy weatherpy
import weatherpy_legacy as wpl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_dataframes(df1, df2):
    """Compare two DataFrames and print differences in a concise way."""
    if df1 is None or df2 is None:
        print("Cannot compare DataFrames: one or both are None")
        return
        
    if df1.equals(df2):
        print("✓ Data outputs are identical.")
        return
    
    print("✗ Data outputs differ.")
    
    # Compare column differences
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    col_diff1 = cols1 - cols2
    col_diff2 = cols2 - cols1
    
    if col_diff1 or col_diff2:
        print("  Column differences:")
        if col_diff1:
            print(f"  - Only in legacy DataFrame: {', '.join(sorted(col_diff1))}")
        if col_diff2:
            print(f"  - Only in OOP DataFrame: {', '.join(sorted(col_diff2))}")
    
    # Compare shapes
    if df1.shape != df2.shape:
        print(f"  Shape difference: {df1.shape} vs {df2.shape}")
    
    # For common columns, compare values
    common_cols = cols1 & cols2
    if common_cols:
        diff_cols = []
        for col in common_cols:
            if not df1[col].equals(df2[col]):
                diff_cols.append(col)
        
        if diff_cols:
            print(f"  Value differences in {len(diff_cols)} columns: {', '.join(sorted(diff_cols))}")

def compare_data_import(stationID, dataType, timeZone, yearStart, yearEnd, interval, verbose=False):
    """
    Compare data import between legacy and class-based methods.
    """
    print(f'Comparing {dataType} import for station {stationID} ({timeZone})...')

    try:
        # Legacy method using weatherpy_legacy.import_data directly
        print(f'Importing legacy {dataType} data...')
        data_old, yearStart_old, yearEnd_old = wpl.import_data(
            stationID,
            dataType,
            timeZone,
            yearStart,
            yearEnd,
            interval
        )
        
        # Class-based model
        print(f'Importing OOP {dataType} data...')
        if dataType == 'BOM':
            importer = BOMWeatherDataImporter(
                station_id=stationID,
                data_type=dataType,
                time_zone=timeZone,
                year_start=yearStart,
                year_end=yearEnd,
                interval=interval
            )
        elif dataType == 'NOAA':
            importer = NOAAWeatherDataImporter(
                station_id=stationID,
                data_type=dataType,
                time_zone=timeZone,
                year_start=yearStart,
                year_end=yearEnd,
                interval=interval
            )
        else:
            raise ValueError(f"Unsupported data type: {dataType}")
            
        weather_data = importer.import_data()
        data_class = weather_data.data
        
        # Compare data
        if data_old is not None and data_class is not None:
            print(f'\nComparing {dataType} import results:')
            compare_dataframes(data_old, data_class)
            
            if verbose:
                print("\nSample data from legacy import:")
                print(data_old.head())
                print("\nSample data from OOP import:")
                print(data_class.head())
        else:
            print("Cannot compare data: one or both datasets are None")
        
        return data_old, data_class
                    
    except Exception as e:
        logging.error('An error occurred during import comparison: %s', str(e))
        traceback.print_exc()
        return None, None

# Example usage
input_data_1 = {
    'stationID': '066037',
    'dataType': 'BOM',
    'timeZone': 'UTC',
    'yearStart': 2010,
    'yearEnd': 2020,
    'interval': 60,
    'verbose': False  # Set to True for more detailed output
}

input_data_2 = {
    'stationID': '72509014739',
    'dataType': 'NOAA',
    'timeZone': 'LocalTime',
    'yearStart': 2010,
    'yearEnd': 2020,
    'interval': 60,
    'verbose': False  # Set to True for more detailed output
}

# Run the comparison
print('\n=== WeatherPy Import Comparison ===')
print('=' * 30)

# Run BOM import comparison
print('\nRunning BOM import comparison...')
data_old_bom, data_class_bom = compare_data_import(**input_data_1)

# Uncomment to run NOAA import comparison
# print('\nRunning NOAA import comparison...')
# data_old_noaa, data_class_noaa = compare_data_import(**input_data_2)

print('\n=== Comparison Complete ===') 