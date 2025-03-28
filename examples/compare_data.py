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
    
    # Check for NaN values
    if not df1.isna().equals(df2.isna()):
        print("  NaN value differences detected.")
    
    # Check for data type consistency
    if not df1.dtypes.equals(df2.dtypes):
        print("  Data type differences detected:")
        # Instead of using compare() which requires identical labels,
        # manually compare the dtypes for common columns
        common_cols = set(df1.columns) & set(df2.columns)
        dtype_diff = {}
        
        for col in common_cols:
            if df1[col].dtype != df2[col].dtype:
                dtype_diff[col] = {'legacy': df1[col].dtype, 'oop': df2[col].dtype}
        
        if dtype_diff:
            print("  Common columns with different dtypes:")
            for col, types in dtype_diff.items():
                print(f"    {col}: {types['legacy']} vs {types['oop']}")
        
        # Show dtypes for columns that exist only in one DataFrame
        only_df1_cols = set(df1.columns) - set(df2.columns)
        if only_df1_cols:
            print("  Columns only in legacy DataFrame:")
            for col in sorted(only_df1_cols):
                print(f"    {col}: {df1[col].dtype}")
        
        only_df2_cols = set(df2.columns) - set(df1.columns)
        if only_df2_cols:
            print("  Columns only in OOP DataFrame:")
            for col in sorted(only_df2_cols):
                print(f"    {col}: {df2[col].dtype}")
    
    # Check for index consistency
    if not df1.index.equals(df2.index):
        print("  Index differences detected.")
    
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
    Compare data import between legacy and OOP versions.
    
    Parameters
    ----------
    stationID : str
        Station ID.
    dataType : str
        Data type (BOM or NOAA).
    timeZone : str
        Time zone (UTC or LocalTime).
    yearStart : int
        Start year.
    yearEnd : int
        End year.
    interval : str
        Data interval (60minute or daily).
    verbose : bool, optional
        Whether to print verbose output. Default is False.
    
    Returns
    -------
    tuple
        (legacy_data, oop_data)
    """
    print(f"Comparing {dataType} import for station {stationID} ({timeZone})...")
    
    # Import legacy data
    print("Importing legacy data...")
    try:
        # Get station info
        if dataType == 'BOM':
            import weatherpy_legacy.data.stations as stations
            station_info = stations.station_info(stationID, printed=True)
            print('legacy timezone: ', timeZone)
            # Import data
            print("\n-----------------------------------------------------\n")
            # Use the main import_data function from weatherpy_legacy
            data_old, _, _ = wpl.import_data(
                stationID,
                dataType,
                timeZone,
                yearStart,
                yearEnd,
                interval,
                save_raw=False
            )
        elif dataType == 'NOAA':
            import weatherpy_legacy.data.stations as stations
            station_info = stations.station_info(stationID, printed=True)
            
            # Import data
            print("\n-----------------------------------------------------\n")
            # Use the main import_data function from weatherpy_legacy
            
            data_old, _, _ = wpl.import_data(
                stationID,
                dataType,
                timeZone,
                yearStart,
                yearEnd,
                interval,
                save_raw=False
            )
        else:
            print(f"Error: Unsupported data type: {dataType}")
            return None, None
    except Exception as e:
        print(f"Error importing legacy data: {e}")
        import traceback
        traceback.print_exc()
        data_old = None
    
    # Import OOP data
    print("\nImporting OOP data...")
    try:
        # Get station info
        if dataType == 'BOM':
            from weatherpy.data.wd_stations import WeatherStationDatabase
            try:
                station_db = WeatherStationDatabase('BOM')
                station_info = station_db.get_station_info(stationID)
            except Exception as e:
                print(f"Warning: Could not get station information: {e}")
            
            # Import data using the class-based approach
            if dataType == 'BOM':
                importer = wp.BOMWeatherDataImporter(
                    station_id=stationID,
                    data_type=dataType,
                    time_zone=timeZone,
                    year_start=yearStart,
                    year_end=yearEnd,
                    interval=interval
                )
            else:  # NOAA
                importer = wp.NOAAWeatherDataImporter(
                    station_id=stationID,
                    data_type=dataType,
                    time_zone=timeZone,
                    year_start=yearStart,
                    year_end=yearEnd,
                    interval=interval
                )
                
            weather_data = importer.import_data()
            data_class = weather_data.data
            
        elif dataType == 'NOAA':
            from weatherpy.data.wd_stations import WeatherStationDatabase
            try:
                station_db = WeatherStationDatabase('NOAA')
                station_info = station_db.get_station_info(stationID)
            except Exception as e:
                print(f"Warning: Could not get station information: {e}")
            
            # Import data using the class-based approach
            if dataType == 'BOM':
                importer = wp.BOMWeatherDataImporter(
                    station_id=stationID,
                    data_type=dataType,
                    time_zone=timeZone,
                    year_start=yearStart,
                    year_end=yearEnd,
                    interval=interval
                )
            else:  # NOAA
                importer = wp.NOAAWeatherDataImporter(
                    station_id=stationID,
                    data_type=dataType,
                    time_zone=timeZone,
                    year_start=yearStart,
                    year_end=yearEnd,
                    interval=interval
                )
                
            weather_data = importer.import_data()
            data_class = weather_data.data
        else:
            print(f"Error: Unsupported data type: {dataType}")
            return None, None
    except Exception as e:
        print(f"Error importing OOP data: {e}")
        import traceback
        traceback.print_exc()
        data_class = None
    
    # Compare results
    if data_old is not None and data_class is not None:
        print("\nComparing import results:")
        compare_dataframes(data_old, data_class)
    
    return data_old, data_class

# Example usage
input_data_1 = {
    'stationID': '066037',  # Using a valid station ID from the database (WYNDHAM AERO)
    'dataType': 'BOM',
    'timeZone': 'UTC',
    'yearStart': 2018,
    'yearEnd': 2020,
    'interval': 60,
    'verbose': False  # Set to True for more detailed output
}

input_data_2 = {
    'stationID': '72509014739',
    'dataType': 'NOAA',
    'timeZone': 'LocalTime',
    'yearStart': 2018,
    'yearEnd': 2020,
    'interval': 30,
    'verbose': False  # Set to True for more detailed output
}

# Run the comparison
print('\n=== WeatherPy Import Comparison ===')
print('=' * 30)


# Run BOM import comparison
print('\nRunning BOM import comparison...')
data_old_bom, data_class_bom = compare_data_import(**input_data_1)

# Uncomment to run NOAA import comparison
print('\nRunning NOAA import comparison...')
data_old_noaa, data_class_noaa = compare_data_import(**input_data_2)

# print('\n=== Comparison Complete ===')

# def test_import_bom_data():

#     """Test case for importing BOM data."""
#     input_data = {
#         'stationID': '066037',
#         'dataType': 'BOM',
#         'timeZone': 'LocalTime',
#         'yearStart': 2010,
#         'yearEnd': 2020,
#         'interval': 60
#     }
#     try:
#         importer = BOMWeatherDataImporter(
#             station_id=input_data['stationID'],
#             time_zone=input_data['timeZone'],
#             year_start=input_data['yearStart'],
#             year_end=input_data['yearEnd'],
#             interval=input_data['interval']
#         )
#         weather_data = importer.import_data()
#         print("BOM data imported successfully.")
#         # print(weather_data.data.head())
#     except Exception as e:
#         print(f"Error importing BOM data: {e}")


# def test_import_noaa_data():
#     """Test case for importing NOAA data."""
#     input_data = {
#         'stationID': '72509014739',
#         'dataType': 'NOAA',
#         'timeZone': 'LocalTime',
#         'yearStart': 2010,
#         'yearEnd': 2020,
#         'interval': 60
#     }
#     try:
#         importer = NOAAWeatherDataImporter(
#             station_id=input_data['stationID'],
#             time_zone=input_data['timeZone'],
#             year_start=input_data['yearStart'],
#             year_end=input_data['yearEnd'],
#             interval=input_data['interval']
#         )
#         weather_data = importer.import_data()
#         print("NOAA data imported successfully.")
#         # print(weather_data.data.head())
#     except Exception as e:
#         print(f"Error importing NOAA data: {e}")


# # Run test cases
# print('\n=== Running Test Cases ===')
# test_import_bom_data()
# # test_import_noaa_data()
# print('=== Test Cases Complete ===') 