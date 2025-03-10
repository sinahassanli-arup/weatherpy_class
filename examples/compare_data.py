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

def list_available_stations(data_type=None):
    """List available station IDs from the station database."""
    try:
        # Create a station database instance
        from weatherpy.data.wd_stations import WeatherStationDatabase
        
        # Get all stations or filter by data type
        if data_type:
            station_db = WeatherStationDatabase(data_type)
            stations = station_db.get_all_stations()
        else:
            # Get BOM stations by default
            station_db = WeatherStationDatabase('BOM')
            stations = station_db.get_all_stations()
        
        if stations:
            station_ids = [station.id for station in stations]
            print(f"\nAvailable station IDs ({len(station_ids)} total):")
            # Print first 10 stations and indicate if there are more
            print(', '.join(station_ids[:10]) + 
                  (f"... and {len(station_ids)-10} more" if len(station_ids) > 10 else ""))
        else:
            print("No stations found in the database.")
            
        return station_ids
    except Exception as e:
        print(f"Error listing stations: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def print_data_paths():
    """Print the paths where the legacy and OOP versions look for data."""
    print("\n=== Data File Locations ===")
    
    # For legacy weatherpy
    try:
        import weatherpy_legacy as wpl
        legacy_path = wpl.get_data_directory() if hasattr(wpl, 'get_data_directory') else "Unknown"
        print(f"Legacy WeatherPy data directory: {legacy_path}")
    except Exception as e:
        print(f"Could not determine legacy data path: {str(e)}")
    
    # For OOP weatherpy
    try:
        import weatherpy as wp
        oop_path = wp.get_data_directory() if hasattr(wp, 'get_data_directory') else "Unknown"
        print(f"OOP WeatherPy data directory: {oop_path}")
    except Exception as e:
        print(f"Could not determine OOP data path: {str(e)}")
    
    print("=" * 30)

# Example usage
input_data_1 = {
    'stationID': '066037',  # Using a valid station ID from the database (WYNDHAM AERO)
    'dataType': 'BOM',
    'timeZone': 'LocalTime',
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

# List available stations
print("\nListing available BOM stations:")
bom_stations = list_available_stations('BOM')

# Print data file paths
print_data_paths()

# Run BOM import comparison
print('\nRunning BOM import comparison...')
data_old_bom, data_class_bom = compare_data_import(**input_data_1)

# Uncomment to run NOAA import comparison
# print('\nRunning NOAA import comparison...')
# data_old_noaa, data_class_noaa = compare_data_import(**input_data_2)

print('\n=== Comparison Complete ===') 