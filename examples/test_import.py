import sys
import os
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime
import pytz

# Add the weatherpy_class directory to the path
sys.path.insert(0, r'C:\Users\Administrator\Documents\weatherpy_class')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import only the necessary functions from weatherpy_legacy
from weatherpy_legacy.data._bom_preparation import _import_bomhistoric
from weatherpy_legacy.data.initialization import _validate_station_years

# Import the OOP-based importer
from weatherpy.data.wd_importer import BOMWeatherDataImporter
from weatherpy.data.wd_stations import WeatherStationDatabase

def test_legacy_import(station_id, year_start, year_end, time_zone, interval):
    """Test the legacy import function."""
    print(f"\nTesting legacy import for BOM station {station_id}...")
    try:
        # Import data using the legacy function
        data = _import_bomhistoric(
            stationID=station_id,
            interval=interval,
            timeZone=time_zone,
            yearStart=year_start,
            yearEnd=year_end
        )
        
        print(f"Legacy import successful. Data shape: {data.shape}")
        print(f"Data index type: {type(data.index)}")
        print(f"Data index timezone: {data.index.tz if hasattr(data.index, 'tz') else 'None'}")
        print(f"Data columns: {data.columns.tolist()}")
        
        return data
    except Exception as e:
        print(f"Error in legacy import: {e}")
        traceback.print_exc()
        return None

def test_oop_import(station_id, year_start, year_end, time_zone, interval):
    """Test the OOP-based import function."""
    print(f"\nTesting OOP import for BOM station {station_id}...")
    try:
        # Create the importer
        importer = BOMWeatherDataImporter(
            station_id=station_id,
            data_type='BOM',
            time_zone=time_zone,
            year_start=year_start,
            year_end=year_end,
            interval=interval
        )
        
        # Import data
        weather_data = importer.import_data()
        
        print(f"OOP import successful. Data shape: {weather_data.data.shape}")
        print(f"Data index type: {type(weather_data.data.index)}")
        print(f"Data index timezone: {weather_data.data.index.tz if hasattr(weather_data.data.index, 'tz') else 'None'}")
        print(f"Data columns: {weather_data.data.columns.tolist()}")
        
        return weather_data.data
    except Exception as e:
        print(f"Error in OOP import: {e}")
        traceback.print_exc()
        return None

def compare_dataframes(df1, df2):
    """Compare two DataFrames and print differences."""
    if df1 is None or df2 is None:
        print("Cannot compare DataFrames: one or both are None")
        return
    
    if df1.equals(df2):
        print("Data outputs are identical.")
        return
    
    print("Data outputs differ.")
    # Compare column differences
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    print("\nColumn differences:")
    print(f"Only in first DataFrame: {cols1 - cols2}")
    print(f"Only in second DataFrame: {cols2 - cols1}")
    print(f"Common columns: {cols1 & cols2}")
    
    # Compare shapes
    print("\nShape comparison:")
    print(f"First DataFrame shape: {df1.shape}")
    print(f"Second DataFrame shape: {df2.shape}")
    
    # For common columns, compare values
    common_cols = cols1 & cols2
    if common_cols:
        print("\nDetailed value comparison:")
        for col in common_cols:
            if not df1[col].equals(df2[col]):
                print(f"\nDifferences in {col}:")
                print("First DataFrame sample values:")
                print(df1[col].head())
                print("Second DataFrame sample values:")
                print(df2[col].head())

# Test parameters
station_id = '066037'
year_start = 2010
year_end = 2020
time_zone = 'UTC'
interval = 60

# Test the station database
print("Testing WeatherStationDatabase...")
try:
    bom_db = WeatherStationDatabase('BOM')
    print(f"BOM database loaded with {len(bom_db._data)} stations")
    
    # Test getting a station
    station_info = bom_db.get_station_info(station_id)
    print(f"Station info for {station_id}: {station_info}")
except Exception as e:
    print(f"Error testing WeatherStationDatabase: {e}")
    traceback.print_exc()

# Test the legacy import
legacy_data = test_legacy_import(station_id, year_start, year_end, time_zone, interval)

# Test the OOP import
oop_data = test_oop_import(station_id, year_start, year_end, time_zone, interval)

# Compare the results
print("\nComparing legacy and OOP import results:")
compare_dataframes(legacy_data, oop_data) 