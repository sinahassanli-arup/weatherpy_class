import logging
import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import requests
import tempfile
import pytz
from datetime import datetime
import argparse

# Uncomment the line below to import weatherpy locally
sys.path.insert(0, r'C:\Users\Administrator\Documents\weatherpy_class')

import weatherpy as wp
from weatherpy.data.wd_base import WeatherData
from weatherpy.data.wd_unifier import WeatherDataUnifier, BOMDataUnifier, NOAADataUnifier
from weatherpy.data.wd_cleaner import BOMDataCleaner, NOAADataCleaner
from weatherpy.data.wd_importer import BOMWeatherDataImporter, NOAAWeatherDataImporter

# Add this line to check if the WeatherStationDatabase is working properly
from weatherpy.data.wd_stations import WeatherStationDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add a test to check if the station database is working
print("Testing WeatherStationDatabase...")
try:
    bom_db = WeatherStationDatabase('BOM')
    print(f"BOM database loaded with {len(bom_db._data)} stations")
    
    # Test getting a station
    test_station_id = '066037'
    station_info = bom_db.get_station_info(test_station_id)
    print(f"Station info for {test_station_id}: {station_info}")
except Exception as e:
    print(f"Error testing WeatherStationDatabase: {e}")
    traceback.print_exc()

def legacy_import_bom(stationID, timeZone, yearStart, yearEnd, interval, verbose=True):
    """Legacy BOM import method."""
    if verbose:
        print(f"Importing BOM data for station {stationID} from {yearStart} to {yearEnd}")
    
    try:
        # Direct implementation of BOM data import
        # API URL
        url = "https://rr0yh3ttf5.execute-api.ap-southeast-2.amazonaws.com/Prod/v1/bomhistoric"

        # Preparing POST argument for request
        stationFile = f"{stationID}.zip" if interval == 1 else f"{stationID}-{interval}minute.zip"

        body = {
            "bucket": f"bomhistoric-{interval}minute",
            "stationID": stationFile
        }
        
        if verbose:
            print(f"Making API request to {url} with body: {body}")
        response_url = requests.post(url, json=body)
        signed_url = response_url.json()['body']

        signed_url_statusCode = response_url.json()['statusCode']
        
        if signed_url_statusCode != 200:
            raise ValueError(f'signed_url: {signed_url} Error code: {signed_url_statusCode}')

        if verbose:
            print(f"Getting data from signed URL")
        response_data = requests.get(signed_url)

        if response_data.status_code == 200:
            # Create a temporary file to save the response content
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response_data.content)
                temp_file_path = temp_file.name
                if verbose:
                    print(f"Reading data from temporary file: {temp_file_path}")
                data = pd.read_pickle(temp_file_path, compression='zip') 
                if verbose:
                    print("Data imported successfully")
            os.remove(temp_file_path) 
        else:
            raise ValueError(f"API request failed with status code: {response_data.status_code}")
        
        # Switch UTC and Local time datetime index if needed 
        if timeZone != data.index.name:
            if verbose:
                print(f"Switching index from {data.index.name} to {timeZone}")
            data = data.reset_index()
            data = data.set_index(data.columns[1])
        
        if verbose:
            print(f"Legacy import successful. Data shape: {data.shape}")
            print(f"Data index type: {type(data.index)}")
            print(f"Data index timezone: {data.index.tz if hasattr(data.index, 'tz') else 'None'}")
            print(f"Data columns: {data.columns.tolist()}")
        
        return data, yearStart, yearEnd
    except Exception as e:
        print(f"Error in legacy import: {e}")
        traceback.print_exc()
        return None, yearStart, yearEnd

def unify_data(data, data_type):
    """Legacy unification method."""
    # Simple implementation to avoid importing from weatherpy_legacy
    if data_type == 'BOM':
        # Rename columns to standardized names
        column_mapping = {
            'DryBulbTemperature': 'Temperature',
            'DewPointTemperature': 'DewPointTemp',
            'WindSpeed': 'WindSpeed',
            'WindDirection': 'WindDir',
            'RelativeHumidity': 'RelativeHumidity',
            'StationLevelPressure': 'StationPressure',
            'Rain': 'RainCumulative'
        }
        
        # Create a copy of the data
        unified_data = data.copy()
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in unified_data.columns:
                unified_data.rename(columns={old_name: new_name}, inplace=True)
        
        return unified_data
    else:  # NOAA
        # For NOAA, just return the data as is for now
        return data

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
            print(f"  - Only in first DataFrame: {', '.join(sorted(col_diff1))}")
        if col_diff2:
            print(f"  - Only in second DataFrame: {', '.join(sorted(col_diff2))}")
    
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

def compare_data_import(stationID, dataType, timeZone, yearStart, yearEnd, interval, verbose=True):
    """
    Compare data import between legacy and class-based methods.
    """
    if verbose:
        print(f'Comparing {dataType} import for station {stationID} ({timeZone})...')

    try:
        # Old method
        if dataType == 'BOM':
            if not verbose:
                print(f'Importing legacy BOM data...')
            data_old, yearStart_old, yearEnd_old = legacy_import_bom(
                stationID,
                timeZone,
                yearStart,
                yearEnd,
                interval,
                verbose
            )
        else:
            print("Legacy NOAA import not implemented in this test script")
            data_old, yearStart_old, yearEnd_old = None, yearStart, yearEnd
        
        # Class-based model
        if not verbose:
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
        yearStart_class = yearStart
        yearEnd_class = yearEnd
        
        # Compare data
        if data_old is not None and data_class is not None:
            print(f'\nComparing {dataType} import results:')
            compare_dataframes(data_old, data_class)
        else:
            print("Cannot compare data: one or both datasets are None")
        
        return data_old, data_class
                    
    except Exception as e:
        logging.error('An error occurred during import comparison: %s', str(e))
        traceback.print_exc()
        return None, None

def compare_data_unify(data_class, data_type):
    """Compare legacy and OOP unification methods."""
    print(f"\nComparing {data_type} unification methods...")
    
    # Run legacy unification method
    legacy_unified = unify_data(data_class, data_type)

    # Run OOP unification method
    try:
        # Create WeatherData instance with required parameters
        weather_data = WeatherData(
            data=data_class,
            station_id="test_station",
            data_type=data_type,
            time_zone="UTC",
            station_info={"name": "Test Station", "latitude": 0, "longitude": 0}
        )
        
        # Create appropriate unifier based on data type
        if data_type == "BOM":
            unifier = BOMDataUnifier(weather_data)
        else:  # NOAA
            unifier = NOAADataUnifier(weather_data)
        
        # Unify the data and get the DataFrame
        unified_weather_data = unifier.unify()
        unified_data = unified_weather_data.data

        # Compare results
        compare_dataframes(legacy_unified, unified_data)

    except Exception as e:
        logging.error(f"An error occurred during unification comparison: {str(e)}")
        traceback.print_exc()

def clean_data_legacy(data, data_type, clean_params):
    """Legacy cleaning method."""
    try:
        # Simple implementation to avoid importing from weatherpy_legacy
        # Create a copy of the data
        cleaned_data = data.copy()
        
        # Get parameters
        clean_invalid = clean_params.get('clean_invalid', True)
        col2valid = clean_params.get('col2valid', ['WindSpeed', 'WindDirection', 'WindType'])
        clean_threshold = clean_params.get('clean_threshold', True)
        thresholds = clean_params.get('thresholds', {'WindSpeed': (0, 50), 'PrePostRatio': (5, 30)})
        
        # Basic cleaning operations
        if clean_invalid and 'WindSpeed' in col2valid and 'WindSpeed' in cleaned_data.columns:
            # Remove rows with invalid wind speed (negative values)
            cleaned_data = cleaned_data[cleaned_data['WindSpeed'] >= 0]
        
        if clean_threshold and 'WindSpeed' in cleaned_data.columns and 'WindSpeed' in thresholds:
            # Apply thresholds to wind speed
            min_val, max_val = thresholds['WindSpeed']
            cleaned_data = cleaned_data[(cleaned_data['WindSpeed'] >= min_val) & (cleaned_data['WindSpeed'] <= max_val)]
        
        print(f"Legacy cleaning completed. Shape: {cleaned_data.shape}")
        return cleaned_data
    except Exception as e:
        print(f"Error in legacy cleaning: {e}")
        traceback.print_exc()
        return data

def compare_data_clean(data_class, data_type, clean_params):
    """Compare legacy and OOP cleaning methods."""
    print(f"\nComparing {data_type} cleaning methods...")

    # Unify data before cleaning
    print("  Unifying data before cleaning...")
    unified_data = unify_data(data_class, data_type)

    # Run legacy cleaning method
    print("  Running legacy cleaning method...")
    legacy_cleaned = clean_data_legacy(unified_data, data_type, clean_params)
    print(f"  Legacy cleaning completed. Shape: {legacy_cleaned.shape}")

    try:
        # Create WeatherData instance with required parameters
        weather_data = WeatherData(
            data=unified_data,  # Use the unified data for cleaning
            station_id="test_station",
            data_type=data_type,
            time_zone="UTC",
            station_info={"name": "Test Station", "latitude": 0, "longitude": 0}
        )

        # Create appropriate cleaner based on data type
        if data_type == "BOM":
            cleaner = BOMDataCleaner(weather_data, clean_params)
        else:  # NOAA
            cleaner = NOAADataCleaner(weather_data, clean_params)

        # Clean the data and get the DataFrame
        # Pass remove_calms parameter to match legacy behavior
        print("  Running OOP cleaning method...")
        cleaned_weather_data = cleaner.clean(remove_calms=clean_params.get('clean_calms', True))
        cleaned_data = cleaned_weather_data.data
        print(f"  OOP cleaning completed. Shape: {cleaned_data.shape}")

        # Compare results
        print("\nComparing cleaning results:")
        compare_dataframes(legacy_cleaned, cleaned_data)

    except Exception as e:
        print(f"An error occurred during cleaning comparison: {str(e)}")
        traceback.print_exc()

def correct_data_legacy(data, data_type, correction_params):
    """Legacy correction method."""
    if data_type == 'BOM':
        # Apply terrain correction if requested
        if correction_params.get('correct_terrain', False):
            data, is_terrain_corrected = wp.correct_terrain(
                data,
                stationID=correction_params.get('station_id'),
                dataType=data_type,
                source=correction_params.get('terrain_source', 'database'),
                correctionFactors=correction_params.get('correction_factors', None)
            )
        
        # Apply BOM offset correction if requested
        if correction_params.get('correct_bom_offset', False):
            data = wp.spdCorrection_bomOffset(data)
        
        # Apply 10min to 1h correction if requested
        if correction_params.get('correct_10min_to_1h', False):
            data = wp.spdCorrection_10minTo1h(
                data,
                factor=correction_params.get('conversion_factor', 1.05)
            )
    
    elif data_type == 'NOAA':
        # Only apply terrain correction if explicitly requested
        if correction_params.get('correct_terrain', False):
            print('\tWarning: Terrain correction is not typically applied to NOAA data')
            data, is_terrain_corrected = wp.correct_terrain(
                data,
                stationID=correction_params.get('station_id'),
                dataType=data_type,
                source=correction_params.get('terrain_source', 'database'),
                correctionFactors=correction_params.get('correction_factors', None)
            )
    
    return data

def compare_data_correct(data_class, data_type, correction_params):
    """Compare legacy and OOP correction methods."""
    print(f"\nComparing {data_type} correction methods...")

    # Run legacy correction method
    print("  Running legacy correction method...")
    legacy_corrected = correct_data_legacy(data_class, data_type, correction_params)
    print(f"  Legacy correction completed. Shape: {legacy_corrected.shape}")

    try:
        # Create WeatherData instance with required parameters
        weather_data = WeatherData(
            data=data_class,
            station_id=correction_params.get('station_id', 'test_station'),
            data_type=data_type,
            time_zone="UTC",
            station_info={"name": "Test Station", "latitude": 0, "longitude": 0}
        )

        # Create appropriate corrector based on data type
        if data_type == "BOM":
            corrector = BOMDataCorrector(weather_data, correction_params)
        else:  # NOAA
            corrector = NOAADataCorrector(weather_data, correction_params)

        # Apply corrections
        print("  Running OOP correction method...")
        corrected_weather_data = corrector.correct()
        corrected_data = corrected_weather_data.data
        print(f"  OOP correction completed. Shape: {corrected_data.shape}")

        # Compare results
        print("\nComparing correction results:")
        compare_dataframes(legacy_corrected, corrected_data)

    except Exception as e:
        print(f"An error occurred during correction comparison: {str(e)}")
        traceback.print_exc()

# Example usage
input_data_1 = {
    'stationID': '066037',
    'dataType': 'BOM',
    'timeZone': 'UTC',
    'yearStart': 2010,
    'yearEnd': 2020,
    'interval': 60,
    'verbose': False  # Can be set to False for minimal output
}

input_data_2 = {
    'stationID': '72509014739',
    'dataType': 'NOAA',
    'timeZone': 'LocalTime',
    'yearStart': 2010,
    'yearEnd': 2020,
    'interval': 60,
    'verbose': False  # Can be set to False for minimal output
}

# Fixed cleaning parameters for detailed comparison
clean_params_bom = {
    'clean_invalid': True,
    'col2valid': ['WindSpeed', 'WindDirection'],
    'clean_threshold': True,
    'thresholds': {
        'WindSpeed': (0, 50),
        'PrePostRatio': (5, 30)
    },
    'clean_calms': False,
    'clean_off_clock': False,
    'round_wind_direction': True,
    'adjust_low_wind_direction': True,
    'zero_calm_direction': True
}

clean_params_noaa = {
    'clean_invalid': False,
    'col2valid': ['WindSpeed', 'WindDirection'],
    'clean_threshold': False,
    'thresholds': {'WindSpeed': (0, 50), 'PrePostRatio': (5, 30)},
    'clean_calms': False,
    'clean_ranked_rows': False,
    'clean_VC_filter': True,
    'clean_direction': False,
    'clean_storms': False,
    'zero_calm_direction': False
}

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare legacy and OOP-based WeatherPy implementations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--noaa', action='store_true', help='Run NOAA comparison')
    parser.add_argument('--unify', action='store_true', help='Run unification comparison')
    parser.add_argument('--clean', action='store_true', help='Run cleaning comparison')
    parser.add_argument('--correct', action='store_true', help='Run correction comparison')
    args = parser.parse_args()

    # Set verbosity based on command line argument
    input_data_1['verbose'] = args.verbose
    input_data_2['verbose'] = args.verbose

    print('\n=== WeatherPy Comparison Tool ===')
    print('BOM Parameters:', clean_params_bom if args.verbose else '...')
    print('NOAA Parameters:', clean_params_noaa if args.verbose else '...')
    print('=' * 30)

    # Always run BOM import comparison
    print('\n[1/4] Running BOM import comparison...')
    data_old_bom, data_class_bom = compare_data_import(**input_data_1)

    # Run NOAA import comparison if requested
    if args.noaa:
        print('\n[2/4] Running NOAA import comparison...')
        data_old_noaa, data_class_noaa = compare_data_import(**input_data_2)
    else:
        data_old_noaa, data_class_noaa = None, None

    # Run unification comparison if requested
    if args.unify and data_class_bom is not None:
        print('\n[3/4] Running unification comparison...')
        compare_data_unify(data_class_bom, 'BOM')
        
        if args.noaa and data_class_noaa is not None:
            compare_data_unify(data_class_noaa, 'NOAA')

    # Run cleaning comparison if requested
    if args.clean and data_class_bom is not None:
        print('\n[4/4] Running cleaning comparison...')
        compare_data_clean(data_class_bom, 'BOM', clean_params_bom)
        
        if args.noaa and data_class_noaa is not None:
            compare_data_clean(data_class_noaa, 'NOAA', clean_params_noaa)

    # Run correction comparison if requested
    if args.correct and data_class_bom is not None:
        # Define correction parameters
        correction_params_bom = {
            'correct_terrain': True,
            'terrain_source': 'database',
            'station_id': '066037',
            'correct_bom_offset': False,
            'correct_10min_to_1h': True,
            'conversion_factor': 1.05
        }
        
        print('\n[Extra] Running correction comparison...')
        compare_data_correct(data_class_bom, 'BOM', correction_params_bom)
        
        if args.noaa and data_class_noaa is not None:
            correction_params_noaa = {
                'correct_terrain': False,
                'station_id': '72509014739'
            }
            compare_data_correct(data_class_noaa, 'NOAA', correction_params_noaa)

    print('\n=== Comparison Complete ===') 