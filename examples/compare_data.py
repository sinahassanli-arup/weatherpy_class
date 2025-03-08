import logging
import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback

# Uncomment the line below to import weatherpy locally
sys.path.insert(0, r'C:\Users\Administrator\Documents\weatherpy_class')

import weatherpy as wp
from weatherpy.data.weather_data_base import WeatherData
from weatherpy.data.weather_data_unifier import WeatherDataUnifier, BOMDataUnifier, NOAADataUnifier
from weatherpy.data.weather_data_cleaner import BOMDataCleaner, NOAADataCleaner
from weatherpy.data.weather_data_importer import BOMWeatherDataImporter, NOAAWeatherDataImporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def unify_data(data, data_type):
    """Legacy unification method."""
    return wp.unify_datatype(data, data_type)

def compare_dataframes(df1, df2):
    """Compare two DataFrames and print differences."""
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

def compare_data_import(stationID, dataType, timeZone, yearStart, yearEnd, interval, verbose=True):
    """
    Compare data import between legacy and class-based methods.
    """
    if verbose:
        print('Starting data import comparison...')

    try:
        # Old method
        data_old, yearStart_old, yearEnd_old = wp.import_data(
            stationID,
            dataType,
            timeZone,
            yearStart,
            yearEnd,
            interval
        )
        
        # Class-based model
        if dataType == 'BOM':
            importer = BOMWeatherDataImporter(
                stationID=stationID,
                dataType=dataType,
                timeZone=timeZone,
                yearStart=yearStart,
                yearEnd=yearEnd,
                interval=interval
            )
        elif dataType == 'NOAA':
            importer = NOAAWeatherDataImporter(
                stationID=stationID,
                dataType=dataType,
                timeZone=timeZone,
                yearStart=yearStart,
                yearEnd=yearEnd,
                interval=interval
            )
        else:
            raise ValueError(f"Unsupported data type: {dataType}")
            
        data_class, yearStart_class, yearEnd_class = importer.import_data()
        
        # Compare data
        if data_old.equals(data_class):
            print('Data outputs are identical.')
        else:
            print('Data outputs differ.')
            # Compare column differences
            old_cols = set(data_old.columns)
            class_cols = set(data_class.columns)
            print('\nColumn differences:')
            print(f'Only in old: {old_cols - class_cols}')
            print(f'Only in class: {class_cols - old_cols}')
            print(f'Common columns: {old_cols & class_cols}')
            
            if verbose:
                # Compare a few sample values for common columns
                common_cols = old_cols & class_cols
                print('\nSample value comparison for common columns:')
                for col in common_cols:
                    if not data_old[col].equals(data_class[col]):
                        print(f'\nDifferences in {col}:')
                        print('Old method first 5 values:')
                        print(data_old[col].head())
                        print('Class method first 5 values:')
                        print(data_class[col].head())
        
        return data_old, data_class
                    
    except Exception as e:
        logging.error('An error occurred during import comparison: %s', str(e))
        traceback.print_exc()
        return None, None

def compare_data_unify(data_class, data_type):
    """Compare legacy and OOP unification methods."""
    
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
    return wp.clean_data(
        data,
        dataType=data_type,
        clean_ranked_rows=clean_params.get('clean_ranked_rows', False),
        clean_VC_filter=clean_params.get('clean_VC_filter', False),
        clean_calms=clean_params.get('clean_calms', True),
        clean_direction=clean_params.get('clean_direction', True),
        clean_off_clock=clean_params.get('clean_off_clock', False),
        clean_storms=clean_params.get('clean_storms', True),
        clean_invalid=clean_params.get('clean_invalid', True),
        clean_threshold=clean_params.get('clean_threshold', True),
        col2valid=clean_params.get('col2valid', ['WindSpeed', 'WindDirection', 'WindType']),
        thresholds=clean_params.get('thresholds', {'WindSpeed': (0, 50), 'PrePostRatio': (5, 30)})
    )[0]  # Return only the cleaned data, ignore removed and calm data

def compare_data_clean(data_class, data_type, clean_params):
    """Compare legacy and OOP cleaning methods."""
    print(f"\nStarting data cleaning comparison...\n")

    # Unify data before cleaning
    print("Unifying data before cleaning...")
    unified_data = unify_data(data_class, data_type)
    print("Data Unified\n")

    # Run legacy cleaning method
    print("Running legacy cleaning method...")
    legacy_cleaned = clean_data_legacy(unified_data, data_type, clean_params)
    print("Legacy cleaning completed.")
    print(f"Legacy cleaned data shape: {legacy_cleaned.shape}")
    print(f"Legacy cleaned columns: {legacy_cleaned.columns.tolist()}\n")

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
        cleaned_weather_data = cleaner.clean(remove_calms=clean_params.get('clean_calms', True))
        cleaned_data = cleaned_weather_data.data
        print("OOP cleaning completed.")
        print(f"OOP cleaned data shape: {cleaned_data.shape}")
        print(f"OOP cleaned columns: {cleaned_data.columns.tolist()}\n")

        # Compare results
        print("Comparing cleaning results:")
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
    print(f"\nStarting data correction comparison...\n")

    # Run legacy correction method
    print("Running legacy correction method...")
    legacy_corrected = correct_data_legacy(data_class, data_type, correction_params)
    print("Legacy correction completed.")
    print(f"Legacy corrected data shape: {legacy_corrected.shape}")
    print(f"Legacy corrected columns: {legacy_corrected.columns.tolist()}\n")

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
        corrected_weather_data = corrector.correct()
        corrected_data = corrected_weather_data.data
        print("OOP correction completed.")
        print(f"OOP corrected data shape: {corrected_data.shape}")
        print(f"OOP corrected columns: {corrected_data.columns.tolist()}\n")

        # Compare results
        print("Comparing correction results:")
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

# Set verbose to True for detailed output
input_data_1['verbose'] = True
input_data_2['verbose'] = True

print('\nRunning detailed comparison with fixed parameters:')
print('BOM Parameters:', clean_params_bom)
print('NOAA Parameters:', clean_params_noaa)
print('\n********************************************\n')

print('\nRunning Importer for BOM comparison...')
data_old_bom, data_class_bom = compare_data_import(**input_data_1)

print('\nRunning Importer for NOAA comparison...')
data_old_noaa, data_class_noaa = compare_data_import(**input_data_2)

# print('\n\n********************************************\n\n')

# print('\nRunning Unifier for BOM comparison...')
# # if data_old_bom is not None and data_class_bom is not None:
# #     compare_data_unify(data_class_bom, 'BOM')

# print('\nRunning Unifier for NOAA comparison...')
# if data_old_noaa is not None and data_class_noaa is not None:
#     compare_data_unify(data_class_noaa, 'NOAA')

# print('\n\n********************************************\n\n')

# print('\nRunning Cleaner for BOM comparison...')
# # if data_old_bom is not None and data_class_bom is not None:
# #     compare_data_clean(data_class_bom, 'BOM', clean_params_bom)

# print('\nRunning Cleaner for NOAA comparison...')
# if data_old_noaa is not None and data_class_noaa is not None:
#     compare_data_clean(data_class_noaa, 'NOAA', clean_params_noaa)

# # Example usage
# correction_params_bom = {
#     'correct_terrain': True,
#     'terrain_source': 'database',
#     'station_id': '066037',
#     'correct_bom_offset': False,  # Not testing BOM offset
#     'correct_10min_to_1h': True,
#     'conversion_factor': 1.05
# }

# print('\n\n********************************************\n\n')

# print('\nRunning Corrector for BOM comparison...')
# if data_old_bom is not None and data_class_bom is not None:
#     print("\nTesting BOM corrections with:")
#     print("- Terrain correction (from database)")
#     print("- 10min to 1h conversion (factor: 1.05)")
#     compare_data_correct(data_class_bom, 'BOM', correction_params_bom)

# Remove NOAA comparison
# print('\nRunning Corrector for NOAA comparison...')
# if data_old_noaa is not None and data_class_noaa is not None:
#     compare_data_correct(data_class_noaa, 'NOAA', correction_params_noaa) 