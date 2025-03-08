"""
Example script demonstrating the full processing pipeline for BOM data using OOP classes.
Shows import, unification, cleaning, and correction steps with processing history.
"""

import sys
sys.path.insert(0, r'C:\Users\Administrator\Documents\weatherpy_class')

import weatherpy as wp

# Import parameters (from compare_data.py)
import_params = {
    'stationID': '066037',
    'dataType': 'BOM',
    'timeZone': 'UTC',
    'yearStart': 2010,
    'yearEnd': 2020,
    'interval': 60
}

# Clean parameters (from compare_data.py)
clean_params = {
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

# Correction parameters (from compare_data.py)
correction_params = {
    'correct_terrain': True,
    'terrain_source': 'database',
    'station_id': '066037',
    'correct_bom_offset': False,
    'correct_10min_to_1h': True,
    'conversion_factor': 1.05
}

def print_processing_history(weather_data):
    """Print the processing history of the weather data."""
    print("\nProcessing History:")
    print("-" * 50)
    for i, step in enumerate(weather_data.processing_history, 1):
        print(f"\nStep {i}: {step['operation'].upper()}")
        print(f"Description: {step['description']}")
        print("Parameters:")
        for param, value in step['parameters'].items():
            print(f"  - {param}: {value}")
    print("-" * 50)

def main():
    print("Starting BOM data processing pipeline...\n")

    # 1. Import data
    print("1. Importing data...")
    data, year_start, year_end = wp.import_data(**import_params)
    
    # Create initial WeatherData instance
    weather_data = wp.WeatherData(
        data=data,
        station_id=import_params['stationID'],
        data_type=import_params['dataType'],
        time_zone=import_params['timeZone'],
        station_info=wp.station_info(import_params['stationID'], printed=False)
    )
    print(f"Imported data shape: {weather_data.data.shape}")

    # 2. Unify data
    print("\n2. Unifying data...")
    unifier = wp.BOMDataUnifier(weather_data)
    unified_data = unifier.unify()
    print(f"Unified data shape: {unified_data.data.shape}")

    # 3. Clean data
    print("\n3. Cleaning data...")
    cleaner = wp.BOMDataCleaner(unified_data, clean_params)
    cleaned_data = cleaner.clean(remove_calms=clean_params.get('clean_calms', True))
    print(f"Cleaned data shape: {cleaned_data.data.shape}")

    # 4. Correct data
    print("\n4. Correcting data...")
    corrector = wp.BOMDataCorrector(cleaned_data, correction_params)
    corrected_data = corrector.correct()
    print(f"Corrected data shape: {corrected_data.data.shape}")

    # Print processing history
    print_processing_history(corrected_data)

if __name__ == "__main__":
    main() 