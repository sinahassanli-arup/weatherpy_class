"""
Minimal test for BOMWeatherDataImporter.
"""

import sys
import os

# Add the parent directory to the path to import weatherpy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Import the BOMWeatherDataImporter class
    from weatherpy.data.wd_importer import BOMWeatherDataImporter
    print("Successfully imported BOMWeatherDataImporter")
    
    # Create an instance of BOMWeatherDataImporter
    importer = BOMWeatherDataImporter(
        station_id='066037',
        time_zone='LocalTime',
        year_start=2010,
        year_end=2020,
        interval=60
    )
    print("Successfully created BOMWeatherDataImporter instance")
    
    # Import data
    print("Importing data...")
    weather_data = importer.import_data()
    print("Successfully imported data")
    print(f"Data shape: {weather_data.data.shape}")
    
    # Create another instance with UTC timezone
    print("\nCreating instance with UTC timezone...")
    importer_utc = BOMWeatherDataImporter(
        station_id='066037',
        time_zone='UTC',
        year_start=2010,
        year_end=2020,
        interval=60
    )
    print("Successfully created BOMWeatherDataImporter instance with UTC timezone")
    
    # Import data with UTC timezone
    print("Importing data with UTC timezone...")
    weather_data_utc = importer_utc.import_data()
    print("Successfully imported data with UTC timezone")
    print(f"Data shape: {weather_data_utc.data.shape}")
    
    # Compare the two datasets
    print("\nComparing LocalTime and UTC datasets...")
    if weather_data.data.shape == weather_data_utc.data.shape:
        print("Both datasets have the same shape")
    else:
        print(f"Datasets have different shapes: {weather_data.data.shape} vs {weather_data_utc.data.shape}")
    
    # Check if the columns are the same
    if set(weather_data.data.columns) == set(weather_data_utc.data.columns):
        print("Both datasets have the same columns")
    else:
        print("Datasets have different columns")
        print(f"Only in LocalTime: {set(weather_data.data.columns) - set(weather_data_utc.data.columns)}")
        print(f"Only in UTC: {set(weather_data_utc.data.columns) - set(weather_data.data.columns)}")
    
    # Check if the data is the same
    if weather_data.data.equals(weather_data_utc.data):
        print("Both datasets have identical data")
    else:
        print("Datasets have different data")
        
        # Print the first few rows of each dataset
        print("\nFirst 5 rows of LocalTime dataset:")
        print(weather_data.data.head())
        print("\nFirst 5 rows of UTC dataset:")
        print(weather_data_utc.data.head())
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc() 