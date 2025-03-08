"""
Test the NOAA weather data importer.
"""

import sys
import pandas as pd
import os
import time
import logging
import pytz
from datetime import datetime

# Add the parent directory to the path to import weatherpy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import new weatherpy
from weatherpy.data.wd_importer import NOAAWeatherDataImporter

# Import legacy weatherpy
from weatherpy_legacy.data._noaa_preparation import _getNOAA_api
from weatherpy_legacy.data.stations import station_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_noaa_importer(station_id, year_start, year_end, time_zone):
    """
    Test the NOAA importer with both legacy and class-based methods.
    
    Parameters
    ----------
    station_id : str
        NOAA station ID
    year_start : int
        Start year
    year_end : int
        End year
    time_zone : str
        Time zone ('LocalTime' or 'UTC')
    """
    print(f"\nTesting NOAA importer for station {station_id}")
    print(f"Parameters: timeZone={time_zone}, yearStart={year_start}, yearEnd={year_end}")
    
    # Test legacy method
    print("\n=== Testing Legacy Method ===")
    start_time = time.time()
    legacy_data = _getNOAA_api(
        ID=station_id,
        yearStart=year_start,
        yearEnd=year_end,
        timeZone=time_zone
    )
    legacy_time = time.time() - start_time
    
    print(f"Legacy method completed in {legacy_time:.2f} seconds")
    print(f"Legacy data shape: {legacy_data.shape}")
    print(f"Legacy data index name: {legacy_data.index.name}")
    print(f"Legacy data columns: {legacy_data.columns.tolist()}")
    print(f"Legacy data first 5 rows:")
    print(legacy_data.head())
    
    # Test class-based method
    print("\n=== Testing Class-Based Method ===")
    start_time = time.time()
    importer = NOAAWeatherDataImporter(
        station_id=station_id,
        time_zone=time_zone,
        year_start=year_start,
        year_end=year_end
    )
    class_data, _, _ = importer.import_data()
    class_time = time.time() - start_time
    
    print(f"Class-based method completed in {class_time:.2f} seconds")
    print(f"Class-based data shape: {class_data.shape}")
    print(f"Class-based data index name: {class_data.index.name}")
    print(f"Class-based data columns: {class_data.columns.tolist()}")
    print(f"Class-based data first 5 rows:")
    print(class_data.head())
    
    # Filter both datasets to the requested year range for fair comparison
    print("\n=== Filtering Data to Requested Year Range ===")
    
    # Get timezone info
    try:
        station_tz_name = station_info(station_id, printed=False)['Timezone Name']
        station_timezone = pytz.timezone(station_tz_name)
    except:
        # Default to America/New_York for La Guardia Airport
        station_timezone = pytz.timezone('America/New_York')
    
    # Create date bounds
    if time_zone == 'LocalTime':
        start_date = pd.Timestamp(f"{year_start}-01-01", tz=station_timezone)
        end_date = pd.Timestamp(f"{year_end}-12-31 23:59:59", tz=station_timezone)
        date_col = 'LocalTime'
    else:  # UTC
        start_date = pd.Timestamp(f"{year_start}-01-01", tz='UTC')
        end_date = pd.Timestamp(f"{year_end}-12-31 23:59:59", tz='UTC')
        date_col = 'UTC'
    
    # Filter legacy data
    if date_col in legacy_data.columns:
        legacy_filtered = legacy_data[(legacy_data[date_col] >= start_date) & 
                                     (legacy_data[date_col] <= end_date)]
    elif legacy_data.index.name == date_col:
        legacy_filtered = legacy_data[(legacy_data.index >= start_date) & 
                                     (legacy_data.index <= end_date)]
    else:
        legacy_filtered = legacy_data
    
    # Filter class-based data
    if date_col in class_data.columns:
        class_filtered = class_data[(class_data[date_col] >= start_date) & 
                                   (class_data[date_col] <= end_date)]
    elif class_data.index.name == date_col:
        class_filtered = class_data[(class_data.index >= start_date) & 
                                   (class_data.index <= end_date)]
    else:
        class_filtered = class_data
    
    print(f"Legacy data shape after filtering: {legacy_filtered.shape}")
    print(f"Class-based data shape after filtering: {class_filtered.shape}")
    
    # Compare filtered data
    print("\n=== Comparing Filtered Data ===")
    
    # Check if shapes match
    shapes_match = legacy_filtered.shape == class_filtered.shape
    print(f"Shapes match: {shapes_match}")
    
    # Check if columns match
    columns_match = set(legacy_filtered.columns) == set(class_filtered.columns)
    print(f"Columns match: {columns_match}")
    
    # Check if index names match
    index_match = legacy_filtered.index.name == class_filtered.index.name
    print(f"Index names match: {index_match}")
    
    # Check if values match (using a lenient comparison for floating point values)
    try:
        # Reset index to avoid issues with index comparison
        legacy_reset = legacy_filtered.reset_index()
        class_reset = class_filtered.reset_index()
        
        # Sort both dataframes by the same columns to ensure same order
        if time_zone == 'LocalTime':
            legacy_reset = legacy_reset.sort_values(['LocalTime'])
            class_reset = class_reset.sort_values(['LocalTime'])
        else:
            legacy_reset = legacy_reset.sort_values(['UTC'])
            class_reset = class_reset.sort_values(['UTC'])
        
        # Compare only common columns
        common_cols = list(set(legacy_reset.columns) & set(class_reset.columns))
        legacy_common = legacy_reset[common_cols].reset_index(drop=True)
        class_common = class_reset[common_cols].reset_index(drop=True)
        
        # Use lenient comparison
        pd.testing.assert_frame_equal(legacy_common, class_common, 
                                     check_dtype=False, atol=1e-3, 
                                     check_index_type=False, check_column_type=False)
        values_match = True
    except AssertionError as e:
        print(f"Value comparison error: {e}")
        values_match = False
    
    print(f"Values match: {values_match}")
    
    # Overall result
    if shapes_match and columns_match and values_match:
        print("\n✅ SUCCESS: Legacy and class-based methods produce identical results for the requested year range")
    else:
        print("\n❌ FAILURE: Legacy and class-based methods produce different results for the requested year range")

if __name__ == "__main__":
    # Test with the provided station ID
    test_noaa_importer(
        station_id="72503014732",  # La Guardia Airport
        year_start=2010,
        year_end=2020,
        time_zone="LocalTime"
    )
    
    # Also test with UTC timezone
    test_noaa_importer(
        station_id="72503014732",  # La Guardia Airport
        year_start=2010,
        year_end=2020,
        time_zone="UTC"
    ) 