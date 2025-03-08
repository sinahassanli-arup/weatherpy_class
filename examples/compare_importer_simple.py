"""
Test the new class-based WeatherDataImporter implementation.
"""

import sys
import pandas as pd
import numpy as np
import os
import time
import logging
import traceback

# Add the parent directory to the path to import weatherpy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import new weatherpy
from weatherpy.data.wd_importer import BOMWeatherDataImporter, NOAAWeatherDataImporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_importer(stationID, dataType, timeZone, yearStart, yearEnd, interval, verbose=True):
    """
    Test the class-based importer implementation.
    
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
        (data, yearStart, yearEnd)
    """
    if verbose:
        print(f'Testing {dataType} importer for station {stationID}...')
        print(f'Parameters: timeZone={timeZone}, yearStart={yearStart}, yearEnd={yearEnd}, interval={interval}')

    try:
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
            
        data, yearStart, yearEnd = importer.import_data()
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nImport completed in {elapsed_time:.2f} seconds")
            print(f"Data shape: {data.shape}")
            print(f"Data years: {yearStart} to {yearEnd}")
            print(f"Data columns: {data.columns.tolist()}")
            print(f"Data index: {data.index.name}")
            print(f"First 5 rows:")
            print(data.head())
        
        return data, yearStart, yearEnd
                    
    except Exception as e:
        logging.error('An error occurred during import: %s', str(e))
        traceback.print_exc()
        return None, None, None

def run_tests(test_cases):
    """
    Run tests for multiple test cases.
    
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
        
        data, yearStart, yearEnd = test_importer(**test_case['params'])
        
        results.append({
            'name': test_case['name'],
            'success': data is not None,
            'shape': data.shape if data is not None else None,
            'years': f"{yearStart} to {yearEnd}" if data is not None else None
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{result['name']}: {status}")
        if result['shape']:
            print(f"  Shape: {result['shape']}, Years: {result['years']}")
    
    # Overall assessment
    all_successful = all(result['success'] for result in results)
    if all_successful:
        print("\n✅ All test cases were successful!")
    else:
        print("\n❌ Some test cases failed.")

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
            'name': 'BOM Sydney Airport (2010-2020) Default TimeZone',
            'params': {
                'stationID': '066037',
                'dataType': 'BOM',
                'timeZone': None,  # Test default behavior
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
        },
        {
            'name': 'NOAA New York (2010-2020) UTC',
            'params': {
                'stationID': '72503014732',
                'dataType': 'NOAA',
                'timeZone': 'UTC',
                'yearStart': 2010,
                'yearEnd': 2020,
                'interval': 60,
                'verbose': True
            }
        },
        {
            'name': 'NOAA New York (2010-2020) Default TimeZone',
            'params': {
                'stationID': '72503014732',
                'dataType': 'NOAA',
                'timeZone': None,  # Test default behavior
                'yearStart': 2010,
                'yearEnd': 2020,
                'interval': 60,
                'verbose': True
            }
        }
    ]
    
    # Run tests
    run_tests(test_cases) 