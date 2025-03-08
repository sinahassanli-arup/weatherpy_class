"""
Script to compare legacy and OOP methods for weather station operations.
Demonstrates different ways to search for weather stations using both methods.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Set

# Add parent directory to path to import weatherpy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weatherpy.data.stations import find_stations as legacy_find_stations
from weatherpy.data.wd_stations import WeatherStationDatabase

def validate_results(legacy_results: pd.DataFrame, oop_results: list) -> Dict[str, bool]:
    """
    Validate that legacy and OOP results match.
    
    Parameters
    ----------
    legacy_results : pd.DataFrame
        Results from legacy method
    oop_results : list
        Results from OOP method
        
    Returns
    -------
    Dict[str, bool]
        Dictionary of validation results
    """
    validation = {
        'count_match': False,
        'codes_match': False,
        'coordinates_match': False
    }
    
    if legacy_results is None or len(oop_results) == 0:
        return validation
        
    # Compare counts
    validation['count_match'] = len(legacy_results) == len(oop_results)
    
    # Compare station codes
    legacy_codes = set(legacy_results['Station Code'].values)
    oop_codes = set(station.code for station in oop_results)
    validation['codes_match'] = legacy_codes == oop_codes
    
    # Compare coordinates (within small tolerance)
    if validation['codes_match']:
        coords_match = True
        for station in oop_results:
            legacy_station = legacy_results[legacy_results['Station Code'] == station.code].iloc[0]
            if not (np.isclose(legacy_station['Latitude'], station.latitude) and 
                   np.isclose(legacy_station['Longitude'], station.longitude)):
                coords_match = False
                break
        validation['coordinates_match'] = coords_match
    
    return validation

def compare_search(search_params: dict, data_type: str = 'BOM', test_name: str = '') -> Dict[str, Any]:
    """
    Compare station search between legacy and OOP methods.
    
    Parameters
    ----------
    search_params : dict
        Search parameters matching the find_stations arguments
    data_type : str
        'BOM' or 'NOAA'
    test_name : str
        Name of the test case
        
    Returns
    -------
    Dict[str, Any]
        Comparison results
    """
    results = {
        'test_name': test_name,
        'data_type': data_type,
        'legacy_time': 0,
        'oop_time': 0,
        'legacy_count': 0,
        'oop_count': 0,
        'validation': {}
    }
    
    print(f"\n{'='*20} {test_name} {'='*20}")
    print(f"Data Source: {data_type}")
    print("\nSearch Parameters:")
    print('-' * 40)
    for key, value in search_params.items():
        if value is not None:
            print(f"{key:12}: {value}")
    
    # Legacy method
    start = time.time()
    legacy_results = legacy_find_stations(
        dataType=data_type,
        printOutput=False,
        **search_params
    )
    results['legacy_time'] = time.time() - start
    results['legacy_count'] = len(legacy_results) if legacy_results is not None else 0
    
    # OOP method
    start = time.time()
    db = WeatherStationDatabase(data_type)
    
    # Convert parameters to OOP method format
    oop_params = {
        'coordinates': search_params.get('coord'),
        'city': search_params.get('city'),
        'nearest': search_params.get('nearest'),
        'radius': search_params.get('radius')
    }
    oop_results = db.find_stations(**oop_params)
    results['oop_time'] = time.time() - start
    results['oop_count'] = len(oop_results)
    
    # Validate results
    results['validation'] = validate_results(legacy_results, oop_results)
    
    print("\nResults:")
    print('-' * 40)
    print(f"{'Method':12} {'Stations':>10} {'Time (s)':>12} {'Faster by':>12}")
    print('-' * 40)
    
    # Calculate which method is faster
    time_diff = abs(results['legacy_time'] - results['oop_time'])
    if results['legacy_time'] < results['oop_time']:
        faster = 'Legacy'
        speedup = f"{(results['oop_time']/results['legacy_time'] - 1)*100:.1f}%"
    else:
        faster = 'OOP'
        speedup = f"{(results['legacy_time']/results['oop_time'] - 1)*100:.1f}%"
    
    print(f"{'Legacy':12} {results['legacy_count']:10d} {results['legacy_time']:12.3f}")
    print(f"{'OOP':12} {results['oop_count']:10d} {results['oop_time']:12.3f}")
    print(f"\n{faster} method is faster by {speedup}")
    
    print("\nValidation Results:")
    print('-' * 40)
    print(f"Station counts match: {results['validation']['count_match']}")
    print(f"Station codes match: {results['validation']['codes_match']}")
    print(f"Coordinates match: {results['validation']['coordinates_match']}")
    
    return results

def test_bom_searches():
    """Run tests with BOM data."""
    print("\n" + "="*20 + " BOM Data Tests " + "="*20)
    
    results = []
    
    # Australian cities and coordinates
    results.append(compare_search({
        'city': 'Sydney',
        'nearest': 5,
        'radius': None,
        'coord': None
    }, test_name="Sydney - Nearest 5 stations"))
    
    results.append(compare_search({
        'city': 'Melbourne',
        'nearest': None,
        'radius': 50,
        'coord': None
    }, test_name="Melbourne - 50km radius"))
    
    results.append(compare_search({
        'city': 'Brisbane',
        'nearest': 3,
        'radius': None,
        'coord': None
    }, test_name="Brisbane - Nearest 3 stations"))
    
    results.append(compare_search({
        'city': None,
        'nearest': 10,
        'radius': None,
        'coord': (-33.8688, 151.2093)  # Sydney
    }, test_name="Sydney coordinates - Nearest 10"))
    
    results.append(compare_search({
        'city': None,
        'nearest': None,
        'radius': 100,
        'coord': (-37.8136, 144.9631)  # Melbourne
    }, test_name="Melbourne coordinates - 100km radius"))
    
    results.append(compare_search({
        'city': None,
        'nearest': None,
        'radius': 75,
        'coord': (-27.4705, 153.0260)  # Brisbane
    }, test_name="Brisbane coordinates - 75km radius"))
    
    return results

def test_noaa_searches():
    """Run tests with NOAA data."""
    print("\n" + "="*20 + " NOAA Data Tests " + "="*20)
    
    results = []
    
    # US cities and coordinates
    results.append(compare_search({
        'city': 'New York',
        'nearest': 5,
        'radius': None,
        'coord': None
    }, data_type='NOAA', test_name="New York - Nearest 5 stations"))
    
    results.append(compare_search({
        'city': 'Los Angeles',
        'nearest': None,
        'radius': 50,
        'coord': None
    }, data_type='NOAA', test_name="Los Angeles - 50km radius"))
    
    results.append(compare_search({
        'city': None,
        'nearest': 10,
        'radius': None,
        'coord': (40.7128, -74.0060)  # New York
    }, data_type='NOAA', test_name="New York coordinates - Nearest 10"))
    
    results.append(compare_search({
        'city': None,
        'nearest': None,
        'radius': 100,
        'coord': (34.0522, -118.2437)  # Los Angeles
    }, data_type='NOAA', test_name="Los Angeles coordinates - 100km radius"))
    
    results.append(compare_search({
        'city': 'Chicago',
        'nearest': 7,
        'radius': None,
        'coord': None
    }, data_type='NOAA', test_name="Chicago - Nearest 7 stations"))
    
    results.append(compare_search({
        'city': None,
        'nearest': None,
        'radius': 60,
        'coord': (41.8781, -87.6298)  # Chicago
    }, data_type='NOAA', test_name="Chicago coordinates - 60km radius"))
    
    return results

def print_summary(bom_results: list, noaa_results: list):
    """Print summary of all test results."""
    print("\n" + "="*20 + " Overall Summary " + "="*20)
    
    all_results = bom_results + noaa_results
    total_tests = len(all_results)
    matching_counts = sum(1 for r in all_results if r['validation']['count_match'])
    matching_codes = sum(1 for r in all_results if r['validation']['codes_match'])
    matching_coords = sum(1 for r in all_results if r['validation']['coordinates_match'])
    
    print(f"\nTotal tests run: {total_tests}")
    print(f"Tests with matching station counts: {matching_counts}/{total_tests}")
    print(f"Tests with matching station codes: {matching_codes}/{total_tests}")
    print(f"Tests with matching coordinates: {matching_coords}/{total_tests}")
    
    # Performance summary
    legacy_faster = sum(1 for r in all_results if r['legacy_time'] < r['oop_time'])
    oop_faster = total_tests - legacy_faster
    
    print(f"\nLegacy method faster in: {legacy_faster}/{total_tests} tests")
    print(f"OOP method faster in: {oop_faster}/{total_tests} tests")

def main():
    """Run all comparison tests."""
    bom_results = test_bom_searches()
    noaa_results = test_noaa_searches()
    print_summary(bom_results, noaa_results)

if __name__ == '__main__':
    main() 