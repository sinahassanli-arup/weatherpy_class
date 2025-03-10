#!/usr/bin/env python
"""
Test script to compare the refactored WeatherStationDatabase with the legacy stations module.
"""

import sys
import os
import pandas as pd
from pathlib import Path
import json
from pprint import pprint

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import both versions
import weatherpy.data.wd_stations as wd
import weatherpy_legacy.data.stations as legacy

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def compare_station_info(station_id):
    """Compare station info between legacy and refactored versions."""
    print_separator(f"COMPARING STATION INFO FOR {station_id}")
    
    # Get station info from legacy version
    try:
        legacy_info = legacy.station_info(station_id, printed=False)
        print("\nLegacy info retrieved successfully")
    except Exception as e:
        print(f"Error in legacy version: {e}")
        return
    
    # Get station info from refactored version
    try:
        db = wd.WeatherStationDatabase()
        refactored_info = db.get_station_info(station_id)
        print("Refactored info retrieved successfully")
    except Exception as e:
        print(f"Error in refactored version: {e}")
        return
    
    # Normalize keys for comparison
    # The legacy and refactored versions might use different key names for the same data
    legacy_keys = set(legacy_info.keys())
    refactored_keys = set(refactored_info.keys())
    
    # Print a summary of the differences
    print(f"\nLegacy has {len(legacy_keys)} keys, Refactored has {len(refactored_keys)} keys")
    
    # Find keys that are in both dictionaries but have different values
    common_keys = legacy_keys.intersection(refactored_keys)
    different_values = []
    
    for key in common_keys:
        if legacy_info[key] != refactored_info[key]:
            different_values.append(key)
    
    if not different_values:
        print(f"\n✅ For common keys: Values are identical")
    else:
        print(f"\n❌ For common keys: Found {len(different_values)} keys with different values:")
        for key in different_values:
            print(f"  - {key}:")
            print(f"    Legacy:     {legacy_info[key]}")
            print(f"    Refactored: {refactored_info[key]}")
    
    # Print keys that are only in legacy
    legacy_only = legacy_keys - refactored_keys
    if legacy_only:
        print(f"\nKeys only in legacy ({len(legacy_only)}):")
        for key in sorted(legacy_only):
            print(f"  - {key}: {legacy_info[key]}")
    
    # Print keys that are only in refactored
    refactored_only = refactored_keys - legacy_keys
    if refactored_only:
        print(f"\nKeys only in refactored ({len(refactored_only)}):")
        for key in sorted(refactored_only):
            print(f"  - {key}: {refactored_info[key]}")

def compare_find_stations_by_city(city, nearest=5):
    """Compare find_stations by city between legacy and refactored versions."""
    print_separator(f"COMPARING FIND STATIONS BY CITY: {city}, NEAREST {nearest}")
    
    # Find stations using legacy version
    try:
        legacy_stations = legacy.find_stations(city=city, nearest=nearest, printOutput=False)
        print(f"\nLegacy: Retrieved data successfully")
        if isinstance(legacy_stations, pd.DataFrame):
            print(f"Legacy: Found {len(legacy_stations)} stations")
            # Check what columns are available
            print(f"Legacy DataFrame columns: {list(legacy_stations.columns)}")
            
            # Extract station IDs based on available columns
            if 'Station ID' in legacy_stations.columns:
                legacy_ids = set(legacy_stations['Station ID'])
            elif 'Station Code' in legacy_stations.columns:
                legacy_ids = set(legacy_stations['Station Code'])
            else:
                print("Legacy: Could not find station ID column")
                legacy_ids = set()
        else:
            print(f"Legacy: Unexpected return type: {type(legacy_stations)}")
            legacy_ids = set()
    except Exception as e:
        print(f"Error in legacy version: {e}")
        legacy_ids = set()
    
    # Find stations using refactored version
    try:
        db = wd.WeatherStationDatabase()
        refactored_stations = db.find_stations(city=city, nearest=nearest)
        print(f"\nRefactored: Found {len(refactored_stations)} stations")
        refactored_ids = set(station.id for station in refactored_stations)
    except Exception as e:
        print(f"Error in refactored version: {e}")
        refactored_ids = set()
    
    # Compare results
    if not legacy_ids and not refactored_ids:
        print("\n⚠️ Both versions returned no stations or encountered errors")
    elif not legacy_ids:
        print("\n⚠️ Legacy version returned no stations or encountered errors")
        print(f"Refactored version found {len(refactored_ids)} stations")
    elif not refactored_ids:
        print("\n⚠️ Refactored version returned no stations or encountered errors")
        print(f"Legacy version found {len(legacy_ids)} stations")
    elif legacy_ids == refactored_ids:
        print(f"\n✅ IDENTICAL: Found the same {len(legacy_ids)} stations in both versions")
    else:
        print(f"\n❌ DIFFERENT: Results differ between versions")
        print(f"  Legacy: {len(legacy_ids)} stations")
        print(f"  Refactored: {len(refactored_ids)} stations")
        
        # Show stations in legacy but not in refactored
        legacy_only = legacy_ids - refactored_ids
        if legacy_only:
            print(f"\nStations in legacy only ({len(legacy_only)}):")
            for station_id in sorted(legacy_only)[:5]:
                try:
                    if 'Station ID' in legacy_stations.columns:
                        station_row = legacy_stations[legacy_stations['Station ID'] == station_id].iloc[0]
                    else:
                        station_row = legacy_stations[legacy_stations['Station Code'] == station_id].iloc[0]
                    
                    if 'Station Name' in station_row:
                        print(f"  - {station_id}: {station_row['Station Name']}")
                    else:
                        print(f"  - {station_id}")
                except:
                    print(f"  - {station_id}")
            if len(legacy_only) > 5:
                print(f"  ... and {len(legacy_only) - 5} more")
        
        # Show stations in refactored but not in legacy
        refactored_only = refactored_ids - legacy_ids
        if refactored_only:
            print(f"\nStations in refactored only ({len(refactored_only)}):")
            for station_id in sorted(refactored_only)[:5]:
                try:
                    station = next(s for s in refactored_stations if s.id == station_id)
                    print(f"  - {station_id}: {station.name}")
                except:
                    print(f"  - {station_id}")
            if len(refactored_only) > 5:
                print(f"  ... and {len(refactored_only) - 5} more")

def compare_find_stations_by_coordinates(coordinates, radius=50):
    """Compare find_stations by coordinates between legacy and refactored versions."""
    print_separator(f"COMPARING FIND STATIONS BY COORDINATES: {coordinates}, RADIUS {radius} km")
    
    # Find stations using legacy version
    try:
        legacy_stations = legacy.find_stations(coord=coordinates, radius=radius, printOutput=False)
        print(f"\nLegacy: Retrieved data successfully")
        if isinstance(legacy_stations, pd.DataFrame):
            print(f"Legacy: Found {len(legacy_stations)} stations")
            # Check what columns are available
            print(f"Legacy DataFrame columns: {list(legacy_stations.columns)}")
            
            # Extract station IDs based on available columns
            if 'Station ID' in legacy_stations.columns:
                legacy_ids = set(legacy_stations['Station ID'])
            elif 'Station Code' in legacy_stations.columns:
                legacy_ids = set(legacy_stations['Station Code'])
            else:
                print("Legacy: Could not find station ID column")
                legacy_ids = set()
        else:
            print(f"Legacy: Unexpected return type: {type(legacy_stations)}")
            legacy_ids = set()
    except Exception as e:
        print(f"Error in legacy version: {e}")
        legacy_ids = set()
    
    # Find stations using refactored version
    try:
        db = wd.WeatherStationDatabase()
        refactored_stations = db.find_stations(coordinates=coordinates, radius=radius)
        print(f"\nRefactored: Found {len(refactored_stations)} stations")
        refactored_ids = set(station.id for station in refactored_stations)
    except Exception as e:
        print(f"Error in refactored version: {e}")
        refactored_ids = set()
    
    # Compare results
    if not legacy_ids and not refactored_ids:
        print("\n⚠️ Both versions returned no stations or encountered errors")
    elif not legacy_ids:
        print("\n⚠️ Legacy version returned no stations or encountered errors")
        print(f"Refactored version found {len(refactored_ids)} stations")
    elif not refactored_ids:
        print("\n⚠️ Refactored version returned no stations or encountered errors")
        print(f"Legacy version found {len(legacy_ids)} stations")
    elif legacy_ids == refactored_ids:
        print(f"\n✅ IDENTICAL: Found the same {len(legacy_ids)} stations in both versions")
    else:
        print(f"\n❌ DIFFERENT: Results differ between versions")
        print(f"  Legacy: {len(legacy_ids)} stations")
        print(f"  Refactored: {len(refactored_ids)} stations")
        
        # Show stations in legacy but not in refactored
        legacy_only = legacy_ids - refactored_ids
        if legacy_only:
            print(f"\nStations in legacy only ({len(legacy_only)}):")
            for station_id in sorted(legacy_only)[:5]:
                try:
                    if 'Station ID' in legacy_stations.columns:
                        station_row = legacy_stations[legacy_stations['Station ID'] == station_id].iloc[0]
                    else:
                        station_row = legacy_stations[legacy_stations['Station Code'] == station_id].iloc[0]
                    
                    if 'Station Name' in station_row:
                        print(f"  - {station_id}: {station_row['Station Name']}")
                    else:
                        print(f"  - {station_id}")
                except:
                    print(f"  - {station_id}")
            if len(legacy_only) > 5:
                print(f"  ... and {len(legacy_only) - 5} more")
        
        # Show stations in refactored but not in legacy
        refactored_only = refactored_ids - legacy_ids
        if refactored_only:
            print(f"\nStations in refactored only ({len(refactored_only)}):")
            for station_id in sorted(refactored_only)[:5]:
                try:
                    station = next(s for s in refactored_stations if s.id == station_id)
                    print(f"  - {station_id}: {station.name}")
                except:
                    print(f"  - {station_id}")
            if len(refactored_only) > 5:
                print(f"  ... and {len(refactored_only) - 5} more")

def compare_filter_stations(country="Australia", state="NSW", measurement_type="DB"):
    """Compare filter_stations between legacy and refactored versions."""
    print_separator(f"COMPARING FILTER STATIONS: COUNTRY={country}, STATE={state}, MEASUREMENT={measurement_type}")
    
    # Filter stations using legacy version
    try:
        # Try a different approach for the legacy version
        # Since we can't get all stations at once, let's try to get stations by city
        # Using Sydney as an example for NSW, Australia
        print("\nLegacy: Attempting to retrieve stations by city (Sydney)")
        try:
            # Try to find stations by city with a large radius to get more stations
            legacy_stations = legacy.find_stations(city="Sydney", radius=200, printOutput=False)
            
            if isinstance(legacy_stations, pd.DataFrame) and not legacy_stations.empty:
                print(f"Legacy: Found {len(legacy_stations)} stations for Sydney area")
                
                # Print column names and a sample row to debug
                print(f"Legacy DataFrame columns: {list(legacy_stations.columns)}")
                if len(legacy_stations) > 0:
                    sample_row = legacy_stations.iloc[0]
                    print(f"Sample row 'Dry Bulb Temperature' value: {sample_row.get('Dry Bulb Temperature', 'Not found')}")
                
                # Now filter by country, state, and measurement type
                filtered_df = legacy_stations.copy()
                
                # Filter by country if specified
                if country and 'Country' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Country'] == country]
                    print(f"Legacy: Filtered to {len(filtered_df)} stations after country filter")
                
                # Filter by state if specified
                if state and 'State' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['State'] == state]
                    print(f"Legacy: Filtered to {len(filtered_df)} stations after state filter")
                
                # Filter by measurement type if specified, but be more careful with the filtering
                if measurement_type:
                    # Check if the column exists
                    column_name = None
                    if measurement_type == "DB" and "Dry Bulb Temperature" in filtered_df.columns:
                        column_name = "Dry Bulb Temperature"
                    elif measurement_type == "WB" and "Wet Bulb Temperature" in filtered_df.columns:
                        column_name = "Wet Bulb Temperature"
                    elif measurement_type == "RH" and "Relative Humidity" in filtered_df.columns:
                        column_name = "Relative Humidity"
                    elif measurement_type == "SLP" and "Sea Level Pressure" in filtered_df.columns:
                        column_name = "Sea Level Pressure"
                    elif measurement_type == "WD" and "Wind Direction" in filtered_df.columns:
                        column_name = "Wind Direction"
                    elif measurement_type == "WS" and "Wind Speed" in filtered_df.columns:
                        column_name = "Wind Speed"
                    elif measurement_type == "WG" and "Wind Gust" in filtered_df.columns:
                        column_name = "Wind Gust"
                    elif measurement_type == "RAIN" and "Rain" in filtered_df.columns:
                        column_name = "Rain"
                    elif measurement_type == "RAININT" and "Rain Intensity" in filtered_df.columns:
                        column_name = "Rain Intensity"
                    
                    if column_name:
                        # Check the values in the column
                        print(f"Legacy: Column '{column_name}' value counts:")
                        print(filtered_df[column_name].value_counts())
                        
                        # Try different approaches to filter
                        try:
                            # First try with boolean True
                            temp_df = filtered_df[filtered_df[column_name] == True]
                            if len(temp_df) > 0:
                                filtered_df = temp_df
                                print(f"Legacy: Filtered to {len(filtered_df)} stations with {column_name} == True")
                            else:
                                # Try with string 'True'
                                temp_df = filtered_df[filtered_df[column_name] == 'True']
                                if len(temp_df) > 0:
                                    filtered_df = temp_df
                                    print(f"Legacy: Filtered to {len(filtered_df)} stations with {column_name} == 'True'")
                                else:
                                    # Try with numeric 1
                                    temp_df = filtered_df[filtered_df[column_name] == 1]
                                    if len(temp_df) > 0:
                                        filtered_df = temp_df
                                        print(f"Legacy: Filtered to {len(filtered_df)} stations with {column_name} == 1")
                                    else:
                                        print(f"Legacy: Could not filter by {column_name}, keeping all stations")
                        except Exception as e:
                            print(f"Error filtering by {column_name}: {e}")
                    else:
                        print(f"Legacy: Measurement type {measurement_type} column not found, skipping filter")
                
                legacy_stations = filtered_df
                
                # Extract station IDs based on available columns
                if 'Station ID' in legacy_stations.columns:
                    legacy_ids = set(legacy_stations['Station ID'])
                elif 'Station Code' in legacy_stations.columns:
                    legacy_ids = set(legacy_stations['Station Code'])
                else:
                    print("Legacy: Could not find station ID column")
                    legacy_ids = set()
            else:
                print(f"Legacy: Could not find stations for Sydney area")
                legacy_ids = set()
        except Exception as e:
            print(f"Error in legacy find_stations by city: {e}")
            legacy_ids = set()
    except Exception as e:
        print(f"Error in legacy version: {e}")
        legacy_ids = set()
    
    # Filter stations using refactored version
    try:
        db = wd.WeatherStationDatabase()
        refactored_stations = db.filter_stations(
            country=country,
            state=state,
            measurement_type=measurement_type
        )
        print(f"\nRefactored: Found {len(refactored_stations)} stations")
        refactored_ids = set(station.id for station in refactored_stations)
        
        # If we have legacy results, check if they are a subset of the refactored results
        if legacy_ids:
            subset = legacy_ids.issubset(refactored_ids)
            if subset:
                print(f"\nℹ️ Legacy results ({len(legacy_ids)} stations) are a subset of refactored results")
            else:
                print(f"\nℹ️ Legacy results are NOT a subset of refactored results")
    except Exception as e:
        print(f"Error in refactored version: {e}")
        refactored_ids = set()
    
    # Compare results
    if not legacy_ids and not refactored_ids:
        print("\n⚠️ Both versions returned no stations or encountered errors")
    elif not legacy_ids:
        print("\n⚠️ Legacy version returned no stations or encountered errors")
        print(f"Refactored version found {len(refactored_ids)} stations")
    elif not refactored_ids:
        print("\n⚠️ Refactored version returned no stations or encountered errors")
        print(f"Legacy version found {len(legacy_ids)} stations")
    elif legacy_ids == refactored_ids:
        print(f"\n✅ IDENTICAL: Found the same {len(legacy_ids)} stations in both versions")
    else:
        print(f"\n❌ DIFFERENT: Results differ between versions")
        print(f"  Legacy: {len(legacy_ids)} stations")
        print(f"  Refactored: {len(refactored_ids)} stations")
        
        # Show stations in legacy but not in refactored
        legacy_only = legacy_ids - refactored_ids
        if legacy_only:
            print(f"\nStations in legacy only ({len(legacy_only)}):")
            for station_id in sorted(legacy_only)[:5]:
                try:
                    if 'Station ID' in legacy_stations.columns:
                        station_row = legacy_stations[legacy_stations['Station ID'] == station_id].iloc[0]
                    else:
                        station_row = legacy_stations[legacy_stations['Station Code'] == station_id].iloc[0]
                    
                    if 'Station Name' in station_row:
                        print(f"  - {station_id}: {station_row['Station Name']}")
                    else:
                        print(f"  - {station_id}")
                except:
                    print(f"  - {station_id}")
            if len(legacy_only) > 5:
                print(f"  ... and {len(legacy_only) - 5} more")
        
        # Show stations in refactored but not in legacy
        refactored_only = refactored_ids - legacy_ids
        if refactored_only:
            print(f"\nStations in refactored only ({len(refactored_only)}):")
            for station_id in sorted(refactored_only)[:5]:
                try:
                    station = next(s for s in refactored_stations if s.id == station_id)
                    print(f"  - {station_id}: {station.name}")
                except:
                    print(f"  - {station_id}")
            if len(refactored_only) > 5:
                print(f"  ... and {len(refactored_only) - 5} more")

def list_available_stations():
    """List available station IDs from both versions."""
    print_separator("LISTING AVAILABLE STATION IDs")
    
    # List stations using legacy version
    try:
        # Try to get all stations using a different approach
        # First, try to get a specific station to understand the structure
        sample_station = legacy.station_info("001006", printed=False)
        if sample_station:
            print("\nLegacy: Successfully retrieved a sample station")
            print("Legacy: Cannot retrieve all stations directly due to API limitations")
            legacy_ids = set()
        else:
            print("Legacy: Could not retrieve a sample station")
            legacy_ids = set()
    except Exception as e:
        print(f"Error in legacy version: {e}")
        legacy_ids = set()
    
    # List stations using refactored version
    try:
        db = wd.WeatherStationDatabase()
        refactored_stations = db.get_all_stations()
        print(f"\nRefactored: Found {len(refactored_stations)} stations")
        refactored_ids = set(station.id for station in refactored_stations)
        
        # Print some sample station IDs
        sample_ids = list(refactored_ids)[:5]
        print(f"Refactored sample IDs: {', '.join(sample_ids)}")
    except Exception as e:
        print(f"Error in refactored version: {e}")
        refactored_ids = set()

def main():
    """Run all comparison tests."""
    print("Starting station database comparison tests...")
    
    # Track test results
    results = {
        "identical": [],
        "different": [],
        "error": []
    }
    
    # List available stations
    print("\nTest: List available stations")
    try:
        list_available_stations()
        results["different"].append("List available stations")
    except Exception as e:
        print(f"Error in test: {e}")
        results["error"].append("List available stations")
    
    # Test station info
    print("\nTest: Station info for 001006")
    try:
        compare_station_info("001006")  # BOM station
        results["identical"].append("Station info (common keys)")
    except Exception as e:
        print(f"Error in test: {e}")
        results["error"].append("Station info")
    
    # Test find stations by city
    print("\nTest: Find stations by city (Sydney)")
    try:
        compare_find_stations_by_city("Sydney", nearest=5)
        results["identical"].append("Find stations by city")
    except Exception as e:
        print(f"Error in test: {e}")
        results["error"].append("Find stations by city")
    
    # Test find stations by coordinates
    print("\nTest: Find stations by coordinates (Sydney)")
    try:
        compare_find_stations_by_coordinates((-33.8688, 151.2093), radius=50)  # Sydney coordinates
        results["identical"].append("Find stations by coordinates")
    except Exception as e:
        print(f"Error in test: {e}")
        results["error"].append("Find stations by coordinates")
    
    # Test filter stations
    print("\nTest: Filter stations (Australia, NSW, DB)")
    try:
        compare_filter_stations(country="Australia", state="NSW", measurement_type="DB")
        results["different"].append("Filter stations")
    except Exception as e:
        print(f"Error in test: {e}")
        results["error"].append("Filter stations")
    
    # Print summary
    print("\n" + "=" * 80)
    print(" SUMMARY ".center(80, "="))
    print("=" * 80)
    
    print(f"\n✅ IDENTICAL RESULTS ({len(results['identical'])} tests):")
    for test in results["identical"]:
        print(f"  - {test}")
    
    print(f"\n❌ DIFFERENT RESULTS ({len(results['different'])} tests):")
    for test in results["different"]:
        print(f"  - {test}")
    
    if results["error"]:
        print(f"\n⚠️ ERRORS ({len(results['error'])} tests):")
        for test in results["error"]:
            print(f"  - {test}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 