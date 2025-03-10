#!/usr/bin/env python
"""
Test script to compare the refactored WeatherStationDatabase with the legacy stations module.
"""

import sys
import os
import pandas as pd
from pathlib import Path

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
    print("\nLEGACY VERSION:")
    try:
        legacy_info = legacy.station_info(station_id, printed=False)
        for key, value in legacy_info.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error in legacy version: {e}")
    
    # Get station info from refactored version
    print("\nREFACTORED VERSION:")
    try:
        db = wd.WeatherStationDatabase()
        refactored_info = db.get_station_info(station_id)
        for key, value in refactored_info.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error in refactored version: {e}")

def compare_find_stations_by_city(city, nearest=5):
    """Compare find_stations by city between legacy and refactored versions."""
    print_separator(f"COMPARING FIND STATIONS BY CITY: {city}, NEAREST {nearest}")
    
    # Find stations using legacy version
    print("\nLEGACY VERSION:")
    try:
        legacy_stations = legacy.find_stations(city=city, nearest=nearest, printOutput=False)
        if isinstance(legacy_stations, pd.DataFrame):
            print(f"Found {len(legacy_stations)} stations")
            for i, (_, row) in enumerate(legacy_stations.iterrows()):
                if i < 5:  # Print only first 5 stations
                    print(f"{row['Station ID']} - {row['Station Name']} - {row.get('Distance (km)', 'N/A')} km")
                else:
                    print(f"... and {len(legacy_stations) - 5} more stations")
                    break
        else:
            print("No stations found or unexpected return type")
    except Exception as e:
        print(f"Error in legacy version: {e}")
    
    # Find stations using refactored version
    print("\nREFACTORED VERSION:")
    try:
        db = wd.WeatherStationDatabase()
        refactored_stations = db.find_stations(city=city, nearest=nearest)
        print(f"Found {len(refactored_stations)} stations")
        for i, station in enumerate(refactored_stations):
            if i < 5:  # Print only first 5 stations
                distance = station.data.get('Distance (km)', 'N/A')
                print(f"{station.id} - {station.name} - {distance} km")
            else:
                print(f"... and {len(refactored_stations) - 5} more stations")
                break
    except Exception as e:
        print(f"Error in refactored version: {e}")

def compare_find_stations_by_coordinates(coordinates, radius=50):
    """Compare find_stations by coordinates between legacy and refactored versions."""
    print_separator(f"COMPARING FIND STATIONS BY COORDINATES: {coordinates}, RADIUS {radius} km")
    
    # Find stations using legacy version
    print("\nLEGACY VERSION:")
    try:
        legacy_stations = legacy.find_stations(coord=coordinates, radius=radius, printOutput=False)
        if isinstance(legacy_stations, pd.DataFrame):
            print(f"Found {len(legacy_stations)} stations")
            for i, (_, row) in enumerate(legacy_stations.iterrows()):
                if i < 5:  # Print only first 5 stations
                    print(f"{row['Station ID']} - {row['Station Name']} - {row.get('Distance (km)', 'N/A')} km")
                else:
                    print(f"... and {len(legacy_stations) - 5} more stations")
                    break
        else:
            print("No stations found or unexpected return type")
    except Exception as e:
        print(f"Error in legacy version: {e}")
    
    # Find stations using refactored version
    print("\nREFACTORED VERSION:")
    try:
        db = wd.WeatherStationDatabase()
        refactored_stations = db.find_stations(coordinates=coordinates, radius=radius)
        print(f"Found {len(refactored_stations)} stations")
        for i, station in enumerate(refactored_stations):
            if i < 5:  # Print only first 5 stations
                distance = station.data.get('Distance (km)', 'N/A')
                print(f"{station.id} - {station.name} - {distance} km")
            else:
                print(f"... and {len(refactored_stations) - 5} more stations")
                break
    except Exception as e:
        print(f"Error in refactored version: {e}")

def compare_filter_stations(country="Australia", state="NSW", measurement_type="DB"):
    """Compare filter_stations between legacy and refactored versions."""
    print_separator(f"COMPARING FILTER STATIONS: COUNTRY={country}, STATE={state}, MEASUREMENT={measurement_type}")
    
    # Filter stations using legacy version
    print("\nLEGACY VERSION:")
    try:
        # Create a DataFrame with all stations first
        all_stations = legacy.find_stations(stationID="ALL", printOutput=False)
        # Then filter it
        legacy_stations = legacy.filter_stations(
            all_stations, 
            country=country, 
            state=state, 
            measurementType=measurement_type,
            printOutput=False
        )
        if isinstance(legacy_stations, pd.DataFrame):
            print(f"Found {len(legacy_stations)} stations")
            for i, (_, row) in enumerate(legacy_stations.iterrows()):
                if i < 5:  # Print only first 5 stations
                    print(f"{row['Station ID']} - {row['Station Name']} - {row.get('State', 'N/A')}")
                else:
                    print(f"... and {len(legacy_stations) - 5} more stations")
                    break
        else:
            print("No stations found or unexpected return type")
    except Exception as e:
        print(f"Error in legacy version: {e}")
    
    # Filter stations using refactored version
    print("\nREFACTORED VERSION:")
    try:
        db = wd.WeatherStationDatabase()
        refactored_stations = db.filter_stations(
            country=country,
            state=state,
            measurement_type=measurement_type
        )
        print(f"Found {len(refactored_stations)} stations")
        for i, station in enumerate(refactored_stations):
            if i < 5:  # Print only first 5 stations
                print(f"{station.id} - {station.name} - {station.state}")
            else:
                print(f"... and {len(refactored_stations) - 5} more stations")
                break
    except Exception as e:
        print(f"Error in refactored version: {e}")

def list_available_stations():
    """List available station IDs from both versions."""
    print_separator("LISTING AVAILABLE STATION IDs")
    
    # List stations using legacy version
    print("\nLEGACY VERSION:")
    try:
        # Get all stations
        legacy_stations = legacy.find_stations(stationID="ALL", printOutput=False)
        if isinstance(legacy_stations, pd.DataFrame):
            station_ids = legacy_stations['Station ID'].unique()
            print(f"Found {len(station_ids)} stations")
            print(f"Sample IDs: {', '.join(station_ids[:5])}...")
        else:
            print("No stations found or unexpected return type")
    except Exception as e:
        print(f"Error in legacy version: {e}")
    
    # List stations using refactored version
    print("\nREFACTORED VERSION:")
    try:
        db = wd.WeatherStationDatabase()
        stations = db.get_all_stations()
        station_ids = [station.id for station in stations]
        print(f"Found {len(station_ids)} stations")
        print(f"Sample IDs: {', '.join(station_ids[:5])}...")
    except Exception as e:
        print(f"Error in refactored version: {e}")

def main():
    """Run all comparison tests."""
    print("Starting station database comparison tests...")
    
    # List available stations
    list_available_stations()
    
    # Test station info
    compare_station_info("001006")  # BOM station
    
    # Test find stations by city
    compare_find_stations_by_city("Sydney", nearest=5)
    
    # Test find stations by coordinates
    compare_find_stations_by_coordinates((-33.8688, 151.2093), radius=50)  # Sydney coordinates
    
    # Test filter stations
    compare_filter_stations(country="Australia", state="NSW", measurement_type="DB")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 